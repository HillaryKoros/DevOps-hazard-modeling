from prefect import flow, task
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import xarray as xr
from dask.distributed import Client
import geopandas as gp
import pandas as pd
import numpy as np
import rioxarray
import xesmf as xe
from dask.diagnostics import ProgressBar

from utils import (
    imerg_list_files_by_date,
    imerg_download_files,
    imerg_read_tiffs_to_dataset,
    get_dask_client_params,
    make_zones_geotif,
    imerg_update_input_data,
    process_zone_from_combined,
    regrid_dataset,
    zone_mean_df
)

load_dotenv()

# Default to yesterday if date is not provided
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

@task
def setup_environment():
    """Set up the environment for data processing"""
    data_path = os.getenv("data_path", "./data/")  # Default to ./data/ if not set
    imerg_store = f'{data_path}geofsm-input/imerg'
    zone_input_path = f"{data_path}zone_wise_txt_files/"
    init_zone_path = f"{data_path}zone_wise_txt_files/init/"
    
    # Create all necessary directories
    os.makedirs(imerg_store, exist_ok=True)
    os.makedirs(zone_input_path, exist_ok=True)
    os.makedirs(init_zone_path, exist_ok=True)
    
    params = get_dask_client_params()
    client = Client(**params)
    
    print(f"Environment setup complete. Using data_path: {data_path}")
    print(f"Created standard output directory: {zone_input_path}")
    print(f"Created base output directory (no forecast): {init_zone_path}")
    return data_path, imerg_store, client

def get_last_date_from_rain(zone_dir, is_init=False):
    """
    Read the existing rain.txt file and determine the last date in the file.
    
    Parameters:
    ----------
    zone_dir : str
        Path to the zone directory containing rain.txt
    is_init : bool
        Flag indicating if this is the init directory (for logging purposes)
        
    Returns:
    -------
    datetime
        The last date in the file, or None if the file doesn't exist or can't be read
    """
    rain_file = os.path.join(zone_dir, 'rain.txt')
    dir_type = "init" if is_init else "standard"
    
    if not os.path.exists(rain_file):
        print(f"No existing rain.txt found at {rain_file} ({dir_type} directory)")
        return None
    
    try:
        # Read the rain.txt file
        df = pd.read_csv(rain_file, sep=",")
        
        # Check if NA column exists (which contains the dates in YYYYDDD format)
        if 'NA' not in df.columns:
            print(f"Invalid format in rain.txt ({dir_type} directory) - missing 'NA' column")
            return None
        
        # Convert the last date to datetime
        last_date_str = df['NA'].iloc[-1]
        last_date = datetime.strptime(str(last_date_str), '%Y%j')
        
        print(f"Last date in {dir_type} rain.txt: {last_date.strftime('%Y-%m-%d')} (Day {last_date_str})")
        return last_date
        
    except Exception as e:
        print(f"Error reading existing rain.txt ({dir_type} directory): {e}")
        return None

def imerg_extend_forecast_improved(df, date_column, days_to_add=16):
    """
    Add a forecast extension by copying the last 15 days of data and appending it
    to create a 16-day forecast.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    date_column (str): Name of the column containing dates in 'YYYYDDD' format
    days_to_add (int): Number of days to add for forecast (default is 16)
    
    Returns:
    pd.DataFrame: DataFrame with additional forecast rows
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    df = df.copy()
    
    # Function to safely convert date string to datetime
    def safe_to_datetime(date_str):
        try:
            return datetime.strptime(str(date_str), '%Y%j')
        except ValueError:
            return None

    # Convert date column to datetime for processing
    df['_temp_date'] = df[date_column].apply(safe_to_datetime)
    
    # Remove any rows where the date conversion failed
    df = df.dropna(subset=['_temp_date'])
    
    if df.empty:
        print(f"No valid dates found in the '{date_column}' column.")
        return df
        
    # Sort by date to ensure correct order
    df = df.sort_values('_temp_date')
    
    # Get the last 15 days of data (or fewer if less available)
    days_to_copy = min(15, len(df))
    historical_pattern = df.iloc[-days_to_copy:].copy()
    
    # Create new rows for forecast
    new_rows = []
    last_date = df['_temp_date'].iloc[-1]
    
    for i in range(days_to_add):
        # Calculate the new date
        new_date = last_date + timedelta(days=i+1)
        
        # Get corresponding historical row (cycling through the pattern)
        historical_idx = i % len(historical_pattern)
        new_row = historical_pattern.iloc[historical_idx].copy()
        
        # Update the date
        new_row['_temp_date'] = new_date
        new_rows.append(new_row)
    
    # Convert new_rows to a DataFrame
    new_rows_df = pd.DataFrame(new_rows)
    
    # Concatenate the new rows to the original DataFrame
    result_df = pd.concat([df, new_rows_df], ignore_index=True)
    
    # Convert date column back to the original string format and remove temp column
    result_df[date_column] = result_df['_temp_date'].dt.strftime('%Y%j')
    result_df = result_df.drop(columns=['_temp_date'])
    
    return result_df

def imerg_update_input_data_improved(z1a, zone_input_path, zone_str, start_date, end_date):
    """
    Processes precipitation data and generates:
    1. Standard rain.txt and zone-specific rain_zone*.txt files (without forecast)
    2. Base rain.txt and zone-specific rain_zone*.txt files (without forecast) in 'init' folder
    
    For IMERG data, we don't add a forecast extension to either set of files.
    
    Parameters:
    ----------
    z1a : pandas.DataFrame
        Dataframe containing IMERG data that needs to be adjusted, pivoted, and formatted.
    zone_input_path : str
        Base path for input and output data files related to specific zones.
    zone_str : str
        Identifier for the specific zone, used for file naming and directory structure.
    start_date : datetime
        Start date for filtering the dataset.
    end_date : datetime
        End date for filtering the dataset.

    Returns:
    -------
    tuple
        Paths to the four generated files (standard rain.txt, zone-specific rain file, 
        base rain.txt without forecast, and base zone-specific rain file without forecast).
    """
    # Ensure all directories exist
    zone_dir = f'{zone_input_path}{zone_str}'
    init_zone_dir = f'{zone_input_path}init/{zone_str}'
    
    # Create directories recursively if they don't exist
    os.makedirs(zone_dir, exist_ok=True)
    os.makedirs(init_zone_dir, exist_ok=True)
    
    print(f"Ensuring output directories exist for {zone_str}:")
    print(f"  - Standard directory: {zone_dir}")
    print(f"  - Base directory (no forecast): {init_zone_dir}")
    
    # Process the data - assuming 'precipitation' is the column name in z1a
    # If the column name is different, adjust this as needed
    if 'precipitation' in z1a.columns:
        # Convert the precipitation to the format needed
        # This is where you would apply any scaling factors if needed
        # Example: z1a['precipitation'] = z1a['precipitation'] * scaling_factor
        pass
    
    # Pivot the DataFrame
    zz1 = z1a.pivot(index='time', columns='group', values='precipitation')
    
    # Apply formatting to the pivoted DataFrame
    zz1 = zz1.apply(lambda row: row.map(lambda x: f'{x:.1f}' if isinstance(x, (int, float)) and pd.notna(x) else x), axis=1)
    
    # Reset the index and adjust columns
    azz1 = zz1.reset_index()
    azz1['NA'] = azz1['time'].dt.strftime('%Y%j')
    azz1.columns = [str(col) if isinstance(col, int) else col for col in azz1.columns]
    azz1 = azz1.rename(columns={'time': 'date'})
    
    # Path to standard rain.txt file in zone_wise directory
    rain_file = f'{zone_dir}/rain.txt'
    
    # Path to base rain.txt file in init directory (without forecast)
    base_rain_file = f'{init_zone_dir}/rain.txt'
    
    # Check if the standard rain.txt file exists
    if os.path.exists(rain_file):
        # If file exists, read and merge with new data
        try:
            ez1 = pd.read_csv(rain_file, sep=",")
            ez1['date'] = pd.to_datetime(ez1['NA'], format='%Y%j')
            
            # Create a mask for filtering data
            mask = (ez1['date'] < start_date) | (ez1['date'] > end_date)
            aez1 = ez1[mask]
            
            # Concatenate DataFrames
            bz1 = pd.concat([aez1, azz1], axis=0)
            
            # Reset index and drop unnecessary columns
            bz1.drop(['date'], axis=1, inplace=True)
            bz1.reset_index(drop=True, inplace=True)
        except Exception as e:
            print(f"Error reading existing rain.txt: {e}")
            print("Creating new rain.txt file instead")
            bz1 = azz1.drop(['date'], axis=1).reset_index(drop=True)
    else:
        # If file doesn't exist, just use the new data
        print(f"No existing rain.txt found at {rain_file}. Creating new file.")
        bz1 = azz1.drop(['date'], axis=1).reset_index(drop=True)
    
    # Do the same for the base rain file (without forecast)
    if os.path.exists(base_rain_file):
        try:
            base_ez1 = pd.read_csv(base_rain_file, sep=",")
            base_ez1['date'] = pd.to_datetime(base_ez1['NA'], format='%Y%j')
            
            # Create a mask for filtering data
            mask = (base_ez1['date'] < start_date) | (base_ez1['date'] > end_date)
            base_aez1 = base_ez1[mask]
            
            # Concatenate DataFrames
            base_bz1 = pd.concat([base_aez1, azz1], axis=0)
            
            # Reset index and drop unnecessary columns
            base_bz1.drop(['date'], axis=1, inplace=True)
            base_bz1.reset_index(drop=True, inplace=True)
        except Exception as e:
            print(f"Error reading existing base rain.txt: {e}")
            print("Creating new base rain.txt file instead")
            base_bz1 = azz1.drop(['date'], axis=1).reset_index(drop=True)
    else:
        # If file doesn't exist, just use the new data
        print(f"No existing base rain.txt found at {base_rain_file}. Creating new file.")
        base_bz1 = azz1.drop(['date'], axis=1).reset_index(drop=True)
    
    # Ensure all values in NA column are strings for consistent sorting
    if 'NA' in bz1.columns:
        bz1['NA'] = bz1['NA'].astype(str)
    if 'NA' in base_bz1.columns:
        base_bz1['NA'] = base_bz1['NA'].astype(str)
    
    # Sort the data by NA column (date) to ensure proper order
    bz1 = bz1.sort_values(by='NA').reset_index(drop=True)
    base_bz1 = base_bz1.sort_values(by='NA').reset_index(drop=True)
    
    # For IMERG, we do NOT add a forecast extension to either set of files
    # Both standard and base files will be identical
    
    # Create standard files (without forecast for IMERG)
    
    # 1. Standard rain.txt file
    bz1.to_csv(rain_file, index=False)
    print(f"Created/updated standard rain.txt file: {rain_file}")
    
    # 2. Zone-specific rain file (rain_zone1.txt)
    zone_specific_file = f'{zone_dir}/rain_{zone_str}.txt'
    bz1.to_csv(zone_specific_file, index=False)
    print(f"Created zone-specific rain file: {zone_specific_file}")
    
    # Create base files (also without forecast)
    
    # 3. Base rain.txt file
    base_bz1.to_csv(base_rain_file, index=False)
    print(f"Created/updated base rain.txt file: {base_rain_file}")
    
    # 4. Zone-specific base rain file
    base_zone_specific_file = f'{init_zone_dir}/rain_{zone_str}.txt'
    base_bz1.to_csv(base_zone_specific_file, index=False)
    print(f"Created base zone-specific rain file: {base_zone_specific_file}")
    
    return rain_file, zone_specific_file, base_rain_file, base_zone_specific_file

@task
def get_imerg_files(start_date, end_date):
    """Get a list of IMERG files for the specified date range"""
    url = "https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early/"
    flt_str = '-S233000-E235959.1410.V07B.1day.tif'
    username = os.getenv("imerg_username")
    password = os.getenv("imerg_password")
    
    if not username or not password:
        raise ValueError("IMERG credentials not found in environment variables")
    
    file_list = imerg_list_files_by_date(url, flt_str, username, password, start_date, end_date)
    print(f"Found {len(file_list)} IMERG files for date range {start_date} to {end_date}")
    return file_list

@task
def download_imerg_files(file_list, imerg_store):
    """Download IMERG files"""
    download_dir = f"{imerg_store}"
    os.makedirs(download_dir, exist_ok=True)
    
    # Check if files already exist
    existing_files = set(os.listdir(download_dir))
    to_download = []
    
    for url in file_list:
        filename = os.path.basename(url)
        if filename not in existing_files:
            to_download.append(url)
    
    if not to_download:
        print(f"All IMERG files already exist in {download_dir}, skipping download.")
    else:
        print(f"Downloading {len(to_download)} new IMERG files...")
        username = os.getenv("imerg_username")
        password = os.getenv("imerg_password")
        imerg_download_files(to_download, username, password, download_dir)
    
    return download_dir

@task
def process_imerg_data(input_path, start_date, end_date):
    """Process IMERG data into xarray format"""
    print(f"Processing IMERG data from {input_path} for {start_date} to {end_date}")
    data = imerg_read_tiffs_to_dataset(input_path, start_date, end_date)
    
    # If the data is a DataArray, assign a name to it
    if isinstance(data, xr.DataArray) and not data.name:
        data = data.rename('precipitation')
        print(f"Assigned name 'precipitation' to DataArray")
    
    return data

@task
def rename_coordinates(imerg_data):
    """
    Rename 'x' and 'y' coordinates to 'lon' and 'lat' if they exist.
    This ensures compatibility with other functions expecting lat/lon.
    Works with both xarray.DataArray and xarray.Dataset objects.
    """
    # Print the object type and dimensions to help with debugging
    print(f"Object type: {type(imerg_data).__name__}")
    print(f"Dimensions: {list(imerg_data.dims)}")
    print(f"Coordinates: {list(imerg_data.coords)}")
    print(f"Data name: {getattr(imerg_data, 'name', 'unnamed')}")
    
    # Check if x and y are present in the dimensions
    if 'x' in imerg_data.dims and 'y' in imerg_data.dims:
        # Create a new object with renamed coordinates
        renamed_data = imerg_data.rename({'x': 'lon', 'y': 'lat'})
        print("Renamed 'x' to 'lon' and 'y' to 'lat'")
    else:
        renamed_data = imerg_data
        print("No renaming needed or coordinates not found")
    
    # Print the dimensions after renaming to confirm
    print(f"Renamed dimensions: {list(renamed_data.dims)}")
    print(f"Renamed coordinates: {list(renamed_data.coords)}")
    
    return renamed_data

@task
def process_zone(data_path, imerg_data, zone_str):
    """Process a zone from the combined shapefile"""
    master_shapefile = f'{data_path}WGS/geofsm-prod-all-zones-20240712.shp'
    km_str = 1
    z1ds, zone_subset_ds, zone_extent = process_zone_from_combined(master_shapefile, zone_str, km_str, imerg_data)
    print(f"Processed zone {zone_str}")
    return z1ds, zone_subset_ds, zone_extent

@task
def regrid_precipitation_data(zone_subset_ds, input_chunk_sizes, output_chunk_sizes, zone_extent):
    """Regrid the precipitation data to match the zone extent at 1km resolution"""
    print(f"Input to regridding - type: {type(zone_subset_ds).__name__}, name: {getattr(zone_subset_ds, 'name', 'unnamed')}")
    
    # Get the result from regrid_dataset function
    result = regrid_dataset(
        zone_subset_ds,
        input_chunk_sizes,
        output_chunk_sizes,
        zone_extent,
        regrid_method="bilinear"
    )
    
    # Ensure the result has a name if it's a DataArray
    if isinstance(result, xr.DataArray) and not result.name:
        result = result.rename('precipitation')
        print("Named regridded DataArray as 'precipitation' in regrid_precipitation_data task")
    
    # Double-check the output
    print(f"Output from regridding - type: {type(result).__name__}, name: {getattr(result, 'name', 'unnamed')}")
    
    return result

@task
def calculate_zone_means(regridded_data, zone_ds):
    """Calculate zonal means for the regridded data"""
    print(f"Input to zone_mean_df - type: {type(regridded_data).__name__}, name: {getattr(regridded_data, 'name', 'unnamed')}")
    
    # Ensure the input DataArray has a name
    if isinstance(regridded_data, xr.DataArray) and not regridded_data.name:
        print("WARNING: Received unnamed DataArray, renaming to 'precipitation'")
        regridded_data = regridded_data.rename('precipitation')
    
    # Now call zone_mean_df with the properly named data
    return zone_mean_df(regridded_data, zone_ds)

@task
def save_imerg_results(results_df, data_path, zone_str, start_date, end_date):
    """
    Save processed IMERG results and update input data.
    This will create both standard files with forecast and base files without forecast.
    """
    try:
        # Create output directory for CSV files
        output_dir = f"{data_path}geofsm-input/processed/{zone_str}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Format dates to ensure they are datetime objects
        if not isinstance(start_date, datetime):
            start_date = pd.to_datetime(start_date)
        if not isinstance(end_date, datetime):
            end_date = pd.to_datetime(end_date)
            
        # Save CSV file for future reference
        date_string = start_date.strftime('%Y%m%d')
        csv_file = f"{output_dir}/imerg_{date_string}.csv"
        results_df.to_csv(csv_file, index=False)
        print(f"CSV results saved to {csv_file}")
        
        # Create zone input path
        zone_input_path = f"{data_path}zone_wise_txt_files/"
        
        # Update IMERG input data - generate both standard files with forecast and base files without forecast
        rain_file, zone_specific_file, base_rain_file, base_zone_specific_file = imerg_update_input_data_improved(
            results_df, zone_input_path, zone_str, start_date, end_date
        )
        
        print(f"IMERG input data updated:")
        print(f"  - Standard files (with forecast): {rain_file} and {zone_specific_file}")
        print(f"  - Base files (without forecast): {base_rain_file} and {base_zone_specific_file}")
        
        return rain_file, zone_specific_file, base_rain_file, base_zone_specific_file
    except Exception as e:
        print(f"Error saving IMERG results: {e}")
        raise

@flow
def process_single_zone(data_path, imerg_data, zone_str, start_date, end_date):
    """Process a single zone across multiple dates"""
    print(f"Processing zone {zone_str} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Standardize zone string format
    if not isinstance(zone_str, str):
        zone_str = str(zone_str)
        
    if zone_str.isdigit():
        zone_str = f'zone{zone_str}'
    elif not zone_str.startswith('zone'):
        zone_str = f'zone{zone_str}'
    
    try:
        # Process this zone
        z1ds, zone_subset_ds, zone_extent = process_zone(data_path, imerg_data, zone_str)
        
        # Adjust input_chunk_sizes based on the dimensions in the data
        if 'lat' in zone_subset_ds.dims and 'lon' in zone_subset_ds.dims:
            input_chunk_sizes = {'time': 10, 'lat': 30, 'lon': 30}
        else:
            input_chunk_sizes = {'time': 10, 'y': 30, 'x': 30}
        
        output_chunk_sizes = {'lat': 300, 'lon': 300}
        regridded_data = regrid_precipitation_data(zone_subset_ds, input_chunk_sizes, output_chunk_sizes, zone_extent)
        zone_means = calculate_zone_means(regridded_data, z1ds)
        
        # Save the results with both standard (with forecast) and base (without forecast) files
        rain_file, zone_specific_file, base_rain_file, base_zone_specific_file = save_imerg_results(
            zone_means, data_path, zone_str, start_date, end_date
        )
        
        return rain_file, zone_specific_file, base_rain_file, base_zone_specific_file
    except Exception as e:
        print(f"Error in process_single_zone for {zone_str}: {e}")
        return None, None, None, None

@flow
def imerg_all_zones_workflow(start_date: str = yesterday, end_date: str = ""):
    """
    Main workflow for processing IMERG data for all zones.
    Creates two sets of output files:
    1. Standard files with a 16-day forecast extension based on the last 15 days
    2. Base files containing only the actual data without any forecast extension
    
    Returns:
        Dict containing the paths to all generated txt files
    """
    # Handle default values for start_date and end_date
    if not start_date:
        start_date = yesterday
    
    if not end_date:
        # Default to same as start_date if not specified
        end_date = start_date
    
    data_path, imerg_store, client = setup_environment()
    
    try:
        # Check if master shapefile exists before continuing
        master_shapefile = f'{data_path}WGS/geofsm-prod-all-zones-20240712.shp'
        if not os.path.exists(master_shapefile):
            print(f"ERROR: Master shapefile not found at {master_shapefile}")
            raise FileNotFoundError(f"Master shapefile not found: {master_shapefile}")
        else:
            print(f"Found master shapefile: {master_shapefile}")
        
        # Process all zones from the shapefile
        all_zones = gp.read_file(master_shapefile)
        unique_zones = all_zones['zone'].unique()
        
        # Initialize variables for collecting output files
        standard_files = []  # Files with forecast
        base_files = []      # Files without forecast
        
        # Create a reference to the task in this scope
        get_imerg_files_task = get_imerg_files
        
        # Process each zone separately
        for zone_str in unique_zones:
            try:
                # Standardize zone string format
                if not isinstance(zone_str, str):
                    zone_str = str(zone_str)
                    
                if zone_str.isdigit():
                    zone_str = f'zone{zone_str}'
                elif not zone_str.startswith('zone'):
                    zone_str = f'zone{zone_str}'
                
                print(f"\n===== Processing {zone_str} =====")
                
                # Ensure all required directories exist for this zone
                zone_dir = f"{data_path}zone_wise_txt_files/{zone_str}"
                init_zone_dir = f"{data_path}zone_wise_txt_files/init/{zone_str}"
                
                # Recursively create all necessary directories
                os.makedirs(zone_dir, exist_ok=True)
                os.makedirs(init_zone_dir, exist_ok=True)
                
                print(f"Ensuring directories exist for {zone_str}:")
                print(f"  - Standard directory: {zone_dir}")
                print(f"  - Base directory: {init_zone_dir}")
                
                # Always check the init directory first as it contains the reliable historical data
                last_date = get_last_date_from_rain(init_zone_dir, is_init=True)
                if last_date is None:
                    # Only if init directory has no data, check the standard directory
                    last_date = get_last_date_from_rain(zone_dir, is_init=False)
                    print(f"No data found in init directory, checking standard directory instead.")
                
                # Check if we have a last date, start from the next day
                # Otherwise, use a default start date (e.g., 30 days ago)
                if last_date:
                    workflow_start_date = last_date + timedelta(days=1)
                    
                    # If the start date is today or in the future, we're already up to date
                    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    if workflow_start_date >= today:
                        print(f"Init directory already up to date (last date: {last_date.strftime('%Y-%m-%d')})")
                        print(f"No new data to process for {zone_str}. Skipping.")
                        continue
                    
                    # Convert to string format for IMERG functions
                    workflow_start_date_str = workflow_start_date.strftime('%Y%m%d')
                    print(f"Starting data collection from {workflow_start_date.strftime('%Y-%m-%d')}")
                else:
                    # If no last date is found, use the provided start_date
                    workflow_start_date_str = start_date
                    workflow_start_date = datetime.strptime(workflow_start_date_str, '%Y%m%d')
                    print(f"No existing data found. Using provided start date: {workflow_start_date.strftime('%Y-%m-%d')}")
                
                # End date is either today or the provided end_date
                if end_date:
                    workflow_end_date_str = end_date
                    workflow_end_date = datetime.strptime(workflow_end_date_str, '%Y%m%d')
                else:
                    workflow_end_date = datetime.now()
                    workflow_end_date_str = workflow_end_date.strftime('%Y%m%d')
                
                # Generate a list of dates to process
                date_range = pd.date_range(start=workflow_start_date, end=workflow_end_date, freq='D')
                dates_to_process = []
                
                # Check which dates need processing (not already processed)
                for process_date in date_range:
                    date_str = process_date.strftime('%Y%m%d')
                    date_ddd = process_date.strftime('%Y%j')
                    
                    # Check if output files already exist for this date
                    output_dir = f"{data_path}geofsm-input/processed/{zone_str}"
                    processed_file = f"{output_dir}/imerg_{date_str}.csv"
                    
                    if os.path.exists(processed_file):
                        print(f"Data for {date_str} already processed. Skipping.")
                        continue
                    
                    dates_to_process.append((date_str, process_date))
                
                if not dates_to_process:
                    print(f"All dates already processed for {zone_str}. Skipping entire zone.")
                    continue
                
                print(f"Found {len(dates_to_process)} dates to process for {zone_str}")
                
                # Process all dates that need processing
                for date_tuple in dates_to_process:
                    date_str, process_date = date_tuple
                    
                    print(f"Processing {zone_str} for date {date_str}")
                    
                    # Get file list for this specific date
                    print(f"Searching for IMERG files for {date_str}")
                    file_list = get_imerg_files_task(date_str, date_str)
                    
                    if not file_list:
                        print(f"No IMERG files found for {date_str}")
                        continue
                    
                    print(f"Found {len(file_list)} IMERG files for {date_str}")
                    
                    # Download files
                    download_dir = download_imerg_files(file_list, imerg_store)
                    
                    # Process data for this specific date
                    imerg_data = process_imerg_data(download_dir, date_str, date_str)
                    
                    # Rename coordinates from x,y to lon,lat if needed
                    imerg_data = rename_coordinates(imerg_data)
                    
                    # Process this zone for this specific date
                    rain_file, zone_specific_file, base_rain_file, base_zone_specific_file = process_single_zone(
                        data_path, imerg_data, zone_str, process_date, process_date
                    )
                    
                    if rain_file and zone_specific_file:
                        standard_files.extend([rain_file, zone_specific_file])
                        base_files.extend([base_rain_file, base_zone_specific_file])
                        print(f"Successfully processed {zone_str} for {date_str}")
                
            except Exception as e:
                print(f"Error processing zone {zone_str}: {e}")
        
        print(f"Workflow completed successfully!")
        print(f"Processed {len(standard_files)//2} zones")
        print(f"Created {len(standard_files)} standard files (with forecast)")
        print(f"Created {len(base_files)} base files (without forecast)")
        
        return {
            'standard_files': standard_files,  # Files with forecast
            'base_files': base_files           # Files without forecast
        }
    except Exception as e:
        print(f"Error in workflow: {e}")
        raise
    finally:
        client.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process IMERG data for hydrological modeling')
    parser.add_argument('--start-date', type=str, default=yesterday, 
                        help=f'Start date in YYYYMMDD format (default: {yesterday})')
    parser.add_argument('--end-date', type=str, default="", 
                        help='End date in YYYYMMDD format (default: same as start-date)')
    
    args = parser.parse_args()
    
    print(f"Processing IMERG data from {args.start_date} to {args.end_date or args.start_date}")
    result = imerg_all_zones_workflow(args.start_date, args.end_date)
    print(f"Generated standard files (with forecast): {result['standard_files']}")
    print(f"Generated base files (without forecast): {result['base_files']}")