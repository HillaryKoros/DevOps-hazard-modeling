from prefect import flow, task
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import xarray as xr
from dask.distributed import Client
import geopandas as gp
import pandas as pd
import numpy as np
import glob

from utils import (
    pet_list_files_by_date,
    pet_download_extract_bilfile,
    pet_bil_netcdf,
    pet_read_netcdf_files_in_date_range,
    get_dask_client_params,
    process_zone_from_combined,
    regrid_dataset,
    zone_mean_df
)

load_dotenv()

@task
def get_current_date():
    """Get the current date in YYYYMMDD format."""
    return datetime.now().strftime('%Y%m%d')

@task
def setup_environment():
    """Set up the environment for data processing"""
    data_path = os.getenv("data_path", "./data/")  # Default to ./data/ if not set
    output_dir = f'{data_path}geofsm-input/pet/dir/'
    netcdf_path = f'{data_path}geofsm-input/pet/netcdf/'
    zone_input_path = f"{data_path}zone_wise_txt_files/"
    init_zone_path = f"{data_path}zone_wise_txt_files/init/"
    
    # Create all necessary directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(netcdf_path, exist_ok=True)
    os.makedirs(zone_input_path, exist_ok=True)
    os.makedirs(init_zone_path, exist_ok=True)
    
    params = get_dask_client_params()
    client = Client(**params)
    
    print(f"Environment setup complete. Using data_path: {data_path}")
    print(f"Created standard output directory: {zone_input_path}")
    print(f"Created base output directory (no forecast): {init_zone_path}")
    return data_path, output_dir, netcdf_path, client

def get_last_date_from_evap(zone_dir, is_init=False):
    """
    Read the existing evap.txt file and determine the last date in the file.
    
    Parameters:
    ----------
    zone_dir : str
        Path to the zone directory containing evap.txt
    is_init : bool
        Flag indicating if this is the init directory (for logging purposes)
        
    Returns:
    -------
    datetime
        The last date in the file, or None if the file doesn't exist or can't be read
    """
    evap_file = os.path.join(zone_dir, 'evap.txt')
    dir_type = "init" if is_init else "standard"
    
    if not os.path.exists(evap_file):
        print(f"No existing evap.txt found at {evap_file} ({dir_type} directory)")
        return None
    
    try:
        # Read the evap.txt file
        df = pd.read_csv(evap_file, sep=",")
        
        # Check if NA column exists (which contains the dates in YYYYDDD format)
        if 'NA' not in df.columns:
            print(f"Invalid format in evap.txt ({dir_type} directory) - missing 'NA' column")
            return None
        
        # Convert the last date to datetime
        last_date_str = df['NA'].iloc[-1]
        last_date = datetime.strptime(str(last_date_str), '%Y%j')
        
        print(f"Last date in {dir_type} evap.txt: {last_date.strftime('%Y-%m-%d')} (Day {last_date_str})")
        return last_date
        
    except Exception as e:
        print(f"Error reading existing evap.txt ({dir_type} directory): {e}")
        return None

def pet_extend_forecast_improved(df, date_column, days_to_add=16):
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

def update_standard_directory_with_forecast(data_path, zone_str):
    """
    Read the existing evap.txt file from init directory (which has accurate historical data),
    add a forecast by copying the last 15 days to create a 16-day forecast,
    and save it to the standard directory.
    
    This function is used when the init directory is already up to date but we still
    want to update the forecast in the standard directory.
    
    Parameters:
    ----------
    data_path : str
        Base path for data files
    zone_str : str
        Zone identifier (e.g., 'zone1')
        
    Returns:
    -------
    tuple
        Paths to the standard evap.txt and zone-specific evap file that were updated
    """
    # Define paths
    zone_input_path = f"{data_path}zone_wise_txt_files/"
    init_zone_dir = f"{zone_input_path}init/{zone_str}"
    standard_zone_dir = f"{zone_input_path}{zone_str}"
    
    # Make sure directories exist
    os.makedirs(standard_zone_dir, exist_ok=True)
    
    # Path to files
    init_evap_file = os.path.join(init_zone_dir, 'evap.txt')
    standard_evap_file = os.path.join(standard_zone_dir, 'evap.txt')
    standard_zone_specific_file = os.path.join(standard_zone_dir, f'evap_{zone_str}.txt')
    
    if not os.path.exists(init_evap_file):
        print(f"Error: No evap.txt found in init directory for {zone_str}")
        return None, None
    
    try:
        # Read the evap.txt from init directory (which has accurate historical data)
        df = pd.read_csv(init_evap_file, sep=",")
        
        # Apply the forecast extension
        df_with_forecast = pet_extend_forecast_improved(df, 'NA')
        
        # Save to standard directory
        df_with_forecast.to_csv(standard_evap_file, index=False)
        df_with_forecast.to_csv(standard_zone_specific_file, index=False)
        
        print(f"Updated standard directory with forecast for {zone_str}")
        print(f"  - Updated standard evap.txt: {standard_evap_file}")
        print(f"  - Updated zone-specific evap file: {standard_zone_specific_file}")
        
        return standard_evap_file, standard_zone_specific_file
        
    except Exception as e:
        print(f"Error updating standard directory with forecast for {zone_str}: {e}")
        return None, None

def pet_update_input_data(z1a, zone_input_path, zone_str, start_date, end_date):
    """
    Processes evaporation data and generates:
    1. Standard evap.txt and zone-specific evap_zone*.txt files with forecast extension
    2. Base evap.txt and zone-specific evap_zone*.txt files with only historical data in 'init' folder
    
    Parameters:
    ----------
    z1a : pandas.DataFrame
        Dataframe containing PET data that needs to be adjusted, pivoted, and formatted.
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
        Paths to the four generated files (standard evap.txt, zone-specific evap file, 
        base evap.txt without forecast, and base zone-specific evap file without forecast).
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
    
    # Adjust the 'pet' column by a factor of 10
    z1a['pet'] = z1a['pet'] / 10
    
    # Get today's date for filtering forecast data
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Make a copy of z1a for historical data only (for init directory)
    historical_df = z1a.copy()
    # Filter to keep only data up to yesterday (no forecasts)
    historical_df = historical_df[historical_df['time'] < today]
    
    # Pivot the DataFrames for both standard and init directories
    zz1 = z1a.pivot(index='time', columns='group', values='pet')
    historical_zz1 = historical_df.pivot(index='time', columns='group', values='pet')
    
    # Apply formatting to the pivoted DataFrames
    zz1 = zz1.apply(lambda row: row.map(lambda x: f'{x:.1f}' if isinstance(x, (int, float)) and pd.notna(x) else x), axis=1)
    historical_zz1 = historical_zz1.apply(lambda row: row.map(lambda x: f'{x:.1f}' if isinstance(x, (int, float)) and pd.notna(x) else x), axis=1)
    
    # Reset the index and adjust columns for standard directory
    azz1 = zz1.reset_index()
    azz1['NA'] = azz1['time'].dt.strftime('%Y%j')
    azz1.columns = [str(col) if isinstance(col, int) else col for col in azz1.columns]
    azz1 = azz1.rename(columns={'time': 'date'})
    
    # Do the same for historical data (init directory)
    historical_azz1 = historical_zz1.reset_index()
    historical_azz1['NA'] = historical_azz1['time'].dt.strftime('%Y%j')
    historical_azz1.columns = [str(col) if isinstance(col, int) else col for col in historical_azz1.columns]
    historical_azz1 = historical_azz1.rename(columns={'time': 'date'})
    
    # Path to standard evap.txt file in zone_wise directory
    evap_file = f'{zone_dir}/evap.txt'
    
    # Path to base evap.txt file in init directory (without forecast)
    base_evap_file = f'{init_zone_dir}/evap.txt'
    
    # STANDARD FILES (with forecast)
    # Check if the standard evap.txt file exists
    if os.path.exists(evap_file):
        # If file exists, read and merge with new data
        try:
            ez1 = pd.read_csv(evap_file, sep=",")
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
            print(f"Error reading existing evap.txt: {e}")
            print("Creating new evap.txt file instead")
            bz1 = azz1.drop(['date'], axis=1).reset_index(drop=True)
    else:
        # If file doesn't exist, just use the new data
        print(f"No existing evap.txt found at {evap_file}. Creating new file.")
        bz1 = azz1.drop(['date'], axis=1).reset_index(drop=True)
    
    # INIT FILES (historical data only)
    # Check if the base evap file exists
    if os.path.exists(base_evap_file):
        try:
            base_ez1 = pd.read_csv(base_evap_file, sep=",")
            base_ez1['date'] = pd.to_datetime(base_ez1['NA'], format='%Y%j')
            
            # For init directory, only include historical data (up to yesterday)
            # Filter out any forecasted data in the init directory
            filter_date = start_date
            if filter_date >= today:
                # If start_date is today or later, don't update init directory
                print(f"No new historical data to add to init directory")
                base_bz1 = base_ez1.copy()
            else:
                # Create a mask for filtering data
                # Keep data that is either before our start_date or after today (filtering the range we're updating)
                mask = (base_ez1['date'] < start_date) | (base_ez1['date'] >= today)
                base_aez1 = base_ez1[mask]
                
                # Concatenate DataFrames - add only historical data to init directory
                base_bz1 = pd.concat([base_aez1, historical_azz1], axis=0)
                
                # Reset index and drop unnecessary columns
                base_bz1.drop(['date'], axis=1, inplace=True)
                base_bz1.reset_index(drop=True, inplace=True)
        except Exception as e:
            print(f"Error reading existing base evap.txt: {e}")
            print("Creating new base evap.txt file instead")
            base_bz1 = historical_azz1.drop(['date'], axis=1).reset_index(drop=True)
    else:
        # If file doesn't exist, just use the historical data
        print(f"No existing base evap.txt found at {base_evap_file}. Creating new file with historical data only.")
        base_bz1 = historical_azz1.drop(['date'], axis=1).reset_index(drop=True)
    
    # Ensure all values in NA column are strings for consistent sorting
    if 'NA' in bz1.columns:
        bz1['NA'] = bz1['NA'].astype(str)
    if 'NA' in base_bz1.columns:
        base_bz1['NA'] = base_bz1['NA'].astype(str)
    
    # Sort the data by NA column (date) to ensure proper order
    bz1 = bz1.sort_values(by='NA').reset_index(drop=True)
    base_bz1 = base_bz1.sort_values(by='NA').reset_index(drop=True)
    
    # Use the improved forecast extension for standard files only
    bz2 = pet_extend_forecast_improved(bz1, 'NA')
    
    # Create standard files (with forecast)
    
    # 1. Standard evap.txt file
    bz2.to_csv(evap_file, index=False)
    print(f"Created/updated standard evap.txt file (with forecast): {evap_file}")
    
    # 2. Zone-specific evap file (evap_zone1.txt)
    zone_specific_file = f'{zone_dir}/evap_{zone_str}.txt'
    bz2.to_csv(zone_specific_file, index=False)
    print(f"Created zone-specific evap file (with forecast): {zone_specific_file}")
    
    # Create base files (historical data only)
    
    # 3. Base evap.txt file
    base_bz1.to_csv(base_evap_file, index=False)
    print(f"Created/updated base evap.txt file (historical only): {base_evap_file}")
    
    # 4. Zone-specific base evap file
    base_zone_specific_file = f'{init_zone_dir}/evap_{zone_str}.txt'
    base_bz1.to_csv(base_zone_specific_file, index=False)
    print(f"Created base zone-specific evap file (historical only): {base_zone_specific_file}")
    
    return evap_file, zone_specific_file, base_evap_file, base_zone_specific_file

@task
def get_pet_files(url, start_date, end_date):
    """Get the list of PET files for the date range"""
    try:
        print(f"Getting PET files from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        pet_list = pet_list_files_by_date(url, start_date, end_date)
        print(f"Found {len(pet_list)} PET files in date range")
        return pet_list
    except Exception as e:
        print(f"Error fetching PET files: {e}")
        raise

@task
def process_pet_files(pet_list, output_dir, netcdf_path):
    """Download and process PET files"""
    print(f"Processing {len(pet_list)} PET files")
    processed_files = 0
    
    for file_url, date in pet_list:
        try:
            date_str = date.strftime('%Y%m%d')
            nc_file = os.path.join(netcdf_path, f"{date_str}.nc")
            
            if os.path.exists(nc_file):
                print(f"NetCDF file already exists for {date_str}, skipping download and conversion")
                processed_files += 1
                continue
            
            print(f"Processing PET file for {date_str}")
            pet_download_extract_bilfile(file_url, output_dir)
            pet_bil_netcdf(file_url, date, output_dir, netcdf_path)
            processed_files += 1
        except Exception as e:
            print(f"Error processing PET file {file_url}: {e}")
    
    print(f"Processed {processed_files} PET files")
    return processed_files

@task
def process_zone(data_path, pds, zone_str):
    """Process zone from combined shapefile and subset data"""
    master_shapefile = f'{data_path}WGS/geofsm-prod-all-zones-20240712.shp'
    
    if not os.path.exists(master_shapefile):
        print(f"Master shapefile not found: {master_shapefile}")
        raise FileNotFoundError(f"Master shapefile not found: {master_shapefile}")
    
    # Standardize zone string format
    if not isinstance(zone_str, str):
        zone_str = str(zone_str)
        
    if zone_str.isdigit():
        zone_str = f'zone{zone_str}'
    elif not zone_str.startswith('zone'):
        zone_str = f'zone{zone_str}'
    
    print(f"Processing {zone_str} from combined shapefile")
    km_str = 1  # 1km resolution
    
    try:
        z1ds, pdsz1, zone_extent = process_zone_from_combined(master_shapefile, zone_str, km_str, pds)
        print(f"Processed zone {zone_str}")
        return z1ds, pdsz1, zone_extent
    except Exception as e:
        print(f"Error processing zone {zone_str}: {e}")
        raise

@task
def regrid_pet_data(pdsz1, zone_extent):
    """Regrid PET data to match zone resolution"""
    print("Regridding PET data")
    try:
        input_chunk_sizes = {'time': 10, 'lat': 30, 'lon': 30}
        output_chunk_sizes = {'lat': 300, 'lon': 300}
        
        # Ensure data is contiguous
        for var in pdsz1.data_vars:
            pdsz1[var] = pdsz1[var].copy(data=np.ascontiguousarray(pdsz1[var].data))
            
        return regrid_dataset(
            pdsz1,
            input_chunk_sizes,
            output_chunk_sizes,
            zone_extent,
            regrid_method="bilinear"
        )
    except Exception as e:
        print(f"Error regridding PET data: {e}")
        raise

@task
def calculate_zone_means(regridded_data, zone_ds):
    """Calculate mean PET values for each zone"""
    print("Calculating zone means")
    try:
        return zone_mean_df(regridded_data, zone_ds)
    except Exception as e:
        print(f"Error calculating zone means: {e}")
        raise

@task
def save_pet_results(results_df, data_path, zone_str, start_date, end_date):
    """Save processed PET results and update input data"""
    try:
        # Format dates to ensure they are datetime objects
        if not isinstance(start_date, datetime):
            start_date = pd.to_datetime(start_date)
        if not isinstance(end_date, datetime):
            end_date = pd.to_datetime(end_date)
            
        # Create zone input path
        zone_input_path = f"{data_path}zone_wise_txt_files/"
        
        # Update PET input data - generate both standard files with forecast and base files without forecast
        evap_file, zone_specific_file, base_evap_file, base_zone_specific_file = pet_update_input_data(
            results_df, zone_input_path, zone_str, start_date, end_date
        )
        
        print(f"PET input data updated:")
        print(f"  - Standard files (with forecast): {evap_file} and {zone_specific_file}")
        print(f"  - Base files (historical only): {base_evap_file} and {base_zone_specific_file}")
        
        return evap_file, zone_specific_file, base_evap_file, base_zone_specific_file
    except Exception as e:
        print(f"Error saving PET results: {e}")
        raise

@task
def read_and_process_single_pet_file(netcdf_path, file_date):
    """Read and process a single PET netCDF file"""
    date_str = file_date.strftime('%Y%m%d')
    nc_file = os.path.join(netcdf_path, f"{date_str}.nc")
    
    if not os.path.exists(nc_file):
        print(f"Warning: NetCDF file {nc_file} does not exist")
        return None
    
    try:
        # Open the single file
        ds = xr.open_dataset(nc_file)
        
        # Process the dataset
        if 'spatial_ref' in ds.variables:
            ds = ds.drop_vars('spatial_ref')
        
        if 'band' in ds.variables:
            ds = ds.drop_vars('band')
        
        if 'date' in ds.variables:
            ds = ds.drop_vars('date')
        
        if 'band' in ds.dims:
            ds = ds.squeeze('band')
        
        if '__xarray_dataarray_variable__' in ds.data_vars:
            ds = ds.rename_vars({'__xarray_dataarray_variable__': 'pet'})
        
        # Add time dimension
        ds = ds.expand_dims(time=[file_date])
        
        # Rename coordinates if needed
        rename_dict = {}
        if 'x' in ds.dims and 'lon' not in ds.dims:
            rename_dict['x'] = 'lon'
        if 'y' in ds.dims and 'lat' not in ds.dims:
            rename_dict['y'] = 'lat'
        
        if rename_dict:
            ds = ds.rename(rename_dict)
        
        return ds
    
    except Exception as e:
        print(f"Error processing {nc_file}: {e}")
        return None

@flow
def process_zone_pet_for_date(data_path, netcdf_path, zone_str, file_date):
    """Process PET data for a single zone and date"""
    try:
        # Create references to tasks within this scope
        process_zone_task = process_zone
        regrid_pet_data_task = regrid_pet_data
        calculate_zone_means_task = calculate_zone_means
        read_process_single_task = read_and_process_single_pet_file
        
        # Read the single file for the date
        ds = read_process_single_task(netcdf_path, file_date)
        
        if ds is None:
            return None, None
        
        # Process the single-date dataset for the zone
        z1ds, pdsz1, zone_extent = process_zone_task(data_path, ds, zone_str)
        regridded_data = regrid_pet_data_task(pdsz1, zone_extent)
        zone_means = calculate_zone_means_task(regridded_data, z1ds)
        
        # Return the results for later aggregation
        return file_date, zone_means
    
    except Exception as e:
        print(f"Error processing {zone_str} for date {file_date.strftime('%Y-%m-%d')}: {e}")
        return None, None

@flow
def process_single_zone_pet(data_path, netcdf_path, zone_str, start_date, end_date):
    """Process PET data for a single zone across multiple dates"""
    print(f"Processing zone {zone_str} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Create a reference to the task
    process_zone_pet_for_date_flow = process_zone_pet_for_date
    
    # Standardize zone string format
    if not isinstance(zone_str, str):
        zone_str = str(zone_str)
        
    if zone_str.isdigit():
        zone_str = f'zone{zone_str}'
    elif not zone_str.startswith('zone'):
        zone_str = f'zone{zone_str}'
    
    try:
        # Generate a date range for all dates in the period
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Process each date individually
        all_results = []
        for file_date in date_range:
            date_str = file_date.strftime('%Y%m%d')
            print(f"Processing {zone_str} for date {date_str}")
            
            result_date, result_df = process_zone_pet_for_date_flow(data_path, netcdf_path, zone_str, file_date)
            
            if result_date is not None and result_df is not None:
                all_results.append(result_df)
        
        # Combine results if we have any
        if all_results:
            # Concatenate all dataframes
            combined_results = pd.concat(all_results, ignore_index=True)
            
            # Save the combined results
            evap_file, zone_specific_file, base_evap_file, base_zone_specific_file = save_pet_results(
                combined_results, data_path, zone_str, start_date, end_date
            )
            
            return evap_file, zone_specific_file, base_evap_file, base_zone_specific_file
        else:
            print(f"No valid results found for {zone_str} in date range")
            return None, None, None, None
    
    except Exception as e:
        print(f"Error in process_single_zone_pet for {zone_str}: {e}")
        return None, None, None, None

@flow
def pet_all_zones_workflow():
    """
    Main workflow for processing PET data for all zones.
    
    Creates two sets of output files:
    1. Standard files with a 16-day forecast extension based on the last 15 days
       - Always updated with forecast, even if init is up to date
    2. Base files containing only the actual historical data (no forecast)
       - Only updated when new historical data is available
    
    Returns:
        Dict containing the paths to all generated txt files
    """
    data_path, output_dir, netcdf_path, client = setup_environment()
    
    try:
        # Base URL for PET data
        url = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/daily/pet/downloads/daily/"
        
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
        base_files = []      # Files with historical data only
        processed_zone_count = 0
        
        # Create a reference to the task in this scope
        get_pet_files_task = get_pet_files
        
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
                last_date = get_last_date_from_evap(init_zone_dir, is_init=True)
                if last_date is None:
                    # Only if init directory has no data, check the standard directory
                    last_date = get_last_date_from_evap(zone_dir, is_init=False)
                    print(f"No data found in init directory, checking standard directory instead.")
                
                # Define today as midnight today
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                
                # Check if we need to update the init directory
                need_new_data = False
                if last_date:
                    # Next day after the last date in file
                    start_date = last_date + timedelta(days=1)
                    
                    # If the start date is today or in the future, init directory is up to date
                    if start_date >= today:
                        print(f"Init directory already up to date (last date: {last_date.strftime('%Y-%m-%d')})")
                        
                        # IMPORTANT: Even though init is up to date, we still update standard directory with forecast
                        standard_evap_file, zone_specific_file = update_standard_directory_with_forecast(data_path, zone_str)
                        
                        if standard_evap_file and zone_specific_file:
                            standard_files.extend([standard_evap_file, zone_specific_file])
                            processed_zone_count += 1
                            print(f"Successfully updated forecast for {zone_str}")
                            continue
                        else:
                            print(f"Failed to update forecast for {zone_str}. Skipping.")
                            continue
                    else:
                        need_new_data = True
                        print(f"Init directory needs updating. Starting from: {start_date.strftime('%Y-%m-%d')}")
                else:
                    # No existing data, use default start date 30 days ago
                    start_date = today - timedelta(days=30)
                    need_new_data = True
                    print(f"No existing data found. Using default start date: {start_date.strftime('%Y-%m-%d')}")
                
                # If we get here, we need to process new data
                if need_new_data:
                    # End date is yesterday (today's data might not be complete)
                    end_date = today - timedelta(days=1)
                    
                    # Get PET files for the date range
                    print(f"Searching for PET files from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                    pet_files = get_pet_files_task(url, start_date, end_date)
                    
                    if not pet_files:
                        print(f"No new PET files found for the date range")
                        continue
                    
                    print(f"Found {len(pet_files)} PET files to process")
                    
                    # Process all files - download and convert to NetCDF
                    process_pet_files(pet_files, output_dir, netcdf_path)
                    
                    # Process this zone using the approach that handles each date separately
                    evap_file, zone_specific_file, base_evap_file, base_zone_specific_file = process_single_zone_pet(
                        data_path, netcdf_path, zone_str, start_date, end_date
                    )
                    
                    if evap_file and zone_specific_file and base_evap_file and base_zone_specific_file:
                        standard_files.extend([evap_file, zone_specific_file])
                        base_files.extend([base_evap_file, base_zone_specific_file])
                        processed_zone_count += 1
                        print(f"Successfully processed {zone_str} with new data")
                    else:
                        print(f"Failed to process new data for {zone_str}")
                
            except Exception as e:
                print(f"Error processing {zone_str}: {e}")
        
        print(f"Workflow completed successfully!")
        print(f"Processed {processed_zone_count} zones")
        print(f"Created {len(standard_files)} standard files (with forecast)")
        print(f"Created {len(base_files)} base files (historical only)")
        
        return {
            'standard_files': standard_files,  # Files with forecast
            'base_files': base_files           # Files with historical data only
        }
    
    except Exception as e:
        print(f"Error in workflow: {e}")
        raise
    finally:
        client.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process PET data for hydrological modeling')
    
    args = parser.parse_args()
    
    print(f"Processing PET data from last available date forward")
    result = pet_all_zones_workflow()
    print(f"Generated standard files (with forecast): {result['standard_files']}")
    print(f"Generated base files (historical only): {result['base_files']}")