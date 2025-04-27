from prefect import flow, task
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import xarray as xr
from dask.distributed import Client
import geopandas as gp
import pandas as pd
import glob
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from utils import (
    gefs_chrips_list_tiff_files,
    gefs_chrips_download_files,
    gefs_chrips_process,
    get_dask_client_params,
    process_zone_from_combined,
    regrid_dataset,
    zone_mean_df,
    make_zones_geotif
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
    download_dir = f'{data_path}geofsm-input/gefs-chirps'
    zone_input_path = f"{data_path}zone_wise_txt_files/"
    init_zone_path = f"{data_path}zone_wise_txt_files/init/"
    
    # Create all necessary directories
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(zone_input_path, exist_ok=True)
    os.makedirs(init_zone_path, exist_ok=True)
    
    params = get_dask_client_params()
    client = Client(**params)
    
    print(f"Environment setup complete. Using data_path: {data_path}")
    print(f"Created standard output directory: {zone_input_path}")
    print(f"Created base output directory (no forecast): {init_zone_path}")
    return data_path, download_dir, client

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
        try:
            last_date = datetime.strptime(str(last_date_str), '%Y%j')
            print(f"Last date in {dir_type} rain.txt: {last_date.strftime('%Y-%m-%d')} (Day {last_date_str})")
            return last_date
        except ValueError:
            print(f"Error parsing date {last_date_str} in {dir_type} rain.txt")
            return None
        
    except Exception as e:
        print(f"Error reading existing rain.txt ({dir_type} directory): {e}")
        return None

def gefs_chirps_update_input_data(results_df, zone_input_path, zone_str, start_date, end_date):
    """
    Processes precipitation data from GEFS-CHIRPS and generates:
    1. Standard rain.txt and zone-specific rain_zone*.txt files - includes both historical and forecast data
    2. Base rain.txt and zone-specific rain_zone*.txt files in 'init' folder - includes ONLY historical data up to yesterday
    
    Parameters:
    ----------
    results_df : pandas.DataFrame
        Dataframe containing GEFS-CHIRPS data that needs to be formatted.
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
        base rain.txt with only historical, and base zone-specific rain file with only historical).
    """
    # Ensure all directories exist
    zone_dir = f'{zone_input_path}{zone_str}'
    init_zone_dir = f'{zone_input_path}init/{zone_str}'
    
    # Create directories recursively if they don't exist
    os.makedirs(zone_dir, exist_ok=True)
    os.makedirs(init_zone_dir, exist_ok=True)
    
    print(f"Ensuring output directories exist for {zone_str}:")
    print(f"  - Standard directory: {zone_dir}")
    print(f"  - Base directory: {init_zone_dir}")
    
    # Process the data
    # Rename the 'precipitation' column to 'rain' if it exists and 'rain' doesn't
    if 'precipitation' in results_df.columns and 'rain' not in results_df.columns:
        results_df = results_df.rename(columns={'precipitation': 'rain'})
    
    # Get today's date for filtering forecast data
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Make a copy of results_df for historical data only (for init directory)
    historical_df = results_df.copy()
    # Filter to keep only data up to yesterday
    historical_df = historical_df[historical_df['time'] < today]
    
    # Pivot the DataFrames
    zz1 = results_df.pivot(index='time', columns='group', values='rain')
    historical_zz1 = historical_df.pivot(index='time', columns='group', values='rain')
    
    # Apply formatting to the pivoted DataFrames
    zz1 = zz1.apply(lambda row: row.map(lambda x: f'{x:.1f}' if isinstance(x, (int, float)) and pd.notna(x) else x), axis=1)
    historical_zz1 = historical_zz1.apply(lambda row: row.map(lambda x: f'{x:.1f}' if isinstance(x, (int, float)) and pd.notna(x) else x), axis=1)
    
    # Reset the index and adjust columns
    azz1 = zz1.reset_index()
    azz1['NA'] = azz1['time'].dt.strftime('%Y%j')
    azz1.columns = [str(col) if isinstance(col, int) else col for col in azz1.columns]
    azz1 = azz1.rename(columns={'time': 'date'})
    
    # Do the same for historical data
    historical_azz1 = historical_zz1.reset_index()
    historical_azz1['NA'] = historical_azz1['time'].dt.strftime('%Y%j')
    historical_azz1.columns = [str(col) if isinstance(col, int) else col for col in historical_azz1.columns]
    historical_azz1 = historical_azz1.rename(columns={'time': 'date'})
    
    # Path to standard rain.txt file in zone_wise directory
    rain_file = f'{zone_dir}/rain.txt'
    
    # Path to base rain.txt file in init directory (historical data only)
    base_rain_file = f'{init_zone_dir}/rain.txt'
    
    # STANDARD FILES (with forecast)
    # Check if the standard rain.txt file exists
    if os.path.exists(rain_file):
        # If file exists, read and merge with new data
        try:
            print(f"Reading existing rain.txt from {rain_file}")
            ez1 = pd.read_csv(rain_file, sep=",")
            
            # Add a date column for filtering
            ez1['date'] = ez1['NA'].apply(lambda x: datetime.strptime(str(x), '%Y%j'))
            
            # Debug information
            print(f"Existing file contains dates from {ez1['date'].min().strftime('%Y-%m-%d')} to {ez1['date'].max().strftime('%Y-%m-%d')}")
            print(f"New data covers {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Create a mask for filtering data
            mask = (ez1['date'] < start_date) | (ez1['date'] > end_date)
            aez1 = ez1[mask]
            
            print(f"Keeping {len(aez1)} rows from existing file (outside new date range)")
            print(f"Adding {len(azz1)} rows of new data (including forecast)")
            
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
        
    # INIT FILES (historical data only)
    # Do the same for the base rain file but using historical_azz1
    if os.path.exists(base_rain_file):
        try:
            print(f"Reading existing base rain.txt from {base_rain_file}")
            base_ez1 = pd.read_csv(base_rain_file, sep=",")
            
            # Add a date column for filtering
            base_ez1['date'] = base_ez1['NA'].apply(lambda x: datetime.strptime(str(x), '%Y%j'))
            
            # Debug information
            print(f"Existing base file contains dates from {base_ez1['date'].min().strftime('%Y-%m-%d')} to {base_ez1['date'].max().strftime('%Y-%m-%d')}")
            print(f"New historical data covers dates up to {today - timedelta(days=1)}")
            
            # For init directory, only include data up to yesterday
            filter_date = start_date
            if filter_date >= today:
                # If start_date is today or later, don't update init directory
                print(f"No new historical data to add to init directory")
                base_bz1 = base_ez1.copy()
            else:
                # Create a mask for filtering data
                mask = (base_ez1['date'] < start_date) | (base_ez1['date'] >= today)
                base_aez1 = base_ez1[mask]
                
                print(f"Keeping {len(base_aez1)} rows from existing base file (outside new date range or forecast)")
                print(f"Adding {len(historical_azz1)} rows of new historical data (no forecast)")
                
                # Concatenate DataFrames
                base_bz1 = pd.concat([base_aez1, historical_azz1], axis=0)
                
                # Reset index and drop unnecessary columns
                base_bz1.drop(['date'], axis=1, inplace=True)
                base_bz1.reset_index(drop=True, inplace=True)
        except Exception as e:
            print(f"Error reading existing base rain.txt: {e}")
            print("Creating new base rain.txt file instead")
            base_bz1 = historical_azz1.drop(['date'], axis=1).reset_index(drop=True)
    else:
        # If file doesn't exist, just use the historical data
        print(f"No existing base rain.txt found at {base_rain_file}. Creating new file with historical data only.")
        base_bz1 = historical_azz1.drop(['date'], axis=1).reset_index(drop=True)
    
    # Ensure all values in NA column are strings for consistent sorting
    if 'NA' in bz1.columns:
        bz1['NA'] = bz1['NA'].astype(str)
    if 'NA' in base_bz1.columns:
        base_bz1['NA'] = base_bz1['NA'].astype(str)
    
    # Sort the data by NA column (date) to ensure proper order
    bz1 = bz1.sort_values(by='NA').reset_index(drop=True)
    base_bz1 = base_bz1.sort_values(by='NA').reset_index(drop=True)
    
    # Create standard files (with forecast)
    
    # 1. Standard rain.txt file
    bz1.to_csv(rain_file, index=False)
    print(f"Created/updated standard rain.txt file (with forecast): {rain_file}")
    
    # 2. Zone-specific rain file (rain_zone1.txt)
    zone_specific_file = f'{zone_dir}/rain_{zone_str}.txt'
    bz1.to_csv(zone_specific_file, index=False)
    print(f"Created zone-specific rain file (with forecast): {zone_specific_file}")
    
    # Create base files (historical data only)
    
    # 3. Base rain.txt file
    base_bz1.to_csv(base_rain_file, index=False)
    print(f"Created/updated base rain.txt file (historical only): {base_rain_file}")
    
    # 4. Zone-specific base rain file
    base_zone_specific_file = f'{init_zone_dir}/rain_{zone_str}.txt'
    base_bz1.to_csv(base_zone_specific_file, index=False)
    print(f"Created base zone-specific rain file (historical only): {base_zone_specific_file}")
    
    return rain_file, zone_specific_file, base_rain_file, base_zone_specific_file

@task
def check_data_availability(base_url, date_string):
    """
    Check if GEFS-CHIRPS data is available for the specified date.
    
    Args:
        base_url: Base URL for the GEFS-CHIRPS data
        date_string: Date in YYYYMMDD format
        
    Returns:
        bool: True if data is available, False otherwise
    """
    try:
        # Parse the date string
        year = date_string[:4]
        month = date_string[4:6]
        day = date_string[6:]
        
        # Construct the URL for the specific date
        url = f"{base_url}{year}/{month}/{day}/"
        
        # Send a request to check if the URL exists
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the content to check if there are TIFF files available
            soup = BeautifulSoup(response.text, 'html.parser')
            tiff_files = [link.get('href') for link in soup.find_all('a') if link.get('href', '').endswith('.tif')]
            
            # If there are TIFF files, data is available
            return len(tiff_files) > 0
        
        return False
    except Exception as e:
        print(f"Error checking data availability for {date_string}: {e}")
        return False

@task
def get_best_available_date(base_url, days_to_check=7):
    """
    Get the most recent date for which data is available.
    Start with today and go back up to 'days_to_check' days.
    
    Args:
        base_url: Base URL for the GEFS-CHIRPS data
        days_to_check: Number of days to check backward
        
    Returns:
        str: Date in YYYYMMDD format, or None if no data is available
    """
    today = datetime.now()
    
    for i in range(days_to_check):
        check_date = today - timedelta(days=i)
        date_string = check_date.strftime('%Y%m%d')
        print(f"Checking data availability for {date_string}...")
        
        if check_data_availability(base_url, date_string):
            print(f"Data found for {date_string}")
            return date_string
    
    print(f"No data found for the last {days_to_check} days")
    return None

@task
def is_already_processed(data_path, zone_str, date_string):
    """
    Check if the data for a specific date has already been processed for a zone.
    
    Args:
        data_path: Base data path
        zone_str: Zone identifier
        date_string: Date in YYYYMMDD format
        
    Returns:
        bool: True if already processed, False otherwise
    """
    # Convert YYYYMMDD to YYYYDDD format
    try:
        date_obj = datetime.strptime(date_string, '%Y%m%d')
        date_ddd = date_obj.strftime('%Y%j')
    except ValueError:
        date_ddd = date_string
    
    # Check for the existence of the processed file
    output_dir = f"{data_path}geofsm-input/processed/{zone_str}"
    processed_file = f"{output_dir}/rain_{date_ddd}.txt"
    
    return os.path.exists(processed_file)

@task
def get_gefs_files(base_url, date_string):
    """Get GEFS-CHIRPS files for the specified date"""
    all_files = gefs_chrips_list_tiff_files(base_url, date_string)
    print(f"Found {len(all_files)} files for date {date_string}")
    return all_files

@task
def download_gefs_files(url_list, date_string, download_dir):
    """Download GEFS-CHIRPS files"""
    date_dir = f"{download_dir}/{date_string}"
    if os.path.exists(date_dir) and os.listdir(date_dir):
        print(f"Data for {date_string} already exists in {date_dir}, skipping download.")
    else:
        print(f"Downloading data for {date_string}...")
        gefs_chrips_download_files(url_list, date_string, download_dir)
    return date_dir

@task
def process_gefs_data(input_path):
    """Process GEFS-CHIRPS data into xarray format"""
    print(f"Processing GEFS-CHIRPS data from {input_path}")
    return gefs_chrips_process(input_path)

@task
def process_zone(data_path, pds, zone_str):
    """Process a zone from the combined shapefile"""
    master_shapefile = f'{data_path}WGS/geofsm-prod-all-zones-20240712.shp'
    km_str = 1
    z1ds, pdsz1, zone_extent = process_zone_from_combined(master_shapefile, zone_str, km_str, pds)
    print(f"Processed zone {zone_str}")
    return z1ds, pdsz1, zone_extent

@task
def regrid_precipitation_data(pdsz1, input_chunk_sizes, output_chunk_sizes, zone_extent):
    """Regrid precipitation data to match zone resolution"""
    return regrid_dataset(
        pdsz1,
        input_chunk_sizes,
        output_chunk_sizes,
        zone_extent,
        regrid_method="bilinear"
    )

@task
def calculate_zone_means(regridded_data, zone_ds):
    """Calculate mean precipitation values for each zone"""
    return zone_mean_df(regridded_data, zone_ds)

@task
def save_csv_results(results_df, data_path, zone_str, date_string):
    """Save processed results to CSV for reference"""
    output_dir = f"{data_path}geofsm-input/processed/{zone_str}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/gefs_chirps_{date_string}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"CSV results saved to {output_file}")
    return output_file

@task
def save_gefs_results(results_df, data_path, zone_str, start_date, end_date):
    """
    Save processed GEFS-CHIRPS results and update input data.
    This will create both standard files and base files (both identical).
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
        csv_file = f"{output_dir}/gefs_chirps_{date_string}.csv"
        results_df.to_csv(csv_file, index=False)
        print(f"CSV results saved to {csv_file}")
        
        # Create zone input path
        zone_input_path = f"{data_path}zone_wise_txt_files/"
        
        # Update GEFS-CHIRPS input data - generate both standard files and base files
        rain_file, zone_specific_file, base_rain_file, base_zone_specific_file = gefs_chirps_update_input_data(
            results_df, zone_input_path, zone_str, start_date, end_date
        )
        
        print(f"GEFS-CHIRPS input data updated:")
        print(f"  - Standard files: {rain_file} and {zone_specific_file}")
        print(f"  - Base files: {base_rain_file} and {base_zone_specific_file}")
        
        return rain_file, zone_specific_file, base_rain_file, base_zone_specific_file
    except Exception as e:
        print(f"Error saving GEFS-CHIRPS results: {e}")
        raise

@flow
def process_single_zone(data_path, pds, zone_str, start_date, end_date):
    """Process a single zone for GEFS-CHIRPS data"""
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
        z1ds, pdsz1, zone_extent = process_zone(data_path, pds, zone_str)
        
        # Set up chunk sizes for regridding
        input_chunk_sizes = {'time': 10, 'lat': 30, 'lon': 30}
        output_chunk_sizes = {'lat': 300, 'lon': 300}
        
        # Process the data
        regridded_data = regrid_precipitation_data(pdsz1, input_chunk_sizes, output_chunk_sizes, zone_extent)
        zone_means = calculate_zone_means(regridded_data, z1ds)
        
        # Save the results with both standard and base files
        rain_file, zone_specific_file, base_rain_file, base_zone_specific_file = save_gefs_results(
            zone_means, data_path, zone_str, start_date, end_date
        )
        
        return rain_file, zone_specific_file, base_rain_file, base_zone_specific_file
    except Exception as e:
        print(f"Error in process_single_zone for {zone_str}: {e}")
        return None, None, None, None

@flow
def gefs_chirps_all_zones_workflow(date_string: str = None):
    """
    Main workflow for processing GEFS-CHIRPS data for all zones.
    Creates two sets of output files:
    1. Standard files in the original directory
    2. Base files (identical to standard) in the 'init' folder
    
    Args:
        date_string: Optional date in YYYYMMDD format. If None, the best available date will be determined.
        
    Returns:
        Dict containing the paths to all generated txt files
    """
    data_path, download_dir, client = setup_environment()
    
    try:
        base_url = "https://data.chc.ucsb.edu/products/EWX/data/forecasts/CHIRPS-GEFS_precip_v12/daily_16day/"
        
        # If date is not provided, find the most recent available date
        if date_string is None:
            date_string = get_best_available_date(base_url)
            if date_string is None:
                print("No data available for processing. Exiting workflow.")
                return {'standard_files': [], 'base_files': []}
        
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
        standard_files = []  # Standard files
        base_files = []      # Base files
        
        # Create references to tasks
        get_gefs_files_task = get_gefs_files
        
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
                
                # Check if we have a last date
                if last_date:
                    # Convert the date_string to a datetime object
                    date_obj = datetime.strptime(date_string, '%Y%m%d')
                    
                    # If the date we want to process is on or before the last date, skip it
                    if date_obj <= last_date:
                        print(f"Data for {date_string} already exists in {zone_str} (last date: {last_date.strftime('%Y-%m-%d')})")
                        print(f"No new data to process for {zone_str}. Skipping.")
                        continue
                
                # Check if the data is already processed
                if is_already_processed(data_path, zone_str, date_string):
                    print(f"Data for zone {zone_str} and date {date_string} already processed. Skipping.")
                    continue
                
                print(f"Processing data for {date_string}")
                
                # Get file list
                url_list = get_gefs_files_task(base_url, date_string)
                
                if not url_list:
                    print(f"No GEFS-CHIRPS files found for {date_string}")
                    continue
                
                print(f"Found {len(url_list)} GEFS-CHIRPS files for {date_string}")
                
                # Download files
                input_path = download_gefs_files(url_list, date_string, download_dir)
                
                # Process data
                pds = process_gefs_data(input_path)
                
                # Convert date_string to datetime for processing
                date_obj = datetime.strptime(date_string, '%Y%m%d')
                
                # Process this zone
                rain_file, zone_specific_file, base_rain_file, base_zone_specific_file = process_single_zone(
                    data_path, pds, zone_str, date_obj, date_obj
                )
                
                if rain_file and zone_specific_file:
                    standard_files.extend([rain_file, zone_specific_file])
                    base_files.extend([base_rain_file, base_zone_specific_file])
                    print(f"Successfully processed {zone_str} for {date_string}")
                
            except Exception as e:
                print(f"Error processing zone {zone_str}: {e}")
        
        print(f"Workflow completed successfully!")
        print(f"Processed {len(standard_files)//2} zones")
        print(f"Created {len(standard_files)} standard files")
        print(f"Created {len(base_files)} base files")
        
        return {
            'standard_files': standard_files,
            'base_files': base_files
        }
    except Exception as e:
        print(f"Error in workflow: {e}")
        raise
    finally:
        client.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process GEFS-CHIRPS data for hydrological modeling')
    parser.add_argument('--date', type=str, default=None, 
                        help='Date in YYYYMMDD format (default: automatically determine best available date)')
    
    args = parser.parse_args()
    
    print(f"Processing GEFS-CHIRPS data for {args.date or 'best available date'}")
    result = gefs_chirps_all_zones_workflow(args.date)
    print(f"Generated standard files: {result['standard_files']}")
    print(f"Generated base files: {result['base_files']}")