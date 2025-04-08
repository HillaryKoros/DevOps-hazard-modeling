from prefect import flow, task
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
from dask.distributed import Client
import logging
import sys
import pandas as pd
import glob
import geopandas as gp

from utils import (
    pet_list_files_by_date,
    pet_download_extract_bilfile,
    pet_bil_netcdf,
    pet_read_netcdf_files_in_date_range,
    get_dask_client_params,
    process_zone_from_combined,
    regrid_dataset,
    zone_mean_df,
    pet_update_input_data
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('pet_workflow')

# Load environment variables
load_dotenv()

@task(name="setup_environment", retries=2, retry_delay_seconds=5)
def setup_environment():
    """Set up the environment for data processing"""
    input_path = os.getenv("data_path")
    if not input_path:
        input_path = "./"  # Default to current directory if not set
    
    output_dir = f'{input_path}PET/dir/'
    netcdf_path = f'{input_path}PET/netcdf/'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(netcdf_path, exist_ok=True)
    
    params = get_dask_client_params()
    client = Client(**params)
    
    logger.info(f"Environment setup complete. Using data_path: {input_path}")
    return input_path, output_dir, netcdf_path, client

@task(name="check_pet_data_availability")
def check_pet_data_availability(url, date_to_check, output_dir, netcdf_path):
    """
    Check if PET data is available for the specified date,
    if not fall back to previous dates
    
    Args:
        url (str): URL for PET data
        date_to_check (datetime): Date to check for data availability 
        output_dir (str): Directory to store PET files
        netcdf_path (str): Directory to store NetCDF files
        
    Returns:
        tuple: (start_date, end_date) for data processing
    """
    logger.info(f"Checking PET data availability for {date_to_check.strftime('%Y-%m-%d')}")
    
    # Ensure date_to_check is not in the future
    current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if date_to_check > current_date:
        logger.warning(f"Requested date {date_to_check.strftime('%Y-%m-%d')} is in the future")
        date_to_check = current_date
        logger.info(f"Using current date instead: {date_to_check.strftime('%Y-%m-%d')}")
    
    # Maximum number of days to check backward
    max_days_to_check = 10
    
    # List to store dates to check in order of preference
    dates_to_check = []
    
    # Start with the requested date
    dates_to_check.append(date_to_check)
    
    # Add previous days
    for i in range(1, max_days_to_check + 1):
        dates_to_check.append(date_to_check - timedelta(days=i))
    
    # Check each date
    for check_date in dates_to_check:
        date_str = check_date.strftime('%Y%m%d')
        logger.info(f"Checking data availability for {date_str}")
        
        # Check if we already have the netCDF file locally
        nc_file = os.path.join(netcdf_path, f"{date_str}.nc")
        if os.path.exists(nc_file):
            logger.info(f"Found local data for {date_str}")
            return check_date - timedelta(days=30), check_date
        
        # Check if the data is available online
        try:
            date_pet_list = pet_list_files_by_date(url, check_date, check_date)
            
            if date_pet_list:
                logger.info(f"Found online data for {date_str}")
                # Download and process the data
                for file_url, file_date in date_pet_list:
                    try:
                        pet_download_extract_bilfile(file_url, output_dir)
                        pet_bil_netcdf(file_url, file_date, output_dir, netcdf_path)
                    except Exception as e:
                        logger.error(f"Error processing file {file_url}: {e}")
                        continue
                
                return check_date - timedelta(days=30), check_date
            else:
                logger.warning(f"No data available for {date_str}")
        except Exception as e:
            logger.error(f"Error checking data for {date_str}: {e}")
    
    # If we've checked all dates and found nothing, 
    # try to use the most recent available netCDF file
    try:
        nc_files = glob.glob(os.path.join(netcdf_path, "*.nc"))
        if nc_files:
            # Extract dates from filenames
            dates = [os.path.basename(f).replace('.nc', '') for f in nc_files]
            # Parse dates and find the most recent
            parsed_dates = [datetime.strptime(d, '%Y%m%d') for d in dates]
            most_recent = max(parsed_dates)
            
            logger.info(f"Using most recent available data from {most_recent.strftime('%Y%m%d')}")
            start_date = most_recent - timedelta(days=30)
            return start_date, most_recent
    except Exception as e:
        logger.error(f"Error finding most recent data: {e}")
    
    # If all else fails, use a fallback date range
    fallback_end = current_date - timedelta(days=1)  # yesterday
    fallback_start = fallback_end - timedelta(days=30)
    logger.warning(f"No recent data found. Using fallback date range {fallback_start.strftime('%Y-%m-%d')} to {fallback_end.strftime('%Y-%m-%d')}")
    return fallback_start, fallback_end

@task(name="get_pet_files", retries=2, retry_delay_seconds=60)
def get_pet_files(url, start_date, end_date):
    """Get the list of PET files for the date range"""
    try:
        logger.info(f"Getting PET files from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        pet_list = pet_list_files_by_date(url, start_date, end_date)
        logger.info(f"Found {len(pet_list)} PET files in date range")
        return pet_list
    except Exception as e:
        logger.error(f"Error fetching PET files: {e}")
        raise

@task(name="process_pet_files", retries=1)
def process_pet_files(pet_list, output_dir, netcdf_path):
    """Download and process PET files"""
    logger.info(f"Processing {len(pet_list)} PET files")
    processed_files = 0
    
    for file_url, date in pet_list:
        try:
            date_str = date.strftime('%Y%m%d')
            nc_file = os.path.join(netcdf_path, f"{date_str}.nc")
            
            if os.path.exists(nc_file):
                logger.info(f"NetCDF file already exists for {date_str}, skipping download and conversion")
                processed_files += 1
                continue
            
            logger.info(f"Processing PET file for {date_str}")
            pet_download_extract_bilfile(file_url, output_dir)
            pet_bil_netcdf(file_url, date, output_dir, netcdf_path)
            processed_files += 1
        except Exception as e:
            logger.error(f"Error processing PET file {file_url}: {e}")
    
    logger.info(f"Processed {processed_files} PET files")
    return processed_files

@task(name="read_pet_data")
def read_pet_data(netcdf_path, start_date, end_date):
    """Read PET data from NetCDF files"""
    try:
        logger.info(f"Reading PET data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        pds = pet_read_netcdf_files_in_date_range(netcdf_path, start_date, end_date)
        
        logger.info(f"Dataset dimensions: {list(pds.dims.keys())}")
        
        rename_dict = {}
        if 'x' in pds.dims:
            rename_dict['x'] = 'lon'
        if 'y' in pds.dims:
            rename_dict['y'] = 'lat'
            
        if not rename_dict and 'lon' not in pds.dims:
            for dim in pds.dims:
                if dim.lower() in ['longitude', 'long', 'x_dim']:
                    rename_dict[dim] = 'lon'
                elif dim.lower() in ['latitude', 'lat', 'y_dim']:
                    rename_dict[dim] = 'lat'
        
        if rename_dict:
            logger.info(f"Renaming dimensions: {rename_dict}")
            pds_renamed = pds.rename(rename_dict)
        else:
            logger.info("No dimensions need renaming")
            pds_renamed = pds
        
        logger.info(f"Successfully read PET data with shape {pds_renamed.dims}")
        return pds_renamed
    except Exception as e:
        logger.error(f"Error reading PET data: {e}")
        if 'pds' in locals():
            logger.error(f"Dataset info - dims: {pds.dims}, coords: {list(pds.coords)}, data_vars: {list(pds.data_vars)}")
        raise

@task(name="process_zone", retries=1)
def process_zone(input_path, pds, zone_str):
    """Process zone from combined shapefile and subset data"""
    master_shapefile = f'{input_path}data/WGS/geofsm-prod-all-zones-20240712.shp'
    
    if not os.path.exists(master_shapefile):
        logger.error(f"Master shapefile not found: {master_shapefile}")
        raise FileNotFoundError(f"Master shapefile not found: {master_shapefile}")
    
    if zone_str.isdigit():
        zone_str = f'zone{zone_str}'
    elif zone_str.startswith('zone '):
        zone_str = zone_str.replace('zone ', 'zone')
    elif not zone_str.startswith('zone'):
        zone_str = f'zone{zone_str}'
    zone_str = zone_str.replace('zonezone', 'zone')
    
    logger.info(f"Processing {zone_str} from combined shapefile")
    
    all_zones = gp.read_file(master_shapefile)
    if zone_str not in all_zones['zone'].values:
        logger.error(f"Zone '{zone_str}' not found in shapefile {master_shapefile}")
        raise ValueError(f"Zone '{zone_str}' not found in the shapefile.")
    
    try:
        z1ds, pdsz1, zone_extent = process_zone_from_combined(master_shapefile, zone_str, 1, pds)
        return z1ds, pdsz1, zone_extent
    except Exception as e:
        logger.error(f"Error processing zone {zone_str}: {e}")
        raise

@task(name="regrid_pet_data")
def regrid_pet_data(pdsz1, input_chunk_sizes, output_chunk_sizes, zone_extent):
    """Regrid PET data to match zone resolution"""
    logger.info("Regridding PET data")
    try:
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
        logger.error(f"Error regridding PET data: {e}")
        raise

@task(name="calculate_zone_means")
def calculate_zone_means(regridded_data, zone_ds):
    """Calculate mean PET values for each zone"""
    logger.info("Calculating zone means")
    try:
        return zone_mean_df(regridded_data, zone_ds)
    except Exception as e:
        logger.error(f"Error calculating zone means: {e}")
        raise

@task(name="save_pet_results")
def save_pet_results(results_df, input_path, zone_str, end_date):
    """Save processed PET results and update input data"""
    try:
        zone_input_path = f"{input_path}zone_wise_txt_files/"
        os.makedirs(f"{zone_input_path}{zone_str}", exist_ok=True)
        
        start_date = pd.to_datetime(results_df['time'].min())
        end_date_obj = pd.to_datetime(results_df['time'].max())
        
        # Update filename to include zone number (e.g., evap_zoneXX.txt)
        evap_filename = f"evap_{zone_str}.txt"
        logger.info(f"Updating PET input data for zone {zone_str}")
        pet_update_input_data(results_df, zone_input_path, zone_str, start_date, end_date_obj)
        
        output_dir = f"{input_path}PET/processed/{zone_str}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure end_date is in the right format for the filename
        if isinstance(end_date, datetime):
            end_date_str = end_date.strftime('%Y%m%d')
        else:
            end_date_str = end_date
        
        csv_file = f"{output_dir}/pet_{end_date_str}.csv"
        results_df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {zone_input_path}{zone_str}/{evap_filename} and {csv_file}")
        return f"{zone_input_path}{zone_str}/{evap_filename}"
    except Exception as e:
        logger.error(f"Error saving PET results: {e}")
        raise

@task(name="check_previously_processed")
def check_previously_processed(input_path, zone_str):
    """Check if PET data for a zone has already been processed"""
    zone_input_path = f"{input_path}zone_wise_txt_files/"
    evap_file = f"{zone_input_path}{zone_str}/evap.txt"
    
    if os.path.exists(evap_file):
        logger.info(f"PET data for zone {zone_str} already processed")
        return True, evap_file
    
    logger.info(f"No previously processed PET data found for zone {zone_str}")
    return False, None

@flow(name="process_single_zone_pet")
def process_single_zone_pet(input_path, pds, zone_str, end_date, force_reprocess=False):
    """Process PET data for a single zone"""
    logger.info(f"Processing PET data for zone {zone_str}")
    
    already_processed, existing_file = check_previously_processed(input_path, zone_str)
    if already_processed and not force_reprocess:
        logger.info(f"Using existing processed data for zone {zone_str}")
        return existing_file
    
    try:
        z1ds, pdsz1, zone_extent = process_zone(input_path, pds, zone_str)
        input_chunk_sizes = {'time': 10, 'lat': 30, 'lon': 30}
        output_chunk_sizes = {'lat': 300, 'lon': 300}
        regridded_data = regrid_pet_data(pdsz1, input_chunk_sizes, output_chunk_sizes, zone_extent)
        zone_means = calculate_zone_means(regridded_data, z1ds)
        result_file = save_pet_results(zone_means, input_path, zone_str, end_date)
        return result_file
    except Exception as e:
        logger.error(f"Error processing zone {zone_str}: {e}")
        return None

@flow(name="process_pet_data")
def process_pet_data(zone_str="6", force_reprocess=False, check_date=None):
    """
    Main flow to process PET data
    
    Args:
        zone_str (str): Zone identifier (e.g., "6" or "zone6")
        force_reprocess (bool): Force reprocessing even if data exists
        check_date (str or datetime): Date to check for data availability (default: today)
    """
    # Set up environment
    input_path, output_dir, netcdf_path, client = setup_environment()
    
    # Base URL for PET data
    url = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/daily/pet/downloads/daily/"
    
    try:
        # Handle the check date parameter
        if check_date is None:
            # Use current date by default
            check_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        elif isinstance(check_date, str):
            # Parse date string in multiple formats
            for fmt in ['%Y%m%d', '%Y-%m-%d', '%d/%m/%Y']:
                try:
                    check_date = datetime.strptime(check_date, fmt)
                    break
                except ValueError:
                    continue
            if isinstance(check_date, str):
                raise ValueError(f"Could not parse check_date: {check_date}")
        
        # Check data availability
        start_date, end_date = check_pet_data_availability(url, check_date, output_dir, netcdf_path)
        
        # Get PET files for the date range
        pet_list = get_pet_files(url, start_date, end_date)
        
        # Process PET files
        process_pet_files(pet_list, output_dir, netcdf_path)
        
        # Read processed data
        pds = read_pet_data(netcdf_path, start_date, end_date)
        
        # Process the specified zone
        result = process_single_zone_pet(input_path, pds, zone_str, end_date, force_reprocess)
        
        if result:
            logger.info(f"PET workflow completed successfully for {zone_str}")
            return result
        else:
            logger.warning(f"PET workflow for {zone_str} returned no results")
            return None
        
    except Exception as e:
        logger.error(f"Error in PET workflow: {e}")
        raise
    finally:
        # Clean up client
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Process PET data for a specific zone')
    parser.add_argument('--zone', type=str, default="6", help='Zone identifier (e.g., "6" or "zone6")')
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if data exists')
    parser.add_argument('--date', type=str, help='Date to check for data availability (format: YYYYMMDD)')
    args = parser.parse_args()
    
    # Run the flow
    process_pet_data(zone_str=args.zone, force_reprocess=args.force, check_date=args.date)
