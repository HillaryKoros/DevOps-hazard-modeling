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
    zone_mean_df,
    pet_update_input_data
)

load_dotenv()

# Default to yesterday if date is not provided
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

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
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(netcdf_path, exist_ok=True)
    
    params = get_dask_client_params()
    client = Client(**params)
    
    print(f"Environment setup complete. Using data_path: {data_path}")
    return data_path, output_dir, netcdf_path, client

@task
def get_most_recent_pet_files(url):
    """
    Get today's or yesterday's PET file from the server listing.
    
    Args:
        url (str): URL for PET data directory
        
    Returns:
        tuple: (file_url, file_date) for the most recent file (today or yesterday)
    """
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin
    
    print(f"Looking for today's or yesterday's PET file from {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all tar.gz file links
        pet_files = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and href.endswith('.tar.gz') and href.startswith('et'):
                # Extract date info from link text or from timestamp
                file_url = urljoin(url, href)
                
                # Try to get date from the filename (et240409.tar.gz -> 2024-04-09)
                try:
                    # Extract date part (et240409 -> 240409)
                    date_part = href.replace('et', '').split('.')[0]
                    year = int('20' + date_part[:2])
                    month = int(date_part[2:4])
                    day = int(date_part[4:6])
                    file_date = datetime(year, month, day)
                    
                    pet_files.append((file_url, file_date, href))
                except Exception as e:
                    print(f"Could not parse date from {href}: {e}")
        
        # Sort by date, most recent first
        pet_files.sort(key=lambda x: x[1], reverse=True)
        
        # Get today's and yesterday's dates
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday = today - timedelta(days=1)
        
        # Find today's file
        for url, date, filename in pet_files:
            if date.date() == today.date():
                print(f"Found today's file: {filename} ({date.strftime('%Y-%m-%d')})")
                return url, date
        
        # If today's file not found, find yesterday's file
        for url, date, filename in pet_files:
            if date.date() == yesterday.date():
                print(f"Today's file not found. Using yesterday's file: {filename} ({date.strftime('%Y-%m-%d')})")
                return url, date
        
        # If neither found, use the most recent file
        if pet_files:
            url, date, filename = pet_files[0]
            print(f"Neither today's nor yesterday's file found. Using most recent: {filename} ({date.strftime('%Y-%m-%d')})")
            return url, date
        else:
            print("No PET files found on server")
            return None, None
    
    except Exception as e:
        print(f"Error fetching PET files: {e}")
        return None, None

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
def read_pet_data(netcdf_path, start_date, end_date):
    """Read PET data from NetCDF files"""
    try:
        print(f"Reading PET data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Handle the case where no files exist for the date range
        nc_files = []
        date_range = pd.date_range(start=start_date, end=end_date)
        
        for date in date_range:
            date_str = date.strftime('%Y%m%d')
            nc_file = os.path.join(netcdf_path, f"{date_str}.nc")
            if os.path.exists(nc_file):
                nc_files.append(nc_file)
        
        if not nc_files:
            print(f"No NetCDF files found in the date range {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            print("Using the most recent file and repeating it for the entire date range")
            
            # Find the most recent file available
            all_nc_files = glob.glob(os.path.join(netcdf_path, "*.nc"))
            if not all_nc_files:
                raise FileNotFoundError("No NetCDF files found in the directory")
            
            dates = [datetime.strptime(os.path.basename(f).replace('.nc', ''), '%Y%m%d') for f in all_nc_files]
            most_recent_idx = dates.index(max(dates))
            most_recent_file = all_nc_files[most_recent_idx]
            
            # Use this file for all dates in the range
            nc_files = [most_recent_file] * len(date_range)
            print(f"Using {most_recent_file} for all dates")
        
        # Read and combine the files
        datasets = []
        for i, (file, date) in enumerate(zip(nc_files, date_range)):
            if i == 0:
                print(f"Reading first file: {file}")
            
            # Open the file
            ds = xr.open_dataset(file)
            
            # Remove spatial_ref if it exists
            if 'spatial_ref' in ds.variables:
                ds = ds.drop_vars('spatial_ref')
                
            # Rename variables if needed
            if 'band' in ds.variables:
                ds = ds.drop_vars('band')
                
            if 'date' in ds.variables:
                ds = ds.drop_vars('date')
                
            # Squeeze dimensions if needed
            if 'band' in ds.dims:
                ds = ds.squeeze('band')
                
            # Rename the data variable if needed
            if '__xarray_dataarray_variable__' in ds.data_vars:
                ds = ds.rename_vars({'__xarray_dataarray_variable__': 'pet'})
                
            # Set the date
            ds = ds.expand_dims(time=[date])
            datasets.append(ds)
            
            if i == 0:
                print(f"First dataset dims: {ds.dims}, coords: {list(ds.coords)}, data_vars: {list(ds.data_vars)}")
        
        # Combine all datasets
        print(f"Combining {len(datasets)} datasets")
        combined_dataset = xr.concat(datasets, dim='time')
        
        # Rename coordinates to ensure compatibility
        rename_dict = {}
        if 'x' in combined_dataset.dims and 'lon' not in combined_dataset.dims:
            rename_dict['x'] = 'lon'
        if 'y' in combined_dataset.dims and 'lat' not in combined_dataset.dims:
            rename_dict['y'] = 'lat'
            
        if rename_dict:
            print(f"Renaming dimensions: {rename_dict}")
            combined_dataset = combined_dataset.rename(rename_dict)
        
        print(f"Successfully read PET data with shape {combined_dataset.dims}")
        return combined_dataset
    except Exception as e:
        print(f"Error reading PET data: {e}")
        raise

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
def save_pet_results(results_df, data_path, zone_str, end_date):
    """Save processed PET results and update input data"""
    try:
        # Create output directories
        output_dir = f"{data_path}geofsm-input/processed/{zone_str}"
        zone_input_path = f"{data_path}zone_wise_txt_files/"
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{zone_input_path}{zone_str}", exist_ok=True)
        
        # Format dates
        start_date = pd.to_datetime(results_df['time'].min())
        end_date_obj = pd.to_datetime(results_df['time'].max())
        
        if isinstance(end_date, datetime):
            end_date_str = end_date.strftime('%Y%m%d')
        else:
            end_date_str = end_date
            
        # Save CSV file
        csv_file = f"{output_dir}/pet_{end_date_str}.csv"
        results_df.to_csv(csv_file, index=False)
        print(f"CSV results saved to {csv_file}")
        
        # Update PET input data
        pet_update_input_data(results_df, zone_input_path, zone_str, start_date, end_date_obj)
        evap_file = f"{zone_input_path}{zone_str}/evap_{end_date_obj.strftime('%Y%j')}.txt"
        print(f"PET input data updated: {evap_file}")
        
        return evap_file
    except Exception as e:
        print(f"Error saving PET results: {e}")
        raise

@task
def copy_to_zone_wise_txt(data_path, zone_str, txt_file):
    """Copy the text file to the zone-wise directory"""
    zone_wise_dir = f"{data_path}zone_wise_txt_files/{zone_str}"
    os.makedirs(zone_wise_dir, exist_ok=True)
    
    # Update filename to include zone number
    dst_file = f"{zone_wise_dir}/pet_{zone_str}.txt"
    
    try:
        with open(txt_file, 'r') as src_f:
            content = src_f.read()
        with open(dst_file, 'w') as dst_f:
            dst_f.write(content)
        print(f"Copied {txt_file} to {dst_file}")
        return dst_file
    except Exception as e:
        print(f"Error copying to zone-wise file: {e}")
        return None

@flow
def process_single_zone_pet(data_path, pds, zone_str, date_string, copy_to_zone_wise=False):
    """Process PET data for a single zone"""
    print(f"Processing zone {zone_str}...")
    
    # Check if data for this zone and date has already been processed
    output_dir = f"{data_path}geofsm-input/processed/{zone_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    z1ds, pdsz1, zone_extent = process_zone(data_path, pds, zone_str)
    regridded_data = regrid_pet_data(pdsz1, zone_extent)
    zone_means = calculate_zone_means(regridded_data, z1ds)
    txt_file = save_pet_results(zone_means, data_path, zone_str, date_string)
    
    if copy_to_zone_wise and txt_file:
        copy_to_zone_wise_txt(data_path, zone_str, txt_file)
    
    return txt_file

@flow
def pet_all_zones_workflow(copy_to_zone_wise: bool = False):
    """
    Main workflow for processing PET data for all zones.
    
    Args:
        copy_to_zone_wise: Whether to copy the results to zone-wise txt files
        
    Returns:
        Dict containing the paths to the generated txt files
    """
    data_path, output_dir, netcdf_path, client = setup_environment()
    
    try:
        # Base URL for PET data
        url = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/daily/pet/downloads/daily/"
        
        # Check if master shapefile exists before continuing
        master_shapefile = f'{data_path}WGS/geofsm-prod-all-zones-20240712.shp'
        if not os.path.exists(master_shapefile):
            print(f"ERROR: Master shapefile not found at {master_shapefile}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Available files in {os.path.dirname(master_shapefile) or '.'}:")
            if os.path.exists(os.path.dirname(master_shapefile) or '.'):
                print(os.listdir(os.path.dirname(master_shapefile) or '.'))
            else:
                print(f"Directory {os.path.dirname(master_shapefile)} does not exist")
            raise FileNotFoundError(f"Master shapefile not found: {master_shapefile}")
        else:
            print(f"Found master shapefile: {master_shapefile}")
        
        # Get today's or yesterday's PET file directly from the server
        most_recent_url, most_recent_date = get_most_recent_pet_files(url)
        
        if not most_recent_url or not most_recent_date:
            print("No suitable PET file found on server. Cannot proceed.")
            return {'txt_files': []}
        
        # Process the file
        print(f"Processing PET file from {most_recent_date.strftime('%Y-%m-%d')}")
        pet_download_extract_bilfile(most_recent_url, output_dir)
        pet_bil_netcdf(most_recent_url, most_recent_date, output_dir, netcdf_path)
        
        # Set date range to just the single day of the file
        date_string = most_recent_date.strftime('%Y%m%d')
        
        # Read processed data for just that day
        print(f"Reading PET data for {most_recent_date.strftime('%Y-%m-%d')}")
        nc_file = os.path.join(netcdf_path, f"{date_string}.nc")
        
        if not os.path.exists(nc_file):
            print(f"ERROR: NetCDF file not found at {nc_file}")
            return {'txt_files': []}
        
        # Open the dataset
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
        ds = ds.expand_dims(time=[most_recent_date])
        
        # Rename coordinates if needed
        rename_dict = {}
        if 'x' in ds.dims and 'lon' not in ds.dims:
            rename_dict['x'] = 'lon'
        if 'y' in ds.dims and 'lat' not in ds.dims:
            rename_dict['y'] = 'lat'
        
        if rename_dict:
            print(f"Renaming dimensions: {rename_dict}")
            ds = ds.rename(rename_dict)
        
        print(f"Dataset ready with dimensions: {ds.dims}")
        
        # Process all zones
        all_zones = gp.read_file(master_shapefile)
        unique_zones = all_zones['zone'].unique()
        output_files = []
        
        for zone_str in unique_zones:
            try:
                txt_file = process_single_zone_pet(data_path, ds, zone_str, date_string, copy_to_zone_wise)
                if txt_file:
                    output_files.append(txt_file)
            except Exception as e:
                print(f"Error processing {zone_str}: {e}")
        
        print(f"Workflow completed successfully! Processed {len(output_files)} zones")
        return {'txt_files': output_files}
    
    except Exception as e:
        print(f"Error in workflow: {e}")
        raise
    finally:
        client.close()
        

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Process PET data for hydrological modeling')
    parser.add_argument('--copy-to-zone-wise', action='store_true', 
                        help='Copy output files to zone_wise_txt_files directory')
    
    args = parser.parse_args()
    
    print(f"Processing most recent PET data")
    result = pet_all_zones_workflow(args.copy_to_zone_wise)
    print(f"Generated files: {result['txt_files']}")