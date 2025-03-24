from prefect import flow, task
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import xarray as xr
from dask.distributed import Client

from utils import (
    gefs_chrips_list_tiff_files,
    gefs_chrips_download_files,
    gefs_chrips_process,
    get_dask_client_params,
    process_zone_and_subset_data,
    regrid_dataset,
    zone_mean_df
)

load_dotenv()

@task
def setup_environment():
    """Set up the environment for data processing"""
    data_path = os.getenv("data_path")
    download_dir = f'{data_path}geofsm-input/gefs-chirps'
    params = get_dask_client_params()
    client = Client(**params)
    return data_path, download_dir, client

@task
def get_gefs_files(base_url, date_string):
    """Get the full list of GEFS-CHIRPS files for the date"""
    all_files = gefs_chrips_list_tiff_files(base_url, date_string)
    print(f"Found {len(all_files)} files for date {date_string}")
    return all_files

@task
def download_gefs_files(url_list, date_string, download_dir):
    """Download GEFS-CHIRPS files"""
    gefs_chrips_download_files(url_list, date_string, download_dir)
    return f"{download_dir}/{date_string}"

@task
def process_gefs_data(input_path):
    """Process downloaded GEFS-CHIRPS data into xarray dataset"""
    return gefs_chrips_process(input_path)

@task
def process_zone(data_path, pds, zone_str):
    """Process zone shapefile and subset data"""
    shapefl_name = f'{data_path}WGS/{zone_str}.shp'
    km_str = 1
    z1ds, pdsz1, zone_extent = process_zone_and_subset_data(shapefl_name, km_str, zone_str, pds)
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
def save_results(results_df, data_path, zone_str, date_string):
    """Save processed results to file"""
    output_dir = f"{data_path}geofsm-input/processed/{zone_str}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/gefs_chirps_{date_string}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    return output_file

@flow
def process_single_zone(data_path, pds, zone_str, date_string):
    """Process a single zone and save results"""
    print(f"Processing zone {zone_str}...")
    
    # Process zone data
    z1ds, pdsz1, zone_extent = process_zone(data_path, pds, zone_str)
    
    # Regrid data
    input_chunk_sizes = {'time': 10, 'lat': 30, 'lon': 30}
    output_chunk_sizes = {'lat': 300, 'lon': 300}
    regridded_data = regrid_precipitation_data(pdsz1, input_chunk_sizes, output_chunk_sizes, zone_extent)
    
    # Calculate zone means
    zone_means = calculate_zone_means(regridded_data, z1ds)
    
    # Save results
    output_file = save_results(zone_means, data_path, zone_str, date_string)
    
    return output_file

@flow
def gefs_chirps_all_zones_workflow(date_string: str = None):
    """
    Process GEFS-CHIRPS data for all zones for a specific date
    
    Parameters:
    -----------
    date_string : str
        Date in format 'YYYYMMDD' for which to process data.
        If None, today's date will be used.
    """
    # Use today's date if not specified
    if date_string is None:
        date_string = datetime.now().strftime('%Y%m%d')
    
    # Setup environment
    data_path, download_dir, client = setup_environment()
    
    try:
        # Get and download GEFS-CHIRPS files
        base_url = "https://data.chc.ucsb.edu/products/EWX/data/forecasts/CHIRPS-GEFS_precip_v12/daily_16day/"
        url_list = get_gefs_files(base_url, date_string)
        
        if not url_list:
            print(f"No files found for date {date_string}")
            client.close()
            return None
            
        input_path = download_gefs_files(url_list, date_string, download_dir)
        
        # Process GEFS-CHIRPS data (only once for all zones)
        print("Processing downloaded files...")
        pds = process_gefs_data(input_path)
        
        # Process each zone
        output_files = []
        for zone_num in range(1, 7):  # Process zones 1 through 6
            zone_str = f"zone{zone_num}"
            try:
                output_file = process_single_zone(data_path, pds, zone_str, date_string)
                output_files.append(output_file)
            except Exception as e:
                print(f"Error processing {zone_str}: {e}")
        
        print(f"Workflow completed successfully! Processed {len(output_files)} zones.")
        return output_files
    
    except Exception as e:
        print(f"Error in workflow: {e}")
        raise
    
    finally:
        # Ensure client is closed even if there's an error
        client.close()

if __name__ == "__main__":
    # Set the date directly in the code - this is yesterday's date
    # You can modify this line directly when you need to process a different date
    date_string = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    
    print(f"Processing data for date: {date_string}")
    
    gefs_chirps_all_zones_workflow(date_string)