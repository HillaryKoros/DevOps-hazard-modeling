from prefect import flow, task
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import xarray as xr
from dask.distributed import Client
import geopandas as gp

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
    """Download GEFS-CHIRPS files if they don't already exist"""
    # Create the directory path for the date
    date_dir = f"{download_dir}/{date_string}"
    
    # Check if the directory exists and has files in it
    if os.path.exists(date_dir) and os.listdir(date_dir):
        print(f"Data for {date_string} already exists in {date_dir}, skipping download.")
    else:
        print(f"Downloading data for {date_string}...")
        gefs_chrips_download_files(url_list, date_string, download_dir)
    
    return date_dir

@task
def process_gefs_data(input_path):
    """Process downloaded GEFS-CHIRPS data into xarray dataset"""
    return gefs_chrips_process(input_path)

@task
def process_zone(data_path, pds, zone_str):
    """Process zone from combined shapefile and subset data"""
    # Use the combined shapefile instead of individual zone files
    master_shapefile = f'{data_path}WGS/geofsm-prod-all-zones-20240712.shp'  # Update with your actual filename
    km_str = 1
    z1ds, pdsz1, zone_extent = process_zone_from_combined(master_shapefile, zone_str, km_str, pds)
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
        
        # Check if the data directory already exists
        existing_dir = f"{download_dir}/{date_string}"
        if os.path.exists(existing_dir) and os.listdir(existing_dir):
            print(f"Data for {date_string} already exists in {existing_dir}, skipping API check and download.")
            input_path = existing_dir
        else:
            # If no data found for the current date, try yesterday's date
            if not url_list:
                print(f"No files found for date {date_string}, trying yesterday's date...")
                yesterday_date = (datetime.strptime(date_string, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
                
                # Check if yesterday's data already exists
                yesterday_dir = f"{download_dir}/{yesterday_date}"
                if os.path.exists(yesterday_dir) and os.listdir(yesterday_dir):
                    print(f"Data for yesterday ({yesterday_date}) already exists in {yesterday_dir}, using that.")
                    date_string = yesterday_date
                    input_path = yesterday_dir
                else:
                    # Try to fetch yesterday's data
                    url_list = get_gefs_files(base_url, yesterday_date)
                    
                    if url_list:
                        print(f"Found {len(url_list)} files for yesterday ({yesterday_date})")
                        date_string = yesterday_date
                        input_path = download_gefs_files(url_list, date_string, download_dir)
                    else:
                        print(f"No files found for yesterday ({yesterday_date}) either. Exiting.")
                        client.close()
                        return None
            else:
                input_path = download_gefs_files(url_list, date_string, download_dir)
        
        # Process GEFS-CHIRPS data (only once for all zones)
        print("Processing downloaded files...")
        pds = process_gefs_data(input_path)
        
        # Get list of unique zone values from the shapefile
        master_shapefile = f'{data_path}WGS/geofsm-prod-all-zones-20240712.shp'  # Update with your actual filename
        all_zones = gp.read_file(master_shapefile)
        unique_zones = all_zones['zone'].unique()
        
        # Process each zone
        output_files = []
        for zone_str in unique_zones:  # Process all zones from the shapefile
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
    # Set the date to today by default
    date_string = datetime.now().strftime('%Y%m%d')
    
    print(f"Processing data for date: {date_string}")
    
    gefs_chirps_all_zones_workflow(date_string)