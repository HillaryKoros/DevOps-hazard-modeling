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
def get_best_available_date(base_url, days_to_check=7):  # Increase from 1 to 7 days
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
def setup_environment():
    data_path = os.getenv("data_path", "./data/")  # Default to ./data/ if not set
    download_dir = f'{data_path}geofsm-input/gefs-chirps'
    params = get_dask_client_params()
    client = Client(**params)
    print(f"Environment setup: data_path={data_path}, download_dir={download_dir}")
    return data_path, download_dir, client

@task
def get_gefs_files(base_url, date_string):
    all_files = gefs_chrips_list_tiff_files(base_url, date_string)
    print(f"Found {len(all_files)} files for date {date_string}")
    return all_files

@task
def download_gefs_files(url_list, date_string, download_dir):
    date_dir = f"{download_dir}/{date_string}"
    if os.path.exists(date_dir) and os.listdir(date_dir):
        print(f"Data for {date_string} already exists in {date_dir}, skipping download.")
    else:
        print(f"Downloading data for {date_string}...")
        gefs_chrips_download_files(url_list, date_string, download_dir)
    return date_dir

@task
def process_gefs_data(input_path):
    print(f"Processing GEFS-CHIRPS data from {input_path}")
    return gefs_chrips_process(input_path)

@task
def process_zone(data_path, pds, zone_str):
    master_shapefile = f'{data_path}WGS/geofsm-prod-all-zones-20240712.shp'
    km_str = 1
    z1ds, pdsz1, zone_extent = process_zone_from_combined(master_shapefile, zone_str, km_str, pds)
    print(f"Processed zone {zone_str}")
    return z1ds, pdsz1, zone_extent

@task
def regrid_precipitation_data(pdsz1, input_chunk_sizes, output_chunk_sizes, zone_extent):
    return regrid_dataset(
        pdsz1,
        input_chunk_sizes,
        output_chunk_sizes,
        zone_extent,
        regrid_method="bilinear"
    )

@task
def calculate_zone_means(regridded_data, zone_ds):
    return zone_mean_df(regridded_data, zone_ds)

@task
def save_csv_results(results_df, data_path, zone_str, date_string):
    output_dir = f"{data_path}geofsm-input/processed/{zone_str}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/gefs_chirps_{date_string}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"CSV results saved to {output_file}")
    return output_file

@task
def convert_csv_to_txt_format(input_csv_path):
    """
    Convert GEFS-CHIRPS CSV data to the required rain_{date}.txt format.
    """
    filename = os.path.basename(input_csv_path)
    if filename.startswith('gefs_chirps_'):
        date_string = filename.replace('gefs_chirps_', '').replace('.csv', '')
        try:
            date_obj = datetime.strptime(date_string, '%Y%m%d')
            date_ddd = date_obj.strftime('%Y%j')
        except ValueError:
            date_ddd = date_string
    else:
        date_ddd = 'converted'

    output_dir = os.path.dirname(input_csv_path)
    output_txt_path = os.path.join(output_dir, f"rain_{date_ddd}.txt")

    try:
        df = pd.read_csv(input_csv_path)
        if 'time' not in df.columns or 'group' not in df.columns or 'rain' not in df.columns:
            print(f"Error: Required columns not found in {input_csv_path}")
            return None

        df['NA'] = df['time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%Y%j'))
        dates = sorted(df['NA'].unique())
        groups = sorted(df['group'].unique())

        result_columns = ['NA'] + [str(int(g)) for g in groups]
        result_df = pd.DataFrame(columns=result_columns)
        result_df['NA'] = dates

        for group in groups:
            group_str = str(int(group))
            temp_df = df[df['group'] == group][['NA', 'rain']].set_index('NA')
            for date in dates:
                if date in temp_df.index:
                    result_df.loc[result_df['NA'] == date, group_str] = temp_df.loc[date, 'rain']
                else:
                    result_df.loc[result_df['NA'] == date, group_str] = 0

        for col in result_df.columns:
            if col != 'NA':
                result_df[col] = result_df[col].apply(lambda x: f"{float(x):.1f}" if pd.notna(x) else "0")

        header_line = ",".join(result_columns)
        with open(output_txt_path, 'w') as f:
            f.write(header_line + '\n')
            result_df.to_csv(f, index=False, header=False, lineterminator='\n')

        print(f"Converted {input_csv_path} to {output_txt_path}")
        return output_txt_path
    except Exception as e:
        print(f"Error converting {input_csv_path}: {e}")
        return None

@task
def copy_to_zone_wise_txt(data_path, zone_str, txt_file):
    zone_wise_dir = f"{data_path}zone_wise_txt_files/{zone_str}"
    os.makedirs(zone_wise_dir, exist_ok=True)
    # Update filename to include zone number (e.g., rain_zoneXX.txt)
    dst_file = f"{zone_wise_dir}/rain_{zone_str}.txt"
    with open(txt_file, 'r') as src_f:
        content = src_f.read()
    with open(dst_file, 'w') as dst_f:
        dst_f.write(content)
    print(f"Copied {txt_file} to {dst_file}")

@flow
def process_single_zone(data_path, pds, zone_str, date_string, copy_to_zone_wise=False):
    print(f"Processing zone {zone_str}...")
    
    # Check if data for this zone and date has already been processed
    date_obj = datetime.strptime(date_string, '%Y%m%d')
    date_ddd = date_obj.strftime('%Y%j')
    output_dir = f"{data_path}geofsm-input/processed/{zone_str}"
    processed_file = f"{output_dir}/rain_{date_ddd}.txt"
    
    if os.path.exists(processed_file):
        print(f"Data for zone {zone_str} and date {date_string} already processed. Skipping.")
        return processed_file
    
    z1ds, pdsz1, zone_extent = process_zone(data_path, pds, zone_str)
    input_chunk_sizes = {'time': 10, 'lat': 30, 'lon': 30}
    output_chunk_sizes = {'lat': 300, 'lon': 300}
    regridded_data = regrid_precipitation_data(pdsz1, input_chunk_sizes, output_chunk_sizes, zone_extent)
    zone_means = calculate_zone_means(regridded_data, z1ds)
    csv_file = save_csv_results(zone_means, data_path, zone_str, date_string)
    txt_file = convert_csv_to_txt_format(csv_file)
    if copy_to_zone_wise and txt_file:
        copy_to_zone_wise_txt(data_path, zone_str, txt_file)
    return txt_file

@flow
def gefs_chirps_all_zones_workflow(date_string: str = None, copy_to_zone_wise: bool = False):
    """
    Main workflow for processing GEFS-CHIRPS data for all zones.
    
    Args:
        date_string: Optional date in YYYYMMDD format. If None, the best available date will be determined.
        copy_to_zone_wise: Whether to copy the results to zone-wise txt files
        
    Returns:
        Dict containing the paths to the generated txt files
    """
    data_path, download_dir, client = setup_environment()
    
    try:
        base_url = "https://data.chc.ucsb.edu/products/EWX/data/forecasts/CHIRPS-GEFS_precip_v12/daily_16day/"
        
        # If date is not provided, find the most recent available date
        if date_string is None:
            date_string = get_best_available_date(base_url)
            if date_string is None:
                print("No data available for processing. Exiting workflow.")
                return {'txt_files': []}
        
        # Check if yesterday's data was already processed
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        yesterday_dir = f"{download_dir}/{yesterday}"
        
        # If we're processing yesterday's data and it's already downloaded, check if it was processed
        if date_string == yesterday and os.path.exists(yesterday_dir) and os.listdir(yesterday_dir):
            # Check if data was already processed for at least one zone
            master_shapefile = f'{data_path}WGS/geofsm-prod-all-zones-20240712.shp'
            all_zones = gp.read_file(master_shapefile)
            sample_zone = str(all_zones['zone'].iloc[0])
            
            if is_already_processed(data_path, sample_zone, yesterday):
                print(f"Data for {yesterday} was already processed. Skipping workflow.")
                return {'txt_files': []}
        
        # Get the list of files for the selected date
        url_list = get_gefs_files(base_url, date_string)
        
        # Check if data is already downloaded
        existing_dir = f"{download_dir}/{date_string}"
        if os.path.exists(existing_dir) and os.listdir(existing_dir):
            print(f"Data for {date_string} already exists in {existing_dir}, skipping download.")
            input_path = existing_dir
        else:
            input_path = download_gefs_files(url_list, date_string, download_dir)
        
        print("Processing downloaded files...")
        pds = process_gefs_data(input_path)
        
        # Process all zones
        master_shapefile = f'{data_path}WGS/geofsm-prod-all-zones-20240712.shp'
        all_zones = gp.read_file(master_shapefile)
        unique_zones = all_zones['zone'].unique()
        output_files = []
        
        for zone_str in unique_zones:
            try:
                txt_file = process_single_zone(data_path, pds, zone_str, date_string, copy_to_zone_wise)
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
    # Let the workflow automatically determine the best available date
    result = gefs_chirps_all_zones_workflow(copy_to_zone_wise=True)
    print(f"Generated files: {result['txt_files']}")