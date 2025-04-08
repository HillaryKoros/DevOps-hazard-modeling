from prefect import flow, task
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import xarray as xr
from dask.distributed import Client
import geopandas as gp
import pandas as pd
import glob

from utils import (
    imerg_list_files_by_date,
    imerg_download_files,
    imerg_read_tiffs_to_dataset,
    get_dask_client_params,
    process_zone_from_combined,
    regrid_dataset,
    zone_mean_df,
    make_zones_geotif
)

load_dotenv()

# Default to yesterday if date is not provided
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

@task
def setup_environment():
    data_path = os.getenv("data_path", "./data/")  # Default to ./data/ if not set
    imerg_store = f'{data_path}geofsm-input/imerg'
    params = get_dask_client_params()
    client = Client(**params)
    print(f"Environment setup: data_path={data_path}, imerg_store={imerg_store}")
    return data_path, imerg_store, client

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
    return imerg_read_tiffs_to_dataset(input_path, start_date, end_date)

@task
def process_zone(data_path, imerg_ds, zone_str):
    """Process a zone from the combined shapefile"""
    master_shapefile = f'{data_path}WGS/geofsm-prod-all-zones-20240712.shp'
    km_str = 1
    z1ds, zone_subset_ds, zone_extent = process_zone_from_combined(master_shapefile, zone_str, km_str, imerg_ds)
    print(f"Processed zone {zone_str}")
    return z1ds, zone_subset_ds, zone_extent

@task
def regrid_precipitation_data(zone_subset_ds, input_chunk_sizes, output_chunk_sizes, zone_extent):
    """Regrid the precipitation data to match the zone extent at 1km resolution"""
    return regrid_dataset(
        zone_subset_ds,
        input_chunk_sizes,
        output_chunk_sizes,
        zone_extent,
        regrid_method="bilinear"
    )

@task
def calculate_zone_means(regridded_data, zone_ds):
    """Calculate zonal means for the regridded data"""
    return zone_mean_df(regridded_data, zone_ds)

@task
def save_csv_results(results_df, data_path, zone_str, date_string):
    """Save the zonal means to a CSV file"""
    output_dir = f"{data_path}geofsm-input/processed/{zone_str}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/imerg_{date_string}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"CSV results saved to {output_file}")
    return output_file

@task
def convert_csv_to_txt_format(input_csv_path):
    """
    Convert IMERG CSV data to the required rain_{date}.txt format.
    """
    filename = os.path.basename(input_csv_path)
    # Extract date string from filename (assumes pattern 'imerg_YYYYMMDD.csv')
    if filename.startswith('imerg_'):
        date_string = filename.replace('imerg_', '').replace('.csv', '')
        try:
            date_obj = datetime.strptime(date_string, '%Y%m%d')
            date_ddd = date_obj.strftime('%Y%j')  # Format as YYYYDDD where DDD is day of year
        except ValueError:
            date_ddd = date_string
    else:
        date_ddd = 'converted'

    output_dir = os.path.dirname(input_csv_path)
    output_txt_path = os.path.join(output_dir, f"imerg_{date_ddd}.txt")

    try:
        df = pd.read_csv(input_csv_path)
        if 'time' not in df.columns or 'group' not in df.columns:
            print(f"Error: Required columns not found in {input_csv_path}")
            return None

        # Use the main precipitation column (should be named similarly to 'precipitation')
        # Find the precipitation column - it might be named 'rain' or similar
        precip_cols = [col for col in df.columns if col not in ['time', 'group']]
        if not precip_cols:
            print(f"Error: No precipitation column found in {input_csv_path}")
            return None
        
        # Use the first precipitation column found
        precip_col = precip_cols[0]
        
        # Convert times to YYYYDDD format
        df['NA'] = df['time'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d').strftime('%Y%j'))
        dates = sorted(df['NA'].unique())
        groups = sorted(df['group'].unique())

        # Create result dataframe
        result_columns = ['NA'] + [str(int(g)) for g in groups]
        result_df = pd.DataFrame(columns=result_columns)
        result_df['NA'] = dates

        # Fill values by group
        for group in groups:
            group_str = str(int(group))
            temp_df = df[df['group'] == group][['NA', precip_col]].set_index('NA')
            for date in dates:
                if date in temp_df.index:
                    result_df.loc[result_df['NA'] == date, group_str] = temp_df.loc[date, precip_col]
                else:
                    result_df.loc[result_df['NA'] == date, group_str] = 0

        # Format precipitation values to one decimal place
        for col in result_df.columns:
            if col != 'NA':
                result_df[col] = result_df[col].apply(lambda x: f"{float(x):.1f}" if pd.notna(x) else "0")

        # Write result to text file
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
    """Copy the text file to the zone-wise directory"""
    zone_wise_dir = f"{data_path}zone_wise_txt_files/{zone_str}"
    os.makedirs(zone_wise_dir, exist_ok=True)
    # Update filename to include zone number
    dst_file = f"{zone_wise_dir}/imerg_{zone_str}.txt"
    with open(txt_file, 'r') as src_f:
        content = src_f.read()
    with open(dst_file, 'w') as dst_f:
        dst_f.write(content)
    print(f"Copied {txt_file} to {dst_file}")
    return dst_file

@flow
def process_single_zone(data_path, imerg_ds, zone_str, date_string, copy_to_zone_wise=False):
    """Process a single zone"""
    print(f"Processing zone {zone_str}...")
    z1ds, zone_subset_ds, zone_extent = process_zone(data_path, imerg_ds, zone_str)
    input_chunk_sizes = {'time': 10, 'lat': 30, 'lon': 30}
    output_chunk_sizes = {'lat': 300, 'lon': 300}
    regridded_data = regrid_precipitation_data(zone_subset_ds, input_chunk_sizes, output_chunk_sizes, zone_extent)
    zone_means = calculate_zone_means(regridded_data, z1ds)
    csv_file = save_csv_results(zone_means, data_path, zone_str, date_string)
    txt_file = convert_csv_to_txt_format(csv_file)
    if copy_to_zone_wise and txt_file:
        copy_to_zone_wise_txt(data_path, zone_str, txt_file)
    return txt_file

@flow
def imerg_all_zones_workflow(start_date: str = yesterday, end_date: str = ""):
    """Process IMERG data for all zones"""
    copy_to_zone_wise = False
    
    # Handle default values for start_date and end_date
    if not start_date:
        start_date = yesterday
    
    if not end_date:
        # Default to same as start_date if not specified
        end_date = start_date
    
    date_string = start_date  # Use start date for output filenames
    
    data_path, imerg_store, client = setup_environment()
    try:
        # Get file list
        file_list = get_imerg_files(start_date, end_date)
        
        # Download files
        download_dir = download_imerg_files(file_list, imerg_store)
        
        # Process data
        imerg_ds = process_imerg_data(download_dir, start_date, end_date)
        
        # Get all zones
        master_shapefile = f'{data_path}WGS/geofsm-prod-all-zones-20240712.shp'
        all_zones = gp.read_file(master_shapefile)
        unique_zones = all_zones['zone'].unique()
        
        # Process each zone
        output_files = []
        for zone_str in unique_zones:
            try:
                txt_file = process_single_zone(data_path, imerg_ds, zone_str, date_string, copy_to_zone_wise)
                if txt_file:
                    output_files.append(txt_file)
            except Exception as e:
                print(f"Error processing zone {zone_str}: {e}")
        
        print(f"Workflow completed successfully! Processed {len(output_files)} zones")
        return {'txt_files': output_files}
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
    parser.add_argument('--copy-to-zone-wise', action='store_true', 
                        help='Copy output files to zone_wise_txt_files directory')
    
    args = parser.parse_args()
    
    print(f"Processing IMERG data from {args.start_date} to {args.end_date or args.start_date}")
    result = imerg_all_zones_workflow(args.start_date, args.end_date)
    print(f"Generated files: {result['txt_files']}")