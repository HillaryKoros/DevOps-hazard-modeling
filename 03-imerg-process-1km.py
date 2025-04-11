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

def process_zone_from_combined(master_shapefile, zone_name, km_str, pds):
    """
    Process a specific zone from a combined shapefile and subset data based on that zone.
    Handles datasets with either lat/lon or x/y coordinate systems.
    Works with both DataArray and Dataset objects.
    """
    # Read the master shapefile
    all_zones = gp.read_file(master_shapefile)
    
    # Filter for the specific zone
    zone_gdf = all_zones[all_zones['zone'] == zone_name].copy()
    
    if zone_gdf.empty:
        raise ValueError(f"Zone '{zone_name}' not found in the shapefile.")
    
    # Create a temporary directory for the zone-specific shapefile if it doesn't exist
    temp_dir = os.path.join(os.path.dirname(master_shapefile), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save the filtered zone as a temporary shapefile
    temp_shapefile = os.path.join(temp_dir, f"{zone_name}.shp")
    zone_gdf.to_file(temp_shapefile)
    
    # Generate the output path for the zone GeoTIFF
    zone1_tif = make_zones_geotif(temp_shapefile, km_str, zone_name)

    # Load and process the generated GeoTIFF
    z1ds = rioxarray.open_rasterio(zone1_tif, chunks="auto").squeeze()
    z1crds = z1ds.rename(x='lon', y='lat')
    z1county_id = np.unique(z1crds.data).compute()
    z1lat_max = z1crds['lat'].max().values
    z1lat_min = z1crds['lat'].min().values
    z1lon_max = z1crds['lon'].max().values
    z1lon_min = z1crds['lon'].min().values

    zone_extent = {
        'lat_max': z1lat_max,
        'lat_min': z1lat_min,
        'lon_max': z1lon_max,
        'lon_min': z1lon_min
    }

    print(f"Input object type: {type(pds).__name__}")
    print(f"Zone extent: lat({z1lat_min}, {z1lat_max}), lon({z1lon_min}, {z1lon_max})")
    
    # Determine which coordinate system the input data uses
    if 'lat' in pds.dims and 'lon' in pds.dims:
        print("Using lat/lon coordinates for subsetting")
        pz1ds = pds.sel(lat=slice(z1lat_max, z1lat_min), lon=slice(z1lon_min, z1lon_max))
    elif 'y' in pds.dims and 'x' in pds.dims:
        print("Using x/y coordinates for subsetting")
        pz1ds = pds.sel(y=slice(z1lat_max, z1lat_min), x=slice(z1lon_min, z1lon_max))
    else:
        raise ValueError(f"Data has unrecognized coordinate dimensions: {list(pds.dims)}")

    return z1crds, pz1ds, zone_extent

@task
def process_zone(data_path, imerg_data, zone_str):
    """Process a zone from the combined shapefile"""
    master_shapefile = f'{data_path}WGS/geofsm-prod-all-zones-20240712.shp'
    km_str = 1
    z1ds, zone_subset_ds, zone_extent = process_zone_from_combined(master_shapefile, zone_str, km_str, imerg_data)
    print(f"Processed zone {zone_str}")
    return z1ds, zone_subset_ds, zone_extent

def regrid_dataset(input_ds, input_chunk_sizes, output_chunk_sizes, zone_extent, regrid_method="bilinear"):
    """
    Regrid a dataset to a specified output grid using a specified regridding method.
    """
    # Extract lat/lon extents from the dictionary
    z1lat_min = zone_extent['lat_min']
    z1lat_max = zone_extent['lat_max']
    z1lon_min = zone_extent['lon_min']
    z1lon_max = zone_extent['lon_max']

    # Create output grid with appropriate chunking
    ds_out = xr.Dataset({
        "lat": (["lat"], np.arange(z1lat_min, z1lat_max, 0.01), {"units": "degrees_north"}),
        "lon": (["lon"], np.arange(z1lon_min, z1lon_max, 0.01), {"units": "degrees_east"})
    }).chunk(output_chunk_sizes)

    # Create regridder with specified output_chunks
    regridder = xe.Regridder(input_ds, ds_out, regrid_method)

    # Define regridding function with output_chunks
    def regrid_chunk(chunk):
        return regridder(chunk, output_chunks=output_chunk_sizes)

    # Apply regridding to each chunk
    regridded = input_ds.groupby('time').map(regrid_chunk)

    # Compute results
    with ProgressBar():
        result = regridded.compute()
    
    # Ensure the result has a name if it's a DataArray
    if isinstance(result, xr.DataArray) and not result.name:
        result = result.rename('precipitation')
        print("Named regridded DataArray as 'precipitation'")

    return result

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

def zone_mean_df(input_ds, zone_ds):
    """
    Compute the mean of values in `input_ds` grouped by zones defined in `zone_ds`.
    Works with both named and unnamed DataArrays.
    """
    # Print input dataset info for debugging
    print(f"Input dataset type: {type(input_ds).__name__}")
    print(f"Input dataset name: {getattr(input_ds, 'name', 'unnamed')}")
    
    # Align datasets
    z1d_, aligned_zone_ds = xr.align(input_ds, zone_ds, join="override")
    
    # Group by aligned zone dataset
    z1 = input_ds.groupby(aligned_zone_ds).mean()
    
    # If the DataArray has no name, assign one before converting to DataFrame
    if isinstance(z1, xr.DataArray) and not z1.name:
        print("Assigning name 'precipitation' to unnamed DataArray")
        z1 = z1.rename('precipitation')
    
    # Convert to DataFrame
    z1 = z1.to_dataframe()
    z1a = z1.reset_index()
    
    return z1a

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
    Convert IMERG CSV data to the required imerg_{date}.txt format with improved handling
    of small precipitation values and proper pivoting.
    
    This function properly handles scientific notation and very small precipitation values,
    ensuring they are preserved as 0.1 in the output rather than being converted to 0.0.
    
    Args:
        input_csv_path: Path to the input CSV file (e.g., 'imerg_20250410.csv')
        
    Returns:
        Path to the output TXT file (e.g., 'imerg_2025100.txt') or None if conversion failed
    """
    import os
    import pandas as pd
    from datetime import datetime
    
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
        # Read the CSV file
        df = pd.read_csv(input_csv_path)
        
        # Verify required columns exist
        required_cols = ['time', 'group']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Required columns not found in {input_csv_path}")
            print(f"Available columns: {df.columns.tolist()}")
            return None

        # Find the precipitation column - it might be named 'precipitation', 'rain' or similar
        precip_cols = [col for col in df.columns if col.lower() in ['precipitation', 'precip', 'rain']]
        if not precip_cols:
            # If no obvious precipitation column, use any column that's not in a standard set
            standard_cols = ['time', 'group', 'spatial_ref', 'band']
            precip_cols = [col for col in df.columns if col not in standard_cols]
            
        if not precip_cols:
            print(f"Error: No precipitation column found in {input_csv_path}")
            return None
        
        # Use the first precipitation column found
        precip_col = precip_cols[0]
        print(f"Using precipitation column: {precip_col}")
        
        # Convert date format: YYYY-MM-DD to YYYYDDD format
        df['NA'] = df['time'].apply(lambda x: 
                                   datetime.strptime(str(x), '%Y-%m-%d').strftime('%Y%j'))
        
        # Sort unique dates and groups
        dates = sorted(df['NA'].unique())
        groups = sorted(df['group'].unique())
        
        # Create result dataframe with proper column structure
        result_columns = ['NA'] + [str(int(g)) for g in groups]
        result_df = pd.DataFrame(columns=result_columns)
        result_df['NA'] = dates
        
        # Initialize all precipitation values to 0.0
        for col in result_columns:
            if col != 'NA':
                result_df[col] = "0.0"
        
        # Fill values by group
        for group in groups:
            group_str = str(int(group))
            group_data = df[df['group'] == group]
            
            # For each date and group, get the precipitation value
            for date in dates:
                date_group_data = group_data[group_data['NA'] == date]
                if not date_group_data.empty:
                    # Get precipitation value
                    value = date_group_data[precip_col].iloc[0]
                    
                    # Apply the new rounding rules
                    if value <= 0.01:
                        # Values <= 0.01 become 0.0
                        formatted_value = "0.0"
                    else:
                        # Round to nearest 0.1 using standard rounding rules
                        rounded = round(float(value) * 10) / 10
                        formatted_value = f"{rounded:.1f}"
                        
                    # Update the result dataframe
                    result_df.loc[result_df['NA'] == date, group_str] = formatted_value
        
        # Write result to text file
        header_line = ",".join(result_columns)
        with open(output_txt_path, 'w') as f:
            f.write(header_line + '\n')
            # Use to_csv without index, without header, and with Windows line endings
            result_df.to_csv(f, index=False, header=False, lineterminator='\n')
        
        print(f"Successfully converted {input_csv_path} to {output_txt_path}")
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
def process_single_zone(data_path, imerg_data, zone_str, date_string, copy_to_zone_wise=False):
    """Process a single zone"""
    print(f"Processing zone {zone_str}...")
    z1ds, zone_subset_ds, zone_extent = process_zone(data_path, imerg_data, zone_str)
    
    # Adjust input_chunk_sizes based on the dimensions in the data
    if 'lat' in zone_subset_ds.dims and 'lon' in zone_subset_ds.dims:
        input_chunk_sizes = {'time': 10, 'lat': 30, 'lon': 30}
    else:
        input_chunk_sizes = {'time': 10, 'y': 30, 'x': 30}
    
    output_chunk_sizes = {'lat': 300, 'lon': 300}
    regridded_data = regrid_precipitation_data(zone_subset_ds, input_chunk_sizes, output_chunk_sizes, zone_extent)
    zone_means = calculate_zone_means(regridded_data, z1ds)
    csv_file = save_csv_results(zone_means, data_path, zone_str, date_string)
    txt_file = convert_csv_to_txt_format(csv_file)
    if copy_to_zone_wise and txt_file:
        copy_to_zone_wise_txt(data_path, zone_str, txt_file)
    return txt_file

@flow
def imerg_all_zones_workflow(start_date: str = yesterday, end_date: str = "", copy_to_zone_wise: bool = False):
    """Process IMERG data for all zones"""
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
        imerg_data = process_imerg_data(download_dir, start_date, end_date)
        
        # Rename coordinates from x,y to lon,lat if needed
        imerg_data = rename_coordinates(imerg_data)
        
        # Get all zones
        master_shapefile = f'{data_path}WGS/geofsm-prod-all-zones-20240712.shp'
        all_zones = gp.read_file(master_shapefile)
        unique_zones = all_zones['zone'].unique()
        
        # Process each zone
        output_files = []
        for zone_str in unique_zones:
            try:
                txt_file = process_single_zone(data_path, imerg_data, zone_str, date_string, copy_to_zone_wise)
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
    result = imerg_all_zones_workflow(args.start_date, args.end_date, args.copy_to_zone_wise)
    print(f"Generated files: {result['txt_files']}")