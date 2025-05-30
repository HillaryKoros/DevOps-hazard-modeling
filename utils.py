import logging
import os
import tarfile
import tempfile
from datetime import datetime, timedelta
from glob import glob
from urllib.parse import urljoin, urlparse
import psutil
import math

import numpy as np
import pandas as pd
import requests
import xarray as xr
import rioxarray
import xesmf as xe
import rasterio
from bs4 import BeautifulSoup
import flox
import flox.xarray
import geopandas as gp
from rasterio.features import rasterize
from rasterio.transform import from_bounds

from distributed import Client
from dask.diagnostics import ProgressBar

###############################################################################
# GEFS-CHIRPS Processing Functions
###############################################################################

def gefs_chrips_list_tiff_files(base_url, date_string):
    '''
    base_url = "https://data.chc.ucsb.edu/products/EWX/data/forecasts/CHIRPS-GEFS_precip_v12/daily_16day/"
    date_string = "20240715"
    tiff_files = gefs_chrips_list_tiff_files(base_url, date_string)

    '''
    # Parse the date string
    year = date_string[:4]
    month = date_string[4:6]
    day = date_string[6:]
    
    # Construct the URL
    url = f"{base_url}{year}/{month}/{day}/"
    
    # Fetch the content of the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"Failed to fetch URL: {url}")

    # Parse the content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all the links to TIFF files and construct full URLs
    tiff_files = [urljoin(url, link.get('href')) for link in soup.find_all('a') if link.get('href').endswith('.tif')]
    
    return tiff_files


def gefs_chrips_download_files(url_list, date_string, download_dir):
    '''
    url_list=tiff_files
    download_dir=f'{data_path}geofsm-input/gefs-chrips'
    date_string='20240715'
    download_files(url_list, download_dir, date_string)
    '''
    # Create the subdirectory for the given date
    sub_dir = os.path.join(download_dir, date_string)
    os.makedirs(sub_dir, exist_ok=True)

    for url in url_list:
        try:
            # Send a GET request to the URL without authentication
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Extract the filename from the URL
            filename = os.path.basename(urlparse(url).path)
            filepath = os.path.join(sub_dir, filename)

            # Download and save the file
            with open(filepath, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f"Successfully downloaded: {filename}")

        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error occurred while downloading {url}: {e}")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while downloading {url}: {e}")


def gefs_extract_date(filename):
    # Extract date from filename (assuming format 'data.YYYY.MMDD.tif')
    parts = filename.split('.')
    return pd.to_datetime(f"{parts[1]}-{parts[2][:2]}-{parts[2][2:]}")



def gefs_chrips_process(input_path):
    # Path to your TIFF files
    tiff_path = f'{input_path}/*.tif'
    # Get list of all TIFF files
    tiff_files = sorted(glob(tiff_path))
    data_arrays = []
    for file in tiff_files:
        with rasterio.open(file) as src:
            # Read the data
            data = src.read(1)  # Assuming single band data

            # Get spatial information
            height, width = src.shape
            left, bottom, right, top = src.bounds

            # Create coordinates
            lons = np.linspace(left, right, width)
            lats = np.linspace(top, bottom, height)

            # Extract time from filename
            time = gefs_extract_date(os.path.basename(file))

            # Create DataArray
            da = xr.DataArray(
                data,
                coords=[('lat', lats), ('lon', lons)],
                dims=['lat', 'lon'],
                name='rain'
            )

            # Add time coordinate
            da = da.expand_dims(time=[time])

            data_arrays.append(da)
    # Combine all DataArrays into a single Dataset
    ds = xr.concat(data_arrays, dim='time')
    # Sort by time
    ds = ds.sortby('time')
    ds1 = ds.to_dataset(name='rain')
    return ds1

###############################################################################
# IMERG Processing Functions
###############################################################################

def imerg_list_files_by_date(url, flt_str, username, password, start_date, end_date):
    """
    List IMERG files from a URL, filtered by date range and file name pattern.
    
    :param url: Base URL to scrape
    :param flt_str: String to filter file names (e.g., '-S233000-E235959.1410.V07B.1day.tif')
    :param username: Username for authentication
    :param password: Password for authentication
    :param start_date: start_date = '20240712'
    :param end_date: end_date = '20240715'
    :return: List of tuples containing (file_url, file_date)
    
    Usage example:
    url = "https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early/"
    flt_str = '-S233000-E235959.1410.V07B.1day.tif'
    username = 'your_username'
    password = 'your_password'
    start_date = '20240701'
    end_date = '20240701'
    file_list = imerg_list_files_by_date(url, flt_str, username, password, start_date, end_date)
    """
    # Send a GET request to the URL with authentication
    response = requests.get(url, auth=(username, password))
    response.raise_for_status()  # Raise an exception for bad status codes

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all links in the page
    links = soup.find_all('a')

    # Convert start_date and end_date to datetime objects
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)

    # Filter and collect file links
    file_links = []
    for link in links:
        href = link.get('href')
        if href and flt_str in href:
            # Correctly extract the date part from the href string
            date_part = href.split('.')[4].split('-')[0]  # This gets only the date part
            try:
                #print(date_part)
                file_date = pd.to_datetime(date_part, format='%Y%m%d')  # Adjust format as necessary
                # Check if the date is within the specified range
                if start_date_dt <= file_date <= end_date_dt:
                    full_url = urljoin(url, href)
                    file_links.append(full_url)
            except ValueError as e:
                print(f"Error parsing date from {href}: {e}")

    return file_links


def imerg_download_files(url_list, username, password, download_dir):
    '''
    imerg_download_files(url_list, username, password, imerg_store)
    '''
    # Create the download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    for url in url_list:
        try:
            # Send a GET request to the URL with authentication
            response = requests.get(url, auth=(username, password), stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Extract the filename from the URL
            filename = os.path.basename(urlparse(url).path)
            filepath = os.path.join(download_dir, filename)

            # Download and save the file
            with open(filepath, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f"Successfully downloaded: {filename}")

        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error occurred while downloading {url}: {e}")
            if e.response.status_code == 401:
                print("Authentication failed. Please check your username and password.")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while downloading {url}: {e}")

def imerg_extract_date_from_filename(filename):
    # Extract date from filename
    date_str = filename.split('3IMERG.')[1][:8]
    return datetime.strptime(date_str, '%Y%m%d')

def imerg_read_tiffs_to_dataset(folder_path, start_date, end_date):
    # Get list of tif files
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif') and '3IMERG.' in f]
    
    # Extract dates and sort files
    date_file_pairs = [(imerg_extract_date_from_filename(f), f) for f in tif_files]
    date_file_pairs.sort(key=lambda x: x[0])
    
    # Create a complete date range
    #start_date = date_file_pairs[0][0]
    #end_date = date_file_pairs[-1][0]
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create dataset
    dataset_list = []
    for date in all_dates:
        print(date)
        matching_files = [f for d, f in date_file_pairs if d == date]
        if matching_files:
            file_path = os.path.join(folder_path, matching_files[0])
            with rioxarray.open_rasterio(file_path) as da:
                da = da.squeeze().drop_vars('band',errors='raise')  # Remove band dimension if it exists
                da = da.astype('float32')  # Convert data to float if it's not already
                da = da.where(da != 29999, np.nan)
                da1=da/10
                da1 = da1.expand_dims(time=[date])  # Add time dimension
                dataset_list.append(da1)
        else:
            pass
            # Create a dummy dataset with NaN values for missing dates
            #dummy_da = xr.full_like(dataset_list[-1] if dataset_list else None, float('nan'))
            #dummy_da = dummy_da.expand_dims(time=[date])
            #dataset_list.append(dummy_da)
    
    # Combine all datasets
    combined_ds = xr.concat(dataset_list, dim='time')
    
    return combined_ds

###############################################################################
# PET Processing Functions
###############################################################################

def pet_list_files_by_date(url, start_date, end_date):
    '''
    no credentials requiered
    url = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/daily/pet/downloads/daily/"
    start_date = datetime(2024, 4, 14)
    end_date = datetime(2024, 7, 13)

    to remove duplicates from source and order it 

    [('https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/daily/pet/downloads/daily/et240624.tar.gz',
     datetime.datetime(2024, 7, 17, 8, 29)),
     ('https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/daily/pet/downloads/daily/et240625.tar.gz',
     datetime.datetime(2024, 7, 17, 8, 34)),
     ('https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/daily/pet/downloads/daily/et240718.tar.gz',
     datetime.datetime(2024, 7, 19, 3, 16)),
     ('https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/daily/pet/downloads/daily/et240719.tar.gz',
     datetime.datetime(2024, 7, 20, 3, 16)),
     ('https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/daily/pet/downloads/daily/et240720.tar.gz',
     datetime.datetime(2024, 7, 21, 3, 16))]



    '''
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)

    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    
    file_links = []
    table = soup.find('table')
    if table:
        rows = table.find_all('tr')[2:]  # Skip the header and hr rows
        for row in rows:
            columns = row.find_all('td')
            if len(columns) >= 3:
                file_name = columns[1].find('a')
                if file_name and file_name.text.endswith('.tar.gz'):
                    file_url = urljoin(url, file_name['href'])
                    date_str = columns[2].text.strip()
                    try:
                        file_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
                        if start_date_dt <= file_date <= end_date_dt:
                            file_links.append((file_url, file_date))
                    except ValueError:
                        print(f"Could not parse date for {file_url}")
    else:
        print("No table found in the HTML")
    unique_days = {}
    for url, dt in file_links:
        day_key = dt.strftime('%Y%m%d')
        if day_key not in unique_days or dt < unique_days[day_key][1]:
            unique_days[day_key] = (url, dt)
    sorted_unique_data = sorted(unique_days.values(), key=lambda x: x[1])

    return sorted_unique_data


def pet_download_extract_bilfile(file_url, output_dir):
    # Download the file
    '''
    output_dir=f'{input_path}PET/dir/'
    netcdf_path=f'{input_path}PET/netcdf/'
    #download_extract_and_process(file_url, output_dir)
    for file_url, date in pet_files:
        xds = download_extract_and_process(file_url,date, output_dir,netcdf_path)
    '''
    response = requests.get(file_url)
    response.raise_for_status()
    
    # Create a temporary file to store the downloaded tar.gz
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name

    try:
        # Extract the tar.gz file
        with tarfile.open(temp_file_path, 'r:gz') as tar:
            tar.extractall(path=output_dir)
        
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

def pet_bil_netcdf(file_url,date,output_dir,netcdf_dir):
    '''
    Pass the url and make the bil file into netcdf with date as file name
    '''
    filename = os.path.basename(file_url)
    base_name = os.path.basename(filename)
    file_name_without_extension = os.path.splitext(os.path.splitext(base_name)[0])[0] + '.bil'
    bil_path = os.path.join(output_dir, file_name_without_extension)
    if not os.path.exists(netcdf_dir):
    # If not, create the directory
       os.makedirs(netcdf_dir) 
    #Open the .bil file as an xarray dataset
    with rioxarray.open_rasterio(bil_path) as xds:
        # Process or save the xarray dataset as needed
        # For example, you can save it as a NetCDF file
       ncname=date.strftime('%Y%m%d')
       nc_path = os.path.join(netcdf_dir, f"{ncname}.nc")
       xds.to_netcdf(nc_path)
       print(f"Converted {bil_path} to {nc_path}")
    return 'xds'  # Return the xarray dataset


def pet_find_missing_dates(folder_path):
    # Get list of files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.nc')]
    
    # Extract dates from filenames
    dates = [datetime.strptime(f[:8], '%Y%m%d') for f in files]
    
    # Create a DataFrame with these dates
    df = pd.DataFrame({'date': dates})
    df = df.sort_values('date')
    
    # Get the start and end dates
    start_date = df['date'].min()
    end_date = df['date'].max()
    
    # Create a complete date range
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Find missing dates
    missing_dates = all_dates[~all_dates.isin(df['date'])]
    
    return missing_dates, all_dates

def pet_find_last_available(date, available_dates):
    last_date = None
    for avail_date in available_dates:
        if avail_date <= date:
            last_date = avail_date
        else:
            break
    return last_date


def pet_read_netcdf_files_in_date_range(folder_path, start_date, end_date):
    # Convert start and end dates to pandas datetime for easy comparison
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # List all NetCDF files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.nc')]
    
    # Filter files by date range
    filtered_files = []
    file_dates = []
    for file in files:
        # Extract date from filename assuming format YYYYMMDD.nc
        date_str = file.split('.')[0]
        file_date = pd.to_datetime(date_str, format='%Y%m%d')
        
        # Check if file date is within the range
        if start_date <= file_date <= end_date:
            filtered_files.append(file)
            file_dates.append(file_date)
    
    # Sort the filtered files and dates by date
    filtered_files = [f for _, f in sorted(zip(file_dates, filtered_files))]
    file_dates.sort()
    
    # Read the filtered files into xarray datasets and combine them
       
    datasets = []
    for file, date in zip(filtered_files, file_dates):
        ds = xr.open_dataset(os.path.join(folder_path, file))
        # Remove the 'spatial_ref' variable if it exists
        if 'spatial_ref' in ds.variables:
            ds = ds.drop_vars('spatial_ref')

        # Rename the 'band' variable to 'pet' if it exists
        if 'band' in ds.variables:
            ds = ds.drop_vars('band')
            
        if 'date' in ds.variables:
            ds = ds.drop_vars('date')

        if 'band' in ds.dims:
            ds = ds.squeeze('band')

        # Rename the data variable if it is '__xarray_dataarray_variable__'
        if '__xarray_dataarray_variable__' in ds.data_vars:
            ds = ds.rename_vars({'__xarray_dataarray_variable__': 'pet'})
        
        ds = ds.expand_dims(time=[date])
        datasets.append(ds)

    combined_dataset = xr.concat(datasets, dim='time')
    combined_dataset1 = combined_dataset.rename(x='lon', y='lat') 
    return combined_dataset1

def pet_extend_forecast(df, date_column, days_to_add=16):
    """
    Add a specified number of days to the last date in a DataFrame, 
    repeating all values from the last row for non-date columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    date_column (str): Name of the column containing dates in 'YYYYDDD' format
    days_to_add (int): Number of days to add (default is 18)
    
    Returns:
    pd.DataFrame: DataFrame with additional rows
    """
    
    # Function to safely convert date string to datetime
    def safe_to_datetime(date_str):
        try:
            return datetime.strptime(str(date_str), '%Y%j')
        except ValueError:
            return None

    # Create a copy of the input DataFrame to avoid modifying the original
    df = df.copy()
    
    # Convert date column to datetime
    df[date_column] = df[date_column].apply(safe_to_datetime)
    
    # Remove any rows where the date conversion failed
    df = df.dropna(subset=[date_column])
    
    if not df.empty:
        # Get the last row
        last_row = df.iloc[-1]
        
        # Create a list of new dates
        last_date = last_row[date_column]
        new_dates = [last_date + timedelta(days=i+1) for i in range(days_to_add)]
        
        # Create new rows
        new_rows = []
        for new_date in new_dates:
            new_row = last_row.copy()
            new_row[date_column] = new_date
            new_rows.append(new_row)
        
        # Convert new_rows to a DataFrame
        new_rows_df = pd.DataFrame(new_rows)
        
        # Concatenate the new rows to the original DataFrame
        df = pd.concat([df, new_rows_df], ignore_index=True)
        
        # Convert date column back to the original string format
        df[date_column] = df[date_column].dt.strftime('%Y%j')
    else:
        print(f"No valid dates found in the '{date_column}' column.")
    
    return df

###############################################################################
# Zone Processing Functions
###############################################################################

def make_zones_geotif(shapefl_name, km_str, zone_str):
    """
    Create a GeoTIFF from a shapefile representing a zone.

    Parameters:
    ----------
    shapefl_name : str
        Name of the shapefile for the zone.
    km_str : int
        Pixel size for the output raster in kilometers.
    zone_str : str
        Identifier for the zone, used in the naming of the output GeoTIFF.

    Returns:
    -------
    output_tiff_path : str
        The path to the generated GeoTIFF file.
    Example:
    -------
    zone1_tif=make_zones_geotif(shapefl_name,km_str,zone_str)
    """
    gdf = gp.read_file(shapefl_name)
    # Define the output raster properties
    pixel_size = km_str/100  # Define the pixel size (adjust as needed)
    minx, miny, maxx, maxy = gdf.total_bounds
    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)
    #width, height
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    # Create an empty array to hold the rasterized data
    raster = np.zeros((height, width), dtype=np.uint16)
    # Generate shapes (geometry, value) for rasterization
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['GRIDCODE']))
    # Rasterize the shapes into the array
    raster = rasterize(shapes, out_shape=raster.shape, transform=transform, fill=0, dtype=np.uint16)
    output_tiff_path = os.path.dirname(shapefl_name)
    # Save the raster to a TIFF file
    output_tiff_path = f'{output_tiff_path}/ea_geofsm_prod_{zone_str}_{km_str}km.tif'
    with rasterio.open(
        output_tiff_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.uint16,
        crs=gdf.crs.to_string(),
        transform=transform,
    ) as dst:
        dst.write(raster, 1)
    #print(f"Raster TIFF file saved to {output_tiff_path}")
    return output_tiff_path


def process_zone_from_combined(master_shapefile, zone_name, km_str, pds):
    """
    Process a specific zone from a combined shapefile and subset data based on that zone.
    
    Parameters:
    ----------
    master_shapefile : str
        Path to the shapefile containing all zones.
    zone_name : str
        Name of the zone to extract (e.g., 'zone1').
    km_str : int
        Pixel size for the output raster in kilometers.
    pds : xarray.Dataset
        The dataset from which a subset is to be extracted based on the zone extent.
        
    Returns:
    -------
    z1crds : xarray.DataArray
        The dataset corresponding to the generated GeoTIFF for the specific zone.
    pz1ds : xarray.Dataset
        The subset of the input dataset 'pds' within the extent of the zone.
    zone_extent : dict
        Dictionary containing the latitude and longitude extents of the zone.
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

    # Subset the provided dataset based on the zone extent
    pz1ds = pds.sel(lat=slice(z1lat_max, z1lat_min), lon=slice(z1lon_min, z1lon_max))

    return z1crds, pz1ds, zone_extent

###############################################################################
# Dask and Regridding Functions
###############################################################################

def get_dask_client_params():
    # Get number of CPU cores (leave 1 for the OS)
    n_workers = max(1, psutil.cpu_count(logical=False) - 1)
    
    # Assuming hyperthreading is available
    threads_per_worker = 2
    
    # Calculate available memory per worker (in GB)
    total_memory = psutil.virtual_memory().total / (1024**3)  # Convert to GB
    memory_per_worker = math.floor(total_memory / n_workers * 0.75)  # Use 75% of available memory
    
    return {
        "n_workers": n_workers,
        "threads_per_worker": threads_per_worker,
        "memory_limit": f"{memory_per_worker}GB"
    }


def regrid_dataset(input_ds, input_chunk_sizes, output_chunk_sizes, zone_extent, regrid_method="bilinear"):
    """
    Regrid a dataset to a specified output grid using a specified regridding method.

    Parameters:
    ----------
    input_ds : xarray.Dataset
        The input dataset to be regridded.

    input_chunk_sizes : dict
        A dictionary specifying the chunk sizes for the input dataset, e.g., {'time': 10, 'lat': 30, 'lon': 30}.

    output_chunk_sizes : dict
        A dictionary specifying the chunk sizes for the output dataset, e.g., {'lat': 300, 'lon': 300}.

    zone_extent : dict
        A dictionary specifying the latitude and longitude extents of the output grid. 
        Should contain the keys 'lat_min', 'lat_max', 'lon_min', 'lon_max' with respective values.

    regrid_method : str, optional
        The method used for regridding. Default is "bilinear". Other methods can be used if supported by `xesmf.Regridder`.

    Returns:
    -------
    xarray.Dataset
        The regridded dataset.

    Example:
    -------
    input_ds = xr.open_dataset("your_input_data.nc")
    input_chunk_sizes = {'time': 10, 'lat': 30, 'lon': 30}
    output_chunk_sizes = {'lat': 300, 'lon': 300}
    zone_extent = {'lat_min': 0, 'lat_max': 30, 'lon_min': 0, 'lon_max': 30}

    regridded_ds = regrid_dataset(input_ds, input_chunk_sizes, output_chunk_sizes, zone_extent)
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

    return result

###############################################################################
# Data Aggregation and Output Functions
###############################################################################

def zone_mean_df(input_ds, zone_ds):
    """
    Compute the mean of values in `input_ds` grouped by zones defined in `zone_ds` using the "split-apply-combine" strategy.
    This method is particularly aligned with the 'flox groupby method' (see xarray.dev/blog/flox) designed to optimize 
    such operations within xarray's framework. The method consists of three primary steps:
        1. Split: The input dataset is aligned and then split according to the zones defined within `zone_ds`.
        2. Apply: A mean reduction is applied to each group of data to summarize the values within each zone.
        3. Combine: The results of the mean calculations are combined into a single DataFrame for further analysis or export.

    Parameters:
    ----------
    input_ds : xarray.Dataset
        The input dataset containing the data to be averaged. This dataset should contain numerical data that can be
        meaningfully averaged.
    zone_ds : xarray.Dataset
        A dataset derived from polygon zones, indicating the areas over which to calculate means. This dataset
        typically originates from geographic or spatial delineations converted into a format compatible with xarray.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with the mean values for each zone, reset with a clean index to facilitate easy use in further
        analysis or visualization.

    Example:
    -------
    # Assuming 'input_ds' is loaded with relevant environmental data and 'zone_ds' represents catchment areas
    input_ds = xr.open_dataset('input_data.nc')
    zone_ds = process_zone_and_subset_data('shapefile.shp', 1, 'zone1', input_ds)[1]
    zone_mean_df = zone_mean_df(input_ds, zone_ds)

    Notes:
    -------
    The function aligns both datasets before grouping to ensure that the data corresponds directly to the defined zones,
    potentially overriding the alignment of `input_ds` with that of `zone_ds` if discrepancies exist. This is crucial for
    maintaining consistency in spatial analyses where precise location alignment is necessary.
    """
    z1d_, aligned_zone_ds = xr.align(input_ds, zone_ds, join="override")
    z1 = input_ds.groupby(aligned_zone_ds).mean()
    z1 = z1.to_dataframe()
    z1a = z1.reset_index()
    return z1a


def pet_update_input_data(z1a, zone_input_path, zone_str, start_date, end_date):
    """
    Processes evaporation data by performing a series of transformations and merging operations,
    culminating in the extension of forecasts and exporting to a CSV file.

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
    str
        Path to the output file in the processed directory.
    """
    # Ensure zone_wise directory exists
    zone_dir = f'{zone_input_path}{zone_str}'
    os.makedirs(zone_dir, exist_ok=True)
    
    # Get base path for processed directory
    base_path = zone_input_path.replace('zone_wise_txt_files/', '')
    processed_dir = f'{base_path}geofsm-input/processed/{zone_str}'
    os.makedirs(processed_dir, exist_ok=True)
    
    # Adjust the 'pet' column by a factor of 10
    z1a['pet'] = z1a['pet'] / 10
    
    # Pivot the DataFrame
    zz1 = z1a.pivot(index='time', columns='group', values='pet')
    
    # Apply formatting to the pivoted DataFrame
    zz1 = zz1.apply(lambda row: row.map(lambda x: f'{x:.1f}' if isinstance(x, (int, float)) and pd.notna(x) else x), axis=1)
    
    # Reset the index and adjust columns
    azz1 = zz1.reset_index()
    azz1['NA'] = azz1['time'].dt.strftime('%Y%j')
    azz1.columns = [str(col) if isinstance(col, int) else col for col in azz1.columns]
    azz1 = azz1.rename(columns={'time': 'date'})
    
    # Path to standard evap.txt file in zone_wise directory
    evap_file = f'{zone_dir}/evap.txt'
    
    # Check if the evap.txt file exists
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
    
    # Extend the forecast data
    bz2 = pet_extend_forecast(bz1, 'NA')
    
    # Format date for filename
    end_date_str = end_date.strftime('%Y%j')  # Formats as "YearDayOfYear", e.g., "2024365"
    
    # 1. Create files in zone_wise_txt_files directory
    
    # Standard evap.txt file
    bz2.to_csv(evap_file, index=False)
    print(f"Created/updated standard evap.txt file: {evap_file}")
    
    # Zone-specific evap file (evap_zone1.txt)
    zone_specific_file = f'{zone_dir}/evap_{zone_str}.txt'
    bz2.to_csv(zone_specific_file, index=False)
    print(f"Created zone-specific evap file: {zone_specific_file}")
    
    # Dated evap file (evap_2024365.txt)
    dated_file = f'{zone_dir}/evap_{end_date_str}.txt'
    bz2.to_csv(dated_file, index=False)
    print(f"Created dated evap file: {dated_file}")
    
    # 2. Create files in geofsm-input/processed directory
    
    # Dated evap file in processed directory (evap_2024365.txt)
    processed_dated_file = f'{processed_dir}/evap_{end_date_str}.txt'
    bz2.to_csv(processed_dated_file, index=False)
    print(f"Created dated evap file in processed directory: {processed_dated_file}")
    
    # Return the path to the processed file (to be consistent with GEFS-CHIRPS)
    return processed_dated_file



def gefs_chirps_update_input_data(results_df, zone_input_path, zone_str, start_date, end_date):
    """
    Updates GEFS-CHIRPS input data by merging new data with existing data,
    extending forecasts, and saving to required locations.

    Parameters:
    ----------
    results_df : pandas.DataFrame
        Dataframe containing GEFS-CHIRPS data that needs to be adjusted, pivoted, and formatted.
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
    str
        Path to the output file in the processed directory.
    """
    
    # Ensure zone_wise directory exists
    zone_dir = f'{zone_input_path}{zone_str}'
    os.makedirs(zone_dir, exist_ok=True)
    
    # Get base path for processed directory
    base_path = zone_input_path.replace('zone_wise_txt_files/', '')
    processed_dir = f'{base_path}geofsm-input/processed/{zone_str}'
    os.makedirs(processed_dir, exist_ok=True)
    
    # Pivot the DataFrame
    zz1 = results_df.pivot(index='time', columns='group', values='rain')
    
    # Apply formatting to the pivoted DataFrame
    zz1 = zz1.apply(lambda row: row.map(lambda x: f'{x:.1f}' if isinstance(x, (int, float)) and pd.notna(x) else x), axis=1)
    
    # Reset the index and adjust columns
    azz1 = zz1.reset_index()
    azz1['NA'] = azz1['time'].dt.strftime('%Y%j')
    azz1.columns = [str(col) if isinstance(col, int) else col for col in azz1.columns]
    azz1 = azz1.rename(columns={'time': 'date'})
    
    # Path to standard rain.txt file in zone_wise directory
    rain_file = f'{zone_dir}/rain.txt'
    
    # Check if the rain.txt file exists
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
    
    # Do not extend forecast data for GEFS-CHIRPS
    bz2 = bz1  # Use data as is without extension
    
    # Format date for filename
    end_date_str = end_date.strftime('%Y%j')  # Formats as "YearDayOfYear", e.g., "2024365"
    
    # 1. Create files in zone_wise_txt_files directory
    
    # Standard rain.txt file
    bz2.to_csv(rain_file, index=False)
    print(f"Created/updated standard rain.txt file: {rain_file}")
    
    # Zone-specific rain file (rain_zone1.txt)
    zone_specific_file = f'{zone_dir}/rain_{zone_str}.txt'
    bz2.to_csv(zone_specific_file, index=False)
    print(f"Created zone-specific rain file: {zone_specific_file}")
    
    # Dated rain file (rain_2024365.txt)
    dated_file = f'{zone_dir}/rain_{end_date_str}.txt'
    bz2.to_csv(dated_file, index=False)
    print(f"Created dated rain file: {dated_file}")
    
    # 2. Create files in geofsm-input/processed directory
    
    # Dated rain file in processed directory (rain_2024365.txt)
    processed_dated_file = f'{processed_dir}/rain_{end_date_str}.txt'
    bz2.to_csv(processed_dated_file, index=False)
    print(f"Created dated rain file in processed directory: {processed_dated_file}")
    
    # Return the path to the processed file
    return processed_dated_file


def imerg_update_input_data(results_df, zone_input_path, zone_str, start_date, end_date):
    """
    Updates IMERG input data by merging new data with existing rain.txt data,
    and saving to required locations. No forecast extension is applied.

    Parameters:
    ----------
    results_df : pandas.DataFrame
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
    str
        Path to the output file in the processed directory.
    """

    
    # Ensure zone_wise directory exists
    zone_dir = f'{zone_input_path}{zone_str}'
    os.makedirs(zone_dir, exist_ok=True)
    
    # Get base path for processed directory
    base_path = zone_input_path.replace('zone_wise_txt_files/', '')
    processed_dir = f'{base_path}geofsm-input/processed/{zone_str}'
    os.makedirs(processed_dir, exist_ok=True)
    
    # Find the precipitation column - it might be named differently across implementations
    precip_cols = [col for col in results_df.columns if col.lower() in ['precipitation', 'precip', 'rain']]
    if not precip_cols:
        standard_cols = ['time', 'group', 'spatial_ref', 'band']
        precip_cols = [col for col in results_df.columns if col not in standard_cols]
        
    if not precip_cols:
        raise ValueError(f"No precipitation column found in the DataFrame")
    
    # Use the first precipitation column found
    precip_col = precip_cols[0]
    print(f"Using precipitation column: {precip_col}")
    
    # Apply special IMERG rounding rules
    def format_imerg_value(x):
        if pd.isna(x):
            return "0.0"
        elif float(x) <= 0.01:
            return "0.0"
        else:
            rounded = round(float(x) * 10) / 10
            return f"{rounded:.1f}"

    # Pivot the DataFrame
    zz1 = results_df.pivot(index='time', columns='group', values=precip_col)
    
    # Apply IMERG-specific formatting to the pivoted DataFrame
    zz1 = zz1.apply(lambda row: row.map(lambda x: format_imerg_value(x)), axis=1)
    
    # Reset the index and adjust columns
    azz1 = zz1.reset_index()
    azz1['NA'] = azz1['time'].dt.strftime('%Y%j')
    azz1.columns = [str(col) if isinstance(col, int) else col for col in azz1.columns]
    azz1 = azz1.rename(columns={'time': 'date'})
    
    # Path to standard rain.txt file in zone_wise directory (using rain.txt instead of imerg.txt)
    rain_file = f'{zone_dir}/rain.txt'
    
    # Check if the rain.txt file exists
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
    
    # Do not extend forecast data for IMERG
    bz2 = bz1  # Use data as is without extension
    
    # Format date for filename
    end_date_str = end_date.strftime('%Y%j')  # Formats as "YearDayOfYear", e.g., "2024365"
    
    # 1. Create files in zone_wise_txt_files directory
    
    # Standard rain.txt file (using rain.txt instead of imerg.txt)
    bz2.to_csv(rain_file, index=False)
    print(f"Created/updated standard rain.txt file: {rain_file}")
    
    # Zone-specific rain file (rain_zone1.txt)
    zone_specific_file = f'{zone_dir}/rain_{zone_str}.txt'
    bz2.to_csv(zone_specific_file, index=False)
    print(f"Created zone-specific rain file: {zone_specific_file}")
    
    # Dated rain file (rain_2024365.txt)
    dated_file = f'{zone_dir}/rain_{end_date_str}.txt'
    bz2.to_csv(dated_file, index=False)
    print(f"Created dated rain file: {dated_file}")
    
    # Also save copies as imerg files for tracking purposes
    imerg_file = f'{zone_dir}/imerg.txt'
    bz2.to_csv(imerg_file, index=False)
    print(f"Created/updated imerg.txt file: {imerg_file}")
    
    imerg_zone_file = f'{zone_dir}/imerg_{zone_str}.txt'
    bz2.to_csv(imerg_zone_file, index=False)
    print(f"Created zone-specific imerg file: {imerg_zone_file}")
    
    imerg_dated_file = f'{zone_dir}/imerg_{end_date_str}.txt'
    bz2.to_csv(imerg_dated_file, index=False)
    print(f"Created dated imerg file: {imerg_dated_file}")
    
    # 2. Create files in geofsm-input/processed directory
    
    # Also save a copy specifically labeled as imerg for tracking purposes
    imerg_processed_file = f'{processed_dir}/imerg_{end_date_str}.txt'
    bz2.to_csv(imerg_processed_file, index=False)
    print(f"Created imerg dated file in processed directory: {imerg_processed_file}")
    
    # Dated rain file in processed directory (rain_2024365.txt)
    processed_dated_file = f'{processed_dir}/rain_{end_date_str}.txt'
    bz2.to_csv(processed_dated_file, index=False)
    print(f"Created dated rain file in processed directory: {processed_dated_file}")
    
    # Return the path to the processed file
    return processed_dated_file