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
import rasterio
from bs4 import BeautifulSoup
import flox
import flox.xarray
import geopandas as gp
from rasterio.features import rasterize
from rasterio.transform import from_bounds



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


def gefs_chrips_download_files(url_list, date_string,download_dir):
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
    
    return combined_dataset

def pet_extend_forecast(df, date_column, days_to_add=18):
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


def make_zones_geotif(shapefl_name,km_str,zone_str):
    gdf=gp.read_file(shapefl_name)
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


def pet_extend_forecast(df, date_column, days_to_add=18):
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

    print("Regridding complete. Result shape:", result.shape)
    return result



