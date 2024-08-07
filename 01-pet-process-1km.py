from prefect import flow, task
from dotenv import load_dotenv
import os
from datetime import datetime
import xarray as xr
from dask.distributed import Client

from utils import (
    pet_list_files_by_date,
    pet_download_extract_bilfile,
    pet_bil_netcdf,
    pet_read_netcdf_files_in_date_range,
    get_dask_client_params,
    process_zone_and_subset_data,
    regrid_dataset
)

load_dotenv()

@task
def setup_environment():
    input_path = os.getenv("data_path")
    output_dir = f'{input_path}PET/dir/'
    netcdf_path = f'{input_path}PET/netcdf/'
    params = get_dask_client_params()
    client = Client(**params)
    return input_path, output_dir, netcdf_path, client

@task
def get_pet_files(url, start_date, end_date):
    return pet_list_files_by_date(url, start_date, end_date)

@task
def process_pet_files(pet_list, output_dir, netcdf_path):
    for file_url, date in pet_list:
        pet_download_extract_bilfile(file_url, output_dir)
        pet_bil_netcdf(file_url, date, output_dir, netcdf_path)

@task
def read_pet_data(netcdf_path, start_date, end_date):
    pds = pet_read_netcdf_files_in_date_range(netcdf_path, start_date, end_date)
    return pds.rename(x='lon', y='lat')

@task
def process_zone(input_path, pds, zone_str):
    shapefl_name = f'{input_path}WGS/{zone_str}.shp'
    km_str = 1
    z1ds, pdsz1, zone_extent = process_zone_and_subset_data(shapefl_name, km_str, zone_str, pds)
    return z1ds, pdsz1, zone_extent

@task
def regrid_pet_data(pds, input_chunk_sizes, output_chunk_sizes, zone_extent):
    return regrid_dataset(
        pds,
        input_chunk_sizes,
        output_chunk_sizes,
        zone_extent,
        regrid_method="bilinear"
    )

@flow
def pet_processing_workflow(start_date: datetime, end_date: datetime, zone_str: str):
    input_path, output_dir, netcdf_path, client = setup_environment()
    
    url = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/daily/pet/downloads/daily/"
    pet_list = get_pet_files(url, start_date, end_date)
    
    process_pet_files(pet_list, output_dir, netcdf_path)
    
    pds = read_pet_data(netcdf_path, start_date, end_date)
    
    z1ds, pdsz1, zone_extent = process_zone(input_path, pds, zone_str)
    
    input_chunk_sizes = {'time': 10, 'lat': 30, 'lon': 30}
    output_chunk_sizes = {'lat': 300, 'lon': 300}
    
    regridded_data = regrid_pet_data(pds, input_chunk_sizes, output_chunk_sizes, zone_extent)
    
    return regridded_data

if __name__ == "__main__":
    start_date = datetime(2024, 4, 1)
    end_date = datetime(2024, 7, 22)
    zone_str = 'zone6'
    
    pet_processing_workflow(start_date, end_date, zone_str)
