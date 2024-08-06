
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import rioxarray
import xarray as xr
import xesmf as xe
import geopandas as gp
from dask.distributed import Client

from utils import pet_list_files_by_date
from utils import pet_download_extract_bilfile
from utils import pet_bil_netcdf
from utils import pet_find_missing_dates
from utils import pet_read_netcdf_files_in_date_range
from utils import pet_extend_forecast
    
from utils import make_zones_geotif
from utils import get_dask_client_params

from utils import process_zone_and_subset_data
from utils import regrid_dataset


load_dotenv()

input_path=os.getenv("data_path")
output_dir=f'{input_path}PET/dir/'
netcdf_path=f'{input_path}PET/netcdf/'

params = get_dask_client_params()

client = Client(**params)

start_date = datetime(2024, 4, 1)
end_date = datetime(2024, 7, 22)


url = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/daily/pet/downloads/daily/"

pet_list=pet_list_files_by_date(url, start_date, end_date)


for file_url, date in pet_list:
    pet_download_extract_bilfile(file_url,output_dir)
    pet_bil_netcdf(file_url,date,output_dir,netcdf_path)

pds=pet_read_netcdf_files_in_date_range(netcdf_path, start_date, end_date)
pds = pds.rename(x='lon', y='lat')


##############
#zone 6
shapefl_name=f'{input_path}WGS/zone6.shp'
km_str=1
zone_str='zone6'
z1ds, pdsz1, zone_extent = process_zone_and_subset_data(shapefl_name, km_str, zone_str, pds)




input_chunk_sizes = {'time': 10, 'lat': 30, 'lon': 30}
output_chunk_sizes = {'lat': 300, 'lon': 300}

# Call the function with the dictionary
z1ds = regrid_dataset(
    pds,
    input_chunk_sizes,
    output_chunk_sizes,
    zone_extent,
    regrid_method="bilinear"
)

