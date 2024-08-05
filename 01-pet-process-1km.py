
from dotenv import load_dotenv
import os


from dask.distributed import Client

from utils import pet_list_files_by_date
from utils import pet_download_extract_bilfile
from utils import pet_bil_netcdf
from utils import pet_find_missing_dates
from utils import pet_read_netcdf_files_in_date_range
from utils import pet_extend_forecast
    
from utils import make_zones_geotif
from utils import get_dask_client_params


load_dotenv()

data_path=os.getenv("geoglows_path")


params = get_dask_client_params()

client = Client(**params)

url = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/daily/pet/downloads/daily/"

pet_list=pet_list_files_by_date(url, start_date, end_date)


for file_url, date in pet_list:
    pet_download_extract_bilfile(file_url,date, output_dir,netcdf_path)



