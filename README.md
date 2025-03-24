# IGAD-ICPAC DevOps Hazard Modeling

This repository contains tools and workflows for processing, analyzing, and modeling meteorological and hydrological hazard data for the IGAD-ICPAC region in East Africa.

## Overview

The system processes several types of meteorological data at 1km resolution:
- Potential Evapotranspiration (PET) data
- GEFS-CHIRPS precipitation forecast data
- IMERG precipitation data

These scripts prepare data for a hydrological model and perform spatial analysis across six geographic zones in the region.

## Repository Structure

```
.
├── 01-pet-process-1km.py         # Workflow for processing PET data
├── 02-gef-chirps-process-1km.py  # Workflow for processing GEFS-CHIRPS data
├── 03-imerg-process-1km.py       # Workflow for processing IMERG data (planned)
├── data/
│   ├── geofsm-input/             # Input and output folders for GeoFSM model
│   │   ├── gefs-chirps/          # Storage for GEFS-CHIRPS data
│   │   ├── imerg/                # Storage for IMERG data
│   │   └── processed/            # Processed output by zone
│   ├── PET/                      # PET data storage
│   │   ├── dir/                  # Original BIL files
│   │   └── netcdf/               # Converted NetCDF files
│   └── WGS/                      # Geographic zone definitions
├── utils.py                      # Utility functions used by all scripts
└── README.md                     # This documentation file
```

## Dependencies

This project requires the following Python libraries:
- prefect (for workflow orchestration)
- xarray, rioxarray (for array data handling)
- dask (for parallel processing)
- pandas (for tabular data)
- numpy (for numerical operations)
- rasterio (for raster data processing)
- geopandas (for vector data processing)
- xesmf (for regridding)
- flox (for optimized groupby operations)
- requests, BeautifulSoup (for web scraping and downloading)
- psutil (for system resource management)

## Data Sources

The system processes data from several sources:

1. **PET (Potential Evapotranspiration)**
   - Source: USGS FEWS NET
   - URL: https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/daily/pet/downloads/daily/

2. **GEFS-CHIRPS (Global Ensemble Forecast System and Climate Hazards Group InfraRed Precipitation with Station data)**
   - Source: Climate Hazards Center, UCSB
   - URL: https://data.chc.ucsb.edu/products/EWX/data/forecasts/CHIRPS-GEFS_precip_v12/daily_16day/

3. **IMERG (Integrated Multi-satellitE Retrievals for GPM)**
   - Source: NASA
   - URL: https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early/
   - Note: Requires authentication

## Scripts

### 1. PET Processing (01-pet-process-1km.py)

This workflow downloads, processes, and regridds Potential Evapotranspiration (PET) data for hydrological modeling.

Key steps:
- Download PET data in BIL format from USGS FEWS NET
- Convert BIL files to NetCDF format
- Process zone shapefiles to define analysis regions
- Regrid PET data to match zone extents at 1km resolution

Usage:
```bash
python 01-pet-process-1km.py
```

### 2. GEFS-CHIRPS Processing (02-gef-chirps-process-1km.py)

This workflow downloads, processes, and analyzes GEFS-CHIRPS precipitation forecast data for all six zones.

Key steps:
- Download GEFS-CHIRPS data for a specified date
- Process data into xarray format
- For each zone (1-6):
  - Process zone shapefile and subset data
  - Regrid data to 1km resolution
  - Calculate zonal statistics
  - Save results as CSV files

Usage:
```bash
python 02-gef-chirps-process-1km.py
```

The script is configured to process yesterday's date by default. To process a different date, modify the date_string variable in the main block.

### 3. IMERG Processing (03-imerg-process-1km.py) - Planned

This workflow will download and process IMERG precipitation data.

Key steps:
- Download IMERG data for specified date range (requires authentication)
- Process data into xarray format
- Process each zone and perform spatial analysis
- Calculate zonal statistics and save results

## Setup and Configuration

1. Clone the repository:
```bash
git clone https://github.com/username/IGAD-ICPAC-DevOps-hazard-modeling.git
cd IGAD-ICPAC-DevOps-hazard-modeling
```

2. Create a `.env` file with the following variables:
```
data_path=/path/to/your/data/
imerg_username=your_imerg_username
imerg_password=your_imerg_password
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Workflow Execution

The scripts use [Prefect](https://www.prefect.io/) for workflow orchestration, which provides:
- Task dependency management
- Parallel execution with Dask
- Logging and monitoring
- Failure handling and retries

Each workflow is designed to be run independently, processing data for all six geographic zones.

## Data Processing Details

### Geographic Zones

The system processes data for six geographic zones in East Africa. Each zone is defined by a shapefile in the WGS directory. The zones are processed into rasterized GeoTIFFs at 1km resolution.

### Regridding

Data is regridded to match the 1km resolution of the zone boundaries using bilinear interpolation through xESMF.

### Zonal Statistics

The system calculates mean values for each variable within each zone, providing summary statistics for hydrological modeling.

## Notes for Developers

- Dask client parameters are automatically optimized based on available system resources
- For large datasets, adjust the chunking parameters in the regridding functions
- When adding new data sources, follow the pattern of the existing modules in utils.py

## License

[Your License Information]

## Contributors

[Your Contributor Information]