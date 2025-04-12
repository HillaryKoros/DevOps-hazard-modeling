FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for geospatial libraries
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY *.py ./
COPY utils.py ./

# Create data directory structure
RUN mkdir -p /app/data/geofsm-input/gefs-chirps \
    /app/data/geofsm-input/imerg \
    /app/data/geofsm-input/pet \
    /app/data/geofsm-input/processed \
    /app/data/PET/dir \
    /app/data/PET/netcdf \
    /app/data/PET/processed \
    /app/data/WGS \
    /app/data/zone_wise_txt_files

# Copy WGS data
COPY data/WGS /app/data/WGS/

# Copy zone text files
COPY zone_wise_txt_files /app/data/zone_wise_txt_files/

# Create an entrypoint script to run the processes
RUN echo '#!/bin/bash\n\
python 01-pet-process-1km.py\n\
python 02-gef-chirps-process-1km.py\n\
python 03-imerg-process-1km.py\n\
echo "All processing completed."' > /app/entrypoint.sh \
&& chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]