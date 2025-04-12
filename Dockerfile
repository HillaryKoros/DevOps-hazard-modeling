FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for geospatial libraries
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies including Prefect
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir prefect==2.14.5

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
    /app/data/zone_wise_txt_files \
    /app/zone_wise_txt_files

# Copy WGS data if it exists
COPY data/WGS* /app/data/WGS/ 2>/dev/null || true

# Create an entrypoint script that will run all the processes
RUN echo '#!/bin/bash\n\
echo "Starting PET processing..."\n\
python 01-pet-process-1km.py\n\
echo "Starting GEF-CHIRPS processing..."\n\
python 02-gef-chirps-process-1km.py\n\
echo "Starting IMERG processing..."\n\
python 03-imerg-process-1km.py\n\
echo "All processing completed."' > /app/entrypoint.sh \
&& chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
