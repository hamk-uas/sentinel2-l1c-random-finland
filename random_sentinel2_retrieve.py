import time
import openeo
import random
import geopandas as gpd
from shapely.geometry import Polygon
import os
import sys
import requests
import numpy as np
import xarray as xr
import glob
import shutil
from pyproj import Transformer
import pandas as pd

# Print instructions
if len(sys.argv) != 4:
    raise Exception("Three command line arguments are required:\n* Integer for the desired dataset number of images and\n* Boolean for whether to avoid spatial overlap with previous images.\n* Path of image folder including trailing slash")

# Parse num_images
try:
    num_images = int(sys.argv[1])
except ValueError:
    raise Exception("The dataset size (the first argument) must be an integer.")
if num_images <= 0:
    raise Exception("The desired dataset size must be > 0.")
print(f"Desired dataset size: {num_images}")

# Parse avoid_overlap
assert sys.argv[2] in ["True", "False"], "Whether to avoid spatial overlap (the second argument) must be either True or False"
avoid_overlap = sys.argv[2] == "True"
print(f"Avoid spatial overlap: {avoid_overlap}")

# Parse folder
assert sys.argv[3][-1] == "/", "Path of image folder (the third argument) must contain a trailing slash"
image_directory = sys.argv[3]
print(f"Image folder: {image_directory}")

# Create folder if it doesn't exist
if not os.path.exists(image_directory):
    os.makedirs(image_directory)

# Download random satellite images over Finland over the given years

start_year = 2015  # The first year of data collection
end_year = 2023  # Data will be collected until the end of end_year - 1
image_size = 512 # Number of pixels in one dimension, shared for width and height. The images collected will be slightly larger due to coordiante system transformations

# Country map. All image center locations will be inside it.
map = gpd.read_file('finland_reprojected.geojson')

# Image filename prefix (for example "image_" but could be empty too)
image_prefix = ""

# Establish the OpenEO connection
connection = openeo.connect("openeo.dataspace.copernicus.eu")
connection.authenticate_oidc()

# Read stored previous tile geometries
geojson_filenames = sorted(filter(lambda filename: filename.endswith('.geojson'), os.listdir(image_directory)))
num_previous_tiles = 0
if len(geojson_filenames) > 0:
    previous_tiles = gpd.read_file(image_directory + geojson_filenames[-1])
    num_previous_tiles = int(geojson_filenames[-1].replace('.geojson', ''))

imgNumber = 0
print("Collecting previous tile geometries")
while True:
    print(imgNumber, end="\r")
    file_name = f"{image_directory}{image_prefix}{str(imgNumber).zfill(6)}.nc"
    # Check if file exists
    if not os.path.exists(file_name):
        if imgNumber < num_previous_tiles:
            raise Exception(f"Image {str(imgNumber).zfill(6)}.nc is missing but should exist according to the geojson files.")
        break
    if imgNumber >= num_previous_tiles:
        nc_data = xr.open_dataset(file_name)
        transformer = Transformer.from_crs(nc_data.crs.crs_wkt, map.crs, always_xy=True)
        xs = nc_data.x.to_numpy()
        ys = nc_data.y.to_numpy()
        corners = np.array([[xs[0], ys[0]], [xs[-1], ys[0]], [xs[-1], ys[-1]], [xs[0], ys[-1]]], dtype=np.float32)
        cornersx, cornersy = transformer.transform(corners[:,0], corners[:,1])
        polygon_geom = Polygon(zip(cornersx, cornersy))
        # Add the polygon to the existing previous_tiles geodataframe
        # Check if previous_tiles exists
        tile = gpd.GeoDataFrame(geometry=[polygon_geom], crs='epsg:32635')
        if "previous_tiles" not in locals():
            previous_tiles = tile
        else:
            previous_tiles = pd.concat([previous_tiles, tile])
    imgNumber += 1
print()
# Save geojson
previous_tiles.to_file(f"{image_directory}{str(imgNumber).zfill(6)}.geojson")

# Get the serial number of the last image downloaded
def find_last_image_number(directory, prefix):
    max_number = 0
    for file in os.listdir(directory):
        if file.startswith(prefix) and file.endswith('.nc'):
            number = int(file.replace(prefix, '').replace('.nc', ''))
            max_number = max(max_number, number)
    return max_number

# Get random date in the given range
def random_date(start_time = f"{start_year}-1-1", end_time = f"{end_year}-1-1", time_format = '%Y-%m-%d'):
    start_time_unix = time.mktime(time.strptime(start_time, time_format))
    end_time_unix = time.mktime(time.strptime(end_time, time_format))
    result_time_unix = start_time_unix + random.random() * (end_time_unix - start_time_unix)
    return time.strftime(time_format, time.localtime(result_time_unix))

# Get a random location inside a (country) map
def random_location(map):
    minx, miny, maxx, maxy = map.geometry.total_bounds
    while True:        
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=[x], y=[y]), crs='epsg:32635')
        within_points = gpd.sjoin(points, map, predicate='within')
        if len(within_points) == 1:
            break  # Accept point because it was inside the map
    return x, y

# Get an extent around a center point
def extent(center, width = image_size, height = image_size, resolution = 10):
    height_m = height * resolution
    width_m = width * resolution

    polygon_geom = Polygon([
        [center[0] + width_m / 2, center[1] + height_m / 2],
        [center[0] - width_m / 2, center[1] + height_m / 2],
        [center[0] - width_m / 2, center[1] - height_m / 2],
        [center[0] + width_m / 2, center[1] - height_m / 2]
    ])
    
    polygon = gpd.GeoDataFrame(index=[0], crs='epsg:32635', geometry=[polygon_geom])
    polygon = polygon.to_crs(4326)

    minx, miny, maxx, maxy = polygon.geometry.total_bounds
    
    odata_polygon = f"{minx} {miny}, {minx} {maxy}, {maxx} {maxy}, {maxx} {miny}, {minx} {miny}"
    openeo_extent = {"west": minx, "east": maxx, "south": miny, "north": maxy}

    return odata_polygon, openeo_extent, polygon_geom

def odata_query(polygon, start_date, end_date):
    with requests.Session() as session:
        session.headers.update({'Connection': 'close'})
        
        # Make GET request to the Copernicus Data Space API to obtain products based on the parameters
        with session.get(
            f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq 'SENTINEL-2' "
            f"and startswith(Name,'S2A_MSIL1C_') and ContentDate/Start gt {start_date} and ContentDate/Start lt "
            f"{end_date} and OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({polygon}))')&$top=1000"
        ) as response:
            api_data = response.json()

        # List available datasets
        id_list = [item['Id'] for item in api_data['value']]  # Takes only the IDs from the content
        return id_list

# Loop to download images
while imgNumber < num_images:
    try:
        # Pick a random location and calculate the extent around it
        while True:
            center = random_location(map)
            odata_polygon, openeo_extent, polygon_geom = extent(center)
            # Reject the tile if it overlaps with previous tiles
            if not avoid_overlap or "previous_tiles" not in locals():
                break
            else:
                # Test intersection between tile and previous tiles
                tile = gpd.GeoDataFrame(geometry=[polygon_geom], crs='epsg:32635')
                intersects = previous_tiles.intersects(tile.geometry[0]).any()
                if not intersects:
                    break

        # Pick a random date
        date = random_date()
        print(f"Image center: ({center[0]:.2f}, {center[1]:.2f}), date: {date}", end=" ")
        start_date = date + "T00:00:00.000Z"
        end_date = date + "T23:59:59.000Z"

        # Query for satellite images at that location and at that date
        id_list = odata_query(odata_polygon, start_date, end_date)
        #print(f"ids: {id_list}")

        # Were there any images?
        if len(id_list) == 0:
            print("No images")
        else:
            # Yes, load the data collection
            s2_cube = connection.load_collection(
                "SENTINEL2_L1C",
                spatial_extent = openeo_extent,
                temporal_extent = [start_date, end_date],
                bands=["B04", "B03", "B02"],
            )

            # Download the data as a file
            file_name = f"{image_directory}{image_prefix}{str(imgNumber).zfill(6)}.nc"
            print(f"\nDownloading {image_prefix}{str(imgNumber).zfill(6)}.nc")
            startTime = time.time()
            s2_cube.download(file_name)
            endTime = time.time()
            print(f"Finished download in {endTime - startTime:.2f} seconds.")
            nc_data = xr.open_dataset(file_name)
            r = nc_data['B04'][0].as_numpy()
            g = nc_data['B03'][0].as_numpy()
            b = nc_data['B02'][0].as_numpy()
            has_nans = np.isnan(r).any() or np.isnan(g).any() or np.isnan(b).any()
            if has_nans:
                print("Rejecting image due to NaNs")
                os.remove(file_name)
            else:
                # Add the tile to previous_tiles
                transformer = Transformer.from_crs(nc_data.crs.crs_wkt, map.crs, always_xy=True)
                xs = nc_data.x.to_numpy()
                ys = nc_data.y.to_numpy()
                corners = np.array([[xs[0], ys[0]], [xs[-1], ys[0]], [xs[-1], ys[-1]], [xs[0], ys[-1]]], dtype=np.float32)
                cornersx, cornersy = transformer.transform(corners[:,0], corners[:,1])
                polygon_geom = Polygon(zip(cornersx, cornersy))
                # Add the polygon to the existing previous_tiles geodataframe
                # Check if previous_tiles exists
                tile = gpd.GeoDataFrame(geometry=[polygon_geom], crs='epsg:32635')
                if "previous_tiles" not in locals():
                    previous_tiles = tile
                else:
                    previous_tiles = pd.concat([previous_tiles, tile])
                print("Accepting image")
                imgNumber += 1

    except Exception as error:
        print(error)
        print("Failure. Trying another time and location in 6 seconds.")
        time.sleep(6)

