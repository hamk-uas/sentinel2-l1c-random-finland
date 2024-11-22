import numpy as np
import xarray as xr
import glob
import shutil
import os

# Pick images (nc files) with only valid pixels, number the images and collect them to zip files.
# Run this after retrieving images with random_sentinel2_retrieve.py which already checks for missing data.
# This script will re-check the images for missing data but should accept all images if they were correctly retrieved.
# Configure the folder names correctly below.

N = 500  # How many images per zip file. Must be a multiple of 100

nc_input_files_path = '/mnt/d/code/env-data-retrieval/images/**/*.nc'  # ** will include any subdirectory
nc_output_files_path = '/mnt/d/code/env-data-retrieval/valid_images/' # The folder must exist
nc_output_zip_files_path = '/mnt/d/code/env-data-retrieval/valid_images/' # The folder must exist
nc_filenames = sorted(glob.glob(nc_input_files_path, recursive=True))  # go through subdirectories recursively to find files
print(f"Found {len(nc_filenames)} files to check")

num_valid = 0

for input_filename in nc_filenames:
    try:                
        nc_data = xr.open_dataset(input_filename)
        r = nc_data['B04'][0].as_numpy()
        g = nc_data['B03'][0].as_numpy()
        b = nc_data['B02'][0].as_numpy()
        has_nans = np.isnan(r).any() or np.isnan(g).any() or np.isnan(b).any()
        if not has_nans:
            output_filename = f"{nc_output_files_path}{str(num_valid).zfill(6)}.nc"
            shutil.copyfile(input_filename, output_filename)
            print(f"Wrote {output_filename}", end="\r")
            num_valid += 1            
            if num_valid % N == 0:
                sub_nc_filenames = map(lambda x: f'{nc_output_files_path}{str(x//100).zfill(4)}??.nc', range(num_valid - N, num_valid, 100))
                os.system(f"zip -j -0 {nc_output_zip_files_path}valid-images-500-{str(num_valid - N).zfill(6)}.zip {' '.join(sub_nc_filenames)}")
    except Exception as e:
        print(f"\nError processing input_filename=\"{input_filename}\", output_filename={output_filename}")
        print(e)
        None  # Some error loading the nc file
