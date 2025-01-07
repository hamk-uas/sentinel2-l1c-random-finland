import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
from pyproj import Transformer
from PIL import Image, ImageDraw
import cv2
import cartopy.crs as ccrs  # Import cartopy for projections
from matplotlib_scalebar.scalebar import ScaleBar  # For adding the scalebar

image_directory = "/data/sentinel2-l1c-random-rgb-image"  # The folder that contains the .nc satellite image files
dpi = 75  # 75 for drafts, 300 minumum for color/grayscale (IEEE publication standard). Keep line art as vectors.
single_column_width = 3.5  # IEEE single column width
two_column_width = 7.16  # IEEE two column width

# Make figure_dataset_grid.pdf
# ----------------------------

if True:
    def linear_to_srgb(value):
        result = np.zeros_like(value)
        # Convert linear RGB values to sRGB format.
        mask = value <= 0.0031308
        result[mask] = value[mask] * 12.92
        result[~mask] = 1.055 * (value[~mask] ** (1.0 / 2.4)) - 0.055
        result = np.clip(result, 0.0, 1.0) * 255.0
        return result.astype(np.uint8)

    # Load Finland 
    finland = gpd.read_file("finland_reprojected.geojson")  # MultiPolygon
    # Extract Finland CRS
    finland_crs = finland.crs
    # Extract Finland bounds
    xmin, ymin, xmax, ymax = finland.total_bounds
    # Fatten for margins
    xmin -= 10000
    xmax += 10000
    ymin -= 10000
    ymax += 10000

    # Draw satellite image dataset with transparent background and save as figure_dataset.png to be use later
    if True:
        # Define scale factor, m/pixel
        scale = 200
        # Calculate canvas size
        canvas_width = int((xmax - xmin) / scale)
        canvas_height = int((ymax - ymin) / scale)
        # Start with a white canvase
        canvas_data = 255*np.ones((canvas_height, canvas_width, 4), dtype=np.uint8)
        # Plot Finland on canvas (optional)
        if True:
            canvas_image = Image.fromarray(canvas_data)
            draw = ImageDraw.Draw(canvas_image)
            # Extract polygons
            finland_polygons = []
            for polygon in finland.geometry:
                if isinstance(polygon, Polygon):
                    finland_polygons.append(polygon)
                elif isinstance(polygon, MultiPolygon):
                    for poly in polygon.geoms:
                        finland_polygons.append(poly)

            # Plot polygons
            for polygon in finland_polygons:
                x, y = polygon.exterior.coords.xy
                x = (x - xmin) / scale
                y = (ymax - y) / scale
                dest_corners = list(zip(x, y))
                draw.polygon(dest_corners, outline="black", fill="black")
        else:
            # Make the canvas background transparent
            canvas_data[:,:,3] = 0
            canvas_image = Image.fromarray(canvas_data)

        # Loop over satellite image dataset
        for i in range(0, 10000):
            # Print counter
            print(f"\r{i}", end="")
            # Load satellite image
            filename = f"{image_directory}/{i:06d}.nc"
            nc_data = xr.open_dataset(filename)
            # Extract satellite image CRS
            data_crs = nc_data.crs.crs_wkt
            # Extract satellite image datetime
            datetime = nc_data.t.to_numpy()[0]
            # Extract x and y coordinate arrays
            xs = nc_data.x.to_numpy()
            ys = nc_data.y.to_numpy()
            # Crop to 512x512 (same crop as in dataloading.py)
            xs = xs[:512]
            ys = ys[:512]
            # Get RGB channels, crop to 512x512, normalize to reflectance
            r = np.clip(nc_data['B04'][0][:512,:512].as_numpy()/10000, 0, 1).astype(np.float32)
            g = np.clip(nc_data['B03'][0][:512,:512].as_numpy()/10000, 0, 1).astype(np.float32)
            b = np.clip(nc_data['B02'][0][:512,:512].as_numpy()/10000, 0, 1).astype(np.float32)
            # Get uint8 sRGBA image with alpha=255 (no transparency)
            rgb = np.flip(linear_to_srgb(np.stack((r, g, b, np.ones_like(r, dtype=np.float32)), axis=-1, dtype=np.float32)), axis=0)
            # Resize image to final resolution, with antialiasing
            rgb = np.array(Image.fromarray(rgb).resize((5120//scale, 5120//scale), Image.BILINEAR))
            # Replicate pad with alpha = 0 to ensure edge antialiasing by cv2.warpPerspective
            rgb = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
            rgb[0, :, 3] = 0
            rgb[-1, :, 3] = 0
            rgb[:, 0, 3] = 0
            rgb[:, -1, 3] = 0
            # Transform the corner coordinates to the map coordinates
            transformer = Transformer.from_crs(data_crs, finland.crs, always_xy=True)
            corners = np.array([[xs[0], ys[0]], [xs[-1], ys[0]], [xs[-1], ys[-1]], [xs[0], ys[-1]]], dtype=np.float32)
            cornersx, cornersy = transformer.transform(corners[:,0], corners[:,1])
            cornersx = (cornersx - xmin) / scale
            cornersy = (ymax - cornersy) / scale
            dest_corners = np.array(list(zip(cornersx, cornersy)), dtype=np.float32)
            corners = np.array([[-1, rgb.shape[1]-1], [rgb.shape[0]-1, rgb.shape[1]-1], [rgb.shape[0]-1, -1], [-1, -1]], dtype=np.float32)
            perspective_transform = cv2.getPerspectiveTransform(corners, dest_corners)
            warped = cv2.warpPerspective(rgb, perspective_transform, dsize=(canvas_width, canvas_height))
            warped_image = Image.fromarray(warped)
            # Splat the transformed image onto the canvas
            canvas_image = Image.alpha_composite(canvas_image, warped_image)
            
        # Save image
        canvas_image.save("figure_dataset.png")

    # Open the PNG image using PIL and convert to NumPy array
    img = np.array(Image.open('figure_dataset.png'))
    # Define the UTM zone 35N projection (EPSG:32635)
    utm_proj = ccrs.UTM(zone=int(finland_crs.utm_zone[:-1]), southern_hemisphere=False)  # UTM 35N
    # Define WGS84 projection for grid overlay
    wgs84_proj = ccrs.PlateCarree()  # WGS84 lat/lon
    # Create an xarray DataArray, assigning coordinates to axes
    data_array = xr.DataArray(img, dims=["y", "x", "c"], coords={"x": np.linspace(xmin, xmax, img.shape[1]), "y": np.linspace(ymax, ymin, img.shape[0])})
    # Create a figure and set the projection to UTM 35N
    fig_width = single_column_width
    fig, ax = plt.subplots(figsize=(fig_width, fig_width*(ymax-ymin)/(xmax-xmin)), subplot_kw={'projection': utm_proj})
    # Add WGS84 gridlines (this will be tilted/distorted because of the projection difference)
    gl = ax.gridlines(crs=wgs84_proj, draw_labels=True, linewidth=0.5, color='black', zorder=0)
    # Configure the WGS84 gridline labels
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    # Set axis labels for UTM coordinates
    ax.set_xlabel('UTM Easting (meters)')
    ax.set_ylabel('UTM Northing (meters)')
    # Plot Finland 
    finland.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5, zorder=1)
    # Add a scale bar with a length of 100 km
    scalebar = ScaleBar(1, units="m", location='lower right', scale_loc='bottom', fixed_value=100, fixed_units="km")
    ax.add_artist(scalebar)
    # Plot the image in UTM coordinates using the extent and the correct projection
    ax.imshow(img, extent=[xmin, xmax, ymin, ymax], transform=utm_proj, origin='upper', cmap='gray', zorder=2)
    # Show the plot
    plt.show()
    # Save the figure as a PDF for use in LaTeX
    fig.savefig("figure_dataset_grid.pdf", dpi=dpi, bbox_inches='tight')
    fig.savefig("figure_dataset_grid.png", dpi=dpi, bbox_inches='tight')
