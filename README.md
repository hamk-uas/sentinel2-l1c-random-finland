# sentinel2-l1c-random-finland

This code is for collecting a database of random Sentinel 2 satellite images. An earlier version of the source code was used to collect and assemble the database:

Niemitalo, O., Anzini, E. Jr., & Liczkoski, V. H. D. (2024). 10k random 512x512 pixel Sentinel 2 Level-1C RGB satellite images over Finland, years 2015–2022. Häme University of Applied Sciences. https://doi.org/10.23729/32a321ac-9012-4f17-a849-a4e7ed6b6c8c

The current version should be able to produce a dataset with the same distribution.

Workflow:
* `random_sentinel2_retrieve.py` -- Collect dataset (configure with command line params)
* `zip_images.py` -- Create zip files of the dataset (configure by editing the source code)
* `make_overview.py` -- Create an overview of the dataset (configure by editing the source code)

Authors: Olli Niemitalo (HAMK Häme University of Applied Sciences) Elias Anzini Junior, and Vinicius Hermann D. Liczkoski.

Source code copyright 2023-2024 HAMK and authors.

`finland_reprojected.geojson` (the approximate geographical area of Finland) is based on free vector data from Natural Earth. https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/ The procedure to simplify the geometry in QGIS was: Select country, Keep N biggest parts (20 parts), Multipart to singleparts, Simplify (tolerance 0.01 deg), save GeoJson at a precision of 2 decimal places.
