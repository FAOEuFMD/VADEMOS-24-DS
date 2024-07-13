# VADEMOS-24-DS

## VADEMOS Data Science Repository
This repository contains data and notebooks pertaining to 2 main data components of VADEMOS: predicting population growth and the creation of the animal density measure. 

## Animal Density Measure
# Overview
This documentation outlines the process of analyzing animal density data using Geographic Information Systems (GIS) tools. The analysis focuses on calculating the percentage of animal densities per species across various administrative regions within countries. The primary data sources utilized include the Global Livestock Production and Health Atlas (GLW4) for animal density and Global Administrative Areas (GADM) for administrative boundary shapefiles.

# Data Sources
1.	Global Livestock Production and Health Atlas (GLW4)
   link to data: https://data.apps.fao.org/catalog/iso/9d1e149b-d63f-4213-978b-317a8eb42d02
  	
o	Provides raster data (TIFF files) representing animal densities per species.

o	Data is derived fr+om various sources including national censuses and estimates.

o	Each pixel in the raster represents animal density (animals per km²) at a specific geographic location.

3.	Global Administrative Areas (GADM)
   link: https://gadm.org/download_world.html
o	Provides vector shapefiles (GeoPackages) defining administrative boundaries (e.g., countries, regions, districts).

o	Contains attributes such as administrative level names (e.g., NAME_1 for first-level administrative units, NAME_2 for second-level).

# Process Overview

**1.	Data Acquisition**
o	GLW4 Data Acquisition: Downloaded TIFF files from GLW4 containing animal density data per species at a global scale.
o	GADM Shapefile Acquisition: Obtained GeoPackage files from GADM containing administrative boundary polygons.

**2.	Data Preparation**
o	Raster Data Processing: Used Python and GIS libraries to extract animal density values from TIFF files.
o	Vector Data Processing: Loaded GADM shapefiles into GIS software (e.g., QGIS) to visualize and manipulate administrative boundaries.

**3.	Data Integration and Analysis**
o	Zonal Statistics Calculation: Applied zonal statistics in GIS to aggregate animal density values by administrative regions (NAME_1 and NAME_2).
o	Merge with FAO Data: Combined zonal statistics results with FAO animal population data by country to calculate percentage animal densities.

**4.	Result Compilation**
o	DataFrame Creation: Constructed a structured DataFrame summarizing animal density percentages per country, NAME_1 (regions), and NAME_2 (districts).
o	Data Quality Considerations: Noted the variability in data quality (e.g., census vs. estimated data) and its impact on density calculations.

**Issues to Consider**
•	Data Variability: Acknowledged differences between census data, official statistics, and estimates impacting the accuracy of density calculations.
•	Assumptions: Assumed that density percentages derived from zonal statistics reflect relative density rather than absolute values due to data variability.
•	Application Assumptions: Considered the use of density percentages in applications like Vademos, assuming density variations may affect modeling outcomes.

**Conclusion**
This documentation outlines a systematic approach to analyze animal density across administrative regions using GIS and statistical methods. By integrating raster and vector data sources, and considering data quality issues, it provides a framework for understanding and utilizing animal density information effectively.
For further details, refer to the specific scripts, data files, and software tools used in the analysis process.
