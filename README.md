# VADEMOS-24-DS

## VADEMOS Data Science Repository
This repository contains data and notebooks pertaining to 2 main data components of VADEMOS: predicting population growth and the creation of the animal density measure. 

## Repository Structure
```
VADEMOS-24-DS/
├── Creating_Animal_Density/         # Animal density analysis
│   ├── animal_density.ipynb         # Transforms GLW cattle density data
│   └── density_DF.ipynb             # Validation and data frame creation
│
├── Predicting_Population_Growth/    # Population growth forecasting
│   ├── population_*.ipynb           # Various population prediction notebooks
│   ├── data/                        # Input data files from FAO
│   ├── evaluation/                  # Model evaluation results
│   ├── helpers/                     # Python utility modules
│   └── plots/                       # Generated forecast plots
│
├── LICENSE                          # MIT License
└── requirements.txt                 # Python dependencies
```

## Installation and Setup
1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Ensure you have Jupyter Notebook installed to run the analysis notebooks

## Python Dependencies
This project relies on several Python libraries:
- Data processing: pandas, numpy
- Time series forecasting: AutoTS, pmdarima
- Visualization: matplotlib, seaborn
- Geospatial analysis: geopandas, rasterio, rasterstats

See `requirements.txt` for the complete list with versions.

## Predicting Population Growth Module

The Population Growth component implements a robust forecasting framework for livestock populations across multiple countries and species. Using historical FAO livestock data spanning several decades, this module employs both traditional ARIMA models and advanced AutoTS ensemble techniques to generate reliable projections through 2033. The methodology includes comprehensive model evaluation with multiple metrics (RMSE, MAE, confidence interval coverage) and handles data quality challenges by implementing fallback strategies for regions with limited historical records. Notable features include automated model selection based on performance criteria, ensemble methods that combine multiple forecasting approaches for improved accuracy, and specialized handling for different animal species. The module is structured with a modular Python codebase that separates core modeling functions (`auto_arima_modelling.py`, `autots_modelling.py`) from analysis notebooks, allowing for easy maintenance and extension. This comprehensive approach enables VADEMOS to incorporate high-quality livestock population projections into disease modeling scenarios.

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

o	All files are uploaded in gdrive as they are too large to be uploaded here link: 
https://drive.google.com/drive/folders/1pT87dA_tAEZAxF_3OUKBW8LEw0ZBBScv?usp=sharing

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

## Population Growth Prediction

### Overview
This component focuses on predicting livestock population growth across various countries and regions using time series forecasting techniques. The analysis leverages historical FAO livestock population data to forecast future population trends.

### Methodology
1. **Data Preparation**: Historical FAO livestock population data is processed and cleaned.
2. **Model Selection**: Two primary approaches are used:
   - **ARIMA Models**: Classical time series forecasting with auto-selected parameters
   - **AutoTS Models**: Automated time series forecasting with ensemble methods

3. **Model Evaluation**: Models are evaluated based on:
   - Accuracy metrics (MAPE, RMSE)
   - Percentage of actual values within prediction confidence intervals
   - Stability and reasonableness of forecasts

4. **Forecasting**: Selected models are used to generate predictions for future years (typically 2023-2033).

### Key Files
- `population_prediction.ipynb`: Main notebook for population forecasting
- `population_prediction_ethiopia.ipynb`: Country-specific analysis for Ethiopia
- `helpers/autots_modelling.py`: AutoTS implementation
- `helpers/auto_arima_modelling.py`: ARIMA implementation
- `evaluation/`: Directory containing model performance results

### Constants
Key parameters for modeling (configured in `helpers/constants.py`):
- Confidence interval: 95%
- Minimum training samples: 20
- Forecasting period: 11 years (2023-2033)

## How to Contribute
Contributions are welcome! Please feel free to submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
