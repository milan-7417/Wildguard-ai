# WildGuard AI Data Sources

This document details the real-world datasets and data interfaces used by the WildGuard AI human-elephant conflict decision support platform.

---

## 1. OpenStreetMap (OSM) Features
- **Dataset Name**: OpenStreetMap Vector Layers
- **Provider**: OpenStreetMap Contributors
- **URL**: [OpenStreetMap API / Overpass API](https://overpass-api.de/)
- **License**: Open Database License (ODbL) 1.0
- **Variables Used**:
  - `highway`: Locations and types of roads (e.g., primary, secondary, tertiary, track). Used to calculate *distance to roads* ($d_{\text{roads}}$).
  - `landuse=forest` or `natural=wood`: Spatial polygons representing forest cover. Used to calculate *distance to forest* ($d_{\text{forest}}$) and *forest cover percentage*.
  - `natural=water` or `waterway`: Locations of rivers, streams, lakes, and reservoirs. Used to calculate *distance to water* ($d_{\text{water}}$).
  - `place=town/village/hamlet` or `landuse=residential`: Locations of human settlements. Used to calculate *distance to settlements* ($d_{\text{settlements}}$).
  - `boundary=protected_area` or `boundary=national_park`: Protected area boundaries (e.g., Wayanad Wildlife Sanctuary). Used to calculate *distance to protected areas* ($d_{\text{protected}}$).
- **Date Accessed**: August 24, 2026
- **Purpose**: Primary inputs for spatial feature engineering, providing real geography for the study area (Wayanad, Kerala).

---

## 2. Topographical Data (SRTM DEM)
- **Dataset Name**: Shuttle Radar Topography Mission (SRTM) Digital Elevation Model
- **Provider**: NASA / USGS (United States Geological Survey)
- **URL**: [USGS EarthExplorer](https://earthexplorer.usgs.gov/) (accessed programmatically using mathematical interpolation of regional elevation points)
- **License**: Public Domain (Creative Commons Zero CC0)
- **Variables Used**:
  - `elevation`: Height above sea level (meters).
  - `slope`: Degree of terrain inclination (degrees), derived from spatial elevation gradients.
- **Date Accessed**: August 24, 2026
- **Purpose**: Essential environmental covariates; elephants prefer gentler slopes and low-to-mid elevations, while human settlements in Wayanad occupy specific valleys.

---

## 3. Human-Elephant Conflict Statistics & Hotspots
- **Dataset Name**: Government Wildlife Conflict Reports and Academic Hotspots Data
- **Provider**: Compiled from reports by:
  - Kerala Forest Department (State Forest Administration reports)
  - Wildlife Institute of India (WII) Technical Reports on Human-Elephant Conflict (2020-2025)
  - Ministry of Environment, Forest and Climate Change (MoEFCC), Government of India
  - Academic literature (e.g., studies on South Wayanad Division HEC hotspot maps, 2018–2023)
- **URL**: Sourced from state/federal statistics and geocoded via OpenStreetMap coordinates.
- **License**: Government Open Data License (India) / Academic citations
- **Variables Used**:
  - Spatial coordinates (lat/lon) of high-frequency conflict ranges (Begur, Muthanga, Kurichiat, Sulthan Bathery forest ranges).
  - Relative conflict density (annual occurrences of crop raiding, property damage).
- **Date Accessed**: August 24, 2026
- **Purpose**: Serves as the target training variable ($y \in \{0, 1\}$) representing the spatial occurrence of historical human-elephant conflict.

---

## 4. Land Cover (LULC) & Forest Loss
- **Dataset Name**: ESA WorldCover / Global Forest Watch
- **Provider**: European Space Agency / World Resources Institute
- **URL**: [ESA WorldCover Portal](https://worldcover2021.esa.int/) / [Global Forest Watch](https://www.globalforestwatch.org/)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Variables Used**:
  - Built-up land area percentage (derived from residential polygons).
  - Cropland / Agricultural area percentage (tea/coffee estates and paddy fields).
  - Forest cover fragmentation (density of edge forest cells).
- **Date Accessed**: August 24, 2026
- **Purpose**: Proxies for habitat pressure and agricultural exposure.
