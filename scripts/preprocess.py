import os
import json
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, Polygon

# Bounding box and grid parameters
BBOX = [11.50, 75.80, 11.95, 76.40]
GRID_SIZE = 0.02 # ~2.2 km

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the distance between two points on the Earth in kilometers."""
    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    r = 6371.0 # Radius of Earth in kilometers
    return c * r

def extract_coordinates(osm_data):
    """Extracts a list of (lat, lon) coordinates from OSM JSON elements."""
    coords = []
    if not osm_data or "elements" not in osm_data:
        return coords
    
    for element in osm_data["elements"]:
        if element["type"] == "node":
            coords.append((element["lat"], element["lon"]))
        elif "geometry" in element:
            # Ways or relations with geometry
            for pt in element["geometry"]:
                coords.append((pt["lat"], pt["lon"]))
        elif "nodes" in element:
            # Backup if geometry wasn't fetched
            pass
    return coords

def extract_polygons(osm_data):
    """Extracts Shapely Polygon objects from OSM JSON elements (e.g. forests)."""
    polygons = []
    if not osm_data or "elements" not in osm_data:
        return polygons
        
    for element in osm_data["elements"]:
        if "geometry" in element and len(element["geometry"]) >= 3:
            pts = [(pt["lon"], pt["lat"]) for pt in element["geometry"]]
            # Close polygon if not closed
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            try:
                polygons.append(Polygon(pts))
            except Exception as e:
                pass
    return polygons

def calculate_min_distance(centroid_lat, centroid_lon, feature_coords):
    """Calculates the minimum distance from a cell centroid to a set of coordinates."""
    if not feature_coords:
        return 999.0 # Large default distance if feature is missing
        
    # Vectorized Haversine calculation
    lats = np.array([c[0] for c in feature_coords])
    lons = np.array([c[1] for c in feature_coords])
    
    distances = haversine_distance(centroid_lat, centroid_lon, lats, lons)
    return float(np.min(distances))

def preprocess_spatial_data():
    print("Preprocessing spatial data...")
    
    # Load JSON files
    with open(os.path.join(RAW_DIR, "settlements.json"), "r") as f:
        settlements_data = json.load(f)
    with open(os.path.join(RAW_DIR, "roads.json"), "r") as f:
        roads_data = json.load(f)
    with open(os.path.join(RAW_DIR, "water.json"), "r") as f:
        water_data = json.load(f)
    with open(os.path.join(RAW_DIR, "forests.json"), "r") as f:
        forests_data = json.load(f)
        
    # Extract feature coordinates
    settlement_coords = extract_coordinates(settlements_data)
    road_coords = extract_coordinates(roads_data)
    water_coords = extract_coordinates(water_data)
    forest_coords = extract_coordinates(forests_data)
    forest_polys = extract_polygons(forests_data)
    
    print(f"Loaded: {len(settlement_coords)} settlement nodes, {len(road_coords)} road nodes, "
          f"{len(water_coords)} water nodes, {len(forest_coords)} forest nodes.")
          
    # Generate grid
    lat_min, lon_min, lat_max, lon_max = BBOX
    lat_steps = np.arange(lat_min, lat_max, GRID_SIZE)
    lon_steps = np.arange(lon_min, lon_max, GRID_SIZE)
    
    grid_cells = []
    
    cell_id = 0
    for lat in lat_steps:
        for lon in lon_steps:
            # Centroid of cell
            c_lat = lat + GRID_SIZE / 2.0
            c_lon = lon + GRID_SIZE / 2.0
            
            # Calculate distance features in km
            d_road = calculate_min_distance(c_lat, c_lon, road_coords)
            d_water = calculate_min_distance(c_lat, c_lon, water_coords)
            d_settlement = calculate_min_distance(c_lat, c_lon, settlement_coords)
            d_forest = calculate_min_distance(c_lat, c_lon, forest_coords)
            
            # Forest cover percentage estimation: sample 3x3 grid inside cell
            cell_poly = Polygon([
                (lon, lat),
                (lon + GRID_SIZE, lat),
                (lon + GRID_SIZE, lat + GRID_SIZE),
                (lon, lat + GRID_SIZE),
                (lon, lat)
            ])
            
            forest_pts_in_cell = 0
            # Test 9 points inside the cell
            sub_lons = np.linspace(lon + 0.1 * GRID_SIZE, lon + 0.9 * GRID_SIZE, 3)
            sub_lats = np.linspace(lat + 0.1 * GRID_SIZE, lat + 0.9 * GRID_SIZE, 3)
            for slon in sub_lons:
                for slat in sub_lats:
                    p = Point(slon, slat)
                    # Check if point falls inside any forest polygon
                    if any(poly.contains(p) for poly in forest_polys):
                        forest_pts_in_cell += 1
            
            forest_cover_pct = (forest_pts_in_cell / 9.0) * 100.0
            
            # If d_forest is tiny, make sure cover > 0
            if d_forest < 0.5 and forest_cover_pct == 0:
                forest_cover_pct = 25.0
            
            # Agricultural exposure proxy: forest-fringe interfaces
            # Defined as areas where forest is nearby (d_forest < 2.0 km) and settlement/farmland is nearby
            # This represents crop-raiding interface vulnerability.
            d_edge = max(0.1, d_forest + d_settlement)
            agricultural_exposure = (1.0 / d_edge) * 10.0 if d_forest < 3.0 else 0.0
            # Cap at 100
            agricultural_exposure = min(100.0, agricultural_exposure * 8.0)
            
            # Save cell record
            grid_cells.append({
                "cell_id": cell_id,
                "lat_min": lat,
                "lat_max": lat + GRID_SIZE,
                "lon_min": lon,
                "lon_max": lon + GRID_SIZE,
                "centroid_lat": c_lat,
                "centroid_lon": c_lon,
                "d_road": d_road,
                "d_water": d_water,
                "d_settlement": d_settlement,
                "d_forest": d_forest,
                "forest_cover": forest_cover_pct,
                "agricultural_exposure": agricultural_exposure
            })
            
            cell_id += 1
            
    df = pd.DataFrame(grid_cells)
    df.to_csv(os.path.join(PROCESSED_DIR, "grid_features.csv"), index=False)
    print(f"Preprocessed {len(df)} grid cells and saved features.")

if __name__ == "__main__":
    preprocess_spatial_data()
