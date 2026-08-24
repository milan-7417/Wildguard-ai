import os
import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join("data", "processed")

# Documented HEC hotspots in Wayanad (lat, lon, weight/intensity)
# Based on forest division reports (Muthanga, Begur, Pulpally/Chekadi, Vythiri fringes, Thirunelli corridor)
HOTSPOTS = [
    {"name": "Muthanga Forest Edge", "lat": 11.6700, "lon": 76.3500, "intensity": 2.5},
    {"name": "Pulpally / Chekadi Farmlands", "lat": 11.7915, "lon": 76.1738, "intensity": 3.0},
    {"name": "Thirunelli Elephant Corridor", "lat": 11.9056, "lon": 76.0125, "intensity": 2.2},
    {"name": "Begur Forest Fringe", "lat": 11.8500, "lon": 75.9900, "intensity": 1.8},
    {"name": "Vythiri / Lakkidi Plantations", "lat": 11.5312, "lon": 76.0425, "intensity": 2.0},
    {"name": "Sulthan Bathery Interface", "lat": 11.6667, "lon": 76.2800, "intensity": 2.4},
    {"name": "Panamaram Valleys", "lat": 11.7258, "lon": 76.0800, "intensity": 1.5}
]

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2.0 * np.arcsin(np.sqrt(a)) * 6371.0

def get_elevation_and_slope(lat, lon):
    """
    Simulates Wayanad's elevation profile. 
    Wayanad contains high peaks in the south/west (e.g. Chembra Peak 2100m, Banasura Hill 2050m) 
    and drops to a plateau of ~700-800m towards the east (Karnataka plains).
    """
    # Base plateau elevation
    elevation = 750.0
    
    # Add Chembra/Banasura massive in the southwest
    # Chembra center: ~11.52 N, 76.08 E
    # Banasura center: ~11.61 N, 75.92 E
    dist_chembra = haversine(lat, lon, 11.52, 76.08)
    dist_banasura = haversine(lat, lon, 11.61, 75.92)
    
    elevation += 1350.0 * np.exp(-(dist_chembra / 8.0)**2)
    elevation += 1300.0 * np.exp(-(dist_banasura / 7.0)**2)
    
    # Add minor ridge variations
    ridge = 150.0 * np.sin(lat * 80.0) * np.cos(lon * 80.0)
    elevation += ridge
    
    # Calculate slope based on spatial gradient (elevation difference over small step)
    step = 0.005 # 500 meters
    e_east = 750.0 + 1350.0 * np.exp(-(haversine(lat, lon + step, 11.52, 76.08) / 8.0)**2) + 1300.0 * np.exp(-(haversine(lat, lon + step, 11.61, 75.92) / 7.0)**2)
    e_north = 750.0 + 1350.0 * np.exp(-(haversine(lat + step, lon, 11.52, 76.08) / 8.0)**2) + 1300.0 * np.exp(-(haversine(lat + step, lon, 11.61, 75.92) / 7.0)**2)
    
    grad_x = (e_east - elevation) / 0.5 # delta elevation / 0.5 km
    grad_y = (e_north - elevation) / 0.5
    
    slope = np.sqrt(grad_x**2 + grad_y**2)
    # Convert gradient to degrees
    slope_deg = np.degrees(np.arctan(slope / 100.0))
    slope_deg = np.clip(slope_deg, 0, 45) # clip to reasonable slope values
    
    return float(elevation), float(slope_deg)

def build_features():
    print("Building model features...")
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "grid_features.csv"))
    
    elevations = []
    slopes = []
    incident_densities = []
    
    for idx, row in df.iterrows():
        lat, lon = row["centroid_lat"], row["centroid_lon"]
        
        # 1. Topography
        elev, slope = get_elevation_and_slope(lat, lon)
        elevations.append(elev)
        slopes.append(slope)
        
        # 2. Historical Conflict Density (from documented hotspots and forest edge proximity)
        density = 0.0
        for hs in HOTSPOTS:
            dist = haversine(lat, lon, hs["lat"], hs["lon"])
            # Decay factor: conflict drops as distance from hotspot increases
            # HEC incidents generally occur within 4km of forest boundaries/hotspot centers
            decay = np.exp(-(dist / 3.5)**2)
            density += hs["intensity"] * decay
            
        # Modulate conflict density by distance to forest
        # Conflicts happen near forest fringes. If deep in forest (d_forest ~ 0) or far away, conflict is lower
        # Peak conflict is at d_forest between 0.1km and 2km
        fringe_mod = np.exp(-((row["d_forest"] - 0.5) / 1.5)**2)
        density *= (fringe_mod + 0.1)
        
        # Modulate conflict density by forest cover: completely deforested urban areas or dense interior forests have less crop raiding
        # Crop raiding peaks in moderate forest cover/fringe cells
        cover_mod = np.exp(-((row["forest_cover"] - 40.0) / 25.0)**2)
        density *= (cover_mod + 0.2)
        
        incident_densities.append(density)
        
    df["elevation"] = elevations
    df["slope"] = slopes
    df["historical_conflict_density"] = incident_densities
    
    # 3. Target variable: High conflict cell = 1, Low conflict cell = 0
    # Let's check the distribution. We want a balanced target where ~20% of cells are high-conflict.
    threshold = np.percentile(incident_densities, 80) # top 20% cells are labeled high risk
    df["target"] = (df["historical_conflict_density"] >= threshold).astype(int)
    
    # Document how the target is created
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    df.to_csv(os.path.join(PROCESSED_DIR, "model_features.csv"), index=False)
    print("Features and target built successfully.")

if __name__ == "__main__":
    build_features()
