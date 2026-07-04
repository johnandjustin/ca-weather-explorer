import zarr
import numpy as np
import pandas as pd
import pvlib
import pyproj
import os
from tqdm import tqdm

# 1. Expanded Geographic Bounds
bounds = {
    "NW": {"lat": 42.5, "lon": -125.5},
    "SE": {"lat": 32.0, "lon": -113.5},
    "SW": {"lat": 32.0, "lon": -125.5},
    "NE": {"lat": 42.5, "lon": -113.5}
}

hrrr_proj = pyproj.Proj(
    proj='lcc', lat_1=38.5, lat_2=38.5, lat_0=38.5, lon_0=-97.5,
    a=6371229.0, b=6371229.0
)

X_MIN, Y_MIN = -2697520.142522, -1587306.152557
DX, DY = 3000.0, 3000.0

x_nw, y_nw = hrrr_proj(bounds["NW"]["lon"], bounds["NW"]["lat"])
x_se, y_se = hrrr_proj(bounds["SE"]["lon"], bounds["SE"]["lat"])
x_sw, y_sw = hrrr_proj(bounds["SW"]["lon"], bounds["SW"]["lat"])
x_ne, y_ne = hrrr_proj(bounds["NE"]["lon"], bounds["NE"]["lat"])

ix_min = min(int(round((x_nw - X_MIN) / DX)), int(round((x_sw - X_MIN) / DX)))
ix_max = max(int(round((x_se - X_MIN) / DX)), int(round((x_ne - X_MIN) / DX)))
iy_min = min(int(round((y_sw - Y_MIN) / DY)), int(round((y_se - Y_MIN) / DY)))
iy_max = max(int(round((y_nw - Y_MIN) / DY)), int(round((y_ne - Y_MIN) / DY)))

ca_width = ix_max - ix_min
ca_height = iy_max - iy_min

print("Mounting California Zarr Database...")
zarr_path = "california_weather_archive.zarr"
root = zarr.open(zarr_path, mode='r')

# We link the datasets but DO NOT load them into RAM yet (removed the [:])
temp_ds = root['temperature']
rad_ds = root['radiation']
u_ds = root['u_wind']
v_ds = root['v_wind']
time_strs = root['timestamps'][:]
time_index = pd.to_datetime(time_strs, utc=True)
total_hours = len(time_index)

# 2. Pre-allocate RAM-efficient 2D running totals
day_hrs = np.zeros((ca_height, ca_width), dtype=np.float32)
sunny_hrs = np.zeros((ca_height, ca_width), dtype=np.float32)
hot_80_hrs = np.zeros((ca_height, ca_width), dtype=np.float32)
hot_85_hrs = np.zeros((ca_height, ca_width), dtype=np.float32)
hot_90_hrs = np.zeros((ca_height, ca_width), dtype=np.float32)

c75_w5_hrs = np.zeros((ca_height, ca_width), dtype=np.float32)
c75_w10_hrs = np.zeros((ca_height, ca_width), dtype=np.float32)
c75_w15_hrs = np.zeros((ca_height, ca_width), dtype=np.float32)
c70_w5_hrs = np.zeros((ca_height, ca_width), dtype=np.float32)
c70_w10_hrs = np.zeros((ca_height, ca_width), dtype=np.float32)
c70_w15_hrs = np.zeros((ca_height, ca_width), dtype=np.float32)
c65_w5_hrs = np.zeros((ca_height, ca_width), dtype=np.float32)
c65_w10_hrs = np.zeros((ca_height, ca_width), dtype=np.float32)
c65_w15_hrs = np.zeros((ca_height, ca_width), dtype=np.float32)

# 3. Generate Astronomical Clear Sky Baseline
print("Calculating Astronomical Clear Sky Index...")
site = pvlib.location.Location(37.1661, -119.4494, tz='UTC')
clearsky_ghi = site.get_clearsky(time_index)['ghi'].values

# 4. Out-of-Core Processing Loop
CHUNK_SIZE = 1000
print(f"Executing Chunked Processing ({CHUNK_SIZE} hours per block)...")

for start_idx in tqdm(range(0, total_hours, CHUNK_SIZE)):
    end_idx = min(start_idx + CHUNK_SIZE, total_hours)
    
    # Safely pull only this specific chunk into RAM
    temp_k = temp_ds[start_idx:end_idx]
    rad = rad_ds[start_idx:end_idx]
    u_wind = u_ds[start_idx:end_idx]
    v_wind = v_ds[start_idx:end_idx]

    # Matrix Math for the isolated chunk
    temp_f = (temp_k - 273.15) * 9/5 + 32
    wind_mph = np.sqrt(u_wind**2 + v_wind**2) * 2.23694

    clearsky_chunk = clearsky_ghi[start_idx:end_idx, np.newaxis, np.newaxis]
    csi_matrix = rad / (clearsky_chunk + 1e-5)

    valid_data_mask = ~np.isnan(rad) & ~np.isnan(u_wind)
    daytime_mask = (clearsky_chunk > 10) & valid_data_mask 
    
    # Add chunk sums directly to the running 2D totals
    day_hrs += np.sum(daytime_mask, axis=0)
    sunny_hrs += np.sum((csi_matrix > 0.80) & daytime_mask, axis=0)

    hot_80_hrs += np.sum((temp_f > 80) & daytime_mask, axis=0)
    hot_85_hrs += np.sum((temp_f > 85) & daytime_mask, axis=0)
    hot_90_hrs += np.sum((temp_f > 90) & daytime_mask, axis=0)

    temp_75_mask = (temp_f < 75) & daytime_mask
    temp_70_mask = (temp_f < 70) & daytime_mask
    temp_65_mask = (temp_f < 65) & daytime_mask

    wind_5_mask = wind_mph > 5
    wind_10_mask = wind_mph > 10
    wind_15_mask = wind_mph > 15

    c75_w5_hrs += np.sum(temp_75_mask & wind_5_mask, axis=0)
    c75_w10_hrs += np.sum(temp_75_mask & wind_10_mask, axis=0)
    c75_w15_hrs += np.sum(temp_75_mask & wind_15_mask, axis=0)

    c70_w5_hrs += np.sum(temp_70_mask & wind_5_mask, axis=0)
    c70_w10_hrs += np.sum(temp_70_mask & wind_10_mask, axis=0)
    c70_w15_hrs += np.sum(temp_70_mask & wind_15_mask, axis=0)

    c65_w5_hrs += np.sum(temp_65_mask & wind_5_mask, axis=0)
    c65_w10_hrs += np.sum(temp_65_mask & wind_10_mask, axis=0)
    c65_w15_hrs += np.sum(temp_65_mask & wind_15_mask, axis=0)

# 5. Mapping Coordinates and Exporting Dataset
print("\nMapping Coordinates and Exporting Dataset...")
x_1d = X_MIN + (np.arange(ix_min, ix_max) * DX)
y_1d = Y_MIN + (np.arange(iy_min, iy_max) * DY)
xv, yv = np.meshgrid(x_1d, y_1d)
lons_2d, lats_2d = hrrr_proj(xv, yv, inverse=True)

df = pd.DataFrame({
    'lat': lats_2d.flatten(),
    'lon': lons_2d.flatten(),
    'day_hrs': day_hrs.flatten(),
    'sunny_hrs': sunny_hrs.flatten(),
    'hot_80_hrs': hot_80_hrs.flatten(),
    'hot_85_hrs': hot_85_hrs.flatten(),
    'hot_90_hrs': hot_90_hrs.flatten(),
    
    'c75_w5': c75_w5_hrs.flatten(),
    'c75_w10': c75_w10_hrs.flatten(),
    'c75_w15': c75_w15_hrs.flatten(),
    'c70_w5': c70_w5_hrs.flatten(),
    'c70_w10': c70_w10_hrs.flatten(),
    'c70_w15': c70_w15_hrs.flatten(),
    'c65_w5': c65_w5_hrs.flatten(),
    'c65_w10': c65_w10_hrs.flatten(),
    'c65_w15': c65_w15_hrs.flatten(),
})

df = df[df['day_hrs'] > 0].reset_index(drop=True)

output_file = "california_weather_scores.parquet"
df.to_parquet(output_file, engine='pyarrow')
print(f"Success! Expanded Sensible Cold dataset saved to {os.path.abspath(output_file)}")