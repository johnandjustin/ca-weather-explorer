import zarr
import s3fs
import pyproj
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# 1. Define the 4 Corners (Expanded with a 0.5-degree safety buffer)
# California's true borders: N: 42.0, S: 32.53, W: -124.41, E: -114.13
bounds = {
    "NW": {"lat": 42.5, "lon": -125.5},
    "SE": {"lat": 32.0, "lon": -113.5},
    "SW": {"lat": 32.0, "lon": -125.5},
    "NE": {"lat": 42.5, "lon": -113.5}
}

time_range = pd.date_range(start="2014-07-01", end="2024-07-01", freq="h", inclusive="left")
num_hours = len(time_range)

# 2. HRRR Grid Setup
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

# Find the true mathematical extremes of the tilted grid
ix_min = min(int(round((x_nw - X_MIN) / DX)), int(round((x_sw - X_MIN) / DX)))
ix_max = max(int(round((x_se - X_MIN) / DX)), int(round((x_ne - X_MIN) / DX)))
iy_min = min(int(round((y_sw - Y_MIN) / DY)), int(round((y_se - Y_MIN) / DY)))
iy_max = max(int(round((y_nw - Y_MIN) / DY)), int(round((y_ne - Y_MIN) / DY)))

ca_width = ix_max - ix_min
ca_height = iy_max - iy_min

print(f"California Matrix Size: {ca_width} x {ca_height} pixels")
print(f"Total Hours to Download: {num_hours}")

# 3. Initialize Local Zarr Store
local_store_path = "california_weather_archive.zarr"
root = zarr.open(local_store_path, mode='a')

if 'temperature' not in root:
    tmp_out = root.create_dataset('temperature', shape=(num_hours, ca_height, ca_width), chunks=(1, ca_height, ca_width), dtype='f4', fill_value=np.nan)
    rad_out = root.create_dataset('radiation', shape=(num_hours, ca_height, ca_width), chunks=(1, ca_height, ca_width), dtype='f4', fill_value=np.nan)
    u_out = root.create_dataset('u_wind', shape=(num_hours, ca_height, ca_width), chunks=(1, ca_height, ca_width), dtype='f4', fill_value=np.nan)
    v_out = root.create_dataset('v_wind', shape=(num_hours, ca_height, ca_width), chunks=(1, ca_height, ca_width), dtype='f4', fill_value=np.nan)
    
    timestamps_array = np.array(time_range.strftime('%Y-%m-%dT%H:%M:%S').tolist(), dtype='<U20')
    root.create_dataset('timestamps', data=timestamps_array, overwrite=True)
else:
    tmp_out = root['temperature']
    rad_out = root['radiation']
    u_out = root['u_wind']
    v_out = root['v_wind']

# 4. The Multithreaded Download Worker Function
def fetch_hour(i, dt):
    fs = s3fs.S3FileSystem(anon=True, skip_instance_cache=True)
    
    date_str = dt.strftime('%Y%m%d')
    hour_str = dt.strftime('%H')
    base_url = f"s3://hrrrzarr/sfc/{date_str}/{date_str}_{hour_str}z_anl.zarr"
    
    try:
        tmp_store = s3fs.S3Map(f"{base_url}/2m_above_ground/TMP/2m_above_ground/TMP", s3=fs)
        rad_store = s3fs.S3Map(f"{base_url}/surface/DSWRF/surface/DSWRF", s3=fs)
        ugrd_store = s3fs.S3Map(f"{base_url}/10m_above_ground/UGRD/10m_above_ground/UGRD", s3=fs)
        vgrd_store = s3fs.S3Map(f"{base_url}/10m_above_ground/VGRD/10m_above_ground/VGRD", s3=fs)
        
        tmp_arr = zarr.open(tmp_store, mode='r')
        rad_arr = zarr.open(rad_store, mode='r')
        u_arr = zarr.open(ugrd_store, mode='r')
        v_arr = zarr.open(vgrd_store, mode='r')
        
        ca_tmp_slice = tmp_arr[iy_min:iy_max, ix_min:ix_max]
        ca_rad_slice = rad_arr[iy_min:iy_max, ix_min:ix_max]
        ca_u_slice = u_arr[iy_min:iy_max, ix_min:ix_max]
        ca_v_slice = v_arr[iy_min:iy_max, ix_min:ix_max]
        
        tmp_out[i, :, :] = ca_tmp_slice
        rad_out[i, :, :] = ca_rad_slice
        u_out[i, :, :] = ca_u_slice
        v_out[i, :, :] = ca_v_slice
        return True
    except Exception:
        return False

# 5. Execute Concurrent Pool
print("Starting High-Speed Concurrent Download...")
errors = 0

with ThreadPoolExecutor(max_workers=15) as executor:
    futures = {executor.submit(fetch_hour, i, dt): i for i, dt in enumerate(time_range)}
    
    for future in tqdm(as_completed(futures), total=num_hours, desc="Downloading", unit="hr"):
        success = future.result()
        if not success:
            errors += 1

print(f"\nDownload Complete! Saved to {os.path.abspath(local_store_path)}")
print(f"Failed/Missing Hours: {errors} out of {num_hours}")