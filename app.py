import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="CA Weather Microclimates")
st.title("☀️ California's Best Weather Microclimates")

@st.cache_data
def load_data():
    df = pd.read_parquet("california_weather_scores.parquet")
    return df.sort_values('score', ascending=False).reset_index(drop=True)

try:
    df = load_data()
except FileNotFoundError:
    st.error("Data file missing. Run analyze_matrix.py first to generate the Parquet file.")
    st.stop()

col1, col2 = st.columns([1, 3])

with col1:
    st.header("Map Controls")
    
    max_slider = len(df) if len(df) < 5000 else 5000
    top_n = st.slider("Show Top 'N' Locations:", min_value=10, max_value=max_slider, value=100, step=10)
    
    map_data = df.head(top_n)
    st.write(f"Displaying the top {top_n} highest scoring microclimates.")
    st.divider()
    
    st.header("Location Lookup")
    st.write("Click **anywhere** on the map to see the historical weather score for that specific area!")
    
    # We create an empty placeholder here so the results always appear nicely in the sidebar
    result_container = st.empty()

with col2:
    # Build the Folium Map (this replaces the basic st.map)
    m = folium.Map(location=[36.7783, -119.4179], zoom_start=6, tiles="CartoDB positron")
    
    # Add the Top N markers
    for _, row in map_data.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5,
            color="#e74c3c",
            fill=True,
            fill_opacity=0.7,
            tooltip=f"Score: {row['score']:.2f}%"
        ).add_to(m)
        
    # Render the map and capture user interactions
    # FIX 1: use_container_width and height=800 completely solves the vertical compression
    map_interaction = st_folium(m, use_container_width=True, height=1600)

# FIX 2: Process the map click dynamically
with result_container.container():
    # map_interaction returns a dictionary of events. We look for 'last_clicked'.
    if map_interaction and map_interaction.get("last_clicked"):
        click_lat = map_interaction["last_clicked"]["lat"]
        click_lon = map_interaction["last_clicked"]["lng"]
        
        # Pythagorean distance to find the closest matrix pixel to your click
        distances = np.sqrt((df['lat'] - click_lat)**2 + (df['lon'] - click_lon)**2)
        closest_point = df.loc[distances.idxmin()]
        
        st.success(f"**Score:** {closest_point['score']:.2f}%")
        st.info(f"Perfect Hours: {closest_point['perf_hrs']} / {closest_point['day_hrs']}")
        
        # Shows exactly where you clicked vs the center of the grid pixel it matched you with
        st.caption(f"Clicked: {click_lat:.4f}, {click_lon:.4f}")
        st.caption(f"Nearest Grid: {closest_point['lat']:.4f}, {closest_point['lon']:.4f}")
    else:
        st.info("Waiting for map click... 👆")