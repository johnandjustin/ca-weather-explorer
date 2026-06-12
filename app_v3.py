import streamlit as st
import pandas as pd
import pydeck as pdk

st.set_page_config(layout="wide", page_title="CA Weather Explorer")
st.title("☀️ California Microclimate Explorer")

@st.cache_data
def load_and_prep_data():
    df = pd.read_parquet("california_weather_scores.parquet")
    
    df['sun_pct'] = (df['sunny_hrs'] / df['day_hrs']) * 100
    df['hot_80_pct'] = (df['hot_80_hrs'] / df['day_hrs']) * 100
    df['hot_85_pct'] = (df['hot_85_hrs'] / df['day_hrs']) * 100
    df['hot_90_pct'] = (df['hot_90_hrs'] / df['day_hrs']) * 100
    df['cold_75_pct'] = (df['cold_75_hrs'] / df['day_hrs']) * 100
    df['cold_70_pct'] = (df['cold_70_hrs'] / df['day_hrs']) * 100
    df['cold_65_pct'] = (df['cold_65_hrs'] / df['day_hrs']) * 100
    
    df['sun_fmt'] = df['sun_pct'].round(1).astype(str) + "%"
    df['hot_80_fmt'] = df['hot_80_pct'].round(1).astype(str) + "%"
    df['hot_85_fmt'] = df['hot_85_pct'].round(1).astype(str) + "%"
    df['hot_90_fmt'] = df['hot_90_pct'].round(1).astype(str) + "%"
    df['cold_75_fmt'] = df['cold_75_pct'].round(1).astype(str) + "%"
    df['cold_70_fmt'] = df['cold_70_pct'].round(1).astype(str) + "%"
    df['cold_65_fmt'] = df['cold_65_pct'].round(1).astype(str) + "%"
    
    return df

try:
    df = load_and_prep_data()
except FileNotFoundError:
    st.error("Data file missing. Run analyze_matrix.py first.")
    st.stop()

col1, col2 = st.columns([1, 3])

with col1:
    st.header("Filter Criteria")
    st.write("Adjust your tolerances to find matching microclimates.")
    
    st.subheader("1. Minimum Sunshine")
    min_sun_val = float(df['sun_pct'].min())
    max_sun_val = float(df['sun_pct'].max())
    sun_target = st.slider(
        "Require at least X% sunny daytime hours:", 
        min_value=min_sun_val, 
        max_value=max_sun_val, 
        value=max_sun_val * 0.8,
        format="%.1f%%"
    )
    
    st.subheader("2. Heat Tolerance")
    hot_temp = st.selectbox("Define 'Hot':", options=["80", "85", "90"], index=1)
    hot_col = f"hot_{hot_temp}_pct"
    
    min_hot_val = float(df[hot_col].min())
    max_hot_val = float(df[hot_col].max())
    hot_target = st.slider(
        f"Maximum allowed daytime hours over {hot_temp}°F:", 
        min_value=min_hot_val, 
        max_value=max_hot_val, 
        value=max_hot_val * 0.2,
        format="%.1f%%"
    )
    
    st.subheader("3. Cold Tolerance")
    cold_temp = st.selectbox("Define 'Cold':", options=["75", "70", "65"], index=1)
    cold_col = f"cold_{cold_temp}_pct"
    
    min_cold_val = float(df[cold_col].min())
    max_cold_val = float(df[cold_col].max())
    cold_target = st.slider(
        f"Maximum allowed daytime hours under {cold_temp}°F:", 
        min_value=min_cold_val, 
        max_value=max_cold_val, 
        value=max_cold_val * 0.3,
        format="%.1f%%"
    )

    filtered_df = df[
        (df['sun_pct'] >= sun_target) &
        (df[hot_col] <= hot_target) &
        (df[cold_col] <= cold_target)
    ].copy()
    
    st.divider()
    st.success(f"**Found {len(filtered_df):,} matching microclimates!**")
    
    result_container = st.empty()

with col2:
    # THE FIX: Swapped ScatterplotLayer for ColumnLayer to create perfect 3000m grid boxes
    # A 3000m wide square has a vertex radius of roughly 2121m
    bg_layer = pdk.Layer(
        "ColumnLayer",
        data=df,
        id="california_base_grid", 
        get_position=["lon", "lat"],
        radius=2121, 
        disk_resolution=4, # 4 sides = Square
        angle=45, # Aligns the flat sides to the map axes
        extruded=False, # Flattens it to the map
        get_fill_color=[200, 200, 200, 15],
        pickable=True,
    )

    dynamic_fg_id = f"match_grid_{sun_target}_{hot_target}_{cold_target}_{hot_temp}_{cold_temp}"
    
    fg_layer = pdk.Layer(
        "ColumnLayer",
        data=filtered_df,
        id=dynamic_fg_id, 
        get_position=["lon", "lat"],
        radius=2121,
        disk_resolution=4,
        angle=45,
        extruded=False,
        get_fill_color=[52, 152, 219, 140], 
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=36.7783,
        longitude=-119.4179,
        zoom=5.5,
        pitch=0,
    )

    tooltip_html = (
        "<b>Sun:</b> {sun_fmt}<br/>"
        f"<b>Over {hot_temp}F:</b> {{hot_{hot_temp}_fmt}}<br/>"
        f"<b>Under {cold_temp}F:</b> {{cold_{cold_temp}_fmt}}"
    )

    r = pdk.Deck(
        layers=[bg_layer, fg_layer],
        initial_view_state=view_state,
        map_style="light",
        tooltip={"html": tooltip_html}
    )

    event = st.pydeck_chart(
        r, 
        on_select="rerun", 
        selection_mode="single-object", 
        height=1600,
        key="california_weather_map"
    )

with result_container.container():
    st.header("Specific Location Check")
    if event and event.selection.objects:
        clicked_point = event.selection.objects[0]
        st.write(f"**Coordinates:** {clicked_point['lat']:.4f}, {clicked_point['lon']:.4f}")
        st.info(f"☀️ Sunny Hours: {clicked_point['sun_pct']:.1f}%")
        st.error(f"🔥 Hours > {hot_temp}°F: {clicked_point[hot_col]:.1f}%")
        st.info(f"❄️ Hours < {cold_temp}°F: {clicked_point[cold_col]:.1f}%")
    else:
        st.write("Click any grid box on the map to see its exact stats.")