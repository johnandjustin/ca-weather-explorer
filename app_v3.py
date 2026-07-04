import streamlit as st
import pandas as pd
import pydeck as pdk

st.set_page_config(layout="wide", page_title="CA Weather Explorer")
st.title("☀️ California Microclimate Explorer")

@st.cache_data
def load_and_prep_data():
    df = pd.read_parquet("california_weather_scores.parquet")
    
    # Calculate base percentages
    df['sun_pct'] = (df['sunny_hrs'] / df['day_hrs']) * 100
    df['hot_80_pct'] = (df['hot_80_hrs'] / df['day_hrs']) * 100
    df['hot_85_pct'] = (df['hot_85_hrs'] / df['day_hrs']) * 100
    df['hot_90_pct'] = (df['hot_90_hrs'] / df['day_hrs']) * 100
    
    # Format base strings for map tooltips
    df['sun_fmt'] = df['sun_pct'].round(1).astype(str) + "%"
    df['hot_80_fmt'] = df['hot_80_pct'].round(1).astype(str) + "%"
    df['hot_85_fmt'] = df['hot_85_pct'].round(1).astype(str) + "%"
    df['hot_90_fmt'] = df['hot_90_pct'].round(1).astype(str) + "%"
    
    # Calculate raw percentages for all cold combinations (math is lightweight, strings are heavy)
    cold_combos = [
        'c75_w5', 'c75_w10', 'c75_w15',
        'c70_w5', 'c70_w10', 'c70_w15',
        'c65_w5', 'c65_w10', 'c65_w15'
    ]
    for combo in cold_combos:
        df[f"{combo}_pct"] = (df[combo] / df['day_hrs']) * 100
        
    return df

try:
    df = load_and_prep_data()
except FileNotFoundError:
    st.error("Data file missing. Run analyze_matrix.py first.")
    st.stop()

col1, col2 = st.columns([1, 3])

with col1:
    st.header("Filter Criteria")
    st.write("Adjust tolerances to isolate ideal California microclimates.")
    
    # 1. Minimum Sunshine
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
    
    # 2. Heat Tolerance
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
    
    # 3. Sensible Cold Tolerance
    st.subheader("3. Sensible Cold Tolerance")
    st.write("Define your threshold for uncomfortable, breezy cold:")
    
    sub_c1, sub_c2 = st.columns(2)
    with sub_c1:
        cold_temp = st.selectbox("Temperature below:", options=["75", "70", "65"], index=1)
    with sub_c2:
        cold_wind = st.selectbox("Wind speed above:", options=["5", "10", "15"], index=1, format_func=lambda x: f"{x} mph")
    
    # Map the selections back to the engineered Parquet matrix columns
    target_cold_combo = f"c{cold_temp}_w{cold_wind}"
    cold_col = f"{target_cold_combo}_pct"
    
    min_cold_val = float(df[cold_col].min())
    max_cold_val = float(df[cold_col].max())
    cold_target = st.slider(
        f"Maximum allowed daytime hours meeting BOTH conditions:", 
        min_value=min_cold_val, 
        max_value=max_cold_val, 
        value=max_cold_val * 0.3,
        format="%.1f%%"
    )

    # Apply the composite filters
    filtered_df = df[
        (df['sun_pct'] >= sun_target) &
        (df[hot_col] <= hot_target) &
        (df[cold_col] <= cold_target)
    ].copy()
    
    st.divider()
    st.success(f"**Found {len(filtered_df):,} matching microclimates!**")
    
    result_container = st.empty()

with col2:
    # ---------------------------------------------------------
    # THE PAYLOAD DIET: 
    # Only generate the tooltip string for the active selection
    active_cold_fmt = f"{target_cold_combo}_fmt"
    if active_cold_fmt not in df.columns:
        df[active_cold_fmt] = df[cold_col].round(1).astype(str) + "%"
        filtered_df[active_cold_fmt] = filtered_df[cold_col].round(1).astype(str) + "%"

    # Define the absolute minimum columns needed for the GPU to render the map and tooltips
    render_cols = [
        'lat', 'lon', 
        'sun_pct', hot_col, cold_col, 
        'sun_fmt', f"hot_{hot_temp}_fmt", active_cold_fmt
    ]
    # ---------------------------------------------------------

    bg_layer = pdk.Layer(
        "ColumnLayer",
        # Pass ONLY the sliced columns, dodging the MessageSizeError
        data=df[render_cols], 
        id="california_base_grid", 
        get_position=["lon", "lat"],
        radius=2121, 
        disk_resolution=4,
        angle=45,
        extruded=False,
        get_fill_color=[200, 200, 200, 15],
        pickable=True,
    )

    dynamic_fg_id = f"match_grid_{sun_target}_{hot_target}_{cold_target}_{hot_temp}_{target_cold_combo}"
    
    fg_layer = pdk.Layer(
        "ColumnLayer",
        # Pass ONLY the sliced columns, dodging the MessageSizeError
        data=filtered_df[render_cols],
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
        f"<b>Over {hot_temp}°F:</b> {{hot_{hot_temp}_fmt}}<br/>"
        f"<b>Cold & Breezy (<{cold_temp}°F & >{cold_wind}mph):</b> {{{active_cold_fmt}}}"
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
        st.info(f"💨 Hours < {cold_temp}°F & > {cold_wind} mph: {clicked_point[cold_col]:.1f}%")
    else:
        st.write("Click any grid box on the map to see its exact stats.")