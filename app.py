# app.py — FINAL: Fully Interactive State-by-State Staffing Crisis Dashboard
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="U.S. Nursing Home Staffing Crisis", layout="wide")

# Epic dark design
st.markdown("""
<style>
    .main {background:#0d1117; color:white; padding:2rem; font-family:'Segoe UI',sans-serif;}
    h1, h2 {text-align:center; color:#f85149;}
    .metric {background:#161b22; padding:1.5rem; border-radius:12px; text-align:center; 
             border:1px solid #30363d; margin:10px 0;}
    .finding {background:#21262d; padding:1.2rem; border-radius:10px; margin:12px 0; 
              border-left:6px solid #f85149; font-size:1.15em;}
</style>
""", unsafe_allow_html=True)

st.title("U.S. Nursing Home Staffing Crisis")
st.markdown("### <span style='color:#58a6ff;'>Interactive State-by-State Comparison</span>", unsafe_allow_html=True)

# Load data
nh = pd.read_csv("https://raw.githubusercontent.com/RABIUL-ALAM-RATUL/medicare-hospital-spending/main/NH_ProviderInfo_Sep2025.csv",
                 usecols=['State', 'Adjusted Total Nurse Staffing Hours per Resident per Day', 
                          'Overall Rating', 'Ownership Type', 'Provider Name', 'City/Town'], 
                 low_memory=False)

nh['Staffing HPRD'] = pd.to_numeric(nh['Adjusted Total Nurse Staffing Hours per Resident per Day'], errors='coerce')
nh['Overall Rating'] = pd.to_numeric(nh['Overall Rating'], errors='coerce')
nh = nh.dropna(subset=['Staffing HPRD', 'Overall Rating'])

# State-level summary
state_summary = nh.groupby('State').agg({
    'Staffing HPRD': 'mean',
    'Overall Rating': 'mean',
    'Provider Name': 'count'
}).round(2).reset_index()
state_summary.columns = ['State', 'Avg Staffing HPRD', 'Avg Rating', 'Total Facilities']
state_summary['Below Federal Minimum'] = state_summary['Avg Staffing HPRD'] < 3.48

# Interactive filters
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("#### Controls")
    sort_by = st.selectbox("Sort States By", ["Avg Staffing HPRD", "Avg Rating", "Total Facilities"])
    show_only_below_min = st.checkbox("Show only states below federal minimum (3.48 HPRD)", value=True)
    highlight_state = st.selectbox("Highlight State", ["None"] + sorted(nh['State'].unique()))

# Filter data
display = state_summary.copy()
if show_only_below_min:
    display = display[display['Below Federal Minimum']]

display = display.sort_values(sort_by, ascending=False)

# 1. Interactive Choropleth Map
fig_map = px.choropleth(
    state_summary,
    locations='State',
    locationmode='USA-states',
    color='Avg Staffing HPRD',
    scope="usa",
    color_continuous_scale="Reds",
    title="Click a state → see details below",
    hover_data=['Avg Rating', 'Total Facilities'],
    labels={'Avg Staffing HPRD': 'Staffing Hours/Day'}
)
fig_map.update_layout(height=600, margin=dict(l=0,r=0,b=0,t=50))
selected_state = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun")

# 2. Selected State Deep Dive
if selected_state and selected_state['selection']['points']:
    state_code = selected_state['selection']['points'][0]['location']
    state_data = nh[nh['State'] == state_code]
    
    st.markdown(f"### {state_code} — In-Depth View")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Avg Staffing HPRD", f"{state_data['Staffing HPRD'].mean():.2f}")
    with col2: st.metric("Avg Rating", f"{state_data['Overall Rating'].mean():.2f}")
    with col3: st.metric("Facilities", len(state_data))
    with col4: st.metric("Below 3.48 HPRD", f"{(state_data['Staffing HPRD'] < 3.48).mean():.0%}")
    
    # Interactive scatter with trendline
    fig_scatter = px.scatter(
        state_data, x='Staffing HPRD', y='Overall Rating',
        color='Ownership Type', size='Staffing HPRD',
        hover_name='Provider Name', hover_data=['City/Town'],
        trendline="ols", trendline_color_override="yellow",
        title=f"{state_code}: Higher Staffing = Higher Quality"
    )
    fig_scatter.add_vline(x=3.48, line_dash="dash", line_color="#58a6ff")
    st.plotly_chart(fig_scatter, use_container_width=True)

# 3. National Interactive Scatter
st.markdown("### National View: Click any point for facility details")
fig_national = px.scatter(
    nh.sample(2000),  # For performance
    x='Staffing HPRD', y='Overall Rating',
    color='State', hover_name='Provider Name',
    hover_data=['City/Town', 'Ownership Type'],
    title="Every Nursing Home: Staffing vs Quality (2,000 sample)"
)
fig_national.add_vline(x=3.48, line_dash="dash", line_color="red", annotation_text="Federal Minimum")
st.plotly_chart(fig_national, use_container_width=True)

# 4. Top/Bottom 10 States Table (Interactive sort)
st.markdown("### State Rankings (Click column to sort)")
display_styled = display.style.background_gradient(cmap='Reds', subset=['Avg Staffing HPRD'])\
                             .format({'Avg Staffing HPRD': '{:.2f}', 'Avg Rating': '{:.2f}'})
st.dataframe(display_styled, use_container_width=True)

# Final verdict
st.markdown("""
<div style='text-align:center; padding:50px; background:#161b22; border-radius:20px; margin-top:50px; border:3px solid #f85149;'>
<h1>Every state fails.</h1>
<h2 style='color:#f85149;'>Some just fail less slowly.</h2>
</div>
""", unsafe_allow_html=True)

st.caption("Fully Interactive U.S. Nursing Home Staffing Crisis • Click • Filter • Explore • September 2025 CMS Data")
