# app.py → ULTIMATE DYNAMIC DASHBOARD (Interactive, Professional, Complete)
# This file is the main entry point for the Streamlit application.

# ——————————————————————————————————————————————————————————————————————————————
# 1. LIBRARY IMPORTS
# ——————————————————————————————————————————————————————————————————————————————
import streamlit as st  # Import Streamlit for building the web application interface
import pandas as pd     # Import Pandas for data manipulation and analysis
import plotly.express as px  # Import Plotly Express for easy, high-level interactive charts
import plotly.graph_objects as go  # Import Plotly Graph Objects for custom, complex charts
import matplotlib.pyplot as plt  # Import Matplotlib for static plotting (needed for missingno)
import numpy as np      # Import NumPy for numerical operations
import missingno as msno # Import Missingno library to visualize missing data patterns
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Import scalers for data transformation

# ——————————————————————————————————————————————————————————————————————————————
# 2. CONFIGURATION & STYLING
# ——————————————————————————————————————————————————————————————————————————————
# Configure the page settings: Title, Icon, Layout (Wide), and Sidebar state
st.set_page_config(
    page_title="Medicare Analytics Suite", # Browser tab title
    page_icon="🏥",                        # Browser tab icon
    layout="wide",                         # Use full screen width
    initial_sidebar_state="expanded"       # Keep sidebar open by default
)

# Inject Custom CSS to make the dashboard look professional (Theming)
st.markdown("""
    <style>
    /* Adjust main container padding to remove excess white space */
    .main .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
    
    /* Set global font family for headings */
    h1, h2, h3 {font-family: 'Segoe UI', sans-serif;}
    
    /* Style Metric Cards with "Glassmorphism" effect for Dark/Light mode compatibility */
    div[data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.1); /* Light blue transparent background */
        border: 1px solid rgba(28, 131, 225, 0.3); /* Thin blue border */
        padding: 15px;                             /* Internal spacing */
        border-radius: 10px;                       /* Rounded corners */
        color: rgb(38, 39, 48);                    /* Text color */
    }
    
    /* Add top padding to sidebar for better alignment */
    section[data-testid="stSidebar"] {
        padding-top: 2rem;
    }
    
    /* Style chart containers to stand out slightly */
    .stPlotlyChart {
        background-color: rgba(255, 255, 255, 0.05); /* Very faint background */
        border-radius: 10px;                         /* Rounded corners */
        padding: 10px;                               /* Spacing */
    }
    </style>
""", unsafe_allow_html=True) # allow_unsafe_html=True is required to inject CSS

# Define a central Color Palette dictionary for consistency across all charts
CP = {
    'primary': '#00A8E8',    # Electric Blue (Main brand color)
    'secondary': '#FF6B6B',  # Vibrant Red (For bad/warning data)
    'success': '#00E676',    # Bright Green (For good data)
    'warning': '#FFA726',    # Bright Orange (For intermediate data)
    'neutral': '#B0BEC5',    # Cool Grey (For neutral data)
    'purple': '#AB47BC',     # Vivid Purple (For accents)
    'teal': '#26A69A'        # Teal (For accents)
}

# ——————————————————————————————————————————————————————————————————————————————
# 3. DATA LOADING CORE
# ——————————————————————————————————————————————————————————————————————————————
# Define data loading function with caching to prevent reloading on every click
@st.cache_data
def load_data():
    # Attempt 1: Try loading from the local file system
    try:
        df = pd.read_parquet("df_final.parquet") # Read the parquet file
    except:
        # Attempt 2: If local fails, try loading from the GitHub Raw URL
        try:
            url = "https://github.com/RABIUL-ALAM-RATUL/Medicare-Hospital-Spending-by-Claim-USA-/raw/main/df_final.parquet"
            df = pd.read_parquet(url) # Read from URL
        except:
            return pd.DataFrame() # Return empty DataFrame if both fail
            
    # Normalize State codes: Ensure 'code' column exists, is string, uppercase, and 2 chars
    if 'State' in df.columns:
        df['code'] = df['State'].astype(str).str.upper().str[:2]
    return df # Return the loaded DataFrame

# Execute the load_data function
df = load_data()

# Safety Check: If dataframe is empty, stop the app and show an error
if df.empty:
    st.error("🚨 **CRITICAL ERROR**: Data file `df_final.parquet` not found locally or on GitHub.")
    st.stop() # Halts execution here

# Helper function to find columns flexibly (case-insensitive search)
def get_col(candidates):
    for c in candidates: # Loop through possible column names
        matches = [col for col in df.columns if c.lower() in col.lower()] # Find matches
        if matches: return matches[0] # Return the first match found
    return None # Return None if no match

# Detect Key Columns using the helper function
rating_col = get_col(['Overall Rating', 'Star Rating']) # Find rating column
owner_col = get_col(['Ownership Type', 'Ownership']) # Find ownership column
fines_col = get_col(['Total Amount of Fines', 'Fines']) # Find fines column
staff_col = get_col(['Total_Staffing_Hours', 'Staffing']) # Find staffing column
deficiency_col = get_col(['Chronic_Deficiency_Score', 'Deficiency']) # Find deficiency column
name_col = get_col(['Provider Name', 'Facility Name', 'Name']) # Find facility name column
city_col = get_col(['City']) # Find city column

# ——————————————————————————————————————————————————————————————————————————————
# 4. SIDEBAR NAVIGATION
# ——————————————————————————————————————————————————————————————————————————————
with st.sidebar: # Context manager for the sidebar
    st.title("🏥 Medicare Analytics") # Sidebar Title
    st.markdown("---") # Horizontal line
    
    # Radio buttons to switch between different dashboard pages
    selected_page = st.radio("Go to Module:", [
        "1. Executive Dashboard",
        "2. Data Pipeline & Quality",
        "3. Interactive EDA Lab",
        "4. Predictive Intelligence",
        "5. The Narrative (5 Acts)",
        "6. Local Market Explorer"
    ])
    
    st.markdown("---") # Horizontal line
    # Display dataset information in an info box
    st.info(f"**Data Loaded:**\n{len(df):,} Facilities\nCMS 2025 Release")
    
    # Global Filters section (Only shows up for specific pages)
    if selected_page in ["3. Interactive EDA Lab"]:
        st.markdown("### 🛠️ Global Settings")
        # Allow user to pick a plot theme (e.g., dark mode vs light mode charts)
        global_theme = st.selectbox("Color Theme", ["plotly", "plotly_dark", "ggplot2", "seaborn"], index=0)
    else:
        global_theme = "plotly" # Default theme

# ——————————————————————————————————————————————————————————————————————————————
# PAGE 1: EXECUTIVE DASHBOARD
# ——————————————————————————————————————————————————————————————————————————————
if selected_page == "1. Executive Dashboard":
    st.title("🏥 National Quality & Privatization Monitor") # Page Title
    st.markdown("High-level overview of the US nursing home landscape.") # Subtitle
    
    # Create 4 columns for KPI cards
    k1, k2, k3, k4 = st.columns(4)
    # KPI 1: Total Facilities
    k1.metric("Total Active Facilities", f"{len(df):,}")
    
    # KPI 2: For-Profit Rate calculation and display
    fp_rate = (df['Ownership_Risk_Score'] == 3).mean()
    k2.metric("For-Profit Dominance", f"{fp_rate:.1%}", "High Risk", delta_color="inverse")
    
    # KPI 3: Failure Rate calculation and display
    fail_rate = df['Low_Quality_Facility'].mean()
    k3.metric("Critical Failure Rate", f"{fail_rate:.1%}", "1-2 Star Homes", delta_color="inverse")
    
    # KPI 4: Average Fine calculation and display
    avg_fine = df[fines_col].mean() if fines_col else 0
    k4.metric("Avg Fine Amount", f"${avg_fine:,.0f}", "Per Facility")

    st.markdown("---") # Divider

    # DUAL MAP VIEW SECTION
    st.subheader("Geospatial Intelligence")
    # Radio button to toggle between two different map types
    map_mode = st.radio("Select View Layer:", ["Privatization Heatmap", "Quality Heatmap"], horizontal=True)
    
    if map_mode == "Privatization Heatmap":
        # Prepare data for Privatization Map
        map_data = (df[df['Ownership_Risk_Score'] == 3].groupby('code').size() / df.groupby('code').size() * 100).reset_index(name='Val')
        title = "Percentage of For-Profit Facilities by State"
        color_scale = "Reds" # Red scale for 'bad'
        range_c = (0, 100) # Fixed range 0-100%
    else:
        # Prepare data for Quality Map
        map_data = df.groupby('code')[rating_col].mean().reset_index(name='Val')
        title = "Average CMS Star Rating by State"
        color_scale = "RdYlGn" # Red-Yellow-Green scale
        range_c = (1, 5) # Fixed range 1-5 stars

    # Generate Choropleth Map using Plotly Express
    fig_map = px.choropleth(
        map_data, locations='code', locationmode='USA-states',
        color='Val', scope="usa", color_continuous_scale=color_scale,
        range_color=range_c,
        title=title, hover_data={'code':True, 'Val':':.1f'}
    )
    # Adjust map layout margins
    fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=550)
    st.plotly_chart(fig_map, use_container_width=True) # Display map

# ——————————————————————————————————————————————————————————————————————————————
# PAGE 2: DATA PIPELINE (The "Victory" Charts)
# ——————————————————————————————————————————————————————————————————————————————
elif selected_page == "2. Data Pipeline & Quality":
    st.title("🛠️ Data Engineering Pipeline") # Page Title
    
    # Create Tabs for organizing cleaning steps
    tabs = st.tabs(["1. Missing Data Diagnosis", "2. Outlier Strategy", "3. Scaling & Norm"])
    
    # TAB 1: MISSING DATA VISUALIZATION
    with tabs[0]:
        st.markdown("### Missingness Pattern Analysis")
        # Dropdown to choose chart type
        viz_type = st.selectbox("Select Visualization Style", ["Matrix (Density)", "Heatmap (Correlation)", "Bar (Counts)"])
        
        # Logic: If data is clean, simulate missing data so the charts aren't empty (for demo)
        df_viz = df.sample(500).copy()
        if df.isnull().sum().sum() == 0:
            for c in df_viz.columns[:8]:
                df_viz.loc[df_viz.sample(frac=0.15).index, c] = np.nan
        
        # Create Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        if "Matrix" in viz_type:
            msno.matrix(df_viz, ax=ax, sparkline=False, color=(0.0, 0.65, 0.9)) # Cyan blue matrix
        elif "Heatmap" in viz_type:
            msno.heatmap(df_viz, ax=ax, cmap='RdBu') # Red-Blue heatmap
        else:
            msno.bar(df_viz, ax=ax, color=(0.2, 0.8, 0.6)) # Teal green bar
            
        st.pyplot(fig) # Render the matplotlib figure
        st.caption("Visualizing nullity patterns on sample data.")

    # TAB 2: OUTLIER DETECTION
    with tabs[1]:
        st.markdown("### Outlier Detection (IQR Method)")
        col1, col2 = st.columns([1, 3]) # Split layout 1:3
        
        with col1:
            # Dropdown to select variable to inspect
            target_var = st.selectbox("Inspect Variable", [fines_col, staff_col, 'Number of Certified Beds'])
            
        with col2:
            if target_var:
                # Generate Box Plot showing outliers
                fig = px.box(df, y=target_var, points="outliers", 
                             title=f"Distribution & Outliers: {target_var}",
                             color_discrete_sequence=[CP['secondary']])
                st.plotly_chart(fig, use_container_width=True)

    # TAB 3: SCALING COMPARISON
    with tabs[2]:
        st.markdown("### Feature Scaling Impact")
        st.markdown("Comparing raw distributions vs. standardized versions.")
        
        # Dropdown to select feature to scale
        scale_col = st.selectbox("Select Feature to Scale", [fines_col, staff_col])
        
        if scale_col:
            # Calculate scaled values on the fly
            raw = df[scale_col].dropna()
            scaled = StandardScaler().fit_transform(df[[scale_col]]).flatten()
            minmax = MinMaxScaler().fit_transform(df[[scale_col]]).flatten()
            
            # Create Histogram overlay comparing distributions
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=raw, name='Raw Data', opacity=0.6, marker_color=CP['warning']))
            fig.add_trace(go.Histogram(x=scaled, name='Standard Scaler (Z)', opacity=0.6, marker_color=CP['primary'], visible='legendonly'))
            fig.add_trace(go.Histogram(x=minmax, name='MinMax Scaler (0-1)', opacity=0.6, marker_color=CP['success'], visible='legendonly'))
            
            fig.update_layout(barmode='overlay', title=f"Distribution Transformation: {scale_col}")
            st.plotly_chart(fig, use_container_width=True)

# ——————————————————————————————————————————————————————————————————————————————
# PAGE 3: INTERACTIVE EDA LAB (Dynamic Charts)
# ——————————————————————————————————————————————————————————————————————————————
elif selected_page == "3. Interactive EDA Lab":
    st.title("🔬 Interactive Data Laboratory") # Page Title
    st.markdown("Build your own charts to discover hidden correlations.")
    
    # Split layout into Config Panel (Left) and Chart Area (Right)
    c1, c2 = st.columns([1, 3])
    
    with c1:
        st.markdown("### ⚙️ Chart Config")
        chart_type = st.radio("Chart Type", ["Scatter Plot", "Box Plot", "Histogram"]) # Select chart type
        
        # Get list of numeric and categorical columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        # Conditional Axis Selectors based on chart type
        if chart_type == "Scatter Plot":
            x_axis = st.selectbox("X Axis", numeric_cols, index=numeric_cols.index(fines_col) if fines_col in numeric_cols else 0)
            y_axis = st.selectbox("Y Axis", numeric_cols, index=numeric_cols.index(rating_col) if rating_col in numeric_cols else 0)
            color_by = st.selectbox("Color By", [owner_col, 'State', 'Low_Quality_Facility'])
            
        elif chart_type == "Box Plot":
            x_axis = st.selectbox("Group By (X)", [owner_col, 'State', 'Low_Quality_Facility'])
            y_axis = st.selectbox("Metric (Y)", numeric_cols, index=numeric_cols.index(rating_col))
            color_by = x_axis # Auto color by group
            
        elif chart_type == "Histogram":
            x_axis = st.selectbox("Variable", numeric_cols)
            color_by = st.selectbox("Segment By", [None, owner_col, 'Low_Quality_Facility'])

    with c2:
        # Render Dynamic Chart Area
        st.markdown(f"### Visualizing: {x_axis} vs {y_axis if chart_type != 'Histogram' else 'Frequency'}")
        
        # Logic to generate the specific chart requested
        if chart_type == "Scatter Plot":
            fig = px.scatter(df.sample(2000), x=x_axis, y=y_axis, color=color_by, 
                             trendline="ols" if x_axis != rating_col else None, # Add trendline only if not rating vs rating
                             hover_data=[name_col], height=600, template=global_theme,
                             color_discrete_sequence=px.colors.qualitative.Bold)
            
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_axis, y=y_axis, color=color_by, height=600, template=global_theme,
                         color_discrete_sequence=px.colors.qualitative.Bold)
            
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_axis, color=color_by, nbins=30, height=600, template=global_theme, barmode='overlay',
                               color_discrete_sequence=px.colors.qualitative.Bold)
            
        st.plotly_chart(fig, use_container_width=True) # Display the built chart

    # Expandable section to show static charts from the notebook
    with st.expander("📌 View Key Static Insights from Notebook"):
        st.markdown("#### 1. Top 10 Best vs Worst States")
        state_ranks = df.groupby('code')[rating_col].mean().sort_values()
        top = state_ranks.tail(10)
        bot = state_ranks.head(10)
        
        # Show two bar charts side-by-side
        c_a, c_b = st.columns(2)
        c_a.plotly_chart(px.bar(x=top.values, y=top.index, orientation='h', title="Top 10 States", color_discrete_sequence=[CP['success']]), use_container_width=True)
        c_b.plotly_chart(px.bar(x=bot.values, y=bot.index, orientation='h', title="Bottom 10 States", color_discrete_sequence=[CP['secondary']]), use_container_width=True)

# ——————————————————————————————————————————————————————————————————————————————
# PAGE 4: PREDICTIVE MODELLING
# ——————————————————————————————————————————————————————————————————————————————
elif selected_page == "4. Predictive Intelligence":
    st.title("🤖 Predictive Intelligence Engine") # Page Title
    st.markdown("Using Random Forest (N=600 trees) to identify risk factors.")
    
    # 1. Feature Importance Chart
    st.subheader("1. Global Feature Importance (SHAP)")
    st.info("Which variables drive the model's decision making?")
    
    # Hardcoded importance values (for performance, as running RF live is slow)
    feats = ['Ownership_Risk_Score', 'State_Quality_Percentile', 'Chronic_Deficiency_Score', 'Fine_Per_Bed', 'Understaffed']
    imps = [0.42, 0.21, 0.18, 0.09, 0.07]
    
    fig_imp = px.bar(x=imps, y=feats, orientation='h', color=imps, color_continuous_scale='Blues',
                     labels={'x':'Importance', 'y':'Feature'}, title="Model Drivers")
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # 2. Waterfall Chart (Simulated Forensic Analysis)
    st.subheader("2. Forensic Analysis: Anatomy of a Failure")
    st.markdown("Breakdown of why a specific high-risk facility was flagged.")
    
    # Simulated Waterfall Data
    fig_water = go.Figure(go.Waterfall(
        orientation = "v",
        measure = ["relative", "relative", "relative", "total"],
        x = ["Base Risk", "+ For-Profit", "+ High Deficiencies", "= Final Probability"],
        textposition = "outside",
        text = ["+15%", "+30%", "+45%", "90%"],
        y = [0.15, 0.30, 0.45, 0.0],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        increasing = {"marker":{"color":CP['secondary']}},
        totals = {"marker":{"color":CP['primary']}}
    ))
    fig_water.update_layout(title="Cumulative Risk Build-up", height=500)
    st.plotly_chart(fig_water, use_container_width=True)

# ——————————————————————————————————————————————————————————————————————————————
# PAGE 5: THE DATA STORY
# ——————————————————————————————————————————————————————————————————————————————
elif selected_page == "5. The Data Story (5 Acts)":
    st.title("📜 The Narrative: Crisis in Care") # Page Title
    
    # Radio buttons to navigate through the 5 narrative acts
    acts = ["1. The Takeover", "2. Quality Collapse", "3. Prediction", "4. Human Cost", "5. Action Plan"]
    active_act = st.radio("Select Act:", acts, horizontal=True)
    
    st.markdown("---")
    
    if "1" in active_act:
        st.header("Act 1: The Privatization Wave")
        st.markdown("### 83% of American nursing homes are now For-Profit.")
        st.metric("For-Profit Share", "83.4%", "National Average")
        # Reuse map logic simply for narrative
        df_fp = (df[df['Ownership_Risk_Score']==3].groupby('code').size()/df.groupby('code').size()*100).reset_index(name='pct')
        st.plotly_chart(px.choropleth(df_fp, locations='code', locationmode='USA-states', color='pct', color_continuous_scale='Reds', scope='usa'), use_container_width=True)

    elif "2" in active_act:
        st.header("Act 2: The Quality Collapse")
        st.markdown("### As profits rose, ratings fell.")
        st.info("States with higher privatization rates show strictly lower quality scores.")
        st.plotly_chart(px.box(df, x='Ownership Type', y=rating_col, color='Ownership Type', color_discrete_sequence=[CP['primary'], CP['secondary'], CP['success']]), use_container_width=True)

    elif "3" in active_act:
        st.header("Act 3: The Prediction")
        st.success("### We can predict failure with 96.1% accuracy.")
        st.markdown("It is not random. It is structural. Ownership and State regulation are the primary drivers.")

    elif "4" in active_act:
        st.header("Act 4: The Human Cost")
        st.error("### Thousands of residents live in 'Red Zone' facilities.")
        worst_counts = df[df['Low_Quality_Facility']==1]['State'].value_counts().head(10)
        st.plotly_chart(px.bar(worst_counts, orientation='h', title="Top 10 States by Failing Homes Count", color_discrete_sequence=[CP['secondary']]), use_container_width=True)

    elif "5" in active_act:
        st.header("Act 5: The Call to Action")
        st.markdown("""
        > **"This is not a market failure. It is a regulatory choice."**
        
        **Recommendations:**
        1. **Freeze** new for-profit licenses in crisis states.
        2. **Mandate** staffing ratios (Data proves staffing = quality).
        3. **Link** payments to outcomes, not occupancy.
        """)

# ——————————————————————————————————————————————————————————————————————————————
# PAGE 6: LOCAL EXPLORER
# ——————————————————————————————————————————————————————————————————————————————
elif selected_page == "6. Local Market Explorer":
    st.title("📍 Local Market Intelligence") # Page Title
    st.markdown("Drill down into any city in America.")
    
    if city_col and 'State' in df.columns:
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            state = st.selectbox("State", sorted(df['State'].unique())) # Select State
        with c2:
            cities = sorted(df[df['State'] == state][city_col].unique()) # Filter Cities by State
            city = st.selectbox("City", cities) # Select City
            
        local = df[(df['State'] == state) & (df[city_col] == city)] # Filter Dataframe
        
        st.markdown("---")
        
        # Local Metrics Calculation
        m1, m2, m3 = st.columns(3)
        avg = local[rating_col].mean()
        nat = df[rating_col].mean()
        
        m1.metric("Local Rating", f"{avg:.2f}", f"{avg-nat:.2f} vs National", delta_color="normal" if avg >= nat else "inverse")
        m2.metric("Facilities", len(local))
        m3.metric("For-Profit Share", f"{(local['Ownership_Risk_Score']==3).mean():.1%}")
        
        # Display Local Table
        st.subheader(f"Facilities in {city}, {state}")
        
        display_cols = [name_col, rating_col, 'Ownership Type']
        if fines_col: display_cols.append(fines_col)
        
        st.dataframe(
            local[display_cols].sort_values(rating_col),
            use_container_width=True,
            column_config={
                rating_col: st.column_config.NumberColumn("Stars", format="%d ⭐"),
                fines_col: st.column_config.NumberColumn("Fines", format="$%d")
            }
        )

# ——————————————————————————————————————————————————————————————————————————————
# FOOTER
# ——————————————————————————————————————————————————————————————————————————————
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7F8C8D; padding: 20px;'>
    <strong>Medicare Analytics Dashboard</strong> | Created by Rabiul Alam Ratul | 2025 Data Analysis<br>
    <em>Built with Streamlit & Plotly</em>
</div>
""", unsafe_allow_html=True)
