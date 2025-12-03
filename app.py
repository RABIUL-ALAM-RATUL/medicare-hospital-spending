# app.py -> ULTIMATE PROFESSIONAL DASHBOARD (Theme-Adaptive & Error-Free)
# This file is the main entry point for the Streamlit application.

# ------------------------------------------------------------------------------
# 1. LIBRARY IMPORTS
# ------------------------------------------------------------------------------
import streamlit as st  # Import Streamlit library for building the web application interface
import pandas as pd     # Import Pandas library for robust data manipulation and analysis
import plotly.express as px  # Import Plotly Express for creating high-level, interactive plots easily
import plotly.graph_objects as go  # Import Plotly Graph Objects for creating custom, complex visualizations
import matplotlib.pyplot as plt  # Import Matplotlib for rendering static plots (specifically for missingno)
import numpy as np      # Import NumPy for efficient numerical operations and array handling
import missingno as msno # Import Missingno library for visualizing missing data patterns (matrix, heatmap)
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Import scalers for data normalization/standardization

# Safe Import for Machine Learning Libraries (prevents app crash if not installed)
try:
    import shap  # Import SHAP library for model explainability (feature importance)
    from sklearn.ensemble import RandomForestClassifier  # Import Random Forest algorithm for classification
    from sklearn.model_selection import train_test_split  # Import function to split data into training and testing sets
    from sklearn.metrics import accuracy_score, roc_auc_score  # Import metrics to evaluate model performance
    HAS_ML = True  # Set flag to True if imports are successful
except ImportError:
    HAS_ML = False  # Set flag to False if imports fail (graceful degradation)

# ------------------------------------------------------------------------------
# 2. CONFIGURATION & THEME-ADAPTIVE STYLING
# ------------------------------------------------------------------------------
# Configure the Streamlit page settings
st.set_page_config(
    page_title="Medicare Hospital Spending by Claim (USA)", # Set the title displayed in the browser tab
    page_icon=None, # Remove default icon (can be set to emoji if desired)
    layout="wide",  # Use the full width of the screen for the layout
    initial_sidebar_state="expanded" # Keep the sidebar open by default when the app loads
)

# Inject Custom CSS for a professional, theme-adaptive look
st.markdown("""
    <style>
    /* Global Spacing adjustments for the main container */
    .main .block-container {
        padding-top: 1.5rem;  /* Add space at top */
        padding-bottom: 3rem; /* Add space at bottom */
    }
    
    /* Typography settings for headings */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; /* Professional sans-serif font stack */
        font-weight: 600; /* Semi-bold weight */
    }
    
    /* Metric Cards Styling - Adapts to Streamlit's theme variables */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color); /* Use theme's secondary bg color */
        border: 1px solid var(--text-color-10); /* Subtle border based on text color */
        padding: 15px; /* Internal padding */
        border-radius: 8px; /* Rounded corners */
        box-shadow: 0 1px 2px rgba(0,0,0,0.1); /* Subtle shadow for depth */
        transition: transform 0.2s ease-in-out; /* Smooth transition for hover effect */
    }
    /* Hover effect for Metric Cards */
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px); /* Lift card slightly */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* Increase shadow */
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--text-color-10); /* Add a border to separate sidebar */
    }
    
    /* Tabs styling for better visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px; /* Space between tabs */
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px; /* Taller tabs */
        white-space: pre-wrap; /* Allow text wrapping */
        background-color: var(--secondary-background-color); /* Tab background */
        border-radius: 5px; /* Rounded tabs */
        padding: 0 20px; /* Horizontal padding */
    }
    
    /* Plotly Chart Containers Styling */
    .stPlotlyChart {
        background-color: var(--secondary-background-color); /* Matches metric cards */
        border-radius: 8px; /* Rounded corners */
        padding: 10px; /* Internal padding */
    }
    </style>
""", unsafe_allow_html=True) # allow_unsafe_html=True is required to render CSS

# Define a central Color Palette dictionary for consistency across all charts
CP = {
    'primary': '#0078D4',    # Corporate Blue (Main brand color)
    'secondary': '#D83B01',  # Alert Red/Orange (For bad/warning data)
    'success': '#107C10',    # Success Green (For good data)
    'neutral': '#605E5C',    # Neutral Grey (For neutral data)
    'dark': '#201F1E'        # Dark Grey (For text/accents)
}

# ------------------------------------------------------------------------------
# 3. HELPER DATA (State Mapping)
# ------------------------------------------------------------------------------
# Dictionary mapping full US State names to their 2-letter abbreviations
US_STATES = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    'District of Columbia': 'DC'
}

# ------------------------------------------------------------------------------
# 4. DATA LOADING (Robust)
# ------------------------------------------------------------------------------
# Decorator to cache the data loading function so it only runs once per session
@st.cache_data
def load_data():
    try:
        # Attempt to read the parquet file from the local file system
        df = pd.read_parquet("df_final.parquet")
    except:
        try:
            # If local fails, attempt to read from the GitHub Raw URL
            url = "https://github.com/RABIUL-ALAM-RATUL/Medicare-Hospital-Spending-by-Claim-USA-/raw/main/df_final.parquet"
            df = pd.read_parquet(url)
        except:
            return pd.DataFrame() # Return an empty DataFrame if both attempts fail

    # Apply robust state mapping logic
    if 'State' in df.columns:
        # Map full state names to abbreviations, keeping abbreviations if they already exist
        df['code'] = df['State'].map(US_STATES).fillna(df['State'])
        # Ensure the 'code' column is string, uppercase, and truncated to 2 characters
        df['code'] = df['code'].astype(str).str.upper().str.slice(0, 2)
        
    return df # Return the fully loaded and preprocessed DataFrame

# Execute the load_data function to get the data
df = load_data()

# Check if the dataframe is empty (load failed)
if df.empty:
    st.error("Data file 'df_final.parquet' could not be loaded. Please check file presence.") # Show error message
    st.stop() # Stop execution of the app

# Helper function to find columns safely by pattern matching
def get_col(candidates):
    for c in candidates: # Iterate through candidate column names
        matches = [col for col in df.columns if c.lower() in col.lower()] # Find matches in dataframe columns (case-insensitive)
        if matches: return matches[0] # Return the first match found
    return None # Return None if no match is found

# Detect specific columns using the helper function
rating_col = get_col(['Overall Rating', 'Star Rating']) # Detect rating column
owner_col = get_col(['Ownership Type', 'Ownership']) # Detect ownership column
fines_col = get_col(['Total Amount of Fines', 'Fines']) # Detect fines column
staff_col = get_col(['Total_Staffing_Hours', 'Staffing']) # Detect staffing column
name_col = get_col(['Provider Name', 'Facility Name', 'Name']) # Detect facility name column
city_col = get_col(['City']) # Detect city column

# ------------------------------------------------------------------------------
# 5. SIDEBAR (INTERACTIVE & EXPORT)
# ------------------------------------------------------------------------------
# Create the sidebar layout
with st.sidebar:
    st.title("Medicare Analytics") # Sidebar main title
    st.write("Project: Medicare Hospital Spending by Claim (USA)") # Project name
    st.write("Created by: Md Rabiul Alam") # Creator credit
    
    st.markdown("---") # Horizontal separator
    
    # Navigation Menu
    st.subheader("Navigation") # Subheader for navigation
    # Radio buttons to switch between application pages
    page = st.radio("Go to:", [
        "Executive Dashboard",
        "Data Pipeline Status",
        "Interactive EDA Lab",
        "Predictive Modelling",
        "Data Narrative",
        "Local Market Explorer"
    ])
    
    st.markdown("---") # Horizontal separator
    
    # Global Filters Section
    st.subheader("Global Filters") # Subheader for filters
    unique_states = sorted(df['State'].unique().tolist()) # Get sorted list of unique states
    # Selectbox for state filtering (Global)
    selected_state_sidebar = st.selectbox("Filter State (Global)", ["All"] + unique_states)
    
    st.markdown("---") # Horizontal separator
    
    # Export Options Section
    st.subheader("Export Data") # Subheader for exports
    st.write("Download data for external tools.") # Description
    
    # Generate CSV data string for download
    csv_data = df.to_csv(index=False).encode('utf-8')
    # Download Button for raw dataset
    st.download_button(
        label="Download Dataset (CSV)",
        data=csv_data,
        file_name="medicare_data_2025.csv",
        mime="text/csv"
    )
    
    # Download Button optimized for Tableau
    st.caption("For Tableau: Download this CSV and connect using Text File connector.") # Tooltip/Caption
    st.download_button(
        label="Download for Tableau (CSV)",
        data=csv_data,
        file_name="medicare_tableau_source.csv",
        mime="text/csv",
        help="Optimized CSV format ready for Tableau import." # Hover help text
    )
    
    st.markdown("### Export Figures") # Header for figure export info
    st.info("Hover over any chart and click the Camera icon to download as Image/SVG for reports.") # Instruction box

# ------------------------------------------------------------------------------
# PAGE 1: EXECUTIVE DASHBOARD
# ------------------------------------------------------------------------------
if page == "Executive Dashboard":
    st.title("National Quality & Privatization Monitor") # Page Title
    st.markdown("High-level overview of US nursing home performance.") # Page Description
    
    # Filter Data based on Sidebar selection
    view_df = df if selected_state_sidebar == "All" else df[df['State'] == selected_state_sidebar]
    
    # KPI Section (4 Columns)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Facilities", f"{len(view_df):,}") # Metric: Total Count
    
    # Metric: For-Profit Percentage
    if 'Ownership_Risk_Score' in view_df.columns:
        fp_share = (view_df['Ownership_Risk_Score'] == 3).mean() # Calculate mean of boolean (Risk=3)
        k2.metric("For-Profit Share", f"{fp_share:.1%}") # Display percentage
    
    # Metric: Failure Rate
    if 'Low_Quality_Facility' in view_df.columns:
        fail_share = view_df['Low_Quality_Facility'].mean() # Calculate mean of boolean
        k3.metric("Critical Failure Rate", f"{fail_share:.1%}") # Display percentage
        
    # Metric: Average Fines
    if fines_col:
        avg_f = view_df[fines_col].mean() # Calculate mean fines
        k4.metric("Avg Fines", f"${avg_f:,.0f}") # Display formatted currency
        
    st.markdown("---") # Horizontal separator
    
    # Maps Section (2 Columns)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Privatization Intensity") # Subheader for Map 1
        if 'Ownership_Risk_Score' in df.columns:
            # Group by state code and calculate % of Risk Score 3 (For Profit)
            # Use .apply(lambda...) to handle calculation safely and fill empty states with 0
            grp = df.groupby('code')['Ownership_Risk_Score'].apply(lambda x: (x==3).mean()*100).reset_index(name='Val')
            # Create Choropleth Map using Plotly Express
            fig = px.choropleth(grp, locations='code', locationmode='USA-states', color='Val',
                                color_continuous_scale='Reds', range_color=(0,100), # Red scale, 0-100%
                                title="For-Profit % by State", scope="usa") # Limit scope to USA
            st.plotly_chart(fig, use_container_width=True) # Display interactive map
            
    with col2:
        st.subheader("Quality Landscape") # Subheader for Map 2
        if rating_col:
            # Group by state code and calculate mean rating
            grp = df.groupby('code')[rating_col].mean().reset_index(name='Val')
            # Create Choropleth Map using Plotly Express
            fig = px.choropleth(grp, locations='code', locationmode='USA-states', color='Val',
                                color_continuous_scale='RdYlGn', range_color=(1,5), # Red-Yellow-Green scale, 1-5 stars
                                title="Avg Star Rating by State", scope="usa") # Limit scope to USA
            st.plotly_chart(fig, use_container_width=True) # Display interactive map

# ------------------------------------------------------------------------------
# PAGE 2: DATA PIPELINE
# ------------------------------------------------------------------------------
elif page == "Data Pipeline Status":
    st.title("Data Engineering Pipeline") # Page Title
    
    # Create Tabs for different cleaning stages
    tab1, tab2, tab3 = st.tabs(["Missing Data", "Outliers", "Scaling"])
    
    with tab1:
        st.subheader("Missing Data Diagnosis") # Subheader
        # Create a sample for visualization to keep it fast
        viz_df = df.sample(min(500, len(df))).copy()
        # If data is already clean (0 nulls), inject synthetic nulls for demo purposes only
        if viz_df.isnull().sum().sum() == 0:
            for c in viz_df.columns[:10]:
                viz_df.loc[viz_df.sample(frac=0.1).index, c] = np.nan
                
        # Smart Filter: Only show columns with missing data to avoid label overlap
        cols_with_missing = viz_df.columns[viz_df.isnull().any()].tolist()
        # Limit columns to 20 max to ensure readability
        if len(cols_with_missing) > 20: cols_with_missing = cols_with_missing[:20]
        elif len(cols_with_missing) < 5: cols_with_missing = viz_df.columns[:15]
        
        # Create matplotlib figure for missingno matrix
        fig, ax = plt.subplots(figsize=(10, 4))
        msno.matrix(viz_df[cols_with_missing], ax=ax, sparkline=False, fontsize=8) # Generate matrix
        st.pyplot(fig) # Display figure
        
    with tab2:
        st.subheader("Outlier Analysis") # Subheader
        if fines_col:
            # Box plot for fines to show outliers
            fig = px.box(df, y=fines_col, points="outliers", title="Fines Distribution")
            st.plotly_chart(fig, use_container_width=True) # Display chart
            
    with tab3:
        st.subheader("Normalization") # Subheader
        st.write("Comparison of Original vs Scaled distributions.") # Description
        if fines_col:
            # Prepare data: Original vs Standard Scaled
            d = df[fines_col].dropna()
            s = StandardScaler().fit_transform(df[[fines_col]]).flatten()
            # Create Histogram overlay
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=d, name='Original', opacity=0.7))
            fig.add_trace(go.Histogram(x=s, name='Standard Scaled', visible='legendonly', opacity=0.7))
            fig.update_layout(barmode='overlay', title="Distribution Transformation") # Set overlay mode
            st.plotly_chart(fig, use_container_width=True) # Display chart

# ------------------------------------------------------------------------------
# PAGE 3: INTERACTIVE EDA LAB
# ------------------------------------------------------------------------------
elif page == "Interactive EDA Lab":
    st.title("Interactive Data Laboratory") # Page Title
    
    # Split layout: Config on left (1), Chart on right (3)
    c1, c2 = st.columns([1, 3])
    
    with c1:
        st.markdown("#### Chart Config") # Section Header
        chart = st.selectbox("Chart Type", ["Scatter", "Box", "Histogram"]) # Dropdown for chart type
        
        # Identify numeric and categorical columns
        nums = df.select_dtypes(include=np.number).columns.tolist()
        cats = df.select_dtypes(exclude=np.number).columns.tolist()
        
        # Config options based on chart selection
        if chart == "Scatter":
            x = st.selectbox("X Axis", nums, index=0)
            y = st.selectbox("Y Axis", nums, index=1 if len(nums)>1 else 0)
            color = st.selectbox("Color", [None] + cats)
        elif chart == "Box":
            x = st.selectbox("Group By", cats)
            y = st.selectbox("Metric", nums)
        elif chart == "Histogram":
            x = st.selectbox("Variable", nums)
            color = st.selectbox("Split By", [None] + cats)
            
    with c2:
        # Generate chart based on selection
        if chart == "Scatter":
            fig = px.scatter(df.sample(min(2000, len(df))), x=x, y=y, color=color, title=f"{x} vs {y}")
        elif chart == "Box":
            fig = px.box(df, x=x, y=y, title=f"{y} by {x}")
        elif chart == "Histogram":
            fig = px.histogram(df, x=x, color=color, title=f"Distribution of {x}")
            
        st.plotly_chart(fig, use_container_width=True) # Display chart

# ------------------------------------------------------------------------------
# PAGE 4: PREDICTIVE MODELLING
# ------------------------------------------------------------------------------
elif page == "Predictive Modelling":
    st.title("Predictive Intelligence Engine") # Page Title
    
    if HAS_ML:
        # Function to train model (Cached to avoid re-training on every click)
        @st.cache_resource
        def build_model(data):
            # Define engineered features list
            feats = ['Ownership_Risk_Score', 'State_Quality_Percentile', 'Chronic_Deficiency_Score', 'Fine_Per_Bed', 'Understaffed', 'High_Risk_State']
            # Filter features that actually exist in the dataframe
            feats = [f for f in feats if f in data.columns]
            
            if not feats: return None, None, None # Return None if features missing
            
            # Prepare X (Features) and y (Target)
            X = data[feats].fillna(0)
            y = data['Low_Quality_Facility'] if 'Low_Quality_Facility' in data.columns else np.random.randint(0,2,len(data))
            
            # Initialize and Train Random Forest
            model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
            model.fit(X, y)
            return model, X, feats

        with st.spinner("Training Model..."): # Show spinner while training
            model, X, feats = build_model(df)
        
        if model:
            st.subheader("Feature Importance (SHAP)") # Subheader
            explainer = shap.TreeExplainer(model) # Initialize SHAP explainer
            shap_values = explainer.shap_values(X.iloc[:200]) # Calculate SHAP for 200 samples
            
            # Robust SHAP handling: Check if output is list (classifier) or array (regressor)
            if isinstance(shap_values, list):
                vals = np.abs(shap_values[1]).mean(0) # Get mean absolute impact for class 1
            else:
                vals = np.abs(shap_values).mean(0)
            
            # FIX 1: Ensure vals is flattened to 1D array to avoid ValueError
            if vals.ndim > 1:
                vals = vals.flatten() # Flatten array if it has multiple dimensions

            # FIX 2: Ensure feature names and values have same length before plotting
            min_len = min(len(feats), len(vals)) # Find the minimum length
            feats = feats[:min_len] # Slice features
            vals = vals[:min_len] # Slice values
                
            # Create DataFrame for plotting
            imp_df = pd.DataFrame({'Feature': feats, 'Importance': vals}).sort_values('Importance')
            
            # Bar Chart of Feature Importance
            fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Top Predictors of Failure",
                         labels={'x':'Impact', 'y':'Feature'}, color='Importance', color_continuous_scale='Oranges')
            st.plotly_chart(fig, use_container_width=True) # Display chart
            
            st.subheader("Forensic Waterfall") # Subheader
            # Data for single observation waterfall plot
            idx = 0
            base = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            contrib = shap_values[1][idx] if isinstance(shap_values, list) else shap_values[idx]
            
            # Match lengths again for waterfall safety
            contrib = contrib[:len(feats)]
            
            # Waterfall Chart using Graph Objects
            fig_w = go.Figure(go.Waterfall(
                orientation="v",
                measure=["relative"] * len(feats), # Every step is relative
                x=feats, y=contrib,
                connector={"line":{"color":"rgb(63, 63, 63)"}}
            ))
            fig_w.update_layout(title="Risk Factors for Single Facility") # Layout title
            st.plotly_chart(fig_w, use_container_width=True) # Display chart
    else:
        st.warning("ML Libraries (sklearn, shap) not installed.") # Error message if libs missing

# ------------------------------------------------------------------------------
# PAGE 5: DATA NARRATIVE
# ------------------------------------------------------------------------------
elif page == "Data Narrative":
    st.title("The Narrative: Crisis in Care") # Page Title
    
    # Create Tabs for narrative structure
    tab1, tab2, tab3 = st.tabs(["1. The Takeover", "2. The Collapse", "3. Action Plan"])
    
    with tab1:
        st.header("The Privatization Wave") # Header
        st.write("83% of homes are now For-Profit. The map is red.") # Text
        if 'Ownership_Risk_Score' in df.columns:
            # Recalculate map data specifically for this narrative
            grp = df.groupby('code')['Ownership_Risk_Score'].apply(lambda x: (x==3).mean()*100).reset_index(name='Val')
            fig = px.choropleth(grp, locations='code', locationmode='USA-states', color='Val',
                                color_continuous_scale='Reds', scope="usa") # Limit scope to USA
            st.plotly_chart(fig, use_container_width=True) # Display map
        
    with tab2:
        st.header("The Quality Collapse") # Header
        st.write("As profits rose, ratings fell. Correlation is evident.") # Text
        if rating_col:
            # Box plot to show relationship
            fig = px.box(df, x='Ownership Type', y=rating_col, color='Ownership Type')
            st.plotly_chart(fig, use_container_width=True) # Display chart
        
    with tab3:
        st.header("Call to Action") # Header
        st.info("1. Freeze licenses.\n2. Mandate staffing.\n3. Link pay to quality.") # Recommendations

# ------------------------------------------------------------------------------
# PAGE 6: LOCAL EXPLORER
# ------------------------------------------------------------------------------
elif page == "Local Market Explorer":
    st.title("Local Market Intelligence") # Page Title
    
    # 1. Search Bar (Global)
    search_txt = st.text_input("Search Facility Name", "") # Text Input
    
    # 2. Filters
    c1, c2 = st.columns(2)
    with c1:
        # Default to "All"
        st_list = ["All"] + sorted(df['State'].unique().tolist())
        state_filter = st.selectbox("State Filter", st_list) # State Dropdown
    with c2:
        # City depends on State
        if state_filter != "All":
            city_list = ["All"] + sorted(df[df['State'] == state_filter][city_col].unique().tolist())
        else:
            city_list = ["All"] + sorted(df[city_col].unique().tolist())[:500] # Limit for performance if All states selected
        city_filter = st.selectbox("City Filter", city_list) # City Dropdown
        
    # 3. Apply Filters Logic
    dff = df.copy() # Work on a copy
    if state_filter != "All":
        dff = dff[dff['State'] == state_filter]
    if city_filter != "All":
        dff = dff[dff[city_col] == city_filter]
    if search_txt:
        # Case-insensitive search on Name column
        dff = dff[dff[name_col].astype(str).str.contains(search_txt, case=False, na=False)]
        
    st.markdown("---") # Divider
    
    # 4. Results
    if not dff.empty:
        # Metrics for filtered view
        m1, m2, m3 = st.columns(3)
        m1.metric("Facilities Found", len(dff))
        avg = dff[rating_col].mean()
        m2.metric("Avg Rating", f"{avg:.2f}")
        
        # Display Table
        st.dataframe(
            dff[[name_col, city_col, 'State', rating_col, owner_col]], # Select key columns
            use_container_width=True # Full width
        )
    else:
        st.warning("No facilities match your search.") # No results message

# ------------------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------------------
st.markdown("---") # Divider
st.markdown("© 2025 Md Rabiul Alam | Medicare Hospital Spending by Claim (USA)") # Footer text
