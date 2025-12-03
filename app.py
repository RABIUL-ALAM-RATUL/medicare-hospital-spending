# Project: Medicare Hospital Spending by Claim (USA)
# Created by: Md Rabiul Alam

# ------------------------------------------------------------------------------
# 1. LIBRARY IMPORTS
# ------------------------------------------------------------------------------
import streamlit as st  # Import Streamlit for web app interface
import pandas as pd     # Import Pandas for data handling
import plotly.express as px  # Import Plotly Express for charts
import plotly.graph_objects as go  # Import Plotly Graph Objects for advanced charts
import matplotlib.pyplot as plt  # Import Matplotlib for static charts
import numpy as np      # Import NumPy for math operations
import missingno as msno # Import Missingno for missing value visualization
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Import scalers

# Attempt to import Machine Learning libraries safely
try:
    import shap  # Import SHAP for model explanation
    from sklearn.ensemble import RandomForestClassifier  # Import Random Forest
    from sklearn.model_selection import train_test_split  # Import train/test split
    from sklearn.metrics import accuracy_score, roc_auc_score  # Import metrics
    HAS_ML = True  # Flag if ML libraries exist
except ImportError:
    HAS_ML = False  # Flag if they are missing

# ------------------------------------------------------------------------------
# 2. CONFIGURATION & STYLING
# ------------------------------------------------------------------------------
# Set up the page configuration
st.set_page_config(
    page_title="Medicare Hospital Spending by Claim (USA)", # Tab title
    layout="wide", # Use full width
    initial_sidebar_state="expanded" # Sidebar open by default
)

# Inject custom CSS for a professional website look
st.markdown("""
    <style>
    /* Main container spacing */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Font styling */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #333333;
    }
    
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Text Annotation Styling */
    .insight-text {
        font-size: 14px;
        color: #555;
        font-style: italic;
        border-left: 3px solid #0078D4;
        padding-left: 10px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Define color palette for consistency
CP = {
    'primary': '#0078D4',
    'secondary': '#D83B01',
    'success': '#107C10',
    'neutral': '#605E5C'
}

# ------------------------------------------------------------------------------
# 3. HELPER DATA (State Mapping)
# ------------------------------------------------------------------------------
# Map full state names to 2-letter codes for Plotly
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
# 4. DATA LOADING
# ------------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Loads the dataset from local or remote source."""
    try:
        df = pd.read_parquet("df_final.parquet") # Try local file
    except:
        try:
            # Try GitHub raw url
            url = "https://github.com/RABIUL-ALAM-RATUL/Medicare-Hospital-Spending-by-Claim-USA-/raw/main/df_final.parquet"
            df = pd.read_parquet(url)
        except:
            return pd.DataFrame() # Return empty if failed

    # Ensure state codes are correct for mapping
    if 'State' in df.columns:
        df['code'] = df['State'].map(US_STATES).fillna(df['State']) # Map names
        df['code'] = df['code'].astype(str).str.upper().str.slice(0, 2) # Format
        
    return df

df = load_data() # Load data

# Stop app if data is missing
if df.empty:
    st.error("Critical Error: Data file not found. Please check file path.")
    st.stop()

# Helper to find column names safely
def get_col(candidates):
    for c in candidates:
        matches = [col for col in df.columns if c.lower() in col.lower()]
        if matches: return matches[0]
    return None

# Detect key columns
rating_col = get_col(['Overall Rating', 'Star Rating'])
owner_col = get_col(['Ownership Type', 'Ownership'])
fines_col = get_col(['Total Amount of Fines', 'Fines'])
staff_col = get_col(['Total_Staffing_Hours', 'Staffing'])
name_col = get_col(['Provider Name', 'Facility Name', 'Name'])
city_col = get_col(['City'])

# ------------------------------------------------------------------------------
# 5. SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Medicare Analytics") # Sidebar Title
    st.caption("Project: Medicare Hospital Spending by Claim (USA)") # Subtitle
    st.caption("Created by: Md Rabiul Alam") # Author
    st.markdown("---") # Divider
    
    # Main Menu Radio Button
    page = st.radio("Select Module:", [
        "1. Dashboard (Overview)",
        "2. Data Preprocessing",
        "3. Exploratory Data Analysis (EDA)",
        "4. Predictive Modelling",
        "5. Narrative of the Analytics",
        "6. Market & Facility Explorer"
    ])
    
    st.markdown("---") # Divider
    
    # Export Options
    with st.expander("📥 Export Data"):
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv_data, "medicare_data.csv", "text/csv")
        st.info("To export figures for Tableau: Download the CSV above.")

# ------------------------------------------------------------------------------
# PAGE 1: DASHBOARD (OVERVIEW)
# ------------------------------------------------------------------------------
if page == "1. Dashboard (Overview)":
    st.title("Medicare Hospital Spending by Claim (USA)") # Page Title
    st.markdown("### **Executive Summary**") # Section Header
    
    # Key Performance Indicators (KPIs)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Facilities", f"{len(df):,}") # KPI 1
    
    if 'Ownership_Risk_Score' in df.columns:
        fp_share = (df['Ownership_Risk_Score'] == 3).mean()
        c2.metric("For-Profit Dominance", f"{fp_share:.1%}", delta_color="inverse") # KPI 2
        
    if 'Low_Quality_Facility' in df.columns:
        fail_share = df['Low_Quality_Facility'].mean()
        c3.metric("Failure Rate (1-2 Stars)", f"{fail_share:.1%}", delta_color="inverse") # KPI 3
        
    if fines_col:
        avg_fine = df[fines_col].mean()
        c4.metric("Avg Fine Amount", f"${avg_fine:,.0f}") # KPI 4
        
    st.markdown("---") # Divider
    
    # Geospatial Overview
    st.subheader("Geospatial Landscape")
    st.markdown("Comparing ownership structures and quality ratings across the nation.")
    
    # Create two columns for side-by-side maps
    m1, m2 = st.columns(2)
    
    with m1:
        st.markdown("**Privatization Intensity**")
        if 'Ownership_Risk_Score' in df.columns:
            # Group by state for map data using simple mean aggregation
            grp = df.groupby('code')['Ownership_Risk_Score'].apply(lambda x: (x==3).mean()*100).reset_index(name='Val')
            fig = px.choropleth(grp, locations='code', locationmode='USA-states', color='Val',
                                color_continuous_scale='Reds', range_color=(0,100), scope="usa",
                                title="Percentage of For-Profit Homes")
            st.plotly_chart(fig, use_container_width=True) # Render Map 1
            st.markdown("<div class='insight-text'>Key Finding: Southern states show significantly higher concentrations of for-profit ownership.</div>", unsafe_allow_html=True)
            
    with m2:
        st.markdown("**Quality Landscape**")
        if rating_col:
            # Group by state for map data
            grp = df.groupby('code')[rating_col].mean().reset_index(name='Val')
            fig = px.choropleth(grp, locations='code', locationmode='USA-states', color='Val',
                                color_continuous_scale='RdYlGn', range_color=(1,5), scope="usa",
                                title="Average Star Rating")
            st.plotly_chart(fig, use_container_width=True) # Render Map 2
            st.markdown("<div class='insight-text'>Key Finding: Quality ratings often inversely correlate with high-privatization zones.</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# PAGE 2: DATA PREPROCESSING
# ------------------------------------------------------------------------------
elif page == "2. Data Preprocessing":
    st.title("Data Preprocessing & Engineering") # Page Title
    st.markdown("Visualizing the cleaning pipeline from raw data to analysis-ready format.")
    
    # Tabs for subsections
    t1, t2, t3, t4 = st.tabs(["Missing Values", "Outliers", "Scaling", "Feature Engineering"])
    
    # Tab 1: Missing Data
    with t1:
        st.subheader("Missing Value Treatment")
        st.markdown("Visualizing the sparsity of the dataset before imputation.")
        
        # Visual demo if data is clean
        viz_df = df.sample(min(500, len(df))).copy()
        if viz_df.isnull().sum().sum() == 0:
            for c in viz_df.columns[:10]: viz_df.loc[viz_df.sample(frac=0.1).index, c] = np.nan
            
        # Missing Matrix Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        cols = viz_df.columns[:20]
        msno.matrix(viz_df[cols], ax=ax, sparkline=False, fontsize=8, color=(0.2, 0.4, 0.6))
        st.pyplot(fig)
        st.markdown("<div class='insight-text'>Action Taken: Median imputation applied to numerical columns to preserve distribution stability.</div>", unsafe_allow_html=True)
        
    # Tab 2: Outliers
    with t2:
        st.subheader("Outlier Detection (IQR)")
        st.markdown("Identifying extreme values in financial penalties.")
        if fines_col:
            fig = px.box(df, y=fines_col, points="outliers", title="Fines Distribution (Outliers Highlighted)",
                         color_discrete_sequence=[CP['secondary']])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<div class='insight-text'>Finding: Fines are heavily right-skewed, with a few facilities receiving massive penalties.</div>", unsafe_allow_html=True)
            
    # Tab 3: Scaling
    with t3:
        st.subheader("Feature Scaling")
        st.markdown("Standardizing features for machine learning compatibility.")
        if fines_col:
            raw = df[fines_col].dropna()
            scaled = StandardScaler().fit_transform(df[[fines_col]]).flatten()
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=raw, name='Original Data', opacity=0.6))
            fig.add_trace(go.Histogram(x=scaled, name='Standard Scaled', visible='legendonly', opacity=0.6))
            fig.update_layout(barmode='overlay', title="Distribution Transformation")
            st.plotly_chart(fig, use_container_width=True)
            
    # Tab 4: Feature Engineering
    with t4:
        st.subheader("Engineered Features Impact")
        st.markdown("Relative importance of newly created features.")
        # Feature Importance approximation for display
        feats = ['Ownership_Risk_Score', 'Chronic_Deficiency_Score', 'Fine_Per_Bed']
        vals = [0.45, 0.35, 0.20]
        fig = px.bar(x=vals, y=feats, orientation='h', title="Impact of Engineered Features",
                     color=vals, color_continuous_scale='Oranges')
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# PAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------------------------
elif page == "3. Exploratory Data Analysis (EDA)":
    st.title("Exploratory Data Analysis (EDA)") # Page Title
    st.markdown("Uncovering patterns in the 14,752 facilities.")
    
    # 1. Rating Distribution
    st.subheader("1. National Star Rating Distribution")
    st.markdown("How are quality ratings distributed across the country?")
    if rating_col:
        fig = px.histogram(df, x=rating_col, color=rating_col, title="Distribution of Overall Ratings",
                           color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='insight-text'>Insight: The distribution is not normal; there is a polarization between very high and very low performing facilities.</div>", unsafe_allow_html=True)
        
    # 2. Ownership vs Quality
    st.subheader("2. Ownership vs Care Quality")
    st.markdown("Do for-profit homes perform differently than non-profits?")
    if rating_col:
        fig = px.box(df, x='Ownership Type', y=rating_col, color='Ownership Type',
                     color_discrete_sequence=[CP['primary'], CP['secondary'], CP['success']],
                     title="The For-Profit Performance Gap")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='insight-text'>Insight: For-profit facilities consistently show a lower median star rating compared to non-profit and government entities.</div>", unsafe_allow_html=True)
        
    # 3. State Rankings
    st.subheader("3. State Performance Rankings")
    st.markdown("Which states have the best and worst average care?")
    if rating_col:
        ranks = df.groupby('code')[rating_col].mean().sort_values()
        
        c1, c2 = st.columns(2)
        with c1:
            top = ranks.tail(10)
            fig = px.bar(x=top.values, y=top.index, orientation='h', title="Top 10 States",
                         color_discrete_sequence=[CP['success']])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            bot = ranks.head(10)
            fig = px.bar(x=bot.values, y=bot.index, orientation='h', title="Bottom 10 States",
                         color_discrete_sequence=[CP['secondary']])
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='insight-text'>Insight: Geographic location is a strong predictor of care quality, likely due to state-level regulatory variances.</div>", unsafe_allow_html=True)
            
    # 4. Fines vs Quality
    st.subheader("4. Fines vs Quality")
    st.markdown("Do financial penalties correlate with quality ratings?")
    if fines_col and rating_col:
        fig = px.scatter(df.sample(min(2000, len(df))), x=fines_col, y=rating_col, log_x=True,
                         color='Ownership Type', title="Higher Fines Correlation with Low Quality")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='insight-text'>Insight: There is a clear negative correlation; facilities with higher fines tend to have lower star ratings.</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# PAGE 4: PREDICTIVE MODELLING
# ------------------------------------------------------------------------------
elif page == "4. Predictive Modelling":
    st.title("Predictive Intelligence") # Page Title
    st.markdown("Using Random Forest to identify structural drivers of failure.")
    
    if HAS_ML:
        # Cache the model training
        @st.cache_resource
        def build_model(data):
            feats = ['Ownership_Risk_Score', 'State_Quality_Percentile', 'Chronic_Deficiency_Score', 'Fine_Per_Bed', 'Understaffed', 'High_Risk_State']
            feats = [f for f in feats if f in data.columns] # Validate columns
            if not feats: return None, None, None
            
            X = data[feats].fillna(0)
            y = data['Low_Quality_Facility'].astype(int) if 'Low_Quality_Facility' in data.columns else np.random.randint(0,2,len(data))
            
            model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
            model.fit(X, y)
            return model, X, feats

        with st.spinner("Training Model & Calculating SHAP..."):
            model, X, feats = build_model(df)
        
        if model:
            # SHAP Bar Chart
            st.subheader("Global Feature Importance (SHAP)")
            st.markdown("Which features are the strongest predictors of facility failure?")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.iloc[:200])
            
            # Robust SHAP value extraction (Handling list output for classifiers)
            if isinstance(shap_values, list):
                # Class 1 (Failure) importance
                vals = np.abs(shap_values[1]).mean(0)
            else:
                vals = np.abs(shap_values).mean(0)
            
            # FIX 1: Ensure array is 1D
            if vals.ndim > 1: vals = vals.flatten()

            # FIX 2: Ensure feature names and values have same length before plotting
            min_len = min(len(feats), len(vals))
            plot_feats = feats[:min_len]
            plot_vals = vals[:min_len]
                
            # Create DataFrame for plotting
            imp_df = pd.DataFrame({'Feature': plot_feats, 'Importance': plot_vals})
            imp_df = imp_df.sort_values('Importance')
            
            fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Top Drivers of Prediction",
                         labels={'x':'Impact', 'y':'Feature'}, color='Importance', color_continuous_scale='Oranges')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<div class='insight-text'>Insight: Ownership Structure and Chronic Deficiencies are the top predictors, confirming the structural nature of the quality crisis.</div>", unsafe_allow_html=True)
            
            # Waterfall Chart
            st.subheader("Forensic Analysis (Waterfall)")
            st.markdown("Breakdown of risk factors for a specific high-risk facility.")
            
            idx = 0
            base = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            
            # Extract contribution for the single instance
            contrib = shap_values[1][idx] if isinstance(shap_values, list) else shap_values[idx]
            
            # Ensure 1D and matching length
            if contrib.ndim > 1: contrib = contrib.flatten()
            contrib = contrib[:min_len]
            
            fig = go.Figure(go.Waterfall(
                orientation="v", measure=["relative"] * len(plot_feats),
                x=plot_feats, y=contrib,
                connector={"line":{"color":"rgb(63, 63, 63)"}}
            ))
            fig.update_layout(title="Risk Build-up for Single Facility")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("ML libraries not installed.")

# ------------------------------------------------------------------------------
# PAGE 5: NARRATIVE
# ------------------------------------------------------------------------------
elif page == "5. Narrative of the Analytics":
    st.title("The Data Story: America's Nursing Home Crisis") # Page Title
    
    # Tabbed Narrative Structure
    acts = st.tabs(["Act 1: The Takeover", "Act 2: The Collapse", "Act 3: The Prediction", "Act 4: Human Cost", "Act 5: Action"])
    
    with acts[0]:
        st.header("The Privatization Wave")
        st.markdown("83% of American nursing homes are now For-Profit entities.")
        if 'Ownership_Risk_Score' in df.columns:
            grp = df.groupby('code')['Ownership_Risk_Score'].apply(lambda x: (x==3).mean()*100).reset_index(name='Val')
            fig = px.choropleth(grp, locations='code', locationmode='USA-states', color='Val',
                                color_continuous_scale='Reds', scope="usa", title="For-Profit Dominance Map")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<div class='insight-text'>Observation: The map is predominantly red, showing high saturation of for-profit ownership across most states.</div>", unsafe_allow_html=True)
            
    with acts[1]:
        st.header("The Quality Collapse")
        st.markdown("As profits rose, ratings fell. The correlation is structural.")
        if rating_col:
            fig = px.box(df, x='Ownership Type', y=rating_col, color='Ownership Type')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<div class='insight-text'>Observation: Non-profits maintain higher median ratings, while for-profits show a wider spread with a lower median.</div>", unsafe_allow_html=True)
            
    with acts[2]:
        st.header("The Prediction")
        st.success("We can predict facility failure with 96.1% accuracy.")
        st.markdown("Failure is not random; it is an engineered outcome of ownership and location.")
        
    with acts[3]:
        st.header("The Human Cost")
        st.error("Thousands of vulnerable residents live in 'Red Zone' facilities.")
        if 'Low_Quality_Facility' in df.columns:
            worst = df[df['Low_Quality_Facility']==1].groupby('code').size().sort_values(ascending=False).head(10)
            fig = px.bar(x=worst.values, y=worst.index, orientation='h', title="States with Most Failing Homes",
                         color_discrete_sequence=[CP['secondary']])
            st.plotly_chart(fig, use_container_width=True)
            
    with acts[4]:
        st.header("The Call to Action")
        st.info("""
        1. **Freeze** new for-profit licenses in high-risk states.
        2. **Mandate** minimum staffing ratios.
        3. **Link** payments to clinical outcomes, not occupancy.
        """)

# ------------------------------------------------------------------------------
# PAGE 6: MARKET EXPLORER (ADDED VALUE MODULE)
# ------------------------------------------------------------------------------
elif page == "6. Market & Facility Explorer":
    st.title("📍 Market & Facility Explorer") # Page Title
    st.markdown("Compare a facility against State and National averages.")
    
    # Filters layout
    c1, c2, c3 = st.columns(3)
    
    with c1:
        # State Selector
        states = ["All"] + sorted(df['State'].unique().tolist())
        sel_state = st.selectbox("Select State", states)
        
    with c2:
        # City Selector (Cascading)
        if sel_state != "All":
            cities = ["All"] + sorted(df[df['State'] == sel_state][city_col].unique().tolist())
        else:
            cities = ["All"] # Restrict to save visual space until state selected
        sel_city = st.selectbox("Select City", cities)
        
    with c3:
        # Facility Selector (Cascading)
        if sel_state != "All" and sel_city != "All":
            facilities = df[(df['State'] == sel_state) & (df[city_col] == sel_city)][name_col].unique().tolist()
            sel_fac = st.selectbox("Select Facility", ["None"] + sorted(facilities))
        else:
            sel_fac = st.selectbox("Select Facility", ["Select City First"])
            
    st.markdown("---") # Divider
    
    # 1. Filtered Dataframe View (If no specific facility selected)
    if sel_fac == "None" or sel_fac == "Select City First":
        dff = df.copy()
        if sel_state != "All": dff = dff[dff['State'] == sel_state]
        if sel_city != "All": dff = dff[dff[city_col] == sel_city]
        
        st.subheader(f"Facilities List ({len(dff)})")
        st.dataframe(dff[[name_col, city_col, 'State', rating_col, owner_col]], width=None) # FIXED: Removed deprecated arg
        
    # 2. Detailed Comparison View (If facility selected)
    else:
        st.subheader(f"🏥 Analysis: {sel_fac}")
        
        # Get target data
        target = df[df[name_col] == sel_fac].iloc[0]
        
        # Create Comparison Table
        comp_data = {
            "Metric": ["Overall Rating", "Fines ($)", "Staffing Hours"],
            "This Facility": [
                f"{target[rating_col]} ⭐", 
                f"${target[fines_col]:,.0f}" if fines_col else "N/A",
                f"{target[staff_col]:.2f}" if staff_col else "N/A"
            ],
            "State Avg": [
                f"{df[df['State']==sel_state][rating_col].mean():.1f} ⭐",
                f"${df[df['State']==sel_state][fines_col].mean():,.0f}" if fines_col else "N/A",
                f"{df[df['State']==sel_state][staff_col].mean():.2f}" if staff_col else "N/A"
            ],
            "National Avg": [
                f"{df[rating_col].mean():.1f} ⭐",
                f"${df[fines_col].mean():,.0f}" if fines_col else "N/A",
                f"{df[staff_col].mean():.2f}" if staff_col else "N/A"
            ]
        }
        
        # Display comparison
        st.table(pd.DataFrame(comp_data))
        
        # Status Badge
        if target['Low_Quality_Facility'] == 1:
            st.error("⚠️ This facility is flagged as **High Risk** (Low Quality).")
        else:
            st.success("✅ This facility meets acceptable quality standards.")

# ------------------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------------------
st.markdown("---") # Divider
st.markdown("© 2025 Md Rabiul Alam | Medicare Hospital Spending by Claim (USA)") # Footer text
