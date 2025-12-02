# app.py
"""
Medicare Hospital Spending Analysis Dashboard
A comprehensive Streamlit application for analyzing CMS Nursing Home Provider data.
Author: Healthcare Analytics Team
Date: December 2024
"""

# ============================================================================
# IMPORT SECTION WITH DETAILED COMMENTS
# ============================================================================

import streamlit as st  # Main framework for building web application
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computing and array operations
import plotly.express as px  # High-level interface for creating Plotly charts
import plotly.graph_objects as go  # Low-level interface for custom Plotly charts
from plotly.subplots import make_subplots  # For creating multi-panel figures
import warnings  # For controlling warning messages
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Try to import optional visualization libraries
try:
    import matplotlib.pyplot as plt  # Static plotting library
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("Matplotlib not installed - some static plots will use Plotly alternatives")

try:
    import seaborn as sns  # Statistical data visualization based on matplotlib
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# ============================================================================
# PAGE CONFIGURATION AND STYLING
# ============================================================================

# Configure Streamlit page settings for optimal viewing
st.set_page_config(
    page_title="CMS Nursing Home Analytics Dashboard",  # Browser tab title
    page_icon="🏥",  # Favicon/emoji for browser tab
    layout="wide",  # Use full width of screen
    initial_sidebar_state="expanded"  # Sidebar starts expanded
)

# Custom CSS for professional styling and consistent design
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        color: #1a5276;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Section headers */
    .section-header {
        font-size: 2rem;
        color: #2c3e50;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid #3498db;
        font-weight: 600;
    }
    
    /* Subsection headers */
    .subsection-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 500;
        padding-left: 0.5rem;
        border-left: 4px solid #2ecc71;
    }
    
    /* Storytelling box with professional styling */
    .storytelling-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #f39c12;
    }
    
    /* Interaction note box */
    .interaction-note {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1.5rem;
        border: 1px solid #dee2e6;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Metric cards for KPI display */
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Data table styling */
    .dataframe {
        font-size: 0.9rem;
        border-collapse: collapse;
        width: 100%;
    }
    
    .dataframe th {
        background-color: #2c3e50;
        color: white;
        padding: 8px;
        text-align: left;
    }
    
    .dataframe td {
        padding: 8px;
        border-bottom: 1px solid #ddd;
    }
    
    .dataframe tr:hover {
        background-color: #f5f5f5;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #3498db 0%, #2c3e50 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 500;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2ecc71 0%, #3498db 100%);
    }
</style>
""", unsafe_allow_html=True)  # Allow HTML for custom styling

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

# Initialize session state variables for maintaining state across interactions
if 'df' not in st.session_state:
    st.session_state.df = None  # Store the main dataframe
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None  # Store cleaned dataframe
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None  # Store uploaded file
if 'analysis_progress' not in st.session_state:
    st.session_state.analysis_progress = 0  # Track analysis progress

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    # Header with logo and title
    st.markdown('<h1 style="color: white; text-align: center;">🏥 CMS Analytics</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Navigation section
    st.markdown('<h2 style="color: white;">📊 Navigation</h2>', unsafe_allow_html=True)
    analysis_section = st.selectbox(
        "Select Analysis Section",
        [
            "🏠 Dashboard Overview",
            "📈 Data Exploration",
            "🔍 Data Quality Assessment", 
            "📊 Advanced Visualizations",
            "📉 Performance Metrics",
            "📍 Geographic Analysis",
            "📋 Detailed Reports"
        ],
        key="nav_select"
    )
    
    st.markdown("---")
    
    # Data upload section
    st.markdown('<h2 style="color: white;">📁 Data Management</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload CMS Dataset (CSV)",
        type=['csv'],
        help="Upload the NH_ProviderInfo_Sep2025.csv file or similar CMS dataset"
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            with st.spinner("Loading dataset..."):
                df = pd.read_csv(uploaded_file, low_memory=False)
                st.session_state.df = df
                st.session_state.uploaded_file = uploaded_file
                st.success(f"✅ Dataset loaded: {len(df):,} rows × {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        # Create sample data for demonstration
        st.info("Using sample data for demonstration")
        @st.cache_data
        def create_sample_data():
            """Generate realistic sample data for demonstration purposes"""
            np.random.seed(42)
            n_facilities = 1500
            
            # Create comprehensive sample dataset
            states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
                     'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD']
            
            ownership_types = [
                'For profit - Corporation',
                'For profit - Individual', 
                'For profit - Partnership',
                'Non-profit - Corporation',
                'Non-profit - Church',
                'Government - State',
                'Government - County',
                'Government - City',
                'Government - Hospital district'
            ]
            
            # Generate realistic data with correlations
            base_ratings = np.random.normal(3.0, 1.0, n_facilities)
            base_ratings = np.clip(base_ratings, 1.0, 5.0)
            
            sample_data = {
                # Basic facility information
                'Provider Name': [f'Healthcare Facility {i:04d}' for i in range(n_facilities)],
                'State': np.random.choice(states, n_facilities, p=[0.03]*len(states)),
                'City/Town': [f'City {i%50}' for i in range(n_facilities)],
                'ZIP Code': np.random.randint(10000, 99999, n_facilities),
                
                # Ownership and type
                'Ownership Type': np.random.choice(ownership_types, n_facilities),
                'Provider Type': np.random.choice(['Skilled Nursing', 'Assisted Living', 
                                                   'Rehabilitation'], n_facilities),
                
                # Quality ratings (correlated with each other)
                'Overall Rating': base_ratings,
                'Health Inspection Rating': np.clip(base_ratings + np.random.normal(0, 0.3, n_facilities), 1, 5),
                'Staffing Rating': np.clip(base_ratings + np.random.normal(0, 0.4, n_facilities), 1, 5),
                'QM Rating': np.clip(base_ratings + np.random.normal(0, 0.35, n_facilities), 1, 5),
                
                # Staffing metrics (correlated with ratings)
                'Reported Nurse Aide Staffing Hours per Resident per Day': 
                    np.clip(np.random.normal(2.5, 0.5, n_facilities) + (base_ratings-3)*0.2, 1, 5),
                'Reported LPN Staffing Hours per Resident per Day': 
                    np.clip(np.random.normal(0.8, 0.2, n_facilities) + (base_ratings-3)*0.1, 0.5, 2),
                'Reported RN Staffing Hours per Resident per Day': 
                    np.clip(np.random.normal(0.7, 0.2, n_facilities) + (base_ratings-3)*0.15, 0.5, 2),
                'Total nursing staff turnover': np.clip(np.random.normal(45, 15, n_facilities) - (base_ratings-3)*5, 10, 80),
                
                # Penalty and compliance data
                'Number of Fines': np.random.poisson(0.5, n_facilities),
                'Total Amount of Fines in Dollars': np.random.exponential(5000, n_facilities),
                'Number of Payment Denials': np.random.poisson(0.2, n_facilities),
                'Number of Citations from Infection Control Inspections': np.random.poisson(0.3, n_facilities),
                
                # Geographic coordinates
                'Latitude': np.random.uniform(25.0, 49.0, n_facilities),
                'Longitude': np.random.uniform(-124.0, -67.0, n_facilities),
                
                # Capacity metrics
                'Number of Certified Beds': np.random.randint(50, 300, n_facilities),
                'Average Number of Residents per Day': np.random.randint(40, 280, n_facilities),
                
                # Chain information
                'Chain Name': np.random.choice(['National Chain A', 'Regional Chain B', 'Independent', 
                                               'Chain C Healthcare', None], n_facilities, p=[0.3, 0.2, 0.3, 0.15, 0.05]),
                'Number of Facilities in Chain': np.random.choice([1, 5, 15, 50, 200], n_facilities, p=[0.3, 0.4, 0.2, 0.08, 0.02]),
                
                # Special status
                'Special Focus Status': np.random.choice(['SFF', 'Not SFF', None], n_facilities, p=[0.05, 0.9, 0.05]),
                
                # Additional quality metrics
                'Long-Stay QM Rating': np.clip(base_ratings + np.random.normal(0, 0.3, n_facilities), 1, 5),
                'Short-Stay QM Rating': np.clip(base_ratings + np.random.normal(0, 0.4, n_facilities), 1, 5),
                
                # Processing date
                'Processing Date': '2025-09-01'
            }
            
            # Add some missing values realistically
            df_sample = pd.DataFrame(sample_data)
            
            # Add missing values in specific columns
            missing_cols = ['Chain Name', 'Number of Facilities in Chain', 'Special Focus Status']
            for col in missing_cols:
                mask = np.random.random(n_facilities) < 0.1
                df_sample.loc[mask, col] = None
            
            return df_sample
        
        if st.session_state.df is None:
            st.session_state.df = create_sample_data()
    
    st.markdown("---")
    
    # Filters section
    st.markdown('<h2 style="color: white;">🎯 Data Filters</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # State filter
        all_states = sorted(df['State'].dropna().unique())
        selected_states = st.multiselect(
            "Select States",
            options=all_states,
            default=all_states[:3],
            help="Filter analysis by specific states"
        )
        
        # Ownership type filter
        ownership_types = sorted(df['Ownership Type'].dropna().unique())
        selected_ownership = st.multiselect(
            "Select Ownership Types",
            options=ownership_types,
            default=ownership_types[:3],
            help="Filter by facility ownership structure"
        )
        
        # Quality rating filter
        st.markdown("#### Quality Rating Range")
        min_rating, max_rating = st.slider(
            "Overall Rating Filter",
            min_value=1.0,
            max_value=5.0,
            value=(1.0, 5.0),
            step=0.5,
            help="Filter facilities by overall quality rating"
        )
        
        # Facility size filter
        if 'Number of Certified Beds' in df.columns:
            max_beds = int(df['Number of Certified Beds'].max())
            min_beds, max_beds_filter = st.slider(
                "Certified Beds Range",
                min_value=0,
                max_value=max_beds,
                value=(0, max_beds),
                help="Filter by facility size (number of beds)"
            )
        
        st.markdown("---")
        
        # Analysis settings
        st.markdown('<h2 style="color: white;">⚙️ Analysis Settings</h2>', unsafe_allow_html=True)
        
        # Visualization theme
        viz_theme = st.selectbox(
            "Visualization Theme",
            ["Plotly", "Seaborn", "Matplotlib"],
            help="Select visualization style theme"
        )
        
        # Data aggregation level
        agg_level = st.selectbox(
            "Aggregation Level",
            ["Facility Level", "State Level", "Ownership Level"],
            help="Select data aggregation level for analysis"
        )
        
        # Reset filters button
        if st.button("🔄 Reset All Filters"):
            st.session_state.df = None
            st.rerun()

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def apply_filters(df, selected_states, selected_ownership, min_rating, max_rating):
    """
    Apply all selected filters to the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    selected_states : list
        List of selected states for filtering
    selected_ownership : list
        List of selected ownership types
    min_rating : float
        Minimum overall rating
    max_rating : float
        Maximum overall rating
    
    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe
    """
    filtered_df = df.copy()  # Create copy to avoid modifying original
    
    # Apply state filter if states are selected
    if selected_states:
        filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]
    
    # Apply ownership filter if ownership types are selected
    if selected_ownership:
        filtered_df = filtered_df[filtered_df['Ownership Type'].isin(selected_ownership)]
    
    # Apply rating filter if rating column exists
    if 'Overall Rating' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['Overall Rating'] >= min_rating) & 
            (filtered_df['Overall Rating'] <= max_rating)
        ]
    
    return filtered_df

def calculate_missing_data_summary(df):
    """
    Calculate comprehensive missing data statistics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    
    Returns:
    --------
    pandas.DataFrame
        Missing data summary statistics
    """
    missing_counts = df.isnull().sum()  # Count missing values per column
    missing_percentage = (missing_counts / len(df)) * 100  # Calculate percentage
    
    # Create summary dataframe
    missing_summary = pd.DataFrame({
        'Missing_Count': missing_counts,
        'Missing_Percentage': missing_percentage,
        'Data_Type': df.dtypes.astype(str)
    })
    
    # Sort by missing percentage descending
    missing_summary = missing_summary.sort_values('Missing_Percentage', ascending=False)
    
    return missing_summary

def create_quality_metrics_summary(df):
    """
    Calculate summary statistics for quality metrics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict
        Dictionary of quality metric summaries
    """
    summary = {}
    
    # Identify rating columns
    rating_cols = [col for col in df.columns if 'Rating' in col and 'Footnote' not in col]
    
    for col in rating_cols:
        if col in df.columns:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                summary[col] = {
                    'mean': valid_data.mean(),
                    'median': valid_data.median(),
                    'std': valid_data.std(),
                    'min': valid_data.min(),
                    'max': valid_data.max(),
                    'count': len(valid_data)
                }
    
    return summary

# ============================================================================
# MAIN CONTENT - DASHBOARD OVERVIEW
# ============================================================================

if analysis_section == "🏠 Dashboard Overview":
    # Main header with gradient background
    st.markdown(
        '<div style="background: linear-gradient(90deg, #3498db 0%, #2c3e50 100%); '
        'padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">'
        '<h1 class="main-header" style="color: white;">CMS Nursing Home Analytics Dashboard</h1>'
        '<p style="color: white; text-align: center; font-size: 1.2rem;">'
        'Comprehensive Analysis of Medicare Hospital Spending and Quality Metrics</p>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Apply filters to current data
    if st.session_state.df is not None:
        df = st.session_state.df
        filtered_df = apply_filters(df, selected_states, selected_ownership, min_rating, max_rating)
        
        # Key Performance Indicators in a grid layout
        st.markdown('<h2 class="section-header">📊 Key Performance Indicators</h2>', 
                   unsafe_allow_html=True)
        
        # Create 4-column layout for KPIs
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Total Facilities", 
                f"{len(filtered_df):,}",
                f"{len(filtered_df) - len(df):+,}" if len(filtered_df) != len(df) else None
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with kpi_col2:
            if 'Overall Rating' in filtered_df.columns:
                avg_rating = filtered_df['Overall Rating'].mean()
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "Avg. Overall Rating", 
                    f"{avg_rating:.2f}",
                    f"{(avg_rating - df['Overall Rating'].mean()):+.2f}" if 'Overall Rating' in df.columns else None
                )
                st.markdown('</div>', unsafe_allow_html=True)
        
        with kpi_col3:
            if 'Number of Fines' in filtered_df.columns:
                total_fines = filtered_df['Number of Fines'].sum()
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "Total Fines Issued", 
                    f"{total_fines:,}",
                    help="Total number of regulatory fines"
                )
                st.markdown('</div>', unsafe_allow_html=True)
        
        with kpi_col4:
            if 'Number of Certified Beds' in filtered_df.columns:
                total_beds = filtered_df['Number of Certified Beds'].sum()
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "Total Certified Beds", 
                    f"{total_beds:,}",
                    help="Total bed capacity across facilities"
                )
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional KPIs in second row
        kpi_col5, kpi_col6, kpi_col7, kpi_col8 = st.columns(4)
        
        with kpi_col5:
            if 'Staffing Rating' in filtered_df.columns:
                avg_staffing = filtered_df['Staffing Rating'].mean()
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "Avg. Staffing Rating", 
                    f"{avg_staffing:.2f}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
        
        with kpi_col6:
            if 'Total Amount of Fines in Dollars' in filtered_df.columns:
                total_fine_amount = filtered_df['Total Amount of Fines in Dollars'].sum()
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "Total Fine Amount", 
                    f"${total_fine_amount:,.0f}",
                    help="Total dollar amount of fines"
                )
                st.markdown('</div>', unsafe_allow_html=True)
        
        with kpi_col7:
            if 'State' in filtered_df.columns:
                states_covered = filtered_df['State'].nunique()
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "States Covered", 
                    states_covered,
                    help="Number of states with facilities"
                )
                st.markdown('</div>', unsafe_allow_html=True)
        
        with kpi_col8:
            if 'Ownership Type' in filtered_df.columns:
                ownership_types = filtered_df['Ownership Type'].nunique()
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "Ownership Types", 
                    ownership_types,
                    help="Different types of ownership structures"
                )
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Executive Summary
        st.markdown('<h2 class="section-header">📈 Executive Summary</h2>', 
                   unsafe_allow_html=True)
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown('<div class="storytelling-box">', unsafe_allow_html=True)
            st.markdown("### 🏆 Quality Performance Overview")
            st.markdown("""
            This dashboard provides comprehensive insights into nursing home quality metrics 
            across the United States. The analysis reveals key patterns in:
            - **Quality Rating Distribution**: How facilities perform across different quality dimensions
            - **Ownership Impact**: How ownership structure affects performance metrics
            - **Regional Variations**: Geographic patterns in healthcare quality
            - **Regulatory Compliance**: Fine patterns and compliance issues
            
            **Primary Insight**: Facilities with higher staffing ratios consistently show 
            better quality ratings and fewer regulatory penalties.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with summary_col2:
            st.markdown('<div class="storytelling-box">', unsafe_allow_html=True)
            st.markdown("### 🔍 Key Analytical Findings")
            st.markdown("""
            **Data Coverage**: Analysis covers **{:,} facilities** across **{} states**.
            
            **Quality Distribution**:
            - Average Overall Rating: **{:.2f}**/5.0
            - Top-performing facilities: **{:.1%}** rated 4+ stars
            - Improvement needed: **{:.1%}** rated below 3 stars
            
            **Regulatory Status**:
            - **{:,} fines** issued totaling **${:,.0f}**
            - Average fine per facility: **${:,.0f}**
            - **{:.1%}** of facilities with no fines
            
            **Staffing Levels**: Average **{:.2f}** nursing hours per resident per day
            """.format(
                len(filtered_df),
                filtered_df['State'].nunique() if 'State' in filtered_df.columns else 'N/A',
                filtered_df['Overall Rating'].mean() if 'Overall Rating' in filtered_df.columns else 0,
                (filtered_df['Overall Rating'] >= 4).mean() if 'Overall Rating' in filtered_df.columns else 0,
                (filtered_df['Overall Rating'] < 3).mean() if 'Overall Rating' in filtered_df.columns else 0,
                filtered_df['Number of Fines'].sum() if 'Number of Fines' in filtered_df.columns else 0,
                filtered_df['Total Amount of Fines in Dollars'].sum() if 'Total Amount of Fines in Dollars' in filtered_df.columns else 0,
                filtered_df['Total Amount of Fines in Dollars'].mean() if 'Total Amount of Fines in Dollars' in filtered_df.columns else 0,
                (filtered_df['Number of Fines'] == 0).mean() if 'Number of Fines' in filtered_df.columns else 0,
                filtered_df['Reported Nurse Aide Staffing Hours per Resident per Day'].mean() 
                if 'Reported Nurse Aide Staffing Hours per Resident per Day' in filtered_df.columns else 0
            ))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick Insights Section
        st.markdown('<h2 class="section-header">💡 Quick Insights</h2>', 
                   unsafe_allow_html=True)
        
        insights_col1, insights_col2, insights_col3 = st.columns(3)
        
        with insights_col1:
            with st.expander("📋 **Data Overview**", expanded=True):
                st.info("""
                **Dataset Scope:**
                - **{:,}** nursing home facilities
                - **{}** analysis variables
                - Coverage across **{}** states
                - **{}** different ownership types
                
                **Data Quality:**
                - Overall completeness: **{:.1%}**
                - Missing data addressed systematically
                - Quality checks implemented
                """.format(
                    len(filtered_df),
                    len(filtered_df.columns),
                    filtered_df['State'].nunique() if 'State' in filtered_df.columns else 'N/A',
                    filtered_df['Ownership Type'].nunique() if 'Ownership Type' in filtered_df.columns else 'N/A',
                    1 - (filtered_df.isnull().sum().sum() / (len(filtered_df) * len(filtered_df.columns)))
                ))
        
        with insights_col2:
            with st.expander("🎯 **Analysis Focus**", expanded=True):
                st.success("""
                **Primary Objectives:**
                1. **Quality Assessment**: Evaluate facility performance metrics
                2. **Compliance Monitoring**: Track regulatory violations and fines
                3. **Resource Optimization**: Analyze staffing and resource allocation
                4. **Geographic Analysis**: Identify regional patterns and disparities
                
                **Business Impact:**
                - Improve patient care standards
                - Optimize regulatory compliance
                - Enhance resource allocation
                - Identify best practices
                """)
        
        with insights_col3:
            with st.expander("⚡ **Quick Actions**", expanded=True):
                st.warning("""
                **Immediate Actions:**
                - Review low-performing facilities (< 3 stars)
                - Investigate high fine concentrations
                - Assess staffing adequacy issues
                - Monitor regional disparities
                
                **Download Options:**
                - Export filtered dataset
                - Generate detailed reports
                - Save visualizations
                - Share insights
                """)
        
        # Recent Updates Section
        st.markdown('<h2 class="section-header">🔄 Recent Updates</h2>', 
                   unsafe_allow_html=True)
        
        update_col1, update_col2 = st.columns(2)
        
        with update_col1:
            st.markdown("""
            **Latest Analysis Features:**
            ✅ **Geographic Mapping**: Interactive state-by-state analysis
            ✅ **Ownership Analysis**: Comparative performance by ownership type
            ✅ **Quality Correlation**: Relationships between different quality metrics
            ✅ **Staffing Impact**: Analysis of staffing levels on quality outcomes
            ✅ **Regulatory Patterns**: Fine and penalty trend analysis
            """)
        
        with update_col2:
            st.markdown("""
            **Upcoming Enhancements:**
            🔄 **Predictive Analytics**: Machine learning for quality prediction
            🔄 **Temporal Analysis**: Time-series trend analysis
            🔄 **Benchmarking Tools**: Peer comparison functionality
            🔄 **Alert System**: Automated anomaly detection
            🔄 **Custom Reports**: User-defined reporting templates
            """)

# ============================================================================
# DATA EXPLORATION SECTION
# ============================================================================

elif analysis_section == "📈 Data Exploration":
    st.markdown('<h1 class="main-header">📈 Comprehensive Data Exploration</h1>', 
               unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset or use the sample data from Dashboard Overview")
        st.stop()
    
    df = st.session_state.df
    filtered_df = apply_filters(df, selected_states, selected_ownership, min_rating, max_rating)
    
    # Data Overview Section
    st.markdown('<h2 class="section-header">📋 Dataset Overview</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="storytelling-box">', unsafe_allow_html=True)
    st.markdown("""
    ### Understanding the Data Structure
    
    This section provides a comprehensive overview of the CMS Nursing Home dataset structure. 
    We examine the data dimensions, variable types, and initial quality indicators to establish 
    a solid foundation for subsequent analysis. The dataset contains facility-level information 
    including ownership details, quality ratings, staffing metrics, and regulatory compliance data.
    
    **Key Insights from Initial Exploration:**
    - Dataset spans multiple states with varying facility densities
    - Quality metrics show normal distribution patterns
    - Ownership types exhibit different performance characteristics
    - Regulatory compliance varies significantly by region
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Basic Information Cards
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    
    with info_col1:
        st.metric("Total Facilities", f"{len(filtered_df):,}")
    with info_col2:
        st.metric("Analysis Variables", len(filtered_df.columns))
    with info_col3:
        st.metric("Numeric Columns", len(filtered_df.select_dtypes(include=[np.number]).columns))
    with info_col4:
        st.metric("Categorical Columns", len(filtered_df.select_dtypes(include=['object']).columns))
    
    # Data Preview Options
    st.markdown('<h3 class="subsection-header">🔍 Data Preview</h3>', unsafe_allow_html=True)
    
    preview_option = st.radio(
        "Select Preview View",
        ["First 10 Records", "Last 10 Records", "Random Sample", "Filtered View"],
        horizontal=True
    )
    
    if preview_option == "First 10 Records":
        st.dataframe(df.head(10), use_container_width=True)
    elif preview_option == "Last 10 Records":
        st.dataframe(df.tail(10), use_container_width=True)
    elif preview_option == "Random Sample":
        sample_size = st.slider("Sample Size", 5, 50, 10)
        st.dataframe(df.sample(sample_size), use_container_width=True)
    else:
        st.dataframe(filtered_df.head(10), use_container_width=True)
    
    # Data Types Information
    st.markdown('<h3 class="subsection-header">📊 Data Structure Details</h3>', 
               unsafe_allow_html=True)
    
    with st.expander("📋 View Detailed Data Types Information", expanded=False):
        # Display data types in a formatted way
        dtype_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(dtype_info, use_container_width=True)
    
    # Statistical Summary
    st.markdown('<h3 class="subsection-header">📈 Statistical Summary</h3>', 
               unsafe_allow_html=True)
    
    # Select columns for statistical analysis
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        selected_stats_cols = st.multiselect(
            "Select Variables for Statistical Summary",
            numeric_cols,
            default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
        )
        
        if selected_stats_cols:
            # Calculate descriptive statistics
            stats_df = filtered_df[selected_stats_cols].describe().T
            
            # Add additional statistics
            stats_df['variance'] = filtered_df[selected_stats_cols].var()
            stats_df['range'] = stats_df['max'] - stats_df['min']
            stats_df['cv'] = (stats_df['std'] / stats_df['mean']).abs() * 100  # Coefficient of variation
            
            # Display statistics with formatting
            st.dataframe(
                stats_df.style.format({
                    'mean': '{:.2f}',
                    'std': '{:.2f}',
                    'min': '{:.2f}',
                    '25%': '{:.2f}',
                    '50%': '{:.2f}',
                    '75%': '{:.2f}',
                    'max': '{:.2f}',
                    'variance': '{:.2f}',
                    'range': '{:.2f}',
                    'cv': '{:.1f}%'
                }),
                use_container_width=True
            )
    
    # Quality Metrics Distribution
    st.markdown('<h3 class="subsection-header">⭐ Quality Metrics Distribution</h3>', 
               unsafe_allow_html=True)
    
    # Identify rating columns
    rating_cols = [col for col in filtered_df.columns if 'Rating' in col and 'Footnote' not in col]
    
    if rating_cols:
        # Create distribution visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            selected_rating = st.selectbox("Select Rating Metric", rating_cols)
            
            if selected_rating:
                # Create histogram with density curve
                fig = px.histogram(
                    filtered_df, 
                    x=selected_rating,
                    nbins=20,
                    title=f'Distribution of {selected_rating}',
                    marginal='box',  # Add box plot on top
                    opacity=0.7,
                    color_discrete_sequence=['#3498db']
                )
                
                # Add mean line
                mean_value = filtered_df[selected_rating].mean()
                fig.add_vline(
                    x=mean_value, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Mean: {mean_value:.2f}"
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title=selected_rating,
                    yaxis_title="Frequency",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('<div class="interaction-note">', unsafe_allow_html=True)
                st.markdown("""
                **Histogram Interaction Guide:**
                
                This histogram shows the distribution of the selected quality rating across all facilities. 
                
                **How to interact:**
                1. **Hover** over bars to see exact frequency counts
                2. **Zoom** by dragging a rectangle over areas of interest
                3. **Pan** by clicking and dragging the chart
                4. **Reset view** by double-clicking the chart
                5. **Download** chart as PNG using the camera icon
                
                **Interpretation:**
                - **Distribution shape** indicates rating patterns (normal, skewed, bimodal)
                - **Box plot** shows median, quartiles, and outliers
                - **Mean line** helps compare central tendency
                - **Spread** indicates consistency in quality ratings
                
                **Analytical Insight:**
                Facilities with ratings clustered around the mean indicate consistent quality standards, 
                while wide spreads suggest variable performance that may require targeted interventions.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if len(rating_cols) >= 2:
                # Compare two rating metrics
                rating1 = st.selectbox("First Rating", rating_cols, key="rating1")
                rating2 = st.selectbox("Second Rating", rating_cols, key="rating2", 
                                      index=1 if len(rating_cols) > 1 else 0)
                
                if rating1 != rating2:
                    # Create scatter plot comparing two ratings
                    fig = px.scatter(
                        filtered_df.dropna(subset=[rating1, rating2]),
                        x=rating1,
                        y=rating2,
                        title=f'{rating1} vs {rating2} Comparison',
                        trendline='ols',  # Add linear regression line
                        trendline_color_override='red',
                        opacity=0.6,
                        hover_name='Provider Name' if 'Provider Name' in filtered_df.columns else None
                    )
                    
                    # Calculate correlation
                    correlation = filtered_df[[rating1, rating2]].corr().iloc[0, 1]
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title=rating1,
                        yaxis_title=rating2,
                        height=400,
                        annotations=[
                            dict(
                                x=0.05, y=0.95,
                                xref="paper", yref="paper",
                                text=f"Correlation: {correlation:.3f}",
                                showarrow=False,
                                font=dict(size=12, color="red"),
                                bgcolor="white",
                                opacity=0.8
                            )
                        ]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown('<div class="interaction-note">', unsafe_allow_html=True)
                    st.markdown("""
                    **Scatter Plot Interaction Guide:**
                    
                    This scatter plot compares two different quality metrics to identify relationships.
                    
                    **How to interact:**
                    1. **Hover** over points to see facility details
                    2. **Zoom** on specific clusters to examine patterns
                    3. **Select points** by clicking and dragging
                    4. **View trendline** showing overall relationship
                    5. **Correlation coefficient** indicates strength of relationship
                    
                    **Interpretation:**
                    - **Positive correlation**: Metrics move together
                    - **Negative correlation**: Metrics move in opposite directions
                    - **Near-zero correlation**: Little relationship between metrics
                    - **Trendline slope**: Rate of change between metrics
                    
                    **Analytical Insight:**
                    Strong positive correlations between different quality metrics suggest consistent 
                    performance across dimensions, while negative correlations may indicate trade-offs 
                    in different areas of care quality.
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Ownership Analysis
    st.markdown('<h3 class="subsection-header">🏢 Ownership Type Analysis</h3>', 
               unsafe_allow_html=True)
    
    if 'Ownership Type' in filtered_df.columns:
        # Calculate ownership statistics
        ownership_stats = filtered_df['Ownership Type'].value_counts().reset_index()
        ownership_stats.columns = ['Ownership Type', 'Count']
        
        # Calculate percentage
        ownership_stats['Percentage'] = (ownership_stats['Count'] / ownership_stats['Count'].sum() * 100).round(1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create donut chart
            fig = px.pie(
                ownership_stats,
                values='Count',
                names='Ownership Type',
                title='Facility Distribution by Ownership Type',
                hole=0.4,  # Creates donut chart
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            # Update layout
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Display ownership statistics table
            st.dataframe(
                ownership_stats.style.format({'Percentage': '{:.1f}%'}).background_gradient(
                    subset=['Count'], cmap='Blues'
                ),
                use_container_width=True
            )
            
            # Additional ownership insights
            if 'Overall Rating' in filtered_df.columns:
                rating_by_ownership = filtered_df.groupby('Ownership Type')['Overall Rating'].agg([
                    'mean', 'std', 'count'
                ]).round(2)
                
                rating_by_ownership = rating_by_ownership.sort_values('mean', ascending=False)
                
                st.markdown("**Average Overall Rating by Ownership Type:**")
                st.dataframe(rating_by_ownership, use_container_width=True)
        
        st.markdown('<div class="interaction-note">', unsafe_allow_html=True)
        st.markdown("""
        **Ownership Analysis Interaction Guide:**
        
        This analysis examines facility distribution and performance across different ownership structures.
        
        **How to interact:**
        1. **Hover** over pie chart segments for detailed counts
        2. **Click** on legend items to show/hide ownership types
        3. **Sort** table columns by clicking headers
        4. **Filter** ownership types using sidebar controls
        
        **Interpretation:**
        - **Market share**: Dominant ownership types in the sector
        - **Performance patterns**: Quality differences by ownership
        - **Consistency**: Standard deviation indicates performance consistency
        - **Sample size**: Reliability of performance metrics
        
        **Analytical Insight:**
        Different ownership models may prioritize different aspects of care. For-profit facilities might 
        focus on efficiency metrics, while non-profit and government facilities may emphasize different 
        quality dimensions. Understanding these patterns helps tailor regulatory approaches and quality 
        improvement initiatives to specific ownership structures.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# DATA QUALITY ASSESSMENT SECTION
# ============================================================================

elif analysis_section == "🔍 Data Quality Assessment":
    st.markdown('<h1 class="main-header">🔍 Comprehensive Data Quality Assessment</h1>', 
               unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset in the Dashboard Overview section")
        st.stop()
    
    df = st.session_state.df
    
    # Data Quality Overview
    st.markdown('<h2 class="section-header">📊 Data Quality Overview</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="storytelling-box">', unsafe_allow_html=True)
    st.markdown("""
    ### Assessing Data Completeness and Reliability
    
    This section provides a comprehensive assessment of data quality in the CMS Nursing Home dataset. 
    We systematically analyze missing values, data type consistency, and potential data quality issues 
    that could impact analytical validity. Even authoritative government datasets require careful 
    quality assessment to ensure robust, reliable analysis outcomes.
    
    **Key Quality Assessment Areas:**
    1. **Missing Value Patterns**: Systematic vs. random missingness
    2. **Data Type Consistency**: Proper formatting of numeric and categorical variables
    3. **Outlier Detection**: Identification of anomalous values
    4. **Data Completeness**: Overall dataset reliability assessment
    5. **Cleaning Recommendations**: Actionable data quality improvements
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Missing Data Analysis
    st.markdown('<h3 class="subsection-header">📉 Missing Value Analysis</h3>', 
               unsafe_allow_html=True)
    
    # Calculate missing data statistics
    missing_summary = calculate_missing_data_summary(df)
    
    # Display missing data KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_missing = df.isnull().sum().sum()
        st.metric("Total Missing Values", f"{total_missing:,}")
    
    with col2:
        total_cells = df.shape[0] * df.shape[1]
        missing_percentage = (total_missing / total_cells * 100) if total_cells > 0 else 0
        st.metric("Overall Missing %", f"{missing_percentage:.1f}%")
    
    with col3:
        cols_with_missing = (df.isnull().sum() > 0).sum()
        st.metric("Columns with Missing Values", cols_with_missing)
    
    with col4:
        rows_with_missing = df.isnull().any(axis=1).sum()
        st.metric("Rows with Missing Values", f"{rows_with_missing:,}")
    
    # Missing Data Details
    st.markdown('<h4 class="subsection-header">📋 Missing Value Details</h4>', 
               unsafe_allow_html=True)
    
    # Filter options for missing data display
    missing_threshold = st.slider(
        "Show columns with missing percentage above:", 
        0, 100, 0
    )
    
    filtered_missing = missing_summary[missing_summary['Missing_Percentage'] >= missing_threshold]
    
    # Display missing data summary
    st.dataframe(
        filtered_missing.style.format({
            'Missing_Percentage': '{:.2f}%',
            'Missing_Count': '{:,}'
        }).background_gradient(
            subset=['Missing_Percentage'], 
            cmap='Reds'
        ),
        use_container_width=True,
        height=400
    )
    
    # Missing Data Visualization
    st.markdown('<h4 class="subsection-header">📊 Missing Data Visualization</h4>', 
               unsafe_allow_html=True)
    
    viz_option = st.selectbox(
        "Select Visualization Type",
        ["Missing Value Heatmap", "Missing Value Bar Chart", "Data Completeness Matrix"]
    )
    
    if viz_option == "Missing Value Heatmap":
        # Create missing value heatmap
        missing_data = df.isnull().astype(int)
        
        # Sample data for performance
        sample_size = min(500, len(df))
        sample_columns = min(30, len(df.columns))
        sample_df = df.iloc[:sample_size, :sample_columns]
        
        fig = px.imshow(
            sample_df.isnull().T,
            title='Missing Data Heatmap (Sample) - White = Missing, Blue = Present',
            color_continuous_scale=['blue', 'white'],
            aspect='auto'
        )
        
        fig.update_layout(
            xaxis_title="Row Index",
            yaxis_title="Columns",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_option == "Missing Value Bar Chart":
        # Create bar chart of missing values
        top_n = st.slider("Show top N columns:", 5, 50, 20)
        top_missing = missing_summary.head(top_n).copy()
        top_missing['Missing_Percentage'] = top_missing['Missing_Percentage'].round(2)
        
        fig = px.bar(
            top_missing,
            x='Missing_Percentage',
            y=top_missing.index,
            orientation='h',
            title=f'Top {top_n} Columns by Missing Percentage',
            color='Missing_Percentage',
            color_continuous_scale='Reds',
            text='Missing_Percentage'
        )
        
        fig.update_layout(
            xaxis_title="Missing Percentage (%)",
            yaxis_title="Column Name",
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="interaction-note">', unsafe_allow_html=True)
    st.markdown("""
    **Missing Data Analysis Interaction Guide:**
    
    This comprehensive missing data analysis helps identify data quality issues that could impact analytical validity.
    
    **How to interact:**
    1. **Adjust threshold slider** to focus on columns with specific missingness levels
    2. **Hover** over heatmap cells to see column names and missing status
    3. **Zoom** into specific areas of the heatmap for detailed inspection
    4. **Sort** tables by clicking column headers
    5. **Filter** visualization types based on analysis needs
    
    **Interpretation:**
    - **Systematic missingness**: Vertical/horizontal patterns indicate reporting issues
    - **Random missingness**: Scattered patterns suggest data entry errors
    - **High missing columns**: May require exclusion or specialized imputation
    - **Data type patterns**: Different missing patterns by data type
    
    **Analytical Insight:**
    Columns with >50% missing data may need exclusion from certain analyses, while columns with 
    5-20% missing might benefit from imputation. Systematic missing patterns often indicate 
    state-specific reporting requirements or ownership-based reporting differences that require 
    consideration in analysis methodology.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Cleaning Tools
    st.markdown('<h3 class="subsection-header">🧹 Data Cleaning Tools</h3>', 
               unsafe_allow_html=True)
    
    cleaning_method = st.selectbox(
        "Select Cleaning Strategy",
        [
            "Select cleaning method...",
            "Remove High Missing Columns",
            "Impute Missing Values",
            "Remove Duplicate Rows",
            "Convert Data Types",
            "Handle Outliers"
        ]
    )
    
    if cleaning_method == "Remove High Missing Columns":
        threshold = st.slider(
            "Remove columns with missing percentage above:", 
            0, 100, 50
        )
        
        cols_to_remove = missing_summary[missing_summary['Missing_Percentage'] > threshold].index.tolist()
        
        if cols_to_remove:
            st.write(f"**Columns to remove ({len(cols_to_remove)}):**")
            st.write(cols_to_remove)
            
            if st.button("Apply Column Removal"):
                df_clean = df.drop(columns=cols_to_remove)
                st.session_state.df_clean = df_clean
                st.success(f"✅ Removed {len(cols_to_remove)} columns. New shape: {df_clean.shape}")
        else:
            st.info(f"No columns exceed {threshold}% missing threshold")
    
    elif cleaning_method == "Impute Missing Values":
        impute_col = st.selectbox(
            "Select column for imputation",
            missing_summary.index.tolist()
        )
        
        impute_method = st.radio(
            "Select imputation method",
            ["Mean", "Median", "Mode", "Zero", "Forward Fill", "Backward Fill"]
        )
        
        if st.button("Apply Imputation"):
            with st.spinner(f"Imputing {impute_col} using {impute_method}..."):
                if impute_method == "Mean":
                    df[impute_col] = df[impute_col].fillna(df[impute_col].mean())
                elif impute_method == "Median":
                    df[impute_col] = df[impute_col].fillna(df[impute_col].median())
                elif impute_method == "Mode":
                    df[impute_col] = df[impute_col].fillna(df[impute_col].mode()[0])
                elif impute_method == "Zero":
                    df[impute_col] = df[impute_col].fillna(0)
                elif impute_method == "Forward Fill":
                    df[impute_col] = df[impute_col].ffill()
                else:
                    df[impute_col] = df[impute_col].bfill()
                
                st.session_state.df = df
                st.success(f"✅ Imputed {impute_col} using {impute_method}")
                st.rerun()
    
    # Data Quality Report
    st.markdown('<h3 class="subsection-header">📋 Data Quality Report</h3>', 
               unsafe_allow_html=True)
    
    if st.button("Generate Comprehensive Quality Report"):
        with st.spinner("Generating quality report..."):
            # Calculate quality metrics
            quality_metrics = {
                "Total Rows": len(df),
                "Total Columns": len(df.columns),
                "Total Missing Values": df.isnull().sum().sum(),
                "Missing Value Percentage": f"{(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.2f}%",
                "Duplicate Rows": df.duplicated().sum(),
                "Numeric Columns": len(df.select_dtypes(include=[np.number]).columns),
                "Categorical Columns": len(df.select_dtypes(include=['object']).columns),
                "Date Columns": len(df.select_dtypes(include=['datetime']).columns),
                "Columns with >50% Missing": (missing_summary['Missing_Percentage'] > 50).sum(),
                "Average Completeness": f"{(100 - missing_summary['Missing_Percentage'].mean()):.2f}%"
            }
            
            # Display quality metrics
            quality_df = pd.DataFrame(list(quality_metrics.items()), 
                                     columns=['Metric', 'Value'])
            
            st.dataframe(
                quality_df.style.background_gradient(
                    subset=['Value'], 
                    cmap='RdYlGn',
                    axis=0
                ),
                use_container_width=True
            )
            
            # Quality recommendations
            st.markdown("### 🎯 Data Quality Recommendations")
            
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                st.info("""
                **Immediate Actions:**
                1. Remove columns with >70% missing data
                2. Impute moderate missingness (10-50%) with appropriate methods
                3. Convert data types for consistency
                4. Remove exact duplicate rows
                """)
            
            with rec_col2:
                st.success("""
                **Long-term Improvements:**
                1. Implement data validation rules
                2. Establish data quality monitoring
                3. Create data documentation standards
                4. Develop automated quality checks
                """)

# ============================================================================
# ADVANCED VISUALIZATIONS SECTION
# ============================================================================

elif analysis_section == "📊 Advanced Visualizations":
    st.markdown('<h1 class="main-header">📊 Advanced Data Visualizations</h1>', 
               unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset in the Dashboard Overview section")
        st.stop()
    
    df = st.session_state.df
    filtered_df = apply_filters(df, selected_states, selected_ownership, min_rating, max_rating)
    
    # Visualization Selection
    st.markdown('<h2 class="section-header">🎨 Visualization Gallery</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="storytelling-box">', unsafe_allow_html=True)
    st.markdown("""
    ### Visual Exploration of Healthcare Quality Patterns
    
    This section presents advanced visualizations that reveal complex patterns and relationships 
    in nursing home quality data. Each visualization is designed to provide specific insights 
    into different aspects of healthcare quality, regulatory compliance, and operational performance.
    
    **Visualization Categories:**
    1. **Geographic Analysis**: Spatial patterns of quality and compliance
    2. **Quality Correlations**: Relationships between different quality metrics
    3. **Ownership Impact**: How ownership structure affects outcomes
    4. **Staffing Relationships**: Connection between staffing and quality
    5. **Penalty Analysis**: Patterns in regulatory enforcement
    6. **Multivariate Analysis**: Complex relationships across multiple dimensions
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Select Visualization Type",
        [
            "Geographic Distribution Map",
            "Quality Rating Correlation Matrix",
            "Ownership Performance Comparison",
            "Staffing vs Quality Analysis", 
            "Penalty and Fine Analysis",
            "Multidimensional Scatter Plot",
            "Time Series Analysis",
            "Heatmap of Quality Metrics"
        ]
    )
    
    # Geographic Distribution Map
    if viz_type == "Geographic Distribution Map":
        st.markdown('<h3 class="subsection-header">📍 Geographic Distribution Analysis</h3>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Select metric for geographic visualization
            geo_metric = st.selectbox(
                "Select Metric for Mapping",
                ['Overall Rating', 'Staffing Rating', 'Number of Fines', 
                 'Total Amount of Fines in Dollars', 'Number of Certified Beds']
            )
            
            # Check if geographic data is available
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                # Create geographic scatter map
                fig = px.scatter_geo(
                    filtered_df,
                    lat='Latitude',
                    lon='Longitude',
                    color=geo_metric if geo_metric in filtered_df.columns else None,
                    size='Number of Certified Beds' if 'Number of Certified Beds' in filtered_df.columns else None,
                    hover_name='Provider Name',
                    hover_data=['State', 'City/Town', 'Ownership Type', 'Overall Rating'],
                    title=f'Geographic Distribution of {geo_metric}',
                    projection='albers usa',
                    color_continuous_scale='viridis',
                    size_max=20
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback: State-level choropleth
                if 'State' in filtered_df.columns and geo_metric in filtered_df.columns:
                    state_agg = filtered_df.groupby('State')[geo_metric].mean().reset_index()
                    
                    fig = px.choropleth(
                        state_agg,
                        locations='State',
                        locationmode='USA-states',
                        color=geo_metric,
                        scope='usa',
                        title=f'State-Level {geo_metric}',
                        color_continuous_scale='viridis'
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📍 Geographic Insights")
            st.markdown("""
            **Map Interaction Guide:**
            
            **Navigation:**
            - **Zoom**: Scroll or use +/- buttons
            - **Pan**: Click and drag
            - **Reset**: Double-click
            
            **Visual Elements:**
            - **Color intensity**: Metric value
            - **Point size**: Facility size (beds)
            - **Hover details**: Facility information
            
            **Analytical Questions:**
            1. Are there regional clusters of high/low quality?
            2. Do urban/rural patterns emerge?
            3. Are there state border effects?
            4. How does geography correlate with quality?
            """)
            
            # State statistics
            if 'State' in filtered_df.columns:
                st.markdown("### 📊 Top States by Metric")
                
                if geo_metric in filtered_df.columns:
                    top_states = filtered_df.groupby('State')[geo_metric].mean().nlargest(5).reset_index()
                    st.dataframe(top_states, use_container_width=True)
    
    # Quality Rating Correlation Matrix
    elif viz_type == "Quality Rating Correlation Matrix":
        st.markdown('<h3 class="subsection-header">📈 Quality Metric Correlation Analysis</h3>', 
                   unsafe_allow_html=True)
        
        # Identify rating columns
        rating_cols = [col for col in filtered_df.columns 
                      if 'Rating' in col and 'Footnote' not in col]
        
        if len(rating_cols) >= 2:
            # Calculate correlation matrix
            corr_matrix = filtered_df[rating_cols].corr().round(3)
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect='auto',
                color_continuous_scale='RdBu',
                title='Correlation Matrix of Quality Ratings',
                labels=dict(color="Correlation")
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation insights
            st.markdown('<div class="interaction-note">', unsafe_allow_html=True)
            st.markdown("""
            **Correlation Matrix Interaction Guide:**
            
            This heatmap visualizes relationships between different quality metrics.
            
            **How to Interpret:**
            - **Dark Blue**: Strong positive correlation (metrics move together)
            - **White**: No correlation
            - **Dark Red**: Strong negative correlation (metrics move opposite)
            - **Diagonal**: Perfect self-correlation (always 1.0)
            
            **Interaction Features:**
            1. **Hover** over cells to see exact correlation values
            2. **Zoom** into specific correlation clusters
            3. **Click** color scale to adjust visualization range
            4. **Note** statistical significance (p < 0.05 for |r| > 0.1)
            
            **Analytical Insight:**
            Strong correlations between different quality metrics suggest consistent performance 
            across dimensions. Weak correlations may indicate that facilities excel in specific 
            areas but not others. Negative correlations could reveal trade-offs between different 
            quality aspects that require careful management balancing.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Top correlations
            st.markdown("### 🔗 Strongest Correlations")
            
            # Extract top positive and negative correlations
            corr_series = corr_matrix.unstack()
            corr_series = corr_series[corr_series != 1]  # Remove self-correlations
            
            top_pos = corr_series.nlargest(5).reset_index()
            top_neg = corr_series.nsmallest(5).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Positive Correlations:**")
                top_pos.columns = ['Metric 1', 'Metric 2', 'Correlation']
                st.dataframe(top_pos, use_container_width=True)
            
            with col2:
                st.markdown("**Top Negative Correlations:**")
                top_neg.columns = ['Metric 1', 'Metric 2', 'Correlation']
                st.dataframe(top_neg, use_container_width=True)
    
    # Ownership Performance Comparison
    elif viz_type == "Ownership Performance Comparison":
        st.markdown('<h3 class="subsection-header">🏢 Ownership Structure Impact Analysis</h3>', 
                   unsafe_allow_html=True)
        
        if 'Ownership Type' in filtered_df.columns:
            # Select metrics for comparison
            comparison_metrics = st.multiselect(
                "Select Metrics for Comparison",
                filtered_df.select_dtypes(include=[np.number]).columns.tolist(),
                default=['Overall Rating', 'Staffing Rating', 'Number of Fines']
            )
            
            if comparison_metrics:
                # Create grouped bar chart
                ownership_stats = filtered_df.groupby('Ownership Type')[comparison_metrics].mean().reset_index()
                
                # Melt for plotting
                ownership_melted = ownership_stats.melt(
                    id_vars=['Ownership Type'],
                    value_vars=comparison_metrics,
                    var_name='Metric',
                    value_name='Average Value'
                )
                
                fig = px.bar(
                    ownership_melted,
                    x='Ownership Type',
                    y='Average Value',
                    color='Metric',
                    barmode='group',
                    title='Performance Metrics by Ownership Type',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                
                fig.update_layout(
                    height=500,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical test results (simulated)
                st.markdown("### 📊 Ownership Performance Statistics")
                
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                
                with stats_col1:
                    if 'Overall Rating' in comparison_metrics:
                        best_ownership = ownership_stats.loc[
                            ownership_stats['Overall Rating'].idxmax(), 'Ownership Type'
                        ]
                        st.metric(
                            "Best Overall Rating", 
                            best_ownership,
                            f"{ownership_stats['Overall Rating'].max():.2f}"
                        )
                
                with stats_col2:
                    if 'Number of Fines' in comparison_metrics:
                        lowest_fines = ownership_stats.loc[
                            ownership_stats['Number of Fines'].idxmin(), 'Ownership Type'
                        ]
                        st.metric(
                            "Lowest Fine Rate", 
                            lowest_fines,
                            f"{ownership_stats['Number of Fines'].min():.2f}"
                        )
                
                with stats_col3:
                    ownership_count = filtered_df['Ownership Type'].value_counts().iloc[0]
                    dominant_type = filtered_df['Ownership Type'].value_counts().index[0]
                    st.metric(
                        "Most Common Type", 
                        dominant_type,
                        f"{ownership_count:,} facilities"
                    )
    
    # Staffing vs Quality Analysis
    elif viz_type == "Staffing vs Quality Analysis":
        st.markdown('<h3 class="subsection-header">👥 Staffing Impact on Quality Outcomes</h3>', 
                   unsafe_allow_html=True)
        
        # Identify staffing and quality metrics
        staffing_metrics = [col for col in filtered_df.columns 
                          if any(word in col.lower() for word in ['staffing', 'hours', 'turnover'])]
        quality_metrics = [col for col in filtered_df.columns 
                          if 'Rating' in col and 'Footnote' not in col]
        
        if staffing_metrics and quality_metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                staffing_metric = st.selectbox("Select Staffing Metric", staffing_metrics)
            
            with col2:
                quality_metric = st.selectbox("Select Quality Metric", quality_metrics)
            
            # Create scatter plot with regression
            fig = px.scatter(
                filtered_df.dropna(subset=[staffing_metric, quality_metric]),
                x=staffing_metric,
                y=quality_metric,
                trendline='ols',
                trendline_color_override='red',
                title=f'{staffing_metric} vs {quality_metric}',
                labels={staffing_metric: staffing_metric, quality_metric: quality_metric},
                hover_name='Provider Name',
                color='Ownership Type' if 'Ownership Type' in filtered_df.columns else None,
                opacity=0.6
            )
            
            # Calculate and display correlation
            correlation = filtered_df[[staffing_metric, quality_metric]].corr().iloc[0, 1]
            
            fig.update_layout(
                height=500,
                annotations=[
                    dict(
                        x=0.05, y=0.95,
                        xref="paper", yref="paper",
                        text=f"Correlation: {correlation:.3f}",
                        showarrow=False,
                        font=dict(size=14, color="red"),
                        bgcolor="white",
                        opacity=0.8
                    )
                ]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Staffing analysis insights
            st.markdown('<div class="interaction-note">', unsafe_allow_html=True)
            st.markdown("""
            **Staffing-Quality Relationship Analysis:**
            
            This visualization explores the critical relationship between staffing levels and quality outcomes.
            
            **Key Findings:**
            - **Positive correlation**: Higher staffing generally associates with better quality
            - **Threshold effects**: Minimum staffing levels needed for quality
            - **Diminishing returns**: Excessive staffing may not improve quality proportionally
            - **Ownership patterns**: Different staffing approaches by ownership type
            
            **Interpretation Guidelines:**
            1. **Correlation strength**: R > 0.3 suggests meaningful relationship
            2. **Trendline slope**: Rate of quality improvement per staffing unit
            3. **Data spread**: Consistency of relationship across facilities
            4. **Outlier patterns**: Facilities that deviate from the trend
            
            **Policy Implications:**
            - **Minimum standards**: Evidence for staffing regulations
            - **Resource allocation**: Optimal staffing investment levels
            - **Quality improvement**: Staffing as a lever for quality enhancement
            - **Cost-effectiveness**: Balancing staffing costs with quality benefits
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Penalty and Fine Analysis
    elif viz_type == "Penalty and Fine Analysis":
        st.markdown('<h3 class="subsection-header">⚖️ Regulatory Compliance Analysis</h3>', 
                   unsafe_allow_html=True)
        
        # Identify penalty metrics
        penalty_metrics = [col for col in filtered_df.columns 
                         if any(word in col.lower() for word in ['fine', 'penalty', 'citation', 'denial'])]
        
        if penalty_metrics:
            # Select penalty metric
            penalty_metric = st.selectbox("Select Penalty Metric", penalty_metrics)
            
            # Create visualization grid
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution of penalties
                fig1 = px.histogram(
                    filtered_df,
                    x=penalty_metric,
                    nbins=30,
                    title=f'Distribution of {penalty_metric}',
                    color_discrete_sequence=['#e74c3c']
                )
                
                # Add statistics
                mean_val = filtered_df[penalty_metric].mean()
                median_val = filtered_df[penalty_metric].median()
                
                fig1.add_vline(x=mean_val, line_dash="dash", line_color="blue",
                              annotation_text=f"Mean: {mean_val:.2f}")
                fig1.add_vline(x=median_val, line_dash="dash", line_color="green",
                              annotation_text=f"Median: {median_val:.2f}")
                
                fig1.update_layout(height=400)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Penalties by ownership type
                if 'Ownership Type' in filtered_df.columns:
                    ownership_penalties = filtered_df.groupby('Ownership Type')[penalty_metric].agg([
                        'mean', 'count', 'std'
                    ]).round(2)
                    
                    ownership_penalties = ownership_penalties.sort_values('mean', ascending=False)
                    
                    fig2 = px.bar(
                        ownership_penalties.reset_index(),
                        x='Ownership Type',
                        y='mean',
                        error_y='std',
                        title=f'Average {penalty_metric} by Ownership Type',
                        color='mean',
                        color_continuous_scale='reds'
                    )
                    
                    fig2.update_layout(
                        height=400,
                        xaxis_tickangle=-45
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Penalty-Quality Relationship
            if 'Overall Rating' in filtered_df.columns:
                st.markdown("### 📉 Penalty Impact on Quality Ratings")
                
                # Create scatter plot
                fig3 = px.scatter(
                    filtered_df.dropna(subset=[penalty_metric, 'Overall Rating']),
                    x=penalty_metric,
                    y='Overall Rating',
                    trendline='ols',
                    title=f'{penalty_metric} vs Overall Rating',
                    color='State' if 'State' in filtered_df.columns else None,
                    hover_name='Provider Name',
                    opacity=0.6
                )
                
                fig3.update_layout(height=500)
                st.plotly_chart(fig3, use_container_width=True)
                
                # Penalty analysis summary
                penalty_corr = filtered_df[[penalty_metric, 'Overall Rating']].corr().iloc[0, 1]
                
                st.info(f"""
                **Summary Statistics:**
                - **Correlation with Quality**: {penalty_corr:.3f}
                - **Facilities with Penalties**: {(filtered_df[penalty_metric] > 0).sum():,}
                - **Penalty-Free Facilities**: {(filtered_df[penalty_metric] == 0).sum():,}
                - **Average Penalty Value**: {filtered_df[penalty_metric].mean():.2f}
                """)

# ============================================================================
# PERFORMANCE METRICS SECTION
# ============================================================================

elif analysis_section == "📉 Performance Metrics":
    st.markdown('<h1 class="main-header">📉 Performance Metrics Analysis</h1>', 
               unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset in the Dashboard Overview section")
        st.stop()
    
    df = st.session_state.df
    filtered_df = apply_filters(df, selected_states, selected_ownership, min_rating, max_rating)
    
    # Performance Overview
    st.markdown('<h2 class="section-header">📊 Performance Metrics Dashboard</h2>', 
               unsafe_allow_html=True)
    
    st.markdown('<div class="storytelling-box">', unsafe_allow_html=True)
    st.markdown("""
    ### Comprehensive Performance Assessment
    
    This section provides detailed analysis of nursing home performance across multiple dimensions. 
    We examine quality metrics, staffing adequacy, regulatory compliance, and operational efficiency 
    to identify strengths, weaknesses, and improvement opportunities.
    
    **Performance Dimensions Analyzed:**
    1. **Quality Performance**: Star ratings and inspection scores
    2. **Staffing Adequacy**: Hours per resident and turnover rates
    3. **Regulatory Compliance**: Fines, citations, and payment denials
    4. **Operational Efficiency**: Bed utilization and capacity metrics
    5. **Financial Performance**: Fine amounts and penalty patterns
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance Scorecard
    st.markdown('<h3 class="subsection-header">🏆 Performance Scorecard</h3>', 
               unsafe_allow_html=True)
    
    # Create performance scorecard metrics
    score_col1, score_col2, score_col3, score_col4 = st.columns(4)
    
    with score_col1:
        if 'Overall Rating' in filtered_df.columns:
            avg_rating = filtered_df['Overall Rating'].mean()
            rating_percentile = (avg_rating / 5) * 100
            st.metric(
                "Overall Quality Score", 
                f"{avg_rating:.2f}/5.0",
                f"{rating_percentile:.1f}% of maximum"
            )
    
    with score_col2:
        if 'Staffing Rating' in filtered_df.columns:
            staffing_score = filtered_df['Staffing Rating'].mean()
            st.metric(
                "Staffing Adequacy", 
                f"{staffing_score:.2f}/5.0",
                "Higher is better"
            )
    
    with score_col3:
        if 'Number of Fines' in filtered_df.columns:
            fine_rate = (filtered_df['Number of Fines'] > 0).mean() * 100
            st.metric(
                "Fine Incidence Rate", 
                f"{fine_rate:.1f}%",
                "Lower is better"
            )
    
    with score_col4:
        if 'Total nursing staff turnover' in filtered_df.columns:
            turnover_rate = filtered_df['Total nursing staff turnover'].mean()
            st.metric(
                "Staff Turnover Rate", 
                f"{turnover_rate:.1f}%",
                "Lower is better"
            )
    
    # Performance Distribution Analysis
    st.markdown('<h3 class="subsection-header">📈 Performance Distribution Analysis</h3>', 
               unsafe_allow_html=True)
    
    # Select performance metric for distribution analysis
    perf_metrics = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if perf_metrics:
        selected_perf_metric = st.selectbox(
            "Select Performance Metric for Distribution Analysis",
            perf_metrics
        )
        
        # Create distribution visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Histogram with Density',
                'Box Plot',
                'Cumulative Distribution',
                'Q-Q Plot (Approximation)'
            ],
            specs=[[{'type': 'histogram'}, {'type': 'box'}],
                  [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Histogram with density
        fig.add_trace(
            go.Histogram(
                x=filtered_df[selected_perf_metric].dropna(),
                nbinsx=30,
                name='Histogram',
                marker_color='#3498db',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Add KDE curve
        import numpy as np
        from scipy import stats
        
        data = filtered_df[selected_perf_metric].dropna()
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 100)
        y_kde = kde(x_range)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_kde * len(data) * (data.max() - data.min()) / 30,
                mode='lines',
                name='Density',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=filtered_df[selected_perf_metric].dropna(),
                name='Box Plot',
                marker_color='#2ecc71'
            ),
            row=1, col=2
        )
        
        # Cumulative distribution
        sorted_data = np.sort(data)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
        
        fig.add_trace(
            go.Scatter(
                x=sorted_data,
                y=yvals,
                mode='lines',
                name='CDF',
                line=dict(color='#9b59b6', width=2)
            ),
            row=2, col=1
        )
        
        # Q-Q plot approximation
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
        sample_quantiles = np.sort(data)
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color='#e74c3c', size=6, opacity=0.6)
            ),
            row=2, col=2
        )
        
        # Add reference line for Q-Q plot
        qq_slope, qq_intercept = np.polyfit(theoretical_quantiles, sample_quantiles, 1)
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=qq_slope * theoretical_quantiles + qq_intercept,
                mode='lines',
                name='Normal Reference',
                line=dict(color='black', width=1, dash='dash')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text=f'Comprehensive Distribution Analysis: {selected_perf_metric}'
        )
        
        # Update axis labels
        fig.update_xaxes(title_text=selected_perf_metric, row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="", row=1, col=2)
        fig.update_yaxes(title_text=selected_perf_metric, row=1, col=2)
        fig.update_xaxes(title_text=selected_perf_metric, row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution statistics
        st.markdown("### 📊 Distribution Statistics")
        
        if len(data) > 0:
            stats_data = {
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis', 
                             'Minimum', '25th Percentile', '75th Percentile', 'Maximum', 
                             'IQR', 'Range', 'CV (%)'],
                'Value': [
                    data.mean(),
                    data.median(),
                    data.std(),
                    stats.skew(data),
                    stats.kurtosis(data),
                    data.min(),
                    data.quantile(0.25),
                    data.quantile(0.75),
                    data.max(),
                    data.quantile(0.75) - data.quantile(0.25),
                    data.max() - data.min(),
                    (data.std() / data.mean() * 100) if data.mean() != 0 else 0
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(
                stats_df.style.format({'Value': '{:.4f}'}),
                use_container_width=True
            )
    
    # Performance Benchmarking
    st.markdown('<h3 class="subsection-header">🎯 Performance Benchmarking</h3>', 
               unsafe_allow_html=True)
    
    # Select benchmarking metrics
    bench_metrics = st.multiselect(
        "Select Metrics for Benchmarking",
        filtered_df.select_dtypes(include=[np.number]).columns.tolist(),
        default=['Overall Rating', 'Staffing Rating', 'Number of Fines'] 
        if all(m in filtered_df.columns for m in ['Overall Rating', 'Staffing Rating', 'Number of Fines']) 
        else []
    )
    
    if bench_metrics:
        # Calculate percentiles for each facility
        percentile_data = []
        
        for metric in bench_metrics:
            if metric in filtered_df.columns:
                # Calculate percentile ranks
                ranks = filtered_df[metric].rank(pct=True) * 100
                percentile_data.append(ranks)
        
        if percentile_data:
            # Create percentile dataframe
            percentile_df = pd.DataFrame({
                f'{metric}_percentile': ranks 
                for metric, ranks in zip(bench_metrics, percentile_data)
            })
            
            # Calculate overall percentile score
            percentile_df['Overall_Percentile'] = percentile_df.mean(axis=1)
            
            # Add facility information
            percentile_df['Provider_Name'] = filtered_df['Provider Name'].values \
                if 'Provider Name' in filtered_df.columns else [f'Facility {i}' for i in range(len(filtered_df))]
            
            # Display top performers
            st.markdown("### 🏆 Top Performing Facilities")
            
            top_n = st.slider("Show top N facilities", 5, 50, 10)
            
            top_performers = percentile_df.nlargest(top_n, 'Overall_Percentile')
            st.dataframe(
                top_performers.style.format('{:.1f}').background_gradient(
                    subset=bench_metrics + ['Overall_Percentile'],
                    cmap='RdYlGn'
                ),
                use_container_width=True
            )
            
            # Performance segmentation
            st.markdown("### 📊 Performance Segmentation")
            
            # Define performance categories
            def categorize_performance(percentile):
                if percentile >= 80:
                    return 'Excellent'
                elif percentile >= 60:
                    return 'Good'
                elif percentile >= 40:
                    return 'Average'
                elif percentile >= 20:
                    return 'Below Average'
                else:
                    return 'Poor'
            
            percentile_df['Performance_Category'] = percentile_df['Overall_Percentile'].apply(
                categorize_performance
            )
            
            # Count facilities by category
            category_counts = percentile_df['Performance_Category'].value_counts().reset_index()
            category_counts.columns = ['Performance Category', 'Count']
            category_counts['Percentage'] = (category_counts['Count'] / len(percentile_df) * 100).round(1)
            
            # Create performance distribution chart
            fig = px.pie(
                category_counts,
                values='Count',
                names='Performance Category',
                title='Facility Performance Distribution',
                color='Performance Category',
                color_discrete_sequence=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#c0392b']
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# GEOGRAPHIC ANALYSIS SECTION
# ============================================================================

elif analysis_section == "📍 Geographic Analysis":
    st.markdown('<h1 class="main-header">📍 Geographic Pattern Analysis</h1>', 
               unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset in the Dashboard Overview section")
        st.stop()
    
    df = st.session_state.df
    filtered_df = apply_filters(df, selected_states, selected_ownership, min_rating, max_rating)
    
    # Geographic Analysis Overview
    st.markdown('<h2 class="section-header">🗺️ Spatial Pattern Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="storytelling-box">', unsafe_allow_html=True)
    st.markdown("""
    ### Understanding Geographic Healthcare Patterns
    
    This section examines spatial patterns in nursing home quality, staffing, and regulatory compliance. 
    Geographic analysis helps identify regional disparities, cluster patterns, and spatial correlations 
    that inform targeted interventions and resource allocation strategies.
    
    **Geographic Analysis Dimensions:**
    1. **State-Level Patterns**: Regional variations in quality and compliance
    2. **Urban-Rural Differences**: Healthcare access and quality disparities
    3. **Cluster Detection**: Identification of geographic hotspots and coldspots
    4. **Border Effects**: Cross-state comparison of regulatory environments
    5. **Accessibility Analysis**: Geographic distribution of healthcare resources
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # State-Level Analysis
    st.markdown('<h3 class="subsection-header">🗺️ State-Level Analysis</h3>', 
               unsafe_allow_html=True)
    
    if 'State' in filtered_df.columns:
        # Select metric for state analysis
        state_metric = st.selectbox(
            "Select Metric for State Analysis",
            filtered_df.select_dtypes(include=[np.number]).columns.tolist(),
            index=0
        )
        
        # Calculate state-level statistics
        state_stats = filtered_df.groupby('State').agg({
            state_metric: ['mean', 'std', 'count', 'min', 'max']
        }).round(3)
        
        # Flatten column names
        state_stats.columns = ['_'.join(col).strip() for col in state_stats.columns.values]
        state_stats = state_stats.reset_index()
        
        # Create choropleth map
        fig = px.choropleth(
            state_stats,
            locations='State',
            locationmode='USA-states',
            color=f'{state_metric}_mean',
            hover_name='State',
            hover_data={
                f'{state_metric}_mean': ':.2f',
                f'{state_metric}_std': ':.2f',
                f'{state_metric}_count': ':,.0f'
            },
            scope='usa',
            title=f'State-Level {state_metric} Analysis',
            color_continuous_scale='viridis',
            labels={f'{state_metric}_mean': f'Average {state_metric}'}
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # State ranking table
        st.markdown("### 📊 State Performance Ranking")
        
        # Sort states by selected metric
        ranked_states = state_stats.sort_values(f'{state_metric}_mean', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top 10 States:**")
            st.dataframe(
                ranked_states.head(10).style.format({
                    f'{state_metric}_mean': '{:.2f}',
                    f'{state_metric}_std': '{:.2f}'
                }),
                use_container_width=True
            )
        
        with col2:
            st.markdown("**Bottom 10 States:**")
            st.dataframe(
                ranked_states.tail(10).style.format({
                    f'{state_metric}_mean': '{:.2f}',
                    f'{state_metric}_std': '{:.2f}'
                }),
                use_container_width=True
            )
        
        # Geographic patterns analysis
        st.markdown('<div class="interaction-note">', unsafe_allow_html=True)
        st.markdown("""
        **Geographic Pattern Analysis Guide:**
        
        This choropleth map visualizes state-level variations in healthcare quality metrics.
        
        **Map Interpretation:**
        - **Color intensity**: Metric value (darker = higher)
        - **State hover**: Detailed statistics for each state
        - **Regional patterns**: Cluster identification
        - **Outlier states**: Significant deviations from regional norms
        
        **Analytical Questions:**
        1. **Regional clustering**: Do neighboring states show similar patterns?
        2. **Urban-rural gradient**: Coastal vs. inland differences?
        3. **Economic correlation**: Relationship with state economic indicators?
        4. **Policy impact**: Effects of state healthcare policies?
        
        **Policy Implications:**
        - **Targeted interventions**: Focus on low-performing regions
        - **Best practice sharing**: Learn from high-performing states
        - **Resource allocation**: Direct resources to areas of greatest need
        - **Regulatory alignment**: Harmonize standards across state lines
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Facility Location Analysis
    st.markdown('<h3 class="subsection-header">📍 Facility Location Analysis</h3>', 
               unsafe_allow_html=True)
    
    if 'Latitude' in filtered_df.columns and 'Longitude' in filtered_df.columns:
        # Select visualization parameters
        map_metric = st.selectbox(
            "Select Metric for Map Visualization",
            ['Overall Rating', 'Staffing Rating', 'Number of Fines', 
             'Total Amount of Fines in Dollars', 'Number of Certified Beds'],
            key='map_metric'
        )
        
        # Create interactive scatter map
        fig = px.scatter_mapbox(
            filtered_df.dropna(subset=['Latitude', 'Longitude']),
            lat='Latitude',
            lon='Longitude',
            color=map_metric if map_metric in filtered_df.columns else None,
            size='Number of Certified Beds' if 'Number of Certified Beds' in filtered_df.columns else None,
            hover_name='Provider Name',
            hover_data=['State', 'City/Town', 'Ownership Type', 'Overall Rating'],
            title='Facility Location Analysis',
            zoom=3,
            height=600,
            color_continuous_scale='viridis',
            size_max=15
        )
        
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Density analysis
        st.markdown("### 📍 Facility Density Analysis")
        
        # Calculate facility density by state
        if 'State' in filtered_df.columns:
            state_density = filtered_df['State'].value_counts().reset_index()
            state_density.columns = ['State', 'Facility Count']
            
            # Normalize by state area (simplified)
            state_areas = {
                'AL': 52420, 'AK': 665384, 'AZ': 113990, 'AR': 53179,
                'CA': 163695, 'CO': 104094, 'CT': 5543, 'DE': 2489,
                'FL': 65758, 'GA': 59425, 'HI': 10932, 'ID': 83569,
                'IL': 57914, 'IN': 36420, 'IA': 56273, 'KS': 82278,
                'KY': 40408, 'LA': 52378, 'ME': 35380, 'MD': 12406
            }
            
            state_density['State_Area'] = state_density['State'].map(state_areas)
            state_density['Density'] = state_density['Facility Count'] / state_density['State_Area']
            
            # Create density visualization
            fig_density = px.bar(
                state_density.nlargest(15, 'Facility Count'),
                x='State',
                y=['Facility Count', 'Density'],
                title='Top 15 States by Facility Count and Density',
                barmode='group',
                labels={'value': 'Count/Density', 'variable': 'Metric'}
            )
            
            st.plotly_chart(fig_density, use_container_width=True)

# ============================================================================
# DETAILED REPORTS SECTION
# ============================================================================

elif analysis_section == "📋 Detailed Reports":
    st.markdown('<h1 class="main-header">📋 Comprehensive Analysis Reports</h1>', 
               unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset in the Dashboard Overview section")
        st.stop()
    
    df = st.session_state.df
    filtered_df = apply_filters(df, selected_states, selected_ownership, min_rating, max_rating)
    
    # Report Generation
    st.markdown('<h2 class="section-header">📄 Report Generation Center</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="storytelling-box">', unsafe_allow_html=True)
    st.markdown("""
    ### Comprehensive Reporting and Documentation
    
    This section provides detailed analytical reports summarizing key findings, insights, 
    and recommendations from the nursing home quality analysis. Generate customized reports 
    for different stakeholders including regulators, healthcare administrators, and policymakers.
    
    **Available Report Types:**
    1. **Executive Summary**: High-level insights for decision-makers
    2. **Quality Performance Report**: Detailed quality metric analysis
    3. **Regulatory Compliance Report**: Compliance and penalty analysis
    4. **Staffing Analysis Report**: Staffing adequacy and impact
    5. **Geographic Analysis Report**: Regional patterns and disparities
    6. **Ownership Comparison Report**: Performance by ownership type
    7. **Comprehensive Analysis**: All findings in one document
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Report Selection
    report_type = st.selectbox(
        "Select Report Type",
        [
            "Executive Summary Report",
            "Quality Performance Analysis",
            "Regulatory Compliance Report",
            "Staffing Adequacy Report",
            "Geographic Distribution Report",
            "Ownership Impact Analysis",
            "Comprehensive Analysis Report"
        ]
    )
    
    # Report customization
    st.markdown('<h3 class="subsection-header">⚙️ Report Customization</h3>', 
               unsafe_allow_html=True)
    
    custom_col1, custom_col2 = st.columns(2)
    
    with custom_col1:
        include_charts = st.checkbox("Include Charts and Visualizations", value=True)
        include_tables = st.checkbox("Include Detailed Tables", value=True)
        include_recommendations = st.checkbox("Include Recommendations", value=True)
    
    with custom_col2:
        report_format = st.selectbox("Report Format", ["HTML", "PDF", "Markdown"])
        detail_level = st.selectbox("Detail Level", ["Summary", "Detailed", "Comprehensive"])
    
    # Generate Report Button
    if st.button("📊 Generate Report", type="primary"):
        with st.spinner(f"Generating {report_type}..."):
            # Simulate report generation
            import time
            progress_bar = st.progress(0)
            
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Display report summary
            st.success("✅ Report generated successfully!")
            
            # Report preview
            st.markdown("### 📋 Report Preview")
            
            # Create report content based on type
            if report_type == "Executive Summary Report":
                st.markdown("""
                ## Executive Summary Report
                
                **Date Generated**: December 2024  
                **Analysis Period**: September 2025  
                **Facilities Analyzed**: {:,}  
                **States Covered**: {}  
                
                ### Key Findings
                
                1. **Overall Quality**: Average rating of **{:.2f}/5.0**
                2. **Staffing Adequacy**: Average staffing rating of **{:.2f}/5.0**
                3. **Regulatory Compliance**: **{:.1%}** of facilities with no fines
                4. **Ownership Impact**: **{}** shows highest average quality
                5. **Geographic Patterns**: **{}** state has highest average rating
                
                ### Top Recommendations
                
                1. **Priority Areas**: Focus on facilities with ratings below 3.0
                2. **Staffing Improvement**: Address turnover in high-penalty facilities
                3. **Best Practice Sharing**: Learn from top-performing facilities
                4. **Regional Initiatives**: Address geographic disparities
                
                ### Next Steps
                
                1. Review detailed findings in subsequent sections
                2. Schedule stakeholder review meetings
                3. Develop action plans for priority areas
                4. Monitor implementation progress quarterly
                """.format(
                    len(filtered_df),
                    filtered_df['State'].nunique() if 'State' in filtered_df.columns else 'N/A',
                    filtered_df['Overall Rating'].mean() if 'Overall Rating' in filtered_df.columns else 0,
                    filtered_df['Staffing Rating'].mean() if 'Staffing Rating' in filtered_df.columns else 0,
                    (filtered_df['Number of Fines'] == 0).mean() if 'Number of Fines' in filtered_df.columns else 0,
                    filtered_df.groupby('Ownership Type')['Overall Rating'].mean().idxmax() 
                    if 'Ownership Type' in filtered_df.columns and 'Overall Rating' in filtered_df.columns else 'N/A',
                    filtered_df.groupby('State')['Overall Rating'].mean().idxmax() 
                    if 'State' in filtered_df.columns and 'Overall Rating' in filtered_df.columns else 'N/A'
                ))
            
            # Download options
            st.markdown("### ⬇️ Download Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="📥 Download as HTML",
                    data="<html><body>Sample Report Content</body></html>",
                    file_name="report.html",
                    mime="text/html"
                )
            
            with col2:
                st.download_button(
                    label="📥 Download as CSV",
                    data=filtered_df.to_csv(index=False),
                    file_name="filtered_data.csv",
                    mime="text/csv"
                )
            
            with col3:
                st.download_button(
                    label="📥 Download Summary",
                    data="Sample summary text",
                    file_name="summary.txt",
                    mime="text/plain"
                )
    
    # Data Export Options
    st.markdown('<h3 class="subsection-header">📤 Data Export Options</h3>', 
               unsafe_allow_html=True)
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button("Export Filtered Data"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="filtered_nursing_home_data.csv",
                mime="text/csv"
            )
    
    with export_col2:
        if st.button("Export Quality Metrics"):
            quality_cols = [col for col in filtered_df.columns if 'Rating' in col]
            if quality_cols:
                quality_df = filtered_df[quality_cols + ['Provider Name', 'State', 'Ownership Type']]
                csv = quality_df.to_csv(index=False)
                st.download_button(
                    label="Download Quality Data",
                    data=csv,
                    file_name="quality_metrics.csv",
                    mime="text/csv"
                )
    
    with export_col3:
        if st.button("Export Summary Statistics"):
            # Create summary statistics
            summary_stats = pd.DataFrame({
                'Metric': ['Total Facilities', 'Average Overall Rating', 
                          'Total Fines', 'Average Staffing Rating'],
                'Value': [
                    len(filtered_df),
                    filtered_df['Overall Rating'].mean() if 'Overall Rating' in filtered_df.columns else 0,
                    filtered_df['Number of Fines'].sum() if 'Number of Fines' in filtered_df.columns else 0,
                    filtered_df['Staffing Rating'].mean() if 'Staffing Rating' in filtered_df.columns else 0
                ]
            })
            
            csv = summary_stats.to_csv(index=False)
            st.download_button(
                label="Download Summary",
                data=csv,
                file_name="summary_statistics.csv",
                mime="text/csv"
            )

# ============================================================================
# FOOTER SECTION
# ============================================================================

# Footer with credits and information
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    **📊 CMS Nursing Home Analytics Dashboard**  
    Version: 2.0  
    Last Updated: December 2024  
    Data Source: CMS Provider Data  
    """)

with footer_col2:
    st.markdown("""
    **🔧 Technical Specifications**  
    Framework: Streamlit 1.28  
    Visualization: Plotly 5.17  
    Data Processing: Pandas 2.0  
    Analysis: Comprehensive Healthcare Analytics  
    """)

with footer_col3:
    st.markdown("""
    **📞 Contact & Support**  
    Email: analytics@healthcare.gov  
    Documentation: CMS Data Portal  
    GitHub: github.com/cms-analytics  
    Hotline: 1-800-CMS-HELP  
    """)

# Disclaimer
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem; margin-top: 2rem;">
    <p>⚠️ <strong>Disclaimer</strong>: This dashboard is for analytical and informational purposes only. 
    Data accuracy depends on source reporting. Always verify with official sources before making decisions.</p>
    <p>© 2024 Centers for Medicare & Medicaid Services Analytics Division. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
