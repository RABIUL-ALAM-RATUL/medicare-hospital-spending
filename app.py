# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Try to import matplotlib (optional)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("Matplotlib not available. Some visualizations may be limited.")

# Try to import missingno (optional)
try:
    import missingno as msno
    MISSINGNO_AVAILABLE = True
except ImportError:
    MISSINGNO_AVAILABLE = False
    st.warning("Missingno library not available. Missing data matrix visualization will use Plotly alternative.")

# Page configuration
st.set_page_config(
    page_title="Medicare Hospital Spending Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    .storytelling {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
    }
    .interaction-note {
        background-color: #e8f4fc;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 class="main-header">🏥 Medicare Hospital Spending by Claim (USA)</h1>', unsafe_allow_html=True)

# Sidebar for navigation and filters
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2107/2107845.png", width=100)
    st.title("Navigation")
    analysis_section = st.selectbox(
        "Select Analysis Section",
        ["🏠 Introduction", "📊 Data Overview", "🧹 Data Cleaning", "📈 Visualizations", "🔍 Deep Dive Analysis"]
    )
    
    st.markdown("---")
    st.title("Filters")
    
    # Load data first to get available states
    @st.cache_data
    def load_sample_data():
        """Load sample data for demonstration"""
        # Create sample data structure similar to CMS dataset
        np.random.seed(42)
        n_facilities = 1000
        
        sample_data = {
            'Provider Name': [f'Facility {i}' for i in range(n_facilities)],
            'State': np.random.choice(['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA'], n_facilities),
            'Ownership Type': np.random.choice(['For profit - Corporation', 'Non-profit - Church', 
                                                 'Government - State', 'For profit - Individual'], n_facilities),
            'Overall Rating': np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, np.nan], n_facilities, p=[0.1, 0.2, 0.3, 0.25, 0.1, 0.05]),
            'Health Inspection Rating': np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], n_facilities),
            'Staffing Rating': np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], n_facilities),
            'QM Rating': np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], n_facilities),
            'Number of Fines': np.random.poisson(1, n_facilities),
            'Total Amount of Fines in Dollars': np.random.exponential(10000, n_facilities),
            'Reported Nurse Aide Staffing Hours per Resident per Day': np.random.normal(2.5, 0.5, n_facilities),
            'Total nursing staff turnover': np.random.normal(50, 15, n_facilities),
            'Latitude': np.random.uniform(25, 49, n_facilities),
            'Longitude': np.random.uniform(-124, -67, n_facilities)
        }
        return pd.DataFrame(sample_data)
    
    # Try to load uploaded file, otherwise use sample data
    if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
        df = st.session_state.df
    else:
        df = load_sample_data()
        st.session_state.df = df
    
    # Get unique values for filters
    all_states = sorted(df['State'].unique())
    selected_states = st.multiselect(
        "Select States",
        options=all_states,
        default=all_states[:3] if len(all_states) > 3 else all_states
    )
    
    ownership_types = sorted(df['Ownership Type'].dropna().unique())
    selected_ownership = st.multiselect(
        "Select Ownership Types",
        options=ownership_types,
        default=ownership_types[:2] if len(ownership_types) > 2 else ownership_types
    )
    
    # Rating filter
    min_rating, max_rating = st.slider(
        "Overall Rating Range",
        min_value=1.0,
        max_value=5.0,
        value=(1.0, 5.0),
        step=0.5
    )
    
    # File upload in sidebar
    st.markdown("---")
    st.title("Data Upload")
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

# Apply filters to data
def apply_filters(df, selected_states, selected_ownership, min_rating, max_rating):
    filtered_df = df.copy()
    
    if selected_states:
        filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]
    
    if selected_ownership:
        filtered_df = filtered_df[filtered_df['Ownership Type'].isin(selected_ownership)]
    
    if 'Overall Rating' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['Overall Rating'] >= min_rating) & 
            (filtered_df['Overall Rating'] <= max_rating)
        ]
    
    return filtered_df

# Main content based on selected section
if analysis_section == "🏠 Introduction":
    st.markdown('<h2 class="section-header">Project Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Facilities", f"{len(st.session_state.df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Dataset Variables", f"{len(st.session_state.df.columns)}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Data Source", "CMS Provider Data")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Business Problem
    
    This analysis addresses critical questions about nursing home quality and regulatory compliance across the United States:
    
    1. **What factors drive changes in provider quality scores over time?**
    2. **How does facility ownership type affect patient care outcomes?**
    3. **What regional patterns exist in regulatory compliance and penalties?**
    
    ### Key Metrics (KPIs)
    
    - **Average Quality Ratings** (Overall, Health Inspection, Staffing, QM)
    - **Incident Reports** per facility
    - **Regional Performance Scores**
    - **Penalty Frequency and Amounts**
    - **Staffing Ratios and Turnover Rates**
    
    ### Data Source
    
    The dataset originates from the **CMS Provider Data** (Centers for Medicare & Medicaid Services), 
    representing nursing home provider information, covering facilities nationwide.
    """)
    
    # Quick insights
    st.markdown('<h3 class="section-header">Quick Insights</h3>', unsafe_allow_html=True)
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.info("""
        **Data Scope:**
        - Sample nursing home facilities
        - Multiple quality metrics
        - National coverage across states
        - Mix of ownership types
        """)
    
    with insights_col2:
        st.success("""
        **Analysis Value:**
        - Identify high-performing facilities
        - Detect compliance issues
        - Optimize resource allocation
        - Improve patient care standards
        """)
        
    # Instructions for users
    with st.expander("How to Use This Dashboard"):
        st.markdown("""
        1. **Upload Data**: Use the file uploader in the sidebar to upload your CMS dataset
        2. **Apply Filters**: Use the sidebar filters to focus on specific states or ownership types
        3. **Navigate Sections**: Use the dropdown to explore different analysis sections
        4. **Interact with Visualizations**: Hover, zoom, and click on charts for detailed insights
        5. **Download Results**: Export filtered data or visualizations as needed
        """)

elif analysis_section == "📊 Data Overview":
    st.markdown('<h2 class="section-header">Data Loading and Initial Exploration</h2>', unsafe_allow_html=True)
    
    # File upload
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, low_memory=False)
        st.session_state.df = df
        st.session_state.uploaded_file = uploaded_file
        st.success(f"Dataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")
    else:
        df = st.session_state.df
        st.info("Using sample data. Upload your CSV file in the sidebar for full analysis.")
    
    # Apply filters
    filtered_df = apply_filters(df, selected_states, selected_ownership, min_rating, max_rating)
    
    # Dataset overview
    st.markdown('<div class="storytelling">The dataset represents nursing home provider information, containing records of facilities across the United States. This initial overview helps us understand the data structure, variable types, and overall quality before diving into detailed analysis.</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", filtered_df.shape[0])
    with col2:
        st.metric("Total Columns", filtered_df.shape[1])
    with col3:
        st.metric("Filtered Facilities", f"{len(filtered_df):,}")
    
    # Data preview
    st.markdown('<h3 class="section-header">Data Preview</h3>', unsafe_allow_html=True)
    
    preview_option = st.radio(
        "Preview Options",
        ["First 10 Rows", "Last 10 Rows", "Random Sample", "Filtered Data"]
    )
    
    if preview_option == "First 10 Rows":
        st.dataframe(df.head(10), use_container_width=True)
    elif preview_option == "Last 10 Rows":
        st.dataframe(df.tail(10), use_container_width=True)
    elif preview_option == "Random Sample":
        st.dataframe(df.sample(10), use_container_width=True)
    else:
        st.dataframe(filtered_df.head(10), use_container_width=True)
    
    # Data types information
    st.markdown('<h3 class="section-header">Data Structure</h3>', unsafe_allow_html=True)
    
    with st.expander("View Data Types and Information"):
        buffer = []
        df.info(buf=buffer)
        st.text("".join(buffer))
    
    # Basic statistics
    st.markdown('<h3 class="section-header">Basic Statistics</h3>', unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        selected_numeric = st.multiselect(
            "Select numeric columns for statistics",
            options=numeric_cols,
            default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
        )
        if selected_numeric:
            st.dataframe(df[selected_numeric].describe(), use_container_width=True)
    
    # Missing values overview
    st.markdown('<h3 class="section-header">Missing Values Overview</h3>', unsafe_allow_html=True)
    
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage': missing_percentage
    }).sort_values('Missing Percentage', ascending=False)
    
    st.dataframe(missing_df[missing_df['Missing Count'] > 0].head(10), use_container_width=True)

elif analysis_section == "🧹 Data Cleaning":
    st.markdown('<h2 class="section-header">Data Cleaning and Preprocessing</h2>', unsafe_allow_html=True)
    
    df = st.session_state.df
    
    st.markdown('<div class="storytelling">Even authoritative government datasets require careful cleaning to ensure analysis validity. This section systematically addresses missing values, data type inconsistencies, and prepares the dataset for robust analytical modeling.</div>', unsafe_allow_html=True)
    
    # Missing values analysis
    st.markdown('<h3 class="section-header">Missing Values Analysis</h3>', unsafe_allow_html=True)
    
    # Calculate missing values
    missing_summary = pd.DataFrame({
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
    })
    
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
    missing_summary = missing_summary.sort_values('Missing_Percentage', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Missing Values", f"{df.isnull().sum().sum():,}")
    with col2:
        st.metric("Columns with Missing Values", len(missing_summary))
    
    # Display missing values summary
    st.dataframe(missing_summary.head(20), use_container_width=True)
    
    # Missing data visualization
    st.markdown('<div class="storytelling">The missing data visualization reveals systematic patterns in data completeness. This helps identify columns that may require imputation or exclusion from certain analyses.</div>', unsafe_allow_html=True)
    
    if MISSINGNO_AVAILABLE and MATPLOTLIB_AVAILABLE:
        if st.button("Generate Missing Data Matrix"):
            fig, ax = plt.subplots(figsize=(12, 8))
            msno.matrix(df.sample(min(500, len(df))), sparkline=False, fontsize=10, 
                       color=(0.25, 0.65, 0.65), ax=ax)
            ax.set_title('Missing Data Matrix\n(White = Missing | Dark = Present)', 
                        fontsize=14, weight='bold', pad=20)
            ax.set_xlabel('Columns', fontsize=12)
            ax.set_ylabel('Facilities (rows)', fontsize=12)
            st.pyplot(fig)
    else:
        # Alternative visualization using Plotly
        st.info("Using Plotly alternative for missing data visualization")
        
        # Sample data for visualization
        sample_df = df.sample(min(100, len(df)))
        sample_cols = df.columns[:min(20, len(df.columns))]
        
        # Create missingness heatmap
        missing_data = sample_df[sample_cols].isnull().astype(int)
        
        fig = go.Figure(data=go.Heatmap(
            z=missing_data.values.T,
            x=sample_df.index,
            y=sample_cols,
            colorscale=[[0, 'rgb(64, 166, 166)'], [1, 'white']],
            showscale=True,
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Missing Data Pattern (Sample)',
            xaxis_title="Facility Index",
            yaxis_title="Variables",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="interaction-note">This visualization shows missing data patterns across variables. White cells indicate missing values, while colored cells indicate present data. Hover over the visualization to see which variables have the most missing data. The pattern can reveal whether missing values are random (scattered white cells) or systematic (vertical white bands), which informs our data cleaning strategy. Systematic missingness often indicates reporting differences between states or ownership types.</div>', unsafe_allow_html=True)
    
    # Data cleaning options
    st.markdown('<h3 class="section-header">Data Cleaning Options</h3>', unsafe_allow_html=True)
    
    cleaning_option = st.selectbox(
        "Select Cleaning Strategy",
        ["Select option...", "Remove High Missing Columns (>50%)", 
         "Impute Missing Values", "Convert Data Types", "Remove Duplicates"]
    )
    
    if cleaning_option == "Remove High Missing Columns (>50%)":
        threshold = st.slider("Missing Percentage Threshold", 0, 100, 50)
        cols_to_drop = missing_summary[missing_summary['Missing_Percentage'] > threshold].index.tolist()
        
        if cols_to_drop:
            st.write(f"Columns to remove ({len(cols_to_drop)}):")
            st.write(cols_to_drop)
            
            if st.button("Apply Removal"):
                df_clean = df.drop(columns=cols_to_drop)
                st.session_state.df_clean = df_clean
                st.success(f"Removed {len(cols_to_drop)} columns. Remaining columns: {df_clean.shape[1]}")
        else:
            st.info("No columns exceed the selected threshold")
    
    elif cleaning_option == "Impute Missing Values":
        if len(missing_summary) > 0:
            col_to_impute = st.selectbox("Select Column to Impute", missing_summary.index.tolist())
            
            impute_method = st.radio(
                "Imputation Method",
                ["Mean", "Median", "Mode", "Zero", "Forward Fill", "Backward Fill"]
            )
            
            if st.button("Apply Imputation"):
                if impute_method == "Mean":
                    df[col_to_impute] = df[col_to_impute].fillna(df[col_to_impute].mean())
                elif impute_method == "Median":
                    df[col_to_impute] = df[col_to_impute].fillna(df[col_to_impute].median())
                elif impute_method == "Mode":
                    df[col_to_impute] = df[col_to_impute].fillna(df[col_to_impute].mode()[0])
                elif impute_method == "Zero":
                    df[col_to_impute] = df[col_to_impute].fillna(0)
                elif impute_method == "Forward Fill":
                    df[col_to_impute] = df[col_to_impute].ffill()
                else:
                    df[col_to_impute] = df[col_to_impute].bfill()
                
                st.success(f"Imputed missing values in {col_to_impute} using {impute_method} method")
                st.session_state.df = df
        else:
            st.info("No missing values to impute")
    
    elif cleaning_option == "Remove Duplicates":
        duplicates = df.duplicated().sum()
        st.metric("Duplicate Rows Found", duplicates)
        
        if duplicates > 0 and st.button("Remove Duplicates"):
            df_clean = df.drop_duplicates()
            st.session_state.df = df_clean
            st.success(f"Removed {duplicates} duplicate rows. New row count: {len(df_clean)}")

elif analysis_section == "📈 Visualizations":
    st.markdown('<h2 class="section-header">Data Visualizations</h2>', unsafe_allow_html=True)
    
    df = st.session_state.df
    
    # Apply filters
    filtered_df = apply_filters(df, selected_states, selected_ownership, min_rating, max_rating)
    
    # Visualization selection
    viz_options = [
        "Geographic Distribution",
        "Ownership Type Analysis", 
        "Quality Ratings Distribution",
        "Staffing Metrics",
        "Penalties and Fines Analysis",
        "Correlation Analysis"
    ]
    
    selected_viz = st.selectbox("Select Visualization", viz_options)
    
    if selected_viz == "Geographic Distribution":
        st.markdown('<div class="storytelling">This geographic visualization reveals the national distribution of nursing home facilities, showing concentration patterns in urban versus rural areas and highlighting regional disparities in healthcare infrastructure.</div>', unsafe_allow_html=True)
        
        # Check if geographic columns exist
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            # Filter out invalid coordinates
            geo_df = filtered_df.dropna(subset=['Latitude', 'Longitude'])
            
            # Create base map
            fig = px.scatter_geo(geo_df,
                                lat='Latitude',
                                lon='Longitude',
                                hover_name='Provider Name',
                                hover_data=['State', 'Overall Rating', 'Ownership Type'],
                                color='Overall Rating',
                                size_max=15,
                                projection='albers usa',
                                title='Nursing Home Facilities Distribution Across USA',
                                color_continuous_scale='viridis')
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add state-level summary
            if 'State' in df.columns and 'Overall Rating' in df.columns:
                state_summary = filtered_df.groupby('State').agg({
                    'Overall Rating': 'mean',
                    'Provider Name': 'count'
                }).reset_index()
                state_summary.columns = ['State', 'Average Rating', 'Facility Count']
                
                fig2 = px.bar(state_summary.sort_values('Facility Count', ascending=False).head(10),
                             x='State', y='Facility Count',
                             title='Top 10 States by Facility Count',
                             color='Average Rating',
                             color_continuous_scale='viridis')
                st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown('<div class="interaction-note">This interactive map allows you to explore nursing home facilities across the United States. Zoom in/out to see regional clusters, hover over individual points to see facility details, and use the color scale to identify quality ratings. Click and drag to pan across different regions. The concentration of facilities in certain areas may indicate population density patterns or healthcare resource allocation. You can also filter by ownership type or rating range in the sidebar to see specific subsets of facilities. The bar chart shows the states with the highest facility counts, helping identify regions with greater healthcare infrastructure.</div>', unsafe_allow_html=True)
        else:
            st.warning("Geographic coordinates not available in the dataset")
            # Alternative: show by state
            if 'State' in df.columns:
                state_counts = filtered_df['State'].value_counts().reset_index()
                state_counts.columns = ['State', 'Count']
                
                fig = px.bar(state_counts, x='State', y='Count',
                            title='Facility Distribution by State',
                            color='Count',
                            color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
    
    elif selected_viz == "Ownership Type Analysis":
        st.markdown('<div class="storytelling">Ownership structure significantly influences healthcare delivery models and financial sustainability. This analysis compares performance metrics across different ownership types to identify best practices and potential areas for improvement.</div>', unsafe_allow_html=True)
        
        if 'Ownership Type' in filtered_df.columns:
            # Ownership distribution
            ownership_counts = filtered_df['Ownership Type'].value_counts().reset_index()
            ownership_counts.columns = ['Ownership Type', 'Count']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.pie(ownership_counts, 
                             values='Count', 
                             names='Ownership Type',
                             title='Facility Distribution by Ownership Type',
                             hole=0.4)
                fig1.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                if 'Overall Rating' in filtered_df.columns:
                    ownership_ratings = filtered_df.groupby('Ownership Type')['Overall Rating'].mean().reset_index()
                    ownership_ratings = ownership_ratings.sort_values('Overall Rating', ascending=False)
                    
                    fig2 = px.bar(ownership_ratings,
                                 x='Ownership Type',
                                 y='Overall Rating',
                                 title='Average Overall Rating by Ownership Type',
                                 color='Overall Rating',
                                 color_continuous_scale='viridis')
                    fig2.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Additional metrics by ownership
            if 'Number of Fines' in filtered_df.columns:
                fines_by_ownership = filtered_df.groupby('Ownership Type')['Number of Fines'].mean().reset_index()
                fines_by_ownership = fines_by_ownership.sort_values('Number of Fines', ascending=False)
                
                fig3 = px.bar(fines_by_ownership,
                             x='Ownership Type',
                             y='Number of Fines',
                             title='Average Number of Fines by Ownership Type',
                             color='Number of Fines',
                             color_continuous_scale='reds')
                fig3.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig3, use_container_width=True)
            
            st.markdown('<div class="interaction-note">These interactive charts allow you to explore how ownership type correlates with facility count, quality ratings, and regulatory compliance. Hover over pie chart segments to see exact counts and percentages. The bar charts can be sorted by clicking on column headers. Try filtering by specific states in the sidebar to see regional ownership patterns. Notice how certain ownership types (like non-profit or government) may have different rating distributions compared to for-profit facilities. The fines analysis reveals whether particular ownership models are more prone to regulatory issues.</div>', unsafe_allow_html=True)
    
    elif selected_viz == "Quality Ratings Distribution":
        st.markdown('<div class="storytelling">Quality ratings serve as critical indicators of patient care standards and regulatory compliance. This visualization examines the distribution of various quality metrics and their relationships, highlighting facilities that excel or require improvement.</div>', unsafe_allow_html=True)
        
        rating_cols = [col for col in filtered_df.columns if 'Rating' in col and 'Footnote' not in col]
        
        if rating_cols:
            selected_ratings = st.multiselect(
                "Select Rating Metrics to Compare",
                rating_cols,
                default=rating_cols[:3] if len(rating_cols) >= 3 else rating_cols
            )
            
            if selected_ratings:
                # Create subplots
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=selected_ratings[:4],
                    specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
                          [{'type': 'box'}, {'type': 'scatter'}]]
                )
                
                for i, rating in enumerate(selected_ratings[:4]):
                    row = (i // 2) + 1
                    col = (i % 2) + 1
                    
                    if i < 2:  # First row: histograms
                        fig.add_trace(
                            go.Histogram(x=filtered_df[rating].dropna(), name=rating, nbinsx=20),
                            row=row, col=col
                        )
                    else:  # Second row: box plot and scatter
                        if i == 2:  # Box plot
                            fig.add_trace(
                                go.Box(y=filtered_df[rating].dropna(), name=rating),
                                row=row, col=col
                            )
                        else:  # Scatter plot
                            if len(selected_ratings) > 1:
                                fig.add_trace(
                                    go.Scatter(
                                        x=filtered_df[selected_ratings[0]].dropna(),
                                        y=filtered_df[rating].dropna(),
                                        mode='markers',
                                        name=f'{selected_ratings[0]} vs {rating}',
                                        marker=dict(size=8, opacity=0.6)
                                    ),
                                    row=row, col=col
                                )
                
                fig.update_layout(height=800, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation matrix
                if len(selected_ratings) > 1:
                    corr_matrix = filtered_df[selected_ratings].corr()
                    
                    fig_corr = px.imshow(corr_matrix,
                                        text_auto=True,
                                        aspect='auto',
                                        color_continuous_scale='RdBu_r',
                                        title='Correlation Matrix of Quality Ratings')
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                st.markdown('<div class="interaction-note">This multi-panel visualization provides comprehensive insights into quality ratings. Interact with each subplot independently: hover over histogram bars to see frequency counts, examine box plots for distribution statistics (median, quartiles, outliers), and explore scatter plots for relationships between different ratings. Use the zoom and pan tools to focus on specific areas. The box plot outliers represent facilities with exceptionally high or low ratings that may warrant further investigation. The correlation matrix shows how different quality metrics relate to each other - strong positive correlations indicate metrics that tend to move together.</div>', unsafe_allow_html=True)
    
    elif selected_viz == "Staffing Metrics":
        st.markdown('<div class="storytelling">Staffing levels directly impact patient care quality and outcomes. This analysis examines staffing ratios, turnover rates, and their relationship with quality metrics, identifying optimal staffing patterns for maintaining high care standards.</div>', unsafe_allow_html=True)
        
        staffing_cols = [col for col in filtered_df.columns if any(keyword in col.lower() for keyword in ['staffing', 'turnover', 'hours'])]
        
        if staffing_cols:
            selected_staffing = st.selectbox("Select Staffing Metric", staffing_cols)
            
            if selected_staffing and 'Overall Rating' in filtered_df.columns:
                # Scatter plot
                fig = px.scatter(filtered_df.dropna(subset=[selected_staffing, 'Overall Rating']),
                                x=selected_staffing,
                                y='Overall Rating',
                                color='Ownership Type' if 'Ownership Type' in filtered_df.columns else 'State',
                                hover_name='Provider Name',
                                title=f'{selected_staffing} vs Overall Rating',
                                trendline='ols')
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribution by ownership
                if 'Ownership Type' in filtered_df.columns:
                    fig2 = px.box(filtered_df.dropna(subset=[selected_staffing]),
                                 x='Ownership Type',
                                 y=selected_staffing,
                                 title=f'{selected_staffing} Distribution by Ownership Type',
                                 color='Ownership Type')
                    fig2.update_layout(xaxis_tickangle=-45, height=400)
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Additional staffing metrics correlation
                if len(staffing_cols) > 1:
                    other_staffing = st.selectbox("Compare with another staffing metric", 
                                                 [col for col in staffing_cols if col != selected_staffing])
                    
                    if other_staffing:
                        fig3 = px.scatter(filtered_df.dropna(subset=[selected_staffing, other_staffing]),
                                         x=selected_staffing,
                                         y=other_staffing,
                                         color='Overall Rating' if 'Overall Rating' in filtered_df.columns else None,
                                         hover_name='Provider Name',
                                         title=f'{selected_staffing} vs {other_staffing}',
                                         color_continuous_scale='viridis')
                        st.plotly_chart(fig3, use_container_width=True)
                
                st.markdown('<div class="interaction-note">This interactive scatter plot reveals the relationship between staffing metrics and quality ratings. Hover over individual points to see facility details. The trend line shows the overall correlation. Use the color legend to filter by ownership type or state - click on categories to show/hide specific groups. Higher staffing hours generally correlate with better ratings, but there may be optimal ranges beyond which additional staffing doesn\'t improve outcomes. The box plot shows how staffing metrics vary across different ownership types, revealing whether certain business models maintain different staffing levels. The comparison scatter plot helps identify if different staffing metrics are correlated with each other.</div>', unsafe_allow_html=True)
    
    elif selected_viz == "Penalties and Fines Analysis":
        st.markdown('<div class="storytelling">Regulatory penalties and fines serve as indicators of compliance issues and quality concerns. This analysis explores penalty patterns across states and ownership types, identifying systemic issues and their financial implications.</div>', unsafe_allow_html=True)
        
        penalty_cols = [col for col in filtered_df.columns if any(keyword.lower() in col.lower() for keyword in ['Fine', 'Penalty', 'Citation', 'Denial'])]
        
        if penalty_cols:
            selected_penalty = st.selectbox("Select Penalty Metric", penalty_cols)
            
            if selected_penalty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution by state
                    if 'State' in filtered_df.columns:
                        state_penalties = filtered_df.groupby('State')[selected_penalty].sum().reset_index()
                        
                        fig = px.bar(state_penalties.sort_values(selected_penalty, ascending=False).head(10),
                                     x='State',
                                     y=selected_penalty,
                                     title=f'Top 10 States by {selected_penalty}',
                                     color=selected_penalty,
                                     color_continuous_scale='reds')
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Distribution by ownership type
                    if 'Ownership Type' in filtered_df.columns:
                        ownership_penalties = filtered_df.groupby('Ownership Type')[selected_penalty].mean().reset_index()
                        ownership_penalties = ownership_penalties.sort_values(selected_penalty, ascending=False)
                        
                        fig2 = px.bar(ownership_penalties,
                                     x='Ownership Type',
                                     y=selected_penalty,
                                     title=f'Average {selected_penalty} by Ownership Type',
                                     color=selected_penalty,
                                     color_continuous_scale='reds')
                        fig2.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig2, use_container_width=True)
                
                # Relationship with quality ratings
                if 'Overall Rating' in filtered_df.columns:
                    fig3 = px.scatter(filtered_df.dropna(subset=[selected_penalty, 'Overall Rating']),
                                     x=selected_penalty,
                                     y='Overall Rating',
                                     color='State',
                                     hover_name='Provider Name',
                                     title=f'{selected_penalty} vs Overall Rating',
                                     trendline='ols')
                    st.plotly_chart(fig3, use_container_width=True)
                
                st.markdown('<div class="interaction-note">These visualizations analyze penalty distribution across states and ownership types. The bar charts show which states and ownership types have the highest penalty metrics. Hover over bars to see exact values. The scatter plot reveals the relationship between penalties and quality ratings - generally, facilities with more penalties have lower quality ratings. Use the sidebar filters to focus on specific states or ownership types. Notice how certain states have significantly higher penalty concentrations, which may indicate stricter enforcement, higher violation rates, or larger facility sizes. The ownership type analysis reveals whether certain business models are more prone to regulatory issues.</div>', unsafe_allow_html=True)
    
    elif selected_viz == "Correlation Analysis":
        st.markdown('<div class="storytelling">Understanding relationships between different quality metrics helps identify key drivers of performance. This correlation matrix reveals interconnected factors that influence overall facility quality and regulatory compliance.</div>', unsafe_allow_html=True)
        
        # Select numeric columns for correlation
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 5:
            # Let user select columns or use common metrics
            common_metrics = [col for col in numeric_cols if any(keyword in col for keyword in 
                               ['Rating', 'Fine', 'Staffing', 'Turnover', 'Hours', 'Count'])]
            
            if common_metrics:
                default_cols = common_metrics[:8] if len(common_metrics) >= 8 else common_metrics
            else:
                default_cols = numeric_cols[:8]
            
            selected_corr_cols = st.multiselect(
                "Select Variables for Correlation Analysis",
                numeric_cols,
                default=default_cols
            )
            
            if selected_corr_cols and len(selected_corr_cols) > 1:
                # Calculate correlation matrix
                corr_matrix = filtered_df[selected_corr_cols].corr()
                
                # Create heatmap
                fig = px.imshow(corr_matrix,
                               text_auto='.2f',
                               aspect='auto',
                               color_continuous_scale='RdBu_r',
                               title='Correlation Matrix of Selected Variables')
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Find strongest correlations
                st.subheader("Strongest Correlations")
                
                # Unstack and sort correlations
                corr_pairs = corr_matrix.unstack()
                corr_pairs = corr_pairs[corr_pairs != 1]  # Remove self-correlations
                strongest_pos = corr_pairs.sort_values(ascending=False).head(5)
                strongest_neg = corr_pairs.sort_values(ascending=True).head(5)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Top Positive Correlations:**")
                    for pair, value in strongest_pos.items():
                        st.write(f"{pair[0]} - {pair[1]}: {value:.3f}")
                
                with col2:
                    st.write("**Top Negative Correlations:**")
                    for pair, value in strongest_neg.items():
                        st.write(f"{pair[0]} - {pair[1]}: {value:.3f}")
                
                # Scatter plot for strongest correlation
                if len(strongest_pos) > 0:
                    strongest_pair = strongest_pos.index[0]
                    fig2 = px.scatter(filtered_df.dropna(subset=[strongest_pair[0], strongest_pair[1]]),
                                     x=strongest_pair[0],
                                     y=strongest_pair[1],
                                     hover_name='Provider Name' if 'Provider Name' in filtered_df.columns else None,
                                     title=f'Strongest Positive Correlation: {strongest_pair[0]} vs {strongest_pair[1]}',
                                     trendline='ols')
                    st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown('<div class="interaction-note">This interactive correlation matrix shows relationships between selected variables. Hover over cells to see exact correlation values. Strong positive correlations (blue) indicate variables that move together, while negative correlations (red) show inverse relationships. Click on the color scale to filter by correlation strength. The diagonal represents perfect self-correlation. Use this visualization to identify which quality metrics are most related - for example, staffing levels might strongly correlate with health inspection scores. Understanding these relationships helps prioritize improvement efforts. The scatter plot visualizes the strongest positive correlation, showing how the two most related variables interact across facilities.</div>', unsafe_allow_html=True)

elif analysis_section == "🔍 Deep Dive Analysis":
    st.markdown('<h2 class="section-header">Deep Dive Analysis</h2>', unsafe_allow_html=True)
    
    df = st.session_state.df
    filtered_df = apply_filters(df, selected_states, selected_ownership, min_rating, max_rating)
    
    # Advanced analysis options
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Performance Benchmarking", "Regional Analysis", "Ownership Comparison", "Quality Drivers"]
    )
    
    if analysis_type == "Performance Benchmarking":
        st.markdown('<div class="storytelling">Performance benchmarking compares facility metrics against regional and national averages, identifying top performers and facilities needing improvement.</div>', unsafe_allow_html=True)
        
        # Select metric for benchmarking
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            benchmark_metric = st.selectbox("Select Metric for Benchmarking", numeric_cols)
            
            # Calculate benchmarks
            national_avg = df[benchmark_metric].mean() if benchmark_metric in df.columns else 0
            filtered_avg = filtered_df[benchmark_metric].mean() if benchmark_metric in filtered_df.columns else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("National Average", f"{national_avg:.2f}")
            with col2:
                st.metric("Filtered Average", f"{filtered_avg:.2f}")
            with col3:
                diff = filtered_avg - national_avg
                st.metric("Difference", f"{diff:+.2f}")
            
            # Distribution visualization
            if benchmark_metric in filtered_df.columns:
                fig = px.histogram(filtered_df, x=benchmark_metric, 
                                  nbins=30, 
                                  title=f'Distribution of {benchmark_metric}',
                                  color_discrete_sequence=['#3498db'])
                
                # Add average lines
                fig.add_vline(x=national_avg, line_dash="dash", line_color="red",
                            annotation_text=f"National Avg: {national_avg:.2f}")
                fig.add_vline(x=filtered_avg, line_dash="dash", line_color="green",
                            annotation_text=f"Filtered Avg: {filtered_avg:.2f}")
                
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Regional Analysis":
        st.markdown('<div class="storytelling">Regional analysis examines geographic patterns in healthcare quality, identifying states or regions with exceptional performance or systemic challenges.</div>', unsafe_allow_html=True)
        
        if 'State' in filtered_df.columns:
            # Select metrics for regional comparison
            metric_options = [col for col in filtered_df.columns if filtered_df[col].dtype in [np.int64, np.float64]]
            
            if metric_options:
                selected_metrics = st.multiselect(
                    "Select Metrics for Regional Comparison",
                    metric_options,
                    default=metric_options[:3] if len(metric_options) >= 3 else metric_options
                )
                
                if selected_metrics:
                    # Create state-level summary
                    state_summary = filtered_df.groupby('State')[selected_metrics].mean().reset_index()
                    
                    # Display as interactive table
                    st.dataframe(state_summary.style.background_gradient(cmap='viridis'), 
                               use_container_width=True)
                    
                    # Parallel coordinates plot for multivariate comparison
                    if len(selected_metrics) > 1:
                        fig = px.parallel_coordinates(state_summary, 
                                                     dimensions=selected_metrics,
                                                     color=selected_metrics[0])
                        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Ownership Comparison":
        st.markdown('<div class="storytelling">Ownership comparison analyzes how different business models affect healthcare delivery, financial performance, and regulatory compliance.</div>', unsafe_allow_html=True)
        
        if 'Ownership Type' in filtered_df.columns:
            # Select metrics for ownership comparison
            metric_options = [col for col in filtered_df.columns if filtered_df[col].dtype in [np.int64, np.float64]]
            
            if metric_options:
                comparison_metric = st.selectbox("Select Metric for Comparison", metric_options)
                
                # Group by ownership type
                ownership_stats = filtered_df.groupby('Ownership Type')[comparison_metric].agg([
                    'mean', 'std', 'count', 'min', 'max'
                ]).reset_index()
                
                ownership_stats.columns = ['Ownership Type', 'Mean', 'Std Dev', 'Count', 'Min', 'Max']
                
                # Display statistics
                st.dataframe(ownership_stats.style.background_gradient(subset=['Mean'], cmap='viridis'),
                           use_container_width=True)
                
                # Visualization
                fig = px.bar(ownership_stats.sort_values('Mean', ascending=False),
                            x='Ownership Type',
                            y='Mean',
                            error_y='Std Dev',
                            title=f'{comparison_metric} by Ownership Type',
                            color='Mean',
                            color_continuous_scale='viridis')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Quality Drivers":
        st.markdown('<div class="storytelling">Quality driver analysis identifies the factors most strongly associated with high performance, guiding resource allocation and improvement strategies.</div>', unsafe_allow_html=True)
        
        if 'Overall Rating' in filtered_df.columns:
            # Select potential driver variables
            potential_drivers = [col for col in filtered_df.columns 
                                if col != 'Overall Rating' 
                                and filtered_df[col].dtype in [np.int64, np.float64]]
            
            if potential_drivers:
                selected_drivers = st.multiselect(
                    "Select Potential Quality Drivers",
                    potential_drivers,
                    default=potential_drivers[:5] if len(potential_drivers) >= 5 else potential_drivers
                )
                
                if selected_drivers:
                    # Calculate correlations with Overall Rating
                    correlations = []
                    for driver in selected_drivers:
                        corr = filtered_df[['Overall Rating', driver]].corr().iloc[0, 1]
                        correlations.append((driver, corr))
                    
                    corr_df = pd.DataFrame(correlations, columns=['Driver', 'Correlation'])
                    corr_df = corr_df.sort_values('Correlation', ascending=False)
                    
                    # Display correlations
                    fig = px.bar(corr_df,
                                x='Driver',
                                y='Correlation',
                                title='Correlation with Overall Rating',
                                color='Correlation',
                                color_continuous_scale='RdYlBu')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show top positive and negative drivers
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Top Positive Drivers")
                        top_positive = corr_df[corr_df['Correlation'] > 0].head(3)
                        for _, row in top_positive.iterrows():
                            st.write(f"**{row['Driver']}**: {row['Correlation']:.3f}")
                    
                    with col2:
                        st.subheader("Top Negative Drivers")
                        top_negative = corr_df[corr_df['Correlation'] < 0].head(3)
                        for _, row in top_negative.iterrows():
                            st.write(f"**{row['Driver']}**: {row['Correlation']:.3f}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>CMS Nursing Home Provider Data Analysis | Healthcare Analytics Dashboard</p>
    <p>Data Source: <a href="https://data.cms.gov/provider-data/dataset/4pq5-n9py" target="_blank">CMS Provider Data</a></p>
    <p>For educational and analytical purposes only</p>
</div>
""", unsafe_allow_html=True)

# Requirements info in sidebar
with st.sidebar.expander("Requirements Information"):
    st.write("""
    **Required Packages:**
    - streamlit
    - pandas
    - numpy
    - plotly
    
    **Optional Packages:**
    - matplotlib (for advanced visualizations)
    - missingno (for missing data matrix)
    
    Install with: `pip install streamlit pandas numpy plotly`
    """)
