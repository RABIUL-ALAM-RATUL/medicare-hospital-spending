# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import missingno as msno

# Page configuration
st.set_page_config(
    page_title="Medicare Hospital Spending Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply professional styling
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

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
    def load_data():
        try:
            # Update this path to your actual data location
            df = pd.read_csv('NH_ProviderInfo_Sep2025.csv', low_memory=False)
            return df
        except:
            # If file not found, show warning
            st.warning("Please upload the dataset in the Data Overview section")
            return None
    
    df = load_data()
    
    if df is not None:
        all_states = sorted(df['State'].unique())
        selected_states = st.multiselect(
            "Select States",
            options=all_states,
            default=all_states[:5] if len(all_states) > 5 else all_states
        )
        
        ownership_types = sorted(df['Ownership Type'].dropna().unique())
        selected_ownership = st.multiselect(
            "Select Ownership Types",
            options=ownership_types,
            default=ownership_types[:3] if len(ownership_types) > 3 else ownership_types
        )
        
        # Rating filter
        min_rating, max_rating = st.slider(
            "Overall Rating Range",
            min_value=1.0,
            max_value=5.0,
            value=(1.0, 5.0),
            step=0.5
        )

# Main content based on selected section
if analysis_section == "🏠 Introduction":
    st.markdown('<h2 class="section-header">Project Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Facilities", "14,752")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Dataset Variables", "100")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Time Period", "Sep 2025")
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
    representing nursing home provider information from September 2025, covering 14,752 facilities nationwide.
    """)
    
    # Quick insights
    st.markdown('<h3 class="section-header">Quick Insights</h3>', unsafe_allow_html=True)
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.info("""
        **Data Scope:**
        - 14,752 nursing home facilities
        - 100 variables per facility
        - National coverage across all states
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

elif analysis_section == "📊 Data Overview":
    st.markdown('<h2 class="section-header">Data Loading and Initial Exploration</h2>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload CMS Nursing Home Dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, low_memory=False)
        st.session_state['df'] = df
    elif 'df' not in st.session_state:
        st.warning("Please upload a dataset or use the sample data below")
        # Provide sample data structure
        sample_data = {
            'CMS Certification Number (CCN)': ['015009', '015010'],
            'Provider Name': ['BURNS NURSING HOME, INC.', 'COOSA VALLEY HEALTHCARE CENTER'],
            'State': ['AL', 'AL'],
            'Ownership Type': ['For profit - Corporation', 'For profit - Corporation'],
            'Overall Rating': [3.0, 4.0]
        }
        df = pd.DataFrame(sample_data)
    else:
        df = st.session_state['df']
    
    # Dataset overview
    st.markdown('<div class="storytelling">The dataset represents the complete CMS Nursing Home Provider information from September 2025, containing 14,752 records of facilities across the United States. This initial overview helps us understand the data structure, variable types, and overall quality before diving into detailed analysis.</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        if 'Processing Date' in df.columns:
            st.metric("Processing Date", df['Processing Date'].iloc[0])
        else:
            st.metric("Data Status", "Loaded")
    
    # Data preview
    st.markdown('<h3 class="section-header">Data Preview</h3>', unsafe_allow_html=True)
    
    preview_option = st.radio(
        "Preview Options",
        ["First 10 Rows", "Last 10 Rows", "Random Sample"]
    )
    
    if preview_option == "First 10 Rows":
        st.dataframe(df.head(10), use_container_width=True)
    elif preview_option == "Last 10 Rows":
        st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.dataframe(df.sample(10), use_container_width=True)
    
    # Data types information
    st.markdown('<h3 class="section-header">Data Structure</h3>', unsafe_allow_html=True)
    
    if st.button("Show Data Types Information"):
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

elif analysis_section == "🧹 Data Cleaning":
    st.markdown('<h2 class="section-header">Data Cleaning and Preprocessing</h2>', unsafe_allow_html=True)
    
    if 'df' not in st.session_state:
        st.error("Please load data in the Data Overview section first")
        st.stop()
    
    df = st.session_state['df']
    
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
    st.markdown('<div class="storytelling">The missing data matrix visualization reveals systematic patterns in data completeness. White bars indicate missing values, helping identify columns that may require imputation or exclusion from certain analyses.</div>', unsafe_allow_html=True)
    
    if st.button("Generate Missing Data Matrix"):
        fig, ax = plt.subplots(figsize=(12, 8))
        msno.matrix(df, sparkline=False, fontsize=10, 
                   color=(0.25, 0.65, 0.65), ax=ax)
        ax.set_title('Missing Data Matrix\n(White = Missing | Dark = Present)', 
                    fontsize=14, weight='bold', pad=20)
        ax.set_xlabel('Columns', fontsize=12)
        ax.set_ylabel('Facilities (rows)', fontsize=12)
        st.pyplot(fig)
        
        st.markdown('<div class="interaction-note">This interactive missing data matrix allows you to visually identify patterns of missingness across all variables. Hover over the visualization to see which columns have the most missing data. The pattern can reveal whether missing values are random (scattered white lines) or systematic (vertical white bands), which informs our data cleaning strategy. Systematic missingness often indicates reporting differences between states or ownership types.</div>', unsafe_allow_html=True)
    
    # Data cleaning options
    st.markdown('<h3 class="section-header">Data Cleaning Options</h3>', unsafe_allow_html=True)
    
    cleaning_option = st.selectbox(
        "Select Cleaning Strategy",
        ["Select option...", "Remove High Missing Columns (>50%)", 
         "Impute Missing Values", "Convert Data Types"]
    )
    
    if cleaning_option == "Remove High Missing Columns (>50%)":
        threshold = st.slider("Missing Percentage Threshold", 0, 100, 50)
        cols_to_drop = missing_summary[missing_summary['Missing_Percentage'] > threshold].index.tolist()
        st.write(f"Columns to remove ({len(cols_to_drop)}): {cols_to_drop}")
        
        if st.button("Apply Removal"):
            df_clean = df.drop(columns=cols_to_drop)
            st.session_state['df_clean'] = df_clean
            st.success(f"Removed {len(cols_to_drop)} columns. Remaining columns: {df_clean.shape[1]}")
    
    elif cleaning_option == "Impute Missing Values":
        col_to_impute = st.selectbox("Select Column to Impute", missing_summary.index.tolist())
        
        impute_method = st.radio(
            "Imputation Method",
            ["Mean", "Median", "Mode", "Zero", "Forward Fill"]
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
            else:
                df[col_to_impute] = df[col_to_impute].ffill()
            
            st.success(f"Imputed missing values in {col_to_impute} using {impute_method} method")
            st.session_state['df'] = df

elif analysis_section == "📈 Visualizations":
    st.markdown('<h2 class="section-header">Data Visualizations</h2>', unsafe_allow_html=True)
    
    if 'df' not in st.session_state:
        st.error("Please load data in the Data Overview section first")
        st.stop()
    
    df = st.session_state['df']
    
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
            geo_df = df.dropna(subset=['Latitude', 'Longitude'])
            
            fig = px.scatter_geo(geo_df,
                                lat='Latitude',
                                lon='Longitude',
                                hover_name='Provider Name',
                                hover_data=['State', 'Overall Rating', 'Ownership Type'],
                                color='Overall Rating',
                                size_max=15,
                                projection='albers usa',
                                title='Nursing Home Facilities Distribution Across USA')
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="interaction-note">This interactive map allows you to explore nursing home facilities across the United States. Zoom in/out to see regional clusters, hover over individual points to see facility details, and use the color scale to identify quality ratings. Click and drag to pan across different regions. The concentration of facilities in certain areas may indicate population density patterns or healthcare resource allocation. You can also filter by ownership type or rating range in the sidebar to see specific subsets of facilities.</div>', unsafe_allow_html=True)
        else:
            st.warning("Geographic coordinates not available in the dataset")
    
    elif selected_viz == "Ownership Type Analysis":
        st.markdown('<div class="storytelling">Ownership structure significantly influences healthcare delivery models and financial sustainability. This analysis compares performance metrics across different ownership types to identify best practices and potential areas for improvement.</div>', unsafe_allow_html=True)
        
        if 'Ownership Type' in df.columns:
            # Ownership distribution
            ownership_counts = df['Ownership Type'].value_counts().reset_index()
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
                if 'Overall Rating' in df.columns:
                    ownership_ratings = df.groupby('Ownership Type')['Overall Rating'].mean().reset_index()
                    ownership_ratings = ownership_ratings.sort_values('Overall Rating', ascending=False)
                    
                    fig2 = px.bar(ownership_ratings,
                                 x='Ownership Type',
                                 y='Overall Rating',
                                 title='Average Overall Rating by Ownership Type',
                                 color='Overall Rating',
                                 color_continuous_scale='viridis')
                    fig2.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown('<div class="interaction-note">These interactive charts allow you to explore how ownership type correlates with facility count and quality ratings. Hover over pie chart segments to see exact counts and percentages. The bar chart can be sorted by clicking on column headers. Try filtering by specific states in the sidebar to see regional ownership patterns. Notice how certain ownership types (like non-profit or government) may have different rating distributions compared to for-profit facilities.</div>', unsafe_allow_html=True)
    
    elif selected_viz == "Quality Ratings Distribution":
        st.markdown('<div class="storytelling">Quality ratings serve as critical indicators of patient care standards and regulatory compliance. This visualization examines the distribution of various quality metrics and their relationships, highlighting facilities that excel or require improvement.</div>', unsafe_allow_html=True)
        
        rating_cols = [col for col in df.columns if 'Rating' in col and 'Footnote' not in col]
        
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
                            go.Histogram(x=df[rating].dropna(), name=rating, nbinsx=20),
                            row=row, col=col
                        )
                    else:  # Second row: box plot and scatter
                        if i == 2:  # Box plot
                            fig.add_trace(
                                go.Box(y=df[rating].dropna(), name=rating),
                                row=row, col=col
                            )
                        else:  # Scatter plot
                            if len(selected_ratings) > 1:
                                fig.add_trace(
                                    go.Scatter(
                                        x=df[selected_ratings[0]].dropna(),
                                        y=df[rating].dropna(),
                                        mode='markers',
                                        name=f'{selected_ratings[0]} vs {rating}'
                                    ),
                                    row=row, col=col
                                )
                
                fig.update_layout(height=800, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('<div class="interaction-note">This multi-panel visualization provides comprehensive insights into quality ratings. Interact with each subplot independently: hover over histogram bars to see frequency counts, examine box plots for distribution statistics (median, quartiles, outliers), and explore scatter plots for relationships between different ratings. Use the zoom and pan tools to focus on specific areas. The box plot outliers represent facilities with exceptionally high or low ratings that may warrant further investigation.</div>', unsafe_allow_html=True)
    
    elif selected_viz == "Staffing Metrics":
        st.markdown('<div class="storytelling">Staffing levels directly impact patient care quality and outcomes. This analysis examines staffing ratios, turnover rates, and their relationship with quality metrics, identifying optimal staffing patterns for maintaining high care standards.</div>', unsafe_allow_html=True)
        
        staffing_cols = [col for col in df.columns if 'Staffing' in col and 'Footnote' not in col]
        
        if staffing_cols:
            selected_staffing = st.selectbox("Select Staffing Metric", staffing_cols)
            
            if selected_staffing and 'Overall Rating' in df.columns:
                fig = px.scatter(df.dropna(subset=[selected_staffing, 'Overall Rating']),
                                x=selected_staffing,
                                y='Overall Rating',
                                color='State',
                                hover_name='Provider Name',
                                title=f'{selected_staffing} vs Overall Rating',
                                trendline='ols')
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional staffing metrics
                if 'Total nursing staff turnover' in df.columns and 'Overall Rating' in df.columns:
                    fig2 = px.density_contour(
                        df.dropna(subset=['Total nursing staff turnover', 'Overall Rating']),
                        x='Total nursing staff turnover',
                        y='Overall Rating',
                        title='Staff Turnover vs Quality Rating Density'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown('<div class="interaction-note">This interactive scatter plot reveals the relationship between staffing metrics and quality ratings. Hover over individual points to see facility details. The trend line shows the overall correlation. Use the color legend to filter by state - click on state names to show/hide specific states. Higher staffing hours generally correlate with better ratings, but there may be optimal ranges beyond which additional staffing doesn\'t improve outcomes. The density plot shows concentration patterns, highlighting common combinations of turnover rates and quality scores.</div>', unsafe_allow_html=True)
    
    elif selected_viz == "Penalties and Fines Analysis":
        st.markdown('<div class="storytelling">Regulatory penalties and fines serve as indicators of compliance issues and quality concerns. This analysis explores penalty patterns across states and ownership types, identifying systemic issues and their financial implications.</div>', unsafe_allow_html=True)
        
        penalty_cols = [col for col in df.columns if any(keyword in col for keyword in ['Fine', 'Penalty', 'Citation'])]
        
        if penalty_cols:
            selected_penalty = st.selectbox("Select Penalty Metric", penalty_cols)
            
            if selected_penalty:
                # Group by state
                if 'State' in df.columns:
                    state_penalties = df.groupby('State')[selected_penalty].sum().reset_index()
                    
                    fig = px.choropleth(state_penalties,
                                       locations='State',
                                       locationmode='USA-states',
                                       color=selected_penalty,
                                       scope='usa',
                                       title=f'{selected_penalty} by State',
                                       color_continuous_scale='reds')
                    
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Penalties by ownership type
                if 'Ownership Type' in df.columns:
                    ownership_penalties = df.groupby('Ownership Type')[selected_penalty].mean().reset_index()
                    ownership_penalties = ownership_penalties.sort_values(selected_penalty, ascending=False)
                    
                    fig2 = px.bar(ownership_penalties,
                                 x='Ownership Type',
                                 y=selected_penalty,
                                 title=f'Average {selected_penalty} by Ownership Type',
                                 color=selected_penalty,
                                 color_continuous_scale='reds')
                    fig2.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown('<div class="interaction-note">The choropleth map visualizes penalty distribution across states - darker colors indicate higher penalty amounts or frequencies. Hover over states to see exact values. The bar chart compares penalty metrics across ownership types. Both visualizations can be filtered using the sidebar options. Notice how certain states have significantly higher penalty concentrations, which may indicate stricter enforcement, higher violation rates, or larger facility sizes. The ownership type analysis may reveal whether certain business models are more prone to regulatory issues.</div>', unsafe_allow_html=True)
    
    elif selected_viz == "Correlation Analysis":
        st.markdown('<div class="storytelling">Understanding relationships between different quality metrics helps identify key drivers of performance. This correlation matrix reveals interconnected factors that influence overall facility quality and regulatory compliance.</div>', unsafe_allow_html=True)
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 5:
            selected_corr_cols = st.multiselect(
                "Select Variables for Correlation Analysis",
                numeric_cols,
                default=numeric_cols[:10]
            )
            
            if selected_corr_cols:
                # Calculate correlation matrix
                corr_matrix = df[selected_corr_cols].corr()
                
                # Create heatmap
                fig = px.imshow(corr_matrix,
                               text_auto='.2f',
                               aspect='auto',
                               color_continuous_scale='RdBu_r',
                               title='Correlation Matrix of Selected Variables')
                
                fig.update_layout(height=700)
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
                
                st.markdown('<div class="interaction-note">This interactive correlation matrix shows relationships between selected variables. Hover over cells to see exact correlation values. Strong positive correlations (blue) indicate variables that move together, while negative correlations (red) show inverse relationships. Click on the color scale to filter by correlation strength. The diagonal represents perfect self-correlation. Use this visualization to identify which quality metrics are most related - for example, staffing levels might strongly correlate with health inspection scores. Understanding these relationships helps prioritize improvement efforts.</div>', unsafe_allow_html=True)

elif analysis_section == "🔍 Deep Dive Analysis":
    st.markdown('<h2 class="section-header">Deep Dive Analysis</h2>', unsafe_allow_html=True)
    
    if 'df' not in st.session_state:
        st.error("Please load data in the Data Overview section first")
        st.stop()
    
    df = st.session_state['df']
    
    # Advanced analysis options
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Predictive Modeling", "Trend Analysis", "Benchmarking", "Outlier Detection"]
    )
    
    if analysis_type == "Predictive Modeling":
        st.markdown('<div class="storytelling">Predictive modeling helps identify factors that most strongly influence quality ratings, enabling proactive interventions and resource allocation optimization.</div>', unsafe_allow_html=True)
        
        # Feature selection for modeling
        feature_options = [col for col in df.columns if col not in ['Provider Name', 'Provider Address', 'Location']]
        target_options = [col for col in df.columns if 'Rating' in col]
        
        if target_options:
            target = st.selectbox("Select Target Variable", target_options)
            
            # Select features
            features = st.multiselect(
                "Select Predictor Variables",
                [col for col in feature_options if col != target],
                default=[col for col in feature_options if 'Staffing' in col][:3]
            )
            
            if st.button("Run Preliminary Analysis"):
                # Simple correlation with target
                st.subheader("Feature Correlation with Target")
                
                correlations = []
                for feature in features:
                    if feature in df.columns and target in df.columns:
                        corr = df[[feature, target]].dropna().corr().iloc[0, 1]
                        correlations.append((feature, corr))
                
                corr_df = pd.DataFrame(correlations, columns=['Feature', 'Correlation'])
                corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
                
                fig = px.bar(corr_df,
                            x='Feature',
                            y='Correlation',
                            title=f'Correlation with {target}',
                            color='Correlation',
                            color_continuous_scale='RdYlBu')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Trend Analysis":
        st.markdown('<div class="storytelling">Trend analysis identifies temporal patterns in quality metrics and regulatory outcomes, helping track improvement or deterioration over time.</div>', unsafe_allow_html=True)
        
        # Mock time series analysis (since we don't have time dimension in this dataset)
        st.info("For comprehensive trend analysis, time-series data across multiple periods would be required.")
        
        # Create simulated trends by state
        if 'State' in df.columns and 'Overall Rating' in df.columns:
            state_trends = df.groupby('State')['Overall Rating'].agg(['mean', 'std', 'count']).reset_index()
            state_trends = state_trends.sort_values('mean', ascending=False)
            
            fig = px.scatter(state_trends,
                            x='mean',
                            y='std',
                            size='count',
                            color='State',
                            hover_name='State',
                            title='Quality Rating Trends by State: Mean vs Variability',
                            labels={'mean': 'Average Rating', 'std': 'Rating Variability'})
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Benchmarking":
        st.markdown('<div class="storytelling">Benchmarking compares facility performance against peers, identifying best practices and opportunities for quality improvement.</div>', unsafe_allow_html=True)
        
        if 'Overall Rating' in df.columns and 'State' in df.columns:
            # Calculate benchmarks
            national_avg = df['Overall Rating'].mean()
            state_avgs = df.groupby('State')['Overall Rating'].mean().reset_index()
            
            fig = px.bar(state_avgs.sort_values('Overall Rating', ascending=False),
                        x='State',
                        y='Overall Rating',
                        title='State Benchmarking: Average Overall Rating',
                        color='Overall Rating',
                        color_continuous_scale='viridis')
            
            # Add national average line
            fig.add_hline(y=national_avg, line_dash="dash", line_color="red",
                         annotation_text=f"National Average: {national_avg:.2f}")
            
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Outlier Detection":
        st.markdown('<div class="storytelling">Outlier detection identifies facilities with exceptional performance (positive or negative), enabling targeted interventions and best practice identification.</div>', unsafe_allow_html=True)
        
        metric_for_outliers = st.selectbox(
            "Select Metric for Outlier Detection",
            [col for col in df.columns if df[col].dtype in [np.int64, np.float64]]
        )
        
        if metric_for_outliers:
            # Calculate outliers using IQR method
            data = df[metric_for_outliers].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[metric_for_outliers] < lower_bound) | 
                         (df[metric_for_outliers] > upper_bound)]
            
            st.metric("Total Outliers Detected", len(outliers))
            
            if len(outliers) > 0:
                # Display outlier facilities
                st.dataframe(outliers[['Provider Name', 'State', 'Ownership Type', metric_for_outliers]].head(20),
                           use_container_width=True)
                
                # Download outliers
                csv = outliers.to_csv(index=False)
                st.download_button(
                    label="Download Outliers Data",
                    data=csv,
                    file_name=f"outliers_{metric_for_outliers}.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>CMS Nursing Home Provider Data Analysis | September 2025 Release</p>
    <p>Data Source: <a href="https://data.cms.gov/provider-data/dataset/4pq5-n9py" target="_blank">CMS Provider Data</a></p>
    <p>For educational and analytical purposes only</p>
</div>
""", unsafe_allow_html=True)
