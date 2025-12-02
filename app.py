"""
Streamlit App for Medicare Hospital Spending Analysis
This app converts the Jupyter notebook analysis to an interactive web application
"""

# ============================================================================
# IMPORT LIBRARIES
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Streamlit imports
import streamlit as st
from io import StringIO
import base64
from pathlib import Path

# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Medicare Hospital Spending Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR BETTER STYLING
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1E40AF;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #60A5FA;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTION
# ============================================================================
@st.cache_data
def load_data(file_path=None):
    """
    Load the dataset either from uploaded file or from a default path
    
    Args:
        file_path: Path to the CSV file (optional)
    
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    if file_path:
        df = pd.read_csv(file_path, low_memory=False)
    else:
        # For demo purposes, you can load a sample dataset
        # In production, you'll want to upload your actual CSV
        st.warning("Please upload your dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, low_memory=False)
        else:
            # Load sample data for demonstration
            st.info("Loading sample data for demonstration...")
            # Create sample data structure similar to your dataset
            sample_data = {
                'CMS Certification Number (CCN)': ['015009', '015010'],
                'Provider Name': ['Sample Hospital 1', 'Sample Hospital 2'],
                'State': ['AL', 'AL'],
                'Ownership Type': ['For profit - Corporation', 'Government - County'],
                'Overall Rating': [3.0, 4.0],
                'Number of Fines': [1, 0],
                'Total Amount of Fines in Dollars': [23989.0, 0.0],
                'Latitude': [34.5149, 33.1637],
                'Longitude': [-87.736, -86.254]
            }
            df = pd.DataFrame(sample_data)
    
    return df

# ============================================================================
# DATA PREPROCESSING FUNCTION
# ============================================================================
def preprocess_data(df):
    """
    Clean and preprocess the dataset
    
    Args:
        df: Raw DataFrame
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Handle missing values - fill numeric columns with median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Handle categorical missing values
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna('Unknown')
    
    return df_clean

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_missing_data_matrix(df):
    """
    Create a visualization of missing data patterns
    
    Args:
        df: DataFrame to analyze
    """
    st.subheader("Missing Data Analysis")
    
    # Calculate missing values
    missing_summary = pd.DataFrame({
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
    })
    
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
    missing_summary = missing_summary.sort_values('Missing_Percentage', ascending=False)
    
    # Display summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Columns", len(df.columns))
        st.metric("Columns with Missing Values", len(missing_summary))
    
    with col2:
        st.metric("Total Rows", len(df))
        st.metric("Total Missing Values", df.isnull().sum().sum())
    
    # Show top columns with missing values
    if len(missing_summary) > 0:
        st.write("**Top 20 Columns with Missing Values:**")
        st.dataframe(missing_summary.head(20))
        
        # Create a simple missing data heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a matrix of missing values (True = missing)
        missing_matrix = df.isnull()
        
        # Take a sample for visualization if dataset is large
        if len(df) > 1000:
            sample_df = df.sample(1000, random_state=42)
            missing_matrix = sample_df.isnull()
        
        # Plot heatmap
        ax.imshow(missing_matrix.T, aspect='auto', cmap='binary', interpolation='none')
        ax.set_xlabel('Facilities (sample)' if len(df) > 1000 else 'Facilities')
        ax.set_ylabel('Columns')
        ax.set_title('Missing Data Matrix\n(White = Missing | Black = Present)')
        
        st.pyplot(fig)
    else:
        st.success("🎉 No missing values found in the dataset!")

def plot_geographic_distribution(df):
    """
    Create geographic visualization of hospitals
    
    Args:
        df: DataFrame with latitude and longitude columns
    """
    st.subheader("Geographic Distribution of Hospitals")
    
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        # Clean coordinates
        df_coords = df.dropna(subset=['Latitude', 'Longitude'])
        df_coords = df_coords[(df_coords['Latitude'] != 0) & (df_coords['Longitude'] != 0)]
        
        if len(df_coords) > 0:
            # Create interactive map with Plotly
            fig = px.scatter_geo(df_coords,
                                lat='Latitude',
                                lon='Longitude',
                                hover_name='Provider Name',
                                hover_data=['State', 'Overall Rating'],
                                title='Hospital Locations Across USA',
                                projection='albers usa')
            
            fig.update_layout(geo=dict(showland=True, landcolor="lightgray"),
                             height=600)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No valid geographic coordinates found.")
    else:
        st.warning("Latitude and Longitude columns not found in the dataset.")

def plot_ownership_analysis(df):
    """
    Analyze and visualize hospital ownership types
    
    Args:
        df: DataFrame with ownership information
    """
    st.subheader("Hospital Ownership Analysis")
    
    if 'Ownership Type' in df.columns:
        # Count ownership types
        ownership_counts = df['Ownership Type'].value_counts().reset_index()
        ownership_counts.columns = ['Ownership Type', 'Count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            colors = plt.cm.Set3(np.arange(len(ownership_counts)))
            ax1.barh(ownership_counts['Ownership Type'], ownership_counts['Count'], color=colors)
            ax1.set_xlabel('Number of Hospitals')
            ax1.set_title('Distribution by Ownership Type')
            ax1.grid(axis='x', alpha=0.3)
            st.pyplot(fig1)
        
        with col2:
            # Pie chart
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            ax2.pie(ownership_counts['Count'], 
                   labels=ownership_counts['Ownership Type'],
                   autopct='%1.1f%%',
                   startangle=90,
                   colors=colors)
            ax2.set_title('Ownership Type Distribution')
            st.pyplot(fig2)
        
        # Ownership vs Quality metrics
        if 'Overall Rating' in df.columns:
            st.write("**Average Quality Ratings by Ownership Type:**")
            ownership_ratings = df.groupby('Ownership Type')['Overall Rating'].agg(['mean', 'count']).round(2)
            ownership_ratings = ownership_ratings.sort_values('mean', ascending=False)
            st.dataframe(ownership_ratings)

def plot_quality_metrics(df):
    """
    Visualize quality metrics and ratings
    
    Args:
        df: DataFrame with quality metrics
    """
    st.subheader("Quality Metrics Analysis")
    
    # Check for quality rating columns
    rating_columns = ['Overall Rating', 'Health Inspection Rating', 
                     'Staffing Rating', 'QM Rating']
    
    available_ratings = [col for col in rating_columns if col in df.columns]
    
    if available_ratings:
        col1, col2 = st.columns(2)
        
        with col1:
            # Overall rating distribution
            if 'Overall Rating' in df.columns:
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                df['Overall Rating'].dropna().hist(bins=20, ax=ax1, edgecolor='black', alpha=0.7)
                ax1.set_xlabel('Overall Rating (1-5 stars)')
                ax1.set_ylabel('Number of Hospitals')
                ax1.set_title('Distribution of Overall Ratings')
                ax1.grid(alpha=0.3)
                st.pyplot(fig1)
        
        with col2:
            # Correlation heatmap of ratings
            rating_df = df[available_ratings].dropna()
            if len(rating_df) > 1:
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                corr_matrix = rating_df.corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           ax=ax2, square=True, cbar_kws={"shrink": .8})
                ax2.set_title('Correlation between Quality Metrics')
                st.pyplot(fig2)
        
        # State-wise performance
        if 'State' in df.columns and 'Overall Rating' in df.columns:
            st.write("**State-wise Average Ratings:**")
            state_ratings = df.groupby('State')['Overall Rating'].agg(['mean', 'count']).round(2)
            state_ratings = state_ratings.sort_values('mean', ascending=False)
            
            # Top 10 states
            fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Top 10 states
            top_states = state_ratings.head(10)
            ax3.barh(top_states.index, top_states['mean'], color='skyblue')
            ax3.set_xlabel('Average Rating')
            ax3.set_title('Top 10 States by Average Rating')
            
            # Bottom 10 states
            bottom_states = state_ratings.tail(10)
            ax4.barh(bottom_states.index, bottom_states['mean'], color='lightcoral')
            ax4.set_xlabel('Average Rating')
            ax4.set_title('Bottom 10 States by Average Rating')
            
            plt.tight_layout()
            st.pyplot(fig3)

def plot_financial_penalties(df):
    """
    Analyze financial penalties and fines
    
    Args:
        df: DataFrame with financial penalty information
    """
    st.subheader("Financial Penalties Analysis")
    
    financial_cols = ['Number of Fines', 'Total Amount of Fines in Dollars',
                     'Number of Payment Denials', 'Total Number of Penalties']
    
    available_financial = [col for col in financial_cols if col in df.columns]
    
    if available_financial:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution of fines
            if 'Total Amount of Fines in Dollars' in df.columns:
                fines_data = df[df['Total Amount of Fines in Dollars'] > 0]
                if len(fines_data) > 0:
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    fines_data['Total Amount of Fines in Dollars'].hist(bins=50, ax=ax1, edgecolor='black', alpha=0.7)
                    ax1.set_xlabel('Fine Amount ($)')
                    ax1.set_ylabel('Number of Hospitals')
                    ax1.set_title('Distribution of Fine Amounts')
                    ax1.grid(alpha=0.3)
                    st.pyplot(fig1)
        
        with col2:
            # Penalties by state
            if 'State' in df.columns and 'Total Number of Penalties' in df.columns:
                penalties_by_state = df.groupby('State')['Total Number of Penalties'].sum().sort_values(ascending=False)
                if len(penalties_by_state) > 0:
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    penalties_by_state.head(15).plot(kind='bar', ax=ax2, color='salmon')
                    ax2.set_xlabel('State')
                    ax2.set_ylabel('Total Number of Penalties')
                    ax2.set_title('Top 15 States by Total Penalties')
                    ax2.tick_params(axis='x', rotation=45)
                    ax2.grid(axis='y', alpha=0.3)
                    st.pyplot(fig2)
        
        # Relationship between fines and quality
        if 'Total Amount of Fines in Dollars' in df.columns and 'Overall Rating' in df.columns:
            st.write("**Relationship Between Fines and Quality Ratings:**")
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            scatter_data = df[['Total Amount of Fines in Dollars', 'Overall Rating']].dropna()
            
            # Log transform for better visualization if fines are skewed
            if scatter_data['Total Amount of Fines in Dollars'].max() > 0:
                scatter_data['Log_Fines'] = np.log1p(scatter_data['Total Amount of Fines in Dollars'])
                ax3.scatter(scatter_data['Log_Fines'], scatter_data['Overall Rating'], 
                          alpha=0.5, s=50)
                ax3.set_xlabel('Log(Fine Amount + 1)')
            else:
                ax3.scatter(scatter_data['Total Amount of Fines in Dollars'], 
                          scatter_data['Overall Rating'], alpha=0.5, s=50)
                ax3.set_xlabel('Fine Amount ($)')
            
            ax3.set_ylabel('Overall Rating')
            ax3.set_title('Fines vs Quality Ratings')
            ax3.grid(alpha=0.3)
            st.pyplot(fig3)

# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================
def main():
    # Title and description
    st.markdown("<h1 class='main-header'>🏥 Medicare Hospital Spending Analysis</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    This interactive dashboard analyzes Medicare hospital spending data across the United States.
    Explore geographic distributions, ownership patterns, quality metrics, and financial penalties.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("📊 Analysis Controls")
        
        st.markdown("---")
        st.subheader("Data Source")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            df = load_data(uploaded_file)
        else:
            st.info("Using sample data for demonstration")
            # For demo, create sample data
            df = load_data()
        
        st.markdown("---")
        st.subheader("Analysis Options")
        
        # Analysis toggles
        show_missing_analysis = st.checkbox("Show Missing Data Analysis", value=True)
        show_geographic = st.checkbox("Show Geographic Distribution", value=True)
        show_ownership = st.checkbox("Show Ownership Analysis", value=True)
        show_quality = st.checkbox("Show Quality Metrics", value=True)
        show_financial = st.checkbox("Show Financial Penalties", value=True)
        
        st.markdown("---")
        st.subheader("Filters")
        
        # State filter
        if 'State' in df.columns:
            states = sorted(df['State'].dropna().unique())
            selected_states = st.multiselect("Filter by State", states, default=states[:5] if len(states) > 5 else states)
            if selected_states:
                df = df[df['State'].isin(selected_states)]
        
        # Ownership filter
        if 'Ownership Type' in df.columns:
            ownership_types = sorted(df['Ownership Type'].dropna().unique())
            selected_ownership = st.multiselect("Filter by Ownership", ownership_types, default=ownership_types)
            if selected_ownership:
                df = df[df['Ownership Type'].isin(selected_ownership)]
        
        st.markdown("---")
        st.caption("Built with Streamlit | Data: CMS Provider Data")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🏥 Hospitals", "📊 Analytics", "📥 Export"])
    
    with tab1:
        st.markdown("<h2 class='section-header'>Dataset Overview</h2>", unsafe_allow_html=True)
        
        # Display dataset info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Hospitals", len(df))
        
        with col2:
            if 'State' in df.columns:
                st.metric("Number of States", df['State'].nunique())
        
        with col3:
            if 'Ownership Type' in df.columns:
                st.metric("Ownership Types", df['Ownership Type'].nunique())
        
        with col4:
            if 'Overall Rating' in df.columns:
                avg_rating = df['Overall Rating'].mean()
                st.metric("Average Rating", f"{avg_rating:.2f}" if not pd.isna(avg_rating) else "N/A")
        
        # Show first few rows
        st.markdown("<h3 class='section-header'>Data Preview</h3>", unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)
        
        # Dataset statistics
        st.markdown("<h3 class='section-header'>Dataset Statistics</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Column Information:**")
            col_info = pd.DataFrame({
                'Column Name': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(col_info, use_container_width=True, height=300)
        
        with col2:
            st.write("**Numeric Column Statistics:**")
            numeric_stats = df.select_dtypes(include=[np.number]).describe().round(2)
            st.dataframe(numeric_stats, use_container_width=True)
    
    with tab2:
        st.markdown("<h2 class='section-header'>Hospital Analysis</h2>", unsafe_allow_html=True)
        
        # Run selected analyses
        if show_missing_analysis:
            plot_missing_data_matrix(df)
        
        if show_geographic:
            plot_geographic_distribution(df)
        
        if show_ownership:
            plot_ownership_analysis(df)
        
        if show_quality:
            plot_quality_metrics(df)
        
        if show_financial:
            plot_financial_penalties(df)
    
    with tab3:
        st.markdown("<h2 class='section-header'>Advanced Analytics</h2>", unsafe_allow_html=True)
        
        # Interactive correlation analysis
        st.subheader("Interactive Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("X-axis variable", numeric_cols, index=0)
            
            with col2:
                y_axis = st.selectbox("Y-axis variable", numeric_cols, index=min(1, len(numeric_cols)-1))
            
            if x_axis and y_axis:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(df[x_axis], df[y_axis], alpha=0.5, s=30)
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f'{x_axis} vs {y_axis}')
                ax.grid(alpha=0.3)
                
                # Add regression line
                try:
                    from scipy import stats
                    mask = ~(df[x_axis].isna() | df[y_axis].isna())
                    if mask.sum() > 1:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            df[x_axis][mask], df[y_axis][mask]
                        )
                        x_range = np.linspace(df[x_axis].min(), df[x_axis].max(), 100)
                        ax.plot(x_range, intercept + slope * x_range, 
                               'r-', label=f'R² = {r_value**2:.3f}')
                        ax.legend()
                except:
                    pass
                
                st.pyplot(fig)
        
        # Time series analysis (if date columns exist)
        date_columns = [col for col in df.columns if 'Date' in col or 'date' in col.lower()]
        if date_columns:
            st.subheader("Time Series Analysis")
            date_col = st.selectbox("Select date column", date_columns)
            
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                if 'Overall Rating' in df.columns:
                    # Aggregate by date
                    time_series = df.groupby(df[date_col].dt.to_period('M'))['Overall Rating'].mean().dropna()
                    
                    if len(time_series) > 0:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        time_series.plot(ax=ax, marker='o')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Average Overall Rating')
                        ax.set_title('Average Hospital Rating Over Time')
                        ax.grid(alpha=0.3)
                        ax.tick_params(axis='x', rotation=45)
                        st.pyplot(fig)
    
    with tab4:
        st.markdown("<h2 class='section-header'>Data Export</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Export Processed Data")
            
            # Show processed data
            processed_df = preprocess_data(df)
            st.dataframe(processed_df.head(), use_container_width=True)
            
            # Download button for processed data
            csv = processed_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="processed_hospital_data.csv">📥 Download Processed Data (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Export Visualizations")
            
            # Generate and offer download for key visualizations
            st.write("Key visualizations generated:")
            
            viz_options = st.multiselect(
                "Select visualizations to generate:",
                ["Missing Data Matrix", "Ownership Distribution", "Quality Metrics", "Geographic Map"]
            )
            
            if st.button("Generate Report"):
                with st.spinner("Generating report..."):
                    # Create a simple PDF report (in production, use reportlab or similar)
                    report_content = f"""
                    Medicare Hospital Spending Analysis Report
                    =========================================
                    
                    Dataset Summary:
                    - Total Hospitals: {len(df)}
                    - Total States: {df['State'].nunique() if 'State' in df.columns else 'N/A'}
                    - Data Columns: {len(df.columns)}
                    
                    Analysis Performed:
                    {', '.join(viz_options) if viz_options else 'No visualizations selected'}
                    
                    Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                    """
                    
                    # Create downloadable report
                    b64_report = base64.b64encode(report_content.encode()).decode()
                    href_report = f'<a href="data:text/plain;base64,{b64_report}" download="hospital_analysis_report.txt">📄 Download Analysis Report</a>'
                    st.markdown(href_report, unsafe_allow_html=True)
                    
                    st.success("Report generated successfully!")

# ============================================================================
# RUN THE APP
# ============================================================================
if __name__ == "__main__":
    main()
