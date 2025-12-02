import streamlit as st  # For creating web application interface
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations and random data generation
import plotly.express as px  # For creating interactive plots
import plotly.graph_objects as go  # For more advanced plot customization
from plotly.subplots import make_subplots  # For creating subplot layouts
import matplotlib  # For plotting (though we're mainly using Plotly)
matplotlib.use('Agg')  # Set matplotlib to use non-interactive backend to avoid conflicts
import warnings  # For managing warning messages
warnings.filterwarnings('ignore')  # Ignore all warning messages for cleaner output

# Set Streamlit page configuration
st.set_page_config(
    page_title="Medicare Hospital Spending Dashboard",  # Browser tab title
    page_icon="🏥",  # Browser tab icon (hospital emoji)
    layout="wide",  # Use wide layout for more horizontal space
    initial_sidebar_state="expanded"  # Start with sidebar expanded
)

# Custom CSS for better styling of the dashboard
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .highlight {
        background-color: #DBEAFE;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)  # unsafe_allow_html allows HTML/CSS in markdown

# Title and description section
st.markdown('<h1 class="main-header">🏥 Medicare Hospital Spending Analysis</h1>', unsafe_allow_html=True)
st.markdown("""
This dashboard provides insights into Medicare spending across hospitals, analyzing costs, quality metrics, 
and geographic variations to identify opportunities for improving healthcare efficiency.
""")

# Create sidebar for filters (left panel)
with st.sidebar:
    st.markdown("### 🎯 Filters")  # Sidebar header
    
    # Year selection dropdown
    years = [2020, 2021, 2022, 2023, 2024]  # Available years
    selected_year = st.selectbox("Select Year", years, index=len(years)-1)  # Default to most recent year
    
    # State selection (multi-select for multiple states)
    states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
              'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
              'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
              'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
              'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC']
    selected_states = st.multiselect("Select States", states, default=['CA', 'TX', 'NY', 'FL'])  # Default to major states
    
    # Hospital type filter dropdown
    hospital_types = ['All Types', 'Acute Care', 'Critical Access', 'Teaching', 'Children\'s', 'Rural']
    selected_type = st.selectbox("Hospital Type", hospital_types, index=0)  # Default to 'All Types'
    
    # Spending range slider (min $10,000 to max $50,000)
    min_spend, max_spend = st.slider(
        "Average Spending per Patient ($)",
        10000, 50000, (20000, 40000)  # Range: 10k-50k, default: 20k-40k
    )
    
    st.markdown("---")  # Horizontal line separator
    st.markdown("### 📊 Dashboard Guide")  # Sidebar guide section
    st.markdown("""
    1. **Overview Metrics**: Key statistics
    2. **Geographic Analysis**: State-level spending
    3. **Cost Analysis**: Spending distribution
    4. **Quality Metrics**: Outcome correlations
    5. **Benchmarking**: Performance comparison
    """)

# Sample data generation function with caching decorator
@st.cache_data  # Cache the data so it doesn't reload on every interaction
def create_sample_data():
    np.random.seed(42)  # Set random seed for reproducible results
    n_facilities = 500  # Create 500 sample hospitals
    
    # List of all US states + DC (51 total)
    states_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
                   'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
                   'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
                   'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
                   'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC']
    
    # Create dictionary with sample data for each column
    data = {
        # Facility IDs: HOSP0001, HOSP0002, etc.
        'Facility_ID': [f'HOSP{i:04d}' for i in range(1, n_facilities + 1)],
        # Facility names: Hospital 1, Hospital 2, etc.
        'Facility_Name': [f'Hospital {i}' for i in range(1, n_facilities + 1)],
        # Randomly assign states with equal probability (uniform distribution)
        'State': np.random.choice(states_list, n_facilities),
        # Assign city types with specified probabilities
        'City': np.random.choice(['Metro', 'Urban', 'Suburban', 'Rural'], n_facilities, p=[0.4, 0.3, 0.2, 0.1]),
        # Assign hospital types with specified probabilities
        'Hospital_Type': np.random.choice(['Acute Care', 'Critical Access', 'Teaching', 'Children\'s', 'Rural'], 
                                          n_facilities, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
        # Random total spending between $10M and $500M
        'Total_Spending': np.random.uniform(10000000, 500000000, n_facilities).round(2),
        # Random average spending per patient between $15K and $45K
        'Avg_Spending_per_Patient': np.random.uniform(15000, 45000, n_facilities).round(2),
        # Random number of Medicare discharges between 100 and 10,000
        'Medicare_Discharges': np.random.randint(100, 10000, n_facilities),
        # Random readmission rate between 10% and 25%
        'Readmission_Rate': np.random.uniform(10, 25, n_facilities).round(1),
        # Random mortality rate between 1% and 10%
        'Mortality_Rate': np.random.uniform(1, 10, n_facilities).round(1),
        # Random patient satisfaction between 60% and 95%
        'Patient_Satisfaction': np.random.uniform(60, 95, n_facilities).round(1),
        # Random quality score between 1 and 5
        'Quality_Score': np.random.uniform(1, 5, n_facilities).round(1),
        # Random cost efficiency score between 0.5 and 1.5
        'Cost_Efficiency_Score': np.random.uniform(0.5, 1.5, n_facilities).round(2),
        # Assign years with more recent years having higher probability
        'Year': np.random.choice([2022, 2023, 2024], n_facilities, p=[0.2, 0.3, 0.5])
    }
    
    # Create DataFrame from the dictionary
    df = pd.DataFrame(data)
    
    # Calculate derived metrics
    df['Spending_per_Discharge'] = (df['Total_Spending'] / df['Medicare_Discharges']).round(2)  # Cost per discharge
    df['Value_Score'] = (df['Quality_Score'] / df['Cost_Efficiency_Score']).round(2)  # Quality per efficiency unit
    
    return df

# Load data with error handling
try:
    # Check if data is already in session state (cached)
    if 'df' not in st.session_state:
        # If not, create and store it
        st.session_state.df = create_sample_data()
    # Get data from session state
    df = st.session_state.df
except Exception as e:
    # Display error and stop execution if data loading fails
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Apply filters to the data
filtered_df = df.copy()  # Start with full dataset

# Year filter: only keep data for selected year
filtered_df = filtered_df[filtered_df['Year'] == selected_year]

# State filter: if states are selected, filter to only those states
if selected_states:
    filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]

# Hospital type filter: if not "All Types", filter to selected type
if selected_type != 'All Types':
    filtered_df = filtered_df[filtered_df['Hospital_Type'] == selected_type]

# Spending range filter: only keep hospitals within selected spending range
filtered_df = filtered_df[
    (filtered_df['Avg_Spending_per_Patient'] >= min_spend) & 
    (filtered_df['Avg_Spending_per_Patient'] <= max_spend)
]

# Display key metrics in columns at the top of the dashboard
st.markdown("## 📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)  # Create 4 equal-width columns

with col1:
    total_hospitals = len(filtered_df)  # Count of hospitals after filtering
    st.metric("Total Hospitals", f"{total_hospitals:,}")  # Format with thousands separator
    
with col2:
    avg_spending = filtered_df['Avg_Spending_per_Patient'].mean()  # Calculate average spending
    st.metric("Avg Spending per Patient", f"${avg_spending:,.0f}")  # Format as currency
    
with col3:
    avg_readmission = filtered_df['Readmission_Rate'].mean()  # Calculate average readmission rate
    st.metric("Avg Readmission Rate", f"{avg_readmission:.1f}%")  # Format as percentage
    
with col4:
    avg_satisfaction = filtered_df['Patient_Satisfaction'].mean()  # Calculate average satisfaction
    st.metric("Avg Patient Satisfaction", f"{avg_satisfaction:.1f}%")  # Format as percentage

# Create tabs for different sections of the dashboard
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 Geographic Analysis",  # Tab 1: Maps and state-level analysis
    "💰 Cost Analysis",        # Tab 2: Spending patterns and efficiency
    "🏆 Quality Metrics",       # Tab 3: Quality indicators and outcomes
    "📊 Benchmarking",         # Tab 4: Hospital comparisons
    "📋 Raw Data"              # Tab 5: Data table view
])

# TAB 1: Geographic Analysis
with tab1:
    st.markdown("### State-Level Spending Analysis")
    
    col1, col2 = st.columns(2)  # Split tab into two columns
    
    with col1:
        # Aggregate data by state
        state_stats = filtered_df.groupby('State').agg({
            'Avg_Spending_per_Patient': 'mean',  # Average spending per state
            'Total_Spending': 'sum',              # Total spending per state
            'Facility_ID': 'count'                # Count of hospitals per state
        }).round(2).reset_index()
        state_stats = state_stats.rename(columns={
            'Facility_ID': 'Hospital_Count',
            'Avg_Spending_per_Patient': 'Avg_Spend_per_Patient'
        })
        
        # Create choropleth map of US states
        fig_map = px.choropleth(
            state_stats,
            locations='State',  # State abbreviations
            locationmode='USA-states',  # US state mode
            color='Avg_Spend_per_Patient',  # Color by average spending
            scope='usa',  # USA map scope
            color_continuous_scale='RdBu_r',  # Color scale (Red-Blue reversed)
            title='Average Medicare Spending per Patient by State',
            hover_data=['Hospital_Count', 'Total_Spending']  # Additional hover info
        )
        fig_map.update_layout(height=500)  # Set map height
        st.plotly_chart(fig_map, width='stretch')  # Display map with updated width parameter
    
    with col2:
        # Get top 10 states by average spending
        top_states = state_stats.nlargest(10, 'Avg_Spend_per_Patient')
        fig_bar = px.bar(
            top_states,
            x='State',
            y='Avg_Spend_per_Patient',
            color='Avg_Spend_per_Patient',  # Color bars by value
            title='Top 10 States by Average Spending per Patient',
            labels={'Avg_Spend_per_Patient': 'Average Spending ($)'},
            color_continuous_scale='Viridis'  # Color scale
        )
        fig_bar.update_layout(height=500)
        st.plotly_chart(fig_bar, width='stretch')  # Updated width parameter
        
        # Display state summary table
        st.markdown("##### State Summary")
        st.dataframe(
            state_stats.sort_values('Avg_Spend_per_Patient', ascending=False),  # Sort high to low
            width='stretch',  # Updated width parameter
            hide_index=True  # Don't show row numbers
        )

# TAB 2: Cost Analysis
with tab2:
    st.markdown("### Cost Distribution and Efficiency Analysis")
    
    col1, col2 = st.columns(2)  # Split tab into two columns
    
    with col1:
        # Histogram of spending distribution
        fig_hist = px.histogram(
            filtered_df,
            x='Avg_Spending_per_Patient',
            nbins=30,  # Number of bins for histogram
            title='Distribution of Average Spending per Patient',
            labels={'Avg_Spending_per_Patient': 'Spending per Patient ($)'},
            color_discrete_sequence=['#636EFA']  # Blue color
        )
        # Add vertical line for mean spending
        fig_hist.add_vline(
            x=filtered_df['Avg_Spending_per_Patient'].mean(),
            line_dash="dash",  # Dashed line
            line_color="red",
            annotation_text=f"Mean: ${filtered_df['Avg_Spending_per_Patient'].mean():,.0f}"  # Annotation text
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, width='stretch')  # Updated width parameter
        
        # Scatter plot: Cost vs Quality
        fig_scatter1 = px.scatter(
            filtered_df,
            x='Avg_Spending_per_Patient',
            y='Quality_Score',
            color='Hospital_Type',  # Color points by hospital type
            size='Medicare_Discharges',  # Size points by number of discharges
            hover_data=['Facility_Name', 'State'],  # Info on hover
            title='Cost vs Quality Score',
            labels={
                'Avg_Spending_per_Patient': 'Avg Spending per Patient ($)',
                'Quality_Score': 'Quality Score (1-5)'
            }
        )
        fig_scatter1.update_layout(height=400)
        st.plotly_chart(fig_scatter1, width='stretch')  # Updated width parameter
    
    with col2:
        # Aggregate data by hospital type
        type_stats = filtered_df.groupby('Hospital_Type').agg({
            'Avg_Spending_per_Patient': 'mean',
            'Quality_Score': 'mean',
            'Facility_ID': 'count'
        }).round(2).reset_index()
        type_stats = type_stats.rename(columns={'Facility_ID': 'Count'})
        
        # Bar chart: Spending by hospital type
        fig_bar2 = px.bar(
            type_stats,
            x='Hospital_Type',
            y='Avg_Spending_per_Patient',
            color='Quality_Score',  # Color by quality score
            title='Average Spending by Hospital Type',
            labels={
                'Avg_Spending_per_Patient': 'Avg Spending per Patient ($)',
                'Quality_Score': 'Avg Quality Score'
            },
            text='Avg_Spending_per_Patient'  # Show values on bars
        )
        fig_bar2.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')  # Format text
        fig_bar2.update_layout(height=400)
        st.plotly_chart(fig_bar2, width='stretch')  # Updated width parameter
        
        # Box plot: Cost efficiency by hospital type
        fig_box = px.box(
            filtered_df,
            y='Cost_Efficiency_Score',
            x='Hospital_Type',
            title='Cost Efficiency Score by Hospital Type',
            labels={'Cost_Efficiency_Score': 'Cost Efficiency Score'}
        )
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, width='stretch')  # Updated width parameter

# TAB 3: Quality Metrics
with tab3:
    st.markdown("### Quality and Outcome Metrics")
    
    col1, col2 = st.columns(2)  # Split tab into two columns
    
    with col1:
        # Scatter plot: Spending vs Readmission rate
        fig_scatter2 = px.scatter(
            filtered_df,
            x='Avg_Spending_per_Patient',
            y='Readmission_Rate',
            color='Mortality_Rate',  # Color by mortality rate
            size='Medicare_Discharges',  # Size by number of discharges
            hover_data=['Facility_Name', 'State', 'Hospital_Type'],
            title='Spending vs Readmission Rate',
            labels={
                'Avg_Spending_per_Patient': 'Avg Spending per Patient ($)',
                'Readmission_Rate': 'Readmission Rate (%)',
                'Mortality_Rate': 'Mortality Rate (%)'
            }
        )
        fig_scatter2.update_layout(height=500)
        st.plotly_chart(fig_scatter2, width='stretch')  # Updated width parameter
        
        # Create correlation matrix
        numeric_cols = ['Avg_Spending_per_Patient', 'Readmission_Rate', 
                       'Mortality_Rate', 'Patient_Satisfaction', 'Quality_Score']
        corr_matrix = filtered_df[numeric_cols].corr().round(2)  # Calculate correlations
        
        # Heatmap of correlation matrix
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=True,  # Show values in cells
            aspect='auto',  # Automatic aspect ratio
            color_continuous_scale='RdBu',  # Red-Blue color scale
            title='Correlation Matrix'
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, width='stretch')  # Updated width parameter
    
    with col2:
        # Violin plot: Patient satisfaction by hospital type
        fig_violin = px.violin(
            filtered_df,
            y='Patient_Satisfaction',
            x='Hospital_Type',
            box=True,  # Show box plot inside
            points='all',  # Show all points
            title='Patient Satisfaction by Hospital Type',
            labels={'Patient_Satisfaction': 'Patient Satisfaction Score (%)'}
        )
        fig_violin.update_layout(height=500)
        st.plotly_chart(fig_violin, width='stretch')  # Updated width parameter
        
        # Display summary statistics for quality metrics
        st.markdown("##### Quality Metrics Summary")
        quality_summary = filtered_df.agg({
            'Readmission_Rate': ['mean', 'min', 'max'],
            'Mortality_Rate': ['mean', 'min', 'max'],
            'Patient_Satisfaction': ['mean', 'min', 'max'],
            'Quality_Score': ['mean', 'min', 'max']
        }).round(2)
        st.dataframe(quality_summary, width='stretch')  # Updated width parameter

# TAB 4: Benchmarking
with tab4:
    st.markdown("### Hospital Benchmarking and Performance Comparison")
    
    # Top performers section header
    st.markdown("#### 🏆 Top Performers by Value Score")
    
    # Calculate value ratio: quality per dollar spent (normalized)
    filtered_df['Value_Ratio'] = (filtered_df['Quality_Score'] / 
                                 filtered_df['Avg_Spending_per_Patient'] * 10000).round(2)
    
    col1, col2 = st.columns(2)  # Split section into two columns
    
    with col1:
        # Top 10 hospitals by value ratio
        top_hospitals = filtered_df.nlargest(10, 'Value_Ratio')[['Facility_Name', 'State', 
                                                                 'Hospital_Type', 'Value_Ratio',
                                                                 'Quality_Score', 'Avg_Spending_per_Patient']]
        st.markdown("##### Top 10 Hospitals (Best Value)")
        st.dataframe(top_hospitals, width='stretch', hide_index=True)  # Updated width parameter
    
    with col2:
        # Bottom 10 hospitals by value ratio
        bottom_hospitals = filtered_df.nsmallest(10, 'Value_Ratio')[['Facility_Name', 'State',
                                                                     'Hospital_Type', 'Value_Ratio',
                                                                     'Quality_Score', 'Avg_Spending_per_Patient']]
        st.markdown("##### Bottom 10 Hospitals (Worst Value)")
        st.dataframe(bottom_hospitals, width='stretch', hide_index=True)  # Updated width parameter
    
    # Benchmarking analysis section
    st.markdown("#### 📊 Benchmarking Analysis")
    
    # Create options for benchmark hospital selection
    benchmark_options = filtered_df[['Facility_ID', 'Facility_Name', 'State']].copy()
    benchmark_options['Display'] = benchmark_options['Facility_Name'] + ' (' + benchmark_options['State'] + ')'
    
    # Dropdown to select a hospital for benchmarking
    selected_benchmark = st.selectbox(
        "Select a hospital to benchmark against:",
        benchmark_options['Display'].tolist(),
        index=0 if len(benchmark_options) > 0 else None  # Default to first hospital
    )
    
    if selected_benchmark:
        # Get the selected hospital's ID
        selected_id = benchmark_options[benchmark_options['Display'] == selected_benchmark]['Facility_ID'].values[0]
        # Get the hospital's data
        benchmark_hospital = filtered_df[filtered_df['Facility_ID'] == selected_id].iloc[0]
        
        # List of metrics to compare
        comparison_metrics = [
            'Avg_Spending_per_Patient',
            'Readmission_Rate', 
            'Mortality_Rate',
            'Patient_Satisfaction',
            'Quality_Score',
            'Cost_Efficiency_Score'
        ]
        
        # Create comparison data for each metric
        comparison_data = []
        for metric in comparison_metrics:
            hospital_value = benchmark_hospital[metric]  # Selected hospital's value
            avg_value = filtered_df[metric].mean()  # Average of all hospitals
            # Calculate percentile: % of hospitals with value <= selected hospital
            percentile = (filtered_df[metric] <= hospital_value).mean() * 100
            
            comparison_data.append({
                'Metric': metric.replace('_', ' ').title(),  # Format metric name
                'Hospital Value': round(hospital_value, 2),
                'Average': round(avg_value, 2),
                'Percentile': round(percentile, 1)
            })
        
        # Create DataFrame from comparison data
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.markdown(f"##### Benchmarking: {selected_benchmark}")
        st.dataframe(comparison_df, width='stretch', hide_index=True)  # Updated width parameter
        
        # Create bar chart comparing selected hospital to average
        fig_benchmark = go.Figure()
        
        # Bar for selected hospital
        fig_benchmark.add_trace(go.Bar(
            name='Selected Hospital',
            x=comparison_df['Metric'],
            y=comparison_df['Hospital Value'],
            marker_color='#3B82F6'  # Blue color
        ))
        
        # Bar for average values
        fig_benchmark.add_trace(go.Bar(
            name='Average',
            x=comparison_df['Metric'],
            y=comparison_df['Average'],
            marker_color='#94A3B8'  # Gray color
        ))
        
        # Update chart layout
        fig_benchmark.update_layout(
            title=f'{selected_benchmark} vs Average Comparison',
            barmode='group',  # Grouped bars
            height=400,
            xaxis_tickangle=-45  # Rotate x-axis labels
        )
        
        st.plotly_chart(fig_benchmark, width='stretch')  # Updated width parameter

# TAB 5: Raw Data
with tab5:
    st.markdown("### 📋 Raw Data View")
    
    # Show data summary
    st.markdown(f"**Displaying {len(filtered_df)} of {len(df)} total hospitals**")
    
    # Create CSV download button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download filtered data as CSV",
        data=csv,
        file_name=f"medicare_hospital_data_{selected_year}.csv",  # Dynamic filename
        mime="text/csv",  # MIME type for CSV
    )
    
    # Column selection for data table
    all_columns = filtered_df.columns.tolist()  # Get all column names
    default_columns = ['Facility_Name', 'State', 'Hospital_Type', 
                       'Avg_Spending_per_Patient', 'Quality_Score', 
                       'Readmission_Rate', 'Patient_Satisfaction']
    
    # Multi-select for choosing which columns to display
    selected_columns = st.multiselect(
        "Select columns to display:",
        all_columns,
        default=default_columns  # Default columns
    )
    
    # Filter DataFrame to selected columns
    if selected_columns:
        display_df = filtered_df[selected_columns]
    else:
        display_df = filtered_df  # Show all columns if none selected
    
    # Display the data table
    st.dataframe(
        display_df,
        width='stretch',  # Updated width parameter
        height=600  # Set table height
    )
    
    # Display summary statistics
    st.markdown("##### Data Statistics")
    st.dataframe(
        filtered_df.describe().round(2),  # Basic statistics
        width='stretch'  # Updated width parameter
    )

# Footer section
st.markdown("---")  # Horizontal line
st.markdown("""
<div style='text-align: center; color: #6B7280;'>
    <p>Medicare Hospital Spending Dashboard • Data is simulated for demonstration purposes</p>
    <p>For official Medicare data, visit: <a href='https://data.cms.gov' target='_blank'>data.cms.gov</a></p>
</div>
""", unsafe_allow_html=True)  # Centered footer with link
