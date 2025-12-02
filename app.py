import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Medicare Hospital Spending Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🏥 Medicare Hospital Spending Analysis</h1>', unsafe_allow_html=True)
st.markdown("""
This dashboard provides insights into Medicare spending across hospitals, analyzing costs, quality metrics, 
and geographic variations to identify opportunities for improving healthcare efficiency.
""")

# Create sidebar for filters
with st.sidebar:
    st.markdown("### 🎯 Filters")
    
    # Year selection
    years = [2020, 2021, 2022, 2023, 2024]
    selected_year = st.selectbox("Select Year", years, index=len(years)-1)
    
    # State selection (multi-select)
    states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
              'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
              'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
              'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
              'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC']
    selected_states = st.multiselect("Select States", states, default=['CA', 'TX', 'NY', 'FL'])
    
    # Hospital type filter
    hospital_types = ['All Types', 'Acute Care', 'Critical Access', 'Teaching', 'Children\'s', 'Rural']
    selected_type = st.selectbox("Hospital Type", hospital_types, index=0)
    
    # Spending range filter
    min_spend, max_spend = st.slider(
        "Average Spending per Patient ($)",
        10000, 50000, (20000, 40000)
    )
    
    st.markdown("---")
    st.markdown("### 📊 Dashboard Guide")
    st.markdown("""
    1. **Overview Metrics**: Key statistics
    2. **Geographic Analysis**: State-level spending
    3. **Cost Analysis**: Spending distribution
    4. **Quality Metrics**: Outcome correlations
    5. **Benchmarking**: Performance comparison
    """)

# Sample data generation with caching
@st.cache_data
def create_sample_data():
    np.random.seed(42)
    n_facilities = 500
    
    # States with equal probability (no p parameter)
    states_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
                   'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
                   'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
                   'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
                   'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC']
    
    data = {
        'Facility_ID': [f'HOSP{i:04d}' for i in range(1, n_facilities + 1)],
        'Facility_Name': [f'Hospital {i}' for i in range(1, n_facilities + 1)],
        'State': np.random.choice(states_list, n_facilities),
        'City': np.random.choice(['Metro', 'Urban', 'Suburban', 'Rural'], n_facilities, p=[0.4, 0.3, 0.2, 0.1]),
        'Hospital_Type': np.random.choice(['Acute Care', 'Critical Access', 'Teaching', 'Children\'s', 'Rural'], 
                                          n_facilities, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
        'Total_Spending': np.random.uniform(10000000, 500000000, n_facilities).round(2),
        'Avg_Spending_per_Patient': np.random.uniform(15000, 45000, n_facilities).round(2),
        'Medicare_Discharges': np.random.randint(100, 10000, n_facilities),
        'Readmission_Rate': np.random.uniform(10, 25, n_facilities).round(1),
        'Mortality_Rate': np.random.uniform(1, 10, n_facilities).round(1),
        'Patient_Satisfaction': np.random.uniform(60, 95, n_facilities).round(1),
        'Quality_Score': np.random.uniform(1, 5, n_facilities).round(1),
        'Cost_Efficiency_Score': np.random.uniform(0.5, 1.5, n_facilities).round(2),
        'Year': np.random.choice([2022, 2023, 2024], n_facilities, p=[0.2, 0.3, 0.5])
    }
    
    df = pd.DataFrame(data)
    
    # Calculate some derived metrics
    df['Spending_per_Discharge'] = (df['Total_Spending'] / df['Medicare_Discharges']).round(2)
    df['Value_Score'] = (df['Quality_Score'] / df['Cost_Efficiency_Score']).round(2)
    
    return df

# Load data
try:
    if 'df' not in st.session_state:
        st.session_state.df = create_sample_data()
    df = st.session_state.df
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Apply filters
filtered_df = df.copy()

# Year filter
filtered_df = filtered_df[filtered_df['Year'] == selected_year]

# State filter
if selected_states:
    filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]

# Hospital type filter
if selected_type != 'All Types':
    filtered_df = filtered_df[filtered_df['Hospital_Type'] == selected_type]

# Spending range filter
filtered_df = filtered_df[
    (filtered_df['Avg_Spending_per_Patient'] >= min_spend) & 
    (filtered_df['Avg_Spending_per_Patient'] <= max_spend)
]

# Display metrics in columns
st.markdown("## 📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_hospitals = len(filtered_df)
    st.metric("Total Hospitals", f"{total_hospitals:,}")
    
with col2:
    avg_spending = filtered_df['Avg_Spending_per_Patient'].mean()
    st.metric("Avg Spending per Patient", f"${avg_spending:,.0f}")
    
with col3:
    avg_readmission = filtered_df['Readmission_Rate'].mean()
    st.metric("Avg Readmission Rate", f"{avg_readmission:.1f}%")
    
with col4:
    avg_satisfaction = filtered_df['Patient_Satisfaction'].mean()
    st.metric("Avg Patient Satisfaction", f"{avg_satisfaction:.1f}%")

# Main dashboard sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 Geographic Analysis", 
    "💰 Cost Analysis", 
    "🏆 Quality Metrics", 
    "📊 Benchmarking",
    "📋 Raw Data"
])

with tab1:
    st.markdown("### State-Level Spending Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # State aggregation
        state_stats = filtered_df.groupby('State').agg({
            'Avg_Spending_per_Patient': 'mean',
            'Total_Spending': 'sum',
            'Facility_ID': 'count'
        }).round(2).reset_index()
        state_stats = state_stats.rename(columns={
            'Facility_ID': 'Hospital_Count',
            'Avg_Spending_per_Patient': 'Avg_Spend_per_Patient'
        })
        
        # Choropleth map
        fig = px.choropleth(
            state_stats,
            locations='State',
            locationmode='USA-states',
            color='Avg_Spend_per_Patient',
            scope='usa',
            color_continuous_scale='RdBu_r',
            title='Average Medicare Spending per Patient by State',
            hover_data=['Hospital_Count', 'Total_Spending']
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top 10 states by spending
        top_states = state_stats.nlargest(10, 'Avg_Spend_per_Patient')
        fig2 = px.bar(
            top_states,
            x='State',
            y='Avg_Spend_per_Patient',
            color='Avg_Spend_per_Patient',
            title='Top 10 States by Average Spending per Patient',
            labels={'Avg_Spend_per_Patient': 'Average Spending ($)'},
            color_continuous_scale='Viridis'
        )
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)
        
        # State summary table
        st.markdown("##### State Summary")
        st.dataframe(
            state_stats.sort_values('Avg_Spend_per_Patient', ascending=False),
            use_container_width=True,
            hide_index=True
        )

with tab2:
    st.markdown("### Cost Distribution and Efficiency Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Spending distribution histogram
        fig1 = px.histogram(
            filtered_df,
            x='Avg_Spending_per_Patient',
            nbins=30,
            title='Distribution of Average Spending per Patient',
            labels={'Avg_Spending_per_Patient': 'Spending per Patient ($)'},
            color_discrete_sequence=['#636EFA']
        )
        fig1.add_vline(
            x=filtered_df['Avg_Spending_per_Patient'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: ${filtered_df['Avg_Spending_per_Patient'].mean():,.0f}"
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Cost vs Quality scatter plot
        fig3 = px.scatter(
            filtered_df,
            x='Avg_Spending_per_Patient',
            y='Quality_Score',
            color='Hospital_Type',
            size='Medicare_Discharges',
            hover_data=['Facility_Name', 'State'],
            title='Cost vs Quality Score',
            labels={
                'Avg_Spending_per_Patient': 'Avg Spending per Patient ($)',
                'Quality_Score': 'Quality Score (1-5)'
            }
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Spending by hospital type
        type_stats = filtered_df.groupby('Hospital_Type').agg({
            'Avg_Spending_per_Patient': 'mean',
            'Quality_Score': 'mean',
            'Facility_ID': 'count'
        }).round(2).reset_index()
        type_stats = type_stats.rename(columns={'Facility_ID': 'Count'})
        
        fig2 = px.bar(
            type_stats,
            x='Hospital_Type',
            y='Avg_Spending_per_Patient',
            color='Quality_Score',
            title='Average Spending by Hospital Type',
            labels={
                'Avg_Spending_per_Patient': 'Avg Spending per Patient ($)',
                'Quality_Score': 'Avg Quality Score'
            },
            text='Avg_Spending_per_Patient'
        )
        fig2.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Cost efficiency distribution
        fig4 = px.box(
            filtered_df,
            y='Cost_Efficiency_Score',
            x='Hospital_Type',
            title='Cost Efficiency Score by Hospital Type',
            labels={'Cost_Efficiency_Score': 'Cost Efficiency Score'}
        )
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.markdown("### Quality and Outcome Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Readmission vs Spending
        fig1 = px.scatter(
            filtered_df,
            x='Avg_Spending_per_Patient',
            y='Readmission_Rate',
            color='Mortality_Rate',
            size='Medicare_Discharges',
            hover_data=['Facility_Name', 'State', 'Hospital_Type'],
            title='Spending vs Readmission Rate',
            labels={
                'Avg_Spending_per_Patient': 'Avg Spending per Patient ($)',
                'Readmission_Rate': 'Readmission Rate (%)',
                'Mortality_Rate': 'Mortality Rate (%)'
            }
        )
        fig1.update_layout(height=500)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Correlation matrix
        numeric_cols = ['Avg_Spending_per_Patient', 'Readmission_Rate', 
                       'Mortality_Rate', 'Patient_Satisfaction', 'Quality_Score']
        corr_matrix = filtered_df[numeric_cols].corr().round(2)
        
        fig3 = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect='auto',
            color_continuous_scale='RdBu',
            title='Correlation Matrix'
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Patient satisfaction distribution
        fig2 = px.violin(
            filtered_df,
            y='Patient_Satisfaction',
            x='Hospital_Type',
            box=True,
            points='all',
            title='Patient Satisfaction by Hospital Type',
            labels={'Patient_Satisfaction': 'Patient Satisfaction Score (%)'}
        )
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Quality metrics summary
        st.markdown("##### Quality Metrics Summary")
        quality_summary = filtered_df.agg({
            'Readmission_Rate': ['mean', 'min', 'max'],
            'Mortality_Rate': ['mean', 'min', 'max'],
            'Patient_Satisfaction': ['mean', 'min', 'max'],
            'Quality_Score': ['mean', 'min', 'max']
        }).round(2)
        st.dataframe(quality_summary, use_container_width=True)

with tab4:
    st.markdown("### Hospital Benchmarking and Performance Comparison")
    
    # Top performers section
    st.markdown("#### 🏆 Top Performers by Value Score")
    
    # Calculate value score (quality per dollar spent)
    filtered_df['Value_Ratio'] = (filtered_df['Quality_Score'] / 
                                 filtered_df['Avg_Spending_per_Patient'] * 10000).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 hospitals by value score
        top_hospitals = filtered_df.nlargest(10, 'Value_Ratio')[['Facility_Name', 'State', 
                                                                 'Hospital_Type', 'Value_Ratio',
                                                                 'Quality_Score', 'Avg_Spending_per_Patient']]
        st.markdown("##### Top 10 Hospitals (Best Value)")
        st.dataframe(top_hospitals, use_container_width=True, hide_index=True)
    
    with col2:
        # Bottom 10 hospitals by value score
        bottom_hospitals = filtered_df.nsmallest(10, 'Value_Ratio')[['Facility_Name', 'State',
                                                                     'Hospital_Type', 'Value_Ratio',
                                                                     'Quality_Score', 'Avg_Spending_per_Patient']]
        st.markdown("##### Bottom 10 Hospitals (Worst Value)")
        st.dataframe(bottom_hospitals, use_container_width=True, hide_index=True)
    
    # Benchmarking visualization
    st.markdown("#### 📊 Benchmarking Analysis")
    
    # Select benchmark hospital
    benchmark_options = filtered_df[['Facility_ID', 'Facility_Name', 'State']].copy()
    benchmark_options['Display'] = benchmark_options['Facility_Name'] + ' (' + benchmark_options['State'] + ')'
    
    selected_benchmark = st.selectbox(
        "Select a hospital to benchmark against:",
        benchmark_options['Display'].tolist(),
        index=0 if len(benchmark_options) > 0 else None
    )
    
    if selected_benchmark:
        selected_id = benchmark_options[benchmark_options['Display'] == selected_benchmark]['Facility_ID'].values[0]
        benchmark_hospital = filtered_df[filtered_df['Facility_ID'] == selected_id].iloc[0]
        
        # Create comparison metrics
        comparison_metrics = [
            'Avg_Spending_per_Patient',
            'Readmission_Rate', 
            'Mortality_Rate',
            'Patient_Satisfaction',
            'Quality_Score',
            'Cost_Efficiency_Score'
        ]
        
        comparison_data = []
        for metric in comparison_metrics:
            hospital_value = benchmark_hospital[metric]
            avg_value = filtered_df[metric].mean()
            percentile = (filtered_df[metric] <= hospital_value).mean() * 100
            
            comparison_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Hospital Value': round(hospital_value, 2),
                'Average': round(avg_value, 2),
                'Percentile': round(percentile, 1)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.markdown(f"##### Benchmarking: {selected_benchmark}")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Visualization of benchmark comparison
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Selected Hospital',
            x=comparison_df['Metric'],
            y=comparison_df['Hospital Value'],
            marker_color='#3B82F6'
        ))
        
        fig.add_trace(go.Bar(
            name='Average',
            x=comparison_df['Metric'],
            y=comparison_df['Average'],
            marker_color='#94A3B8'
        ))
        
        fig.update_layout(
            title=f'{selected_benchmark} vs Average Comparison',
            barmode='group',
            height=400,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.markdown("### 📋 Raw Data View")
    
    # Show data summary
    st.markdown(f"**Displaying {len(filtered_df)} of {len(df)} total hospitals**")
    
    # Data download option
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download filtered data as CSV",
        data=csv,
        file_name=f"medicare_hospital_data_{selected_year}.csv",
        mime="text/csv",
    )
    
    # Column selection
    all_columns = filtered_df.columns.tolist()
    default_columns = ['Facility_Name', 'State', 'Hospital_Type', 
                       'Avg_Spending_per_Patient', 'Quality_Score', 
                       'Readmission_Rate', 'Patient_Satisfaction']
    
    selected_columns = st.multiselect(
        "Select columns to display:",
        all_columns,
        default=default_columns
    )
    
    if selected_columns:
        display_df = filtered_df[selected_columns]
    else:
        display_df = filtered_df
    
    # Data table with pagination
    st.dataframe(
        display_df,
        use_container_width=True,
        height=600
    )
    
    # Data statistics
    st.markdown("##### Data Statistics")
    st.dataframe(
        filtered_df.describe().round(2),
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280;'>
    <p>Medicare Hospital Spending Dashboard • Data is simulated for demonstration purposes</p>
    <p>For official Medicare data, visit: <a href='https://data.cms.gov' target='_blank'>data.cms.gov</a></p>
</div>
""", unsafe_allow_html=True)
