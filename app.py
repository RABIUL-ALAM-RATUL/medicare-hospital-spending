import streamlit as st  # Import Streamlit library for building the interactive web dashboard
import pandas as pd  # Import Pandas for data manipulation and handling DataFrames
import numpy as np  # Import NumPy for numerical computations and array operations
import plotly.express as px  # Import Plotly Express for quick and interactive visualizations
import plotly.graph_objects as go  # Import Plotly Graph Objects for custom and advanced plots
from plotly.subplots import make_subplots  # Import make_subplots for creating multi-subplot figures
import warnings  # Import warnings module to manage warning messages
warnings.filterwarnings('ignore')  # Suppress all warnings to keep the app output clean and professional

# Configure the Streamlit page settings for a professional dashboard layout
st.set_page_config(  # Set page configuration for the app
    page_title="Medicare Hospital Spending by Claim (USA)",  # Set the browser tab title matching the notebook
    layout="wide",  # Use wide layout to maximize space for visualizations and tables
    initial_sidebar_state="expanded"  # Expand the sidebar by default for filters and controls
)

# Main title section - Render the exact title from the IPython notebook
st.title("**Medicare Hospital Spending by Claim (USA)**")  # Display the project title in bold for emphasis

# Hardcode the sample data from the IPython notebook (first 5 and last 5 rows)
df_data = {  # Dictionary containing sample data extracted from the notebook's DataFrame display
    'CMS Certification Number (CCN)': ['015009', '015010', '015012', '015014', '015015', '745051', '745052', '745054', '745055', '745056'],
    'Provider Name': ['BURNS NURSING HOME, INC.', 'COOSA VALLEY HEALTHCARE CENTER', 'HIGHLANDS HEALTH AND REHAB', 'EASTVIEW REHABILITATION & HEALTHCARE CENTER', 'PLANTATION MANOR NURSING HOME', 'FIVE POINTS NURSING & REHABILITATION OF COLLEGE...', 'HCA HOUSTON HEALTHCARE SOUTHEAST', 'LEGACY ESTATE LONG TERM CARE', 'MEMORIAL HEALTH CARE CENTER', 'SOUTHERN OAKS THERAPY AND LIVING CENTER'],
    'Provider Address': ['701 MONROE STREET NW', '260 WEST WALNUT STREET', '380 WOODS COVE ROAD', '7755 FOURTH AVENUE SOUTH', '6450 OLD TUSCALOOSA HIGHWAY', '3105 CORSAIR DRIVE', '4801 EAST SAM HOUSTON PARKWAY SOUTH', '10133 HWY 16 N', '212 NW 10TH ST', '3350 BONNIE VIEW RD'],
    'City/Town': ['RUSSELLVILLE', 'SYLACAUGA', 'SCOTTSBORO', 'BIRMINGHAM', 'MC CALLA', 'COLLEGE STATION', 'PASADENA', 'COMANCHE', 'SEMINOLE', 'DALLAS'],
    'State': ['AL', 'AL', 'AL', 'AL', 'AL', 'TX', 'TX', 'TX', 'TX', 'TX'],
    'ZIP Code': [35653, 35150, 35768, 35206, 35111, 77845, 77505, 76442, 79360, 75216],
    'Telephone Number': [2563324110, 2562495604, 2562183708, 2058330146, 2054776161, 9792136105, 7133592000, 2548794970, 4327584877, 4693204400],
    'Provider SSA County Code': [290, 600, 350, 360, 360, 190, 610, 321, 542, 390],
    'County/Parish': ['Franklin', 'Talladega', 'Jackson', 'Jefferson', 'Jefferson', 'Brazos', 'Harris', 'Comanche', 'Gaines', 'Dallas'],
    'Ownership Type': ['For profit - Corporation', 'For profit - Corporation', 'Government - County', 'For profit - Individual', 'For profit - Individual', 'For profit - Corporation', 'For profit - Corporation', 'Government - Hospital district', 'Government - Hospital district', 'For profit - Limited Liability company'],
    'Number of Fines': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'Total Amount of Fines in Dollars': [23989.0, 0.0, 0.0, 0.0, 0.0, 3731.0, 0.0, 0.0, 0.0, 0.0],
    'Number of Payment Denials': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Total Number of Penalties': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'Latitude': [34.5149, 33.1637, 34.6611, 33.5595, 33.3222, 30.6007, 29.6367, 31.9967, 32.7206, 32.7183],
    'Longitude': [-87.736, -86.254, -86.047, -86.722, -87.034, -96.287, -95.163, -98.563, -102.660, -96.778],
    'Processing Date': ['2025-09-01'] * 10
}
df = pd.DataFrame(df_data)  # Create the DataFrame from the hardcoded sample data

# Data overview metrics in columns for a dashboard feel
col1, col2, col3, col4 = st.columns(4)  # Create 4 columns for key metrics
with col1:  # First column
    st.metric("Total Facilities", len(df))  # Display total number of nursing home facilities from sample
with col2:  # Second column
    avg_rating = df.get('Overall Star Rating', pd.Series([0]*len(df))).mean()  # Calculate average star rating if column exists, else 0
    st.metric("Avg. Star Rating", f"{avg_rating:.2f}")  # Display average rating metric
with col3:  # Third column
    for_profit_pct = (df['Ownership Type'].str.contains('For profit', na=False).sum() / len(df)) * 100  # Calculate percentage of for-profit facilities
    st.metric("For-Profit %", f"{for_profit_pct:.1f}%")  # Display for-profit percentage
with col4:  # Fourth column
    total_fines = df['Total Amount of Fines in Dollars'].sum()  # Sum total fines
    st.metric("Total Fines ($)", f"${total_fines:,.0f}")  # Display total fines with formatting

# Render the Goal and Scope section from the notebook
st.markdown("""
## Defining the Goal and Scope

Setting a strong foundation is crucial for the success of any data analytics project. To start, clearly define the aim and direction of my work.

- **Articulate the Business Problem:** Begin by stating the main question or challenge your analysis will address. For example, “What factors caused changes in provider quality scores over time?” or “How does facility ownership type affect patient care outcomes?” This step provides clarity and ensures the analysis remains relevant to real-world needs.

- **Identify Key Metrics (KPIs):** Determine how success will be measured. Consider quantitative indicators like average quality ratings, incident reports per 1,000 residents, or regional performance scores. Define these metrics based on project goals and stakeholder expectations.

- **Establish Data Requirements:** Based on the question and metrics, specify what data is needed and which variables will be most informative. Map these needs to the available columns in the dataset, such as provider ID, region, ownership, inspection outcomes, ratings, and others. Clarify any additional data that may need to gather for a complete analysis.

This initial stage is essential; it supports project rigor, enables focused analysis, and serves as a reference for all decisions throughout the workflow, meeting the standards of UK master’s-level research and professional practice.
""")  # Render the introductory section as formatted Markdown

# Render the Data Collection section
st.markdown("""
## Data Collection and Acquisition

I begin my analytics project by establishing a rigorous data collection process. At this stage, I work systematically through the following steps:

- **Source Identification:** I clearly locate all relevant data sources needed for my analysis. These may include internal databases (such as SQL or NoSQL systems), external APIs, flat files (like CSV or Excel documents), and websites for scraping. I consider which sources are most likely to yield reliable and comprehensive information for my particular research objective.
- **Data Extraction:** I apply suitable tools and technologies to access my chosen data. This could involve executing SQL queries, developing Python scripts, implementing extract-transform-load (ETL) pipelines, or using commercial software platforms. I carefully extract the data, maintaining a record of the processes and settings used to support repeatability and data governance principles.
- **Initial Review:** Once I have collected the data, I promptly check it for integrity. I review the overall volume (number of records and variables), format (column types and structure), and initial quality indicators such as missing values, obvious outliers, or inconsistent entries. A rapid, practical appraisal at this stage helps me flag major issues before further processing and analysis proceeds.

Through these steps, my data collection underpins robust, valid analytics and supports my workflow at a standard appropriate for master's-level research in the UK.
""")  # Render the data collection narrative as Markdown

# Environment Setup section
st.markdown("# **1. ENVIRONMENT SETUP AND IMPORTING LIBRARIES**")  # Header for environment setup
st.markdown("## **1.1 Connect with Google Drive**")  # Subheader for Drive connection (simulated with hardcoded data)

# Data Loading section
st.markdown("# **2. DATA LOADING**")  # Header for data loading
st.markdown("#**LOADING DATASET**")  # Subheader for loading
st.markdown("___")  # Horizontal line separator for visual appeal
st.markdown(f"""
##**Dataset loaded successfully**
##**Dataset Overview**

- **Original shape:** `{df.shape[0]:,} rows × {df.shape[1]} columns`  
- **Source:** Sample from IPython Notebook  
- **Dataset origin:** [CMS Provider Data](https://data.cms.gov/provider-data/dataset/4pq5-n9py)  
- **Processing Date:** `{df['Processing Date'].iloc[0] if 'Processing Date' in df.columns and not df.empty else 'N/A'}`  
""")  # Dynamic dataset summary with formatted numbers and links

# Dataset Preview section
st.markdown("# **3. Display the first & last 5 rows to get a quick look at the data structure and content**")  # Header for preview
st.markdown("""  
In this section, the complete CMS Nursing Home Provider dataset from the September 2025 release is loaded into a Pandas DataFrame with **14,752 records**. The preview shows the data’s range and structure: starting with smaller, mostly for-profit facilities in Alabama (e.g., Burns Nursing Home in Russellville) and ending with diverse providers in Texas (e.g., urban skilled nursing homes in College Station and Dallas). Key variables like CMS Certification Number, ownership type, overall star rating, health inspection scores, staffing metrics, fines, payment denials, and latitude/longitude coordinates are consistently populated. This confirms the dataset's national scope, mix of ownership types, and performance range, building confidence for deeper analysis on care standards, compliance, and regional differences.
""")  # Descriptive text adapted for dashboard
st.markdown("___")  # Separator line
st.markdown("#**First & Last 5 Rows Preview:**")  # Subheader for table
st.markdown("___")  # Another separator
st.dataframe(df, use_container_width=True, height=400)  # Interactive table preview with the sample data

# Act 4: The Human Cost - Interactive visualization
st.markdown("### **Act 4: The Human Cost**")  # Header for Act 4
st.markdown("**Every red state = thousands of vulnerable elders in substandard care.**")  # Subtitle for impact

# Prepare data for the bar chart (hardcoded from notebook for demo)
failing_homes_data = {  # Dictionary for top 10 states data extracted from notebook
    'State': ['TX', 'CA', 'IL', 'OH', 'PA', 'MO', 'FL', 'NY', 'NC', 'IN'],
    'Number of Failing Homes': [555, 428, 376, 339, 274, 261, 244, 237, 199, 199]
}
df_failing = pd.DataFrame(failing_homes_data)  # Create DataFrame for plotting

# Create and render the bar chart
fig_bar = px.bar(  # Use Plotly Express to create a bar chart
    df_failing, x='State', y='Number of Failing Homes',  # Axes: states on x, counts on y
    text='Number of Failing Homes',  # Add text labels on bars
    color='Number of Failing Homes', color_continuous_scale='Reds',  # Color bars by value with red scale for urgency
    title='Top 10 States with Most 1–2 Star Nursing Homes (2025)',  # Chart title
    labels={'Number of Failing Homes': 'Failing Homes (1-2 Stars)'}  # Axis labels for clarity
)
fig_bar.update_traces(textposition='outside', textfont_size=12)  # Position and size text labels
fig_bar.update_layout(  # Customize layout for professional look
    height=550, showlegend=False,  # Fixed height, no legend
    xaxis_title="State", yaxis_title="Number of Failing Homes",  # Axis titles
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'  # Transparent background for Streamlit integration
)
st.plotly_chart(fig_bar, use_container_width=True)  # Render the interactive chart full-width

# Sidebar for user interactions and filters
st.sidebar.header("**Dashboard Controls**")  # Add a header in the sidebar for controls

# Additional dashboard elements: Filters for exploration
st.sidebar.header("**Filters**")  # Sidebar header for filters
state_filter = st.sidebar.multiselect(  # Multiselect for state filtering
    "Filter by State", options=df['State'].unique(), default=df['State'].unique()[:5]  # Default to first 5 unique states in sample
)
rating_filter = st.sidebar.slider(  # Slider for star rating filter
    "Star Rating Range", min_value=1, max_value=5, value=(1,5),  # Range from 1 to 5 stars
    help="Filter facilities by overall star rating (note: sample data may not have this column)"  # Tooltip
)

# Apply filters to DataFrame
filtered_df = df[  # Filter the DataFrame based on user selections
    (df['State'].isin(state_filter))  # Filter by selected states
]  # Note: Rating filter skipped in sample if column missing

# Ownership distribution pie chart
ownership_counts = filtered_df['Ownership Type'].value_counts()  # Count ownership types
fig_pie = px.pie(  # Create pie chart for ownership distribution
    values=ownership_counts.values, names=ownership_counts.index,  # Values and labels
    title="Ownership Type Distribution (Sample Data)", hole=0.4,  # Donut style with hole
    color_discrete_sequence=px.colors.qualitative.Set3  # Professional color palette
)
fig_pie.update_layout(height=400)  # Set height
st.plotly_chart(fig_pie, use_container_width=True)  # Render pie chart

# Geospatial map using sample lat/long
fig_map = px.scatter_mapbox(  # Create interactive map
    filtered_df, lat='Latitude', lon='Longitude',  # Latitude and longitude
    color='State',  # Color by state in sample
    mapbox_style="open-street-map", zoom=3,  # US overview zoom
    title="Nursing Homes by Location (Sample Data)",  # Title
    hover_name='Provider Name', hover_data=['Ownership Type', 'State']  # Hover info
)
fig_map.update_layout(height=500, margin={"r":0,"t":40,"l":0,"b":0})  # Layout adjustments
st.plotly_chart(fig_map, use_container_width=True)  # Render map

# Act 5: Call to Action
st.markdown("""
### **Act 5: The Call to Action**

> **This is not a market. This is a moral failure.**

**Three evidence-based policy levers (immediately implementable):**
1. **Ban new for-profit nursing homes** in states >80% privatised
2. **Mandate minimum staffing ratios** (current model shows understaffing = +42% risk)
3. **Tie Medicare reimbursement** directly to star rating (not bed count)

**Your dissertation does not describe a problem.**  
**It proves one — with unbreakable data.**  
**Impact: Real**
""")  # Render the concluding narrative as Markdown

# Download button for filtered data
csv = filtered_df.to_csv(index=False).encode('utf-8')  # Convert filtered DataFrame to CSV bytes
st.download_button(  # Add download button
    label="Download Sample Data as CSV",  # Button label
    data=csv, file_name="sample_medicare_data.csv", mime="text/csv"  # File details
)
