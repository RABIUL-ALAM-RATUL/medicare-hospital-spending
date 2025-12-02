import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- Configuration and Setup ---

# Set the page configuration for a professional look
st.set_page_config(
    page_title="Healthcare Facility Performance Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the path to the Parquet file
DATA_FILE = "/home/ubuntu/upload/df_final.parquet"

# --- Data Loading and Caching ---

@st.cache_data
def load_data():
    """
    Loads the data from the Parquet file and performs initial cleaning/preparation.
    Streamlit's cache_data decorator ensures the data is loaded only once.
    """
    try:
        # Read the Parquet file into a pandas DataFrame
        df = pd.read_parquet(DATA_FILE)
        
        # Rename columns for better display in the dashboard
        df.columns = df.columns.str.replace(r'\(CCN\)', '', regex=True).str.strip()
        df.rename(columns={
            "Overall Rating": "Overall_Rating",
            "Health Inspection Rating": "Health_Inspection_Rating",
            "Staffing Rating": "Staffing_Rating",
            "QM Rating": "QM_Rating",
            "Total Amount of Fines in Dollars": "Total_Fines_USD",
            "Ownership Type": "Ownership_Type",
            "Reported RN Staffing Hours per Resident per Day": "RN_Staffing_Hours",
            "Total nursing staff turnover": "Total_Staff_Turnover",
            "CMS Certification Number": "CCN",
            "Provider Name": "Provider_Name",
            "State": "State",
            "Latitude": "Latitude",
            "Longitude": "Longitude"
        }, inplace=True)

        # Convert key rating columns to string/category for better filtering/plotting
        rating_cols = ["Overall_Rating", "Health_Inspection_Rating", "Staffing_Rating", "QM_Rating"]
        for col in rating_cols:
            # Replace -1 (often used for not applicable/not rated) with NaN
            df[col] = df[col].replace(-1, np.nan)
            # Convert to integer type for proper sorting/display
            df[col] = df[col].astype('Int64')
        
        # Filter out rows where the primary ratings are missing
        df.dropna(subset=["Overall_Rating"], inplace=True)

        return df
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {DATA_FILE}. Please ensure the file is uploaded correctly.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return pd.DataFrame()

# Load the data
df = load_data()

# Check if data loaded successfully
if df.empty:
    st.stop()

# --- Sidebar for Filtering ---

# Add a title to the sidebar
st.sidebar.header("Filter Options")

# State Multi-Select Filter
# Get a sorted list of unique states
states = sorted(df['State'].unique().tolist())
selected_states = st.sidebar.multiselect(
    "Select State(s)",
    options=states,
    default=states[:5] # Select the first 5 states by default
)

# Ownership Type Multi-Select Filter
ownership_types = sorted(df['Ownership_Type'].unique().tolist())
selected_ownership = st.sidebar.multiselect(
    "Select Ownership Type(s)",
    options=ownership_types,
    default=ownership_types # Select all by default
)

# Apply filters to the DataFrame
df_filtered = df[
    df['State'].isin(selected_states) & 
    df['Ownership_Type'].isin(selected_ownership)
]

# Display the number of facilities selected
st.sidebar.info(f"Showing {len(df_filtered)} facilities out of {len(df)} total.")

# --- Main Dashboard Content ---

st.title("🏥 U.S. Healthcare Facility Performance Dashboard")
st.markdown("A comprehensive overview of facility ratings, staffing, and financial penalties.")

# 1. Key Performance Indicators (KPIs)
st.header("Key Performance Indicators (KPIs)")

# Calculate KPIs on the filtered data
total_facilities = len(df_filtered)
avg_overall_rating = df_filtered['Overall_Rating'].mean()
total_fines = df_filtered['Total_Fines_USD'].sum()
avg_rn_staffing = df_filtered['RN_Staffing_Hours'].mean()

# Create columns for the metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Facilities (Filtered)", 
        value=f"{total_facilities:,}"
    )

with col2:
    st.metric(
        label="Average Overall Rating (1-5 Stars)", 
        value=f"{avg_overall_rating:.2f}",
        delta="5.00 is best" # Contextual information for the user
    )

with col3:
    st.metric(
        label="Total Fines Imposed (USD)", 
        value=f"${total_fines:,.0f}"
    )

with col4:
    st.metric(
        label="Avg. RN Staffing Hours/Resident/Day", 
        value=f"{avg_rn_staffing:.2f} hrs"
    )

st.markdown("---")

# 2. Geographical Map of Facilities (if coordinates are available)
if 'Latitude' in df_filtered.columns and 'Longitude' in df_filtered.columns:
    st.header("Facility Locations and Overall Rating")
    
    # Clean coordinates and filter out invalid entries
    map_data = df_filtered.dropna(subset=['Latitude', 'Longitude', 'Overall_Rating']).copy()
    
    # Use Plotly for an interactive map for better user experience
    fig_map = px.scatter_mapbox(
        map_data,
        lat="Latitude",
        lon="Longitude",
        color="Overall_Rating", # Color points by the overall rating
        size="Overall_Rating",  # Size points by the overall rating
        hover_name="Provider_Name",
        hover_data={"State": True, "Overall_Rating": True, "Total_Fines_USD": ":,.0f", "Latitude": False, "Longitude": False},
        color_continuous_scale=px.colors.sequential.Viridis, # Use a professional color scale
        zoom=3,
        height=500
    )

    # Set map style to a professional, non-default look
    fig_map.update_layout(
        mapbox_style="carto-positron",
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown("---")

# 3. Detailed Visualizations
st.header("Detailed Performance Analysis")

# Use two columns for side-by-side charts
chart_col1, chart_col2 = st.columns(2)

# Chart 1: Distribution of Overall Ratings
with chart_col1:
    st.subheader("Overall Rating Distribution")
    
    # Calculate the count for each rating (1 to 5)
    rating_counts = df_filtered['Overall_Rating'].value_counts().sort_index()
    rating_df = rating_counts.reset_index()
    rating_df.columns = ['Rating', 'Count']
    
    # Create a bar chart using Plotly
    fig_rating = px.bar(
        rating_df, 
        x='Rating', 
        y='Count', 
        color='Rating',
        title="Count of Facilities by Overall Star Rating",
        labels={'Rating': 'Overall Star Rating', 'Count': 'Number of Facilities'},
        color_continuous_scale=px.colors.sequential.Plasma
    )
    fig_rating.update_layout(xaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_rating, use_container_width=True)

# Chart 2: Total Fines by Ownership Type
with chart_col2:
    st.subheader("Total Fines by Ownership Type")
    
    # Group by Ownership Type and sum the fines
    fines_by_ownership = df_filtered.groupby('Ownership_Type')['Total_Fines_USD'].sum().reset_index()
    fines_by_ownership.columns = ['Ownership_Type', 'Total_Fines']
    
    # Create a pie chart to show proportion
    fig_fines = px.pie(
        fines_by_ownership, 
        values='Total_Fines', 
        names='Ownership_Type', 
        title='Proportion of Total Fines by Ownership Type',
        hole=.3 # Make it a donut chart for better aesthetics
    )
    fig_fines.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_fines, use_container_width=True)

# 4. Scatter Plot: Staffing vs. Health Inspection
st.subheader("Staffing vs. Health Inspection Rating")
st.markdown("Exploring the relationship between staffing quality and health inspection outcomes.")

# Create a scatter plot
fig_scatter = px.scatter(
    df_filtered,
    x="Staffing_Rating",
    y="Health_Inspection_Rating",
    color="Overall_Rating", # Use Overall Rating as color for context
    hover_name="Provider_Name",
    title="Staffing Rating vs. Health Inspection Rating",
    labels={
        "Staffing_Rating": "Staffing Rating (1-5 Stars)",
        "Health_Inspection_Rating": "Health Inspection Rating (1-5 Stars)"
    },
    color_continuous_scale=px.colors.sequential.Plasma,
    opacity=0.6
)

# Ensure axes are clearly defined for 1-5 star ratings
fig_scatter.update_xaxes(tickvals=[1, 2, 3, 4, 5], range=[0.5, 5.5])
fig_scatter.update_yaxes(tickvals=[1, 2, 3, 4, 5], range=[0.5, 5.5])

st.plotly_chart(fig_scatter, use_container_width=True)

# --- Raw Data Section (Optional but professional) ---
st.markdown("---")
st.header("Raw Data Table")
st.caption("Displaying the first 1,000 rows of the filtered data.")

# Display the raw data table
st.dataframe(df_filtered.head(1000))

# --- Footer ---
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #888;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
    </style>
    <div class="footer">
        Data Source: df_final.parquet | Dashboard created with Streamlit and Plotly.
    </div>
    """, 
    unsafe_allow_html=True
)
