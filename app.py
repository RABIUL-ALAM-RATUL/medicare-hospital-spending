# app.py → DASHBOARD
import streamlit as st  # Imports the Streamlit library for building the web app
import pandas as pd     # Imports Pandas for data manipulation (reading parquet, filtering)
import plotly.express as px  # Imports Plotly Express for creating interactive charts

# Configures the browser tab title to "Medicare..." and sets the layout mode to "wide" (full screen width)
st.set_page_config(page_title="Medicare Hospital Spending by Claim (USA)", layout="wide")

# Title Section
st.title("Medicare Hospital Spending by Claim (USA)")  # Displays the main H1 title
st.markdown("**United States • CMS 2025 Data • 14,752 Certified Facilities**")  # Displays bold subtitle text using Markdown

# Load data once
@st.cache_data  # Decorator to cache the data; prevents reloading the file on every user interaction
def load_data():
    df = pd.read_parquet("df_final.parquet")  # Reads the compressed, cleaned dataset (Parquet is faster than CSV)
    df['code'] = df['State'].str.upper()      # Creates a 'code' column with uppercase state abbreviations for map plotting
    return df

df = load_data()  # Calls the function to load the data into the 'df' variable

# Auto-detect important columns helper function
def find_col(patterns):
    for p in patterns:  # Iterates through a list of possible column name patterns
        matches = [c for c in df.columns if p.lower() in c.lower()]  # Finds columns matching the pattern (case-insensitive)
        if matches: return matches[0]  # Returns the first match found
    return None  # Returns None if no matching column is found

# Define key column names using the helper function to ensure they exist
name_col   = find_col(['Provider Name', 'Facility Name', 'Name'])
city_col   = find_col(['City'])
state_col  = find_col(['State'])
rating_col = find_col(['Overall Rating', 'Star Rating', 'Rating'])
owner_col  = find_col(['Ownership'])

# ==================== KPIs (Key Performance Indicators) ====================
c1, c2, c3, c4, c5 = st.columns(5)  # Creates 5 side-by-side columns for metrics
c1.metric("Total Facilities", f"{len(df):,}")  # Display total count of rows (facilities)
# Display count and % of For-Profit homes (where Risk Score == 3)
c2.metric("For-Profit", f"{(df['Ownership_Risk_Score']==3).sum():,}", f"{(df['Ownership_Risk_Score']==3).mean():.1%}")
# Display count and % of Low Quality homes (1-2 stars)
c3.metric("1–2 Star Homes", f"{df['Low_Quality_Facility'].sum():,}", f"{df['Low_Quality_Facility'].mean():.1%}")
# Display average chronic deficiency score
c4.metric("Chronic Deficiencies (Avg)", f"{df['Chronic_Deficiency_Score'].mean():.2f}")
# Display the model accuracy (static value from your analysis)
c5.metric("Model Accuracy", "96.1%", "Random Forest")

st.markdown("---")  # Horizontal separator line

# ==================== Maps Section ====================
col1, col2 = st.columns(2)  # Creates 2 columns for the maps

with col1:
    st.subheader("For-Profit Ownership by State (%)")  # Header for Map 1
    # Calculate % of For-Profit homes per state (mean of boolean mask * 100)
    fp = (df['Ownership_Risk_Score']==3).groupby(df['code']).mean()*100
    fp_df = fp.reset_index(name='Percent')  # Reset index to make it a plot-ready DataFrame
    # Create Choropleth map for ownership
    fig1 = px.choropleth(fp_df, locations='code', locationmode='USA-states',
                         color='Percent', scope="usa", color_continuous_scale="Reds",
                         range_color=(0,100), title="Higher = More Privatized")
    st.plotly_chart(fig1, use_container_width=True)  # Render Map 1

with col2:
    st.subheader("Average Star Rating by State")  # Header for Map 2
    # Calculate average rating per state
    rating_avg = df.groupby('code')[rating_col].mean().reset_index()
    rating_avg.columns = ['code', 'Rating']  # Rename columns for clarity
    # Create Choropleth map for ratings (Red to Green scale)
    fig2 = px.choropleth(rating_avg, locations='code', locationmode='USA-states',
                         color='Rating', scope="usa", color_continuous_scale="RdYlGn_r",
                         range_color=(1,5), title="Higher = Better Quality")
    st.plotly_chart(fig2, use_container_width=True)  # Render Map 2

st.markdown("---")  # Horizontal separator

# ==================== Full Interactive Table ====================
st.subheader("Complete Facility Explorer (All 14,752 Facilities)")  # Section header

# Search + filters
search = st.text_input("Search by name, city, county, or any field", "")  # Text input box for searching

# List of default columns to show initially
default_cols = ['Provider Name', 'City', 'State', 'Overall Rating', 'Ownership Type',
                'Ownership_Risk_Score', 'Low_Quality_Facility', 'Chronic_Deficiency_Score',
                'Fine_Per_Bed', 'Understaffed', 'High_Risk_State']
# Determine which columns are actually available in the dataframe
available_cols = [c for c in df.columns if c in default_cols or st.checkbox(f"Show {c}", False)]
# Multiselect widget to let users pick which columns to view
selected_cols = st.multiselect("Choose columns to display", df.columns.tolist(), default=default_cols)

# Apply search logic
if search:
    # Create a boolean mask where ANY column contains the search string (case-insensitive)
    mask = df.apply(lambda row: row.astype(str).str.contains(search, case=False, na=False).any(), axis=1)
    display_df = df.loc[mask, selected_cols]  # Filter the dataframe using the mask
else:
    display_df = df[selected_cols]  # No search? Show all data (filtered by selected columns)

st.dataframe(display_df, use_container_width=True, height=600)  # Render the interactive dataframe

# ==================== Top/Bottom Lists ====================
col1, col2 = st.columns(2)  # Create 2 columns
with col1:
    st.subheader("Top 20 Worst-Rated Homes")
    # Get the 20 rows with the lowest values in 'rating_col'
    worst = df.nsmallest(20, rating_col)[['Provider Name','City','State',rating_col,'Ownership_Risk_Score']]
    st.dataframe(worst, use_container_width=True)  # Show table

with col2:
    st.subheader("Top 20 Best-Rated Homes")
    # Get the 20 rows with the highest values in 'rating_col'
    best = df.nlargest(20, rating_col)[['Provider Name','City','State',rating_col,'Ownership_Risk_Score']]
    st.dataframe(best, use_container_width=True)  # Show table

# ==================== SHAP Explanation ====================
st.subheader("Why Homes Fail: Model Explanation (SHAP)")
# Define the top 6 engineered features
features = ['Ownership_Risk_Score','State_Quality_Percentile','Chronic_Deficiency_Score',
            'Fine_Per_Bed','Understaffed','High_Risk_State']
# Hardcoded importance scores from the Random Forest model (for speed)
importance = [0.42, 0.21, 0.18, 0.09, 0.07, 0.03]

# Create horizontal bar chart of feature importance
fig = px.bar(y=features, x=importance, orientation='h',
             color=importance, color_continuous_scale="Oranges",
             title="Top Drivers of 1–2 Star Ratings")
st.plotly_chart(fig, use_container_width=True)  # Render chart

# ==================== Download ====================
st.subheader("Download Data")
csv = display_df.to_csv(index=False).encode()  # Convert the currently visible (filtered) data to CSV
# Button to download only the filtered view
st.download_button("Download Current View (CSV)", csv, "filtered_facilities.csv", "text/csv")

# Button to download the ENTIRE dataset
st.download_button("Download Full Dataset (CSV)", 
                   df.to_csv(index=False).encode(), 
                   "complete_medicare_nursing_homes_2025.csv", "text/csv")

# ==================== Footer ====================
st.markdown("---")
st.markdown("""
**Rabiul Alam Ratul** • Analysis of Medicare Nursing Homes & Hospital Spending  
**Data**: Centers for Medicare & Medicaid Services (CMS) • 2025  
**GitHub**: https://github.com/RABIUL-ALAM-RATUL/Medicare-Hospital-Spending-by-Claim-USA-  
**Live Dashboard**: https://medicare-ultimate-dashboard.streamlit.app
""")
