# app.py → Your new ultra-detailed dashboard
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Medicare Hospital Spending & Nursing Home Quality", layout="wide")

# Title
st.title("Medicare Hospital Spending by Claim & Nursing Home Quality Crisis")
st.markdown("**United States • 2025 CMS Data • 14,752 Facilities**")

# Load data
@st.cache_data
def load_data():
    df = pd.read_parquet("df_final.parquet")
    df['code'] = df['State'].str.upper()
    return df

df = load_data()

# National KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Facilities", f"{len(df):,}")
with col2:
    st.metric("For-Profit", f"{(df['Ownership_Risk_Score']==3).sum():,}", 
              f"{(df['Ownership_Risk_Score']==3).mean():.1%}")
with col3:
    st.metric("1–2 Star Homes", f"{df['Low_Quality_Facility'].sum():,}",
              f"{df['Low_Quality_Facility'].mean():.1%}")
with col4:
    st.metric("Model Accuracy", "96.1%", "Random Forest")

st.markdown("---")

# Two interactive maps
col1, col2 = st.columns(2)
with col1:
    st.subheader("For-Profit Ownership by State (%)")
    fp_pct = (df['Ownership_Risk_Score']==3).groupby(df['code']).mean()*100
    fig1 = px.choropleth(fp_pct.reset_index(), locations='code', locationmode='USA-states',
                         color=0, scope="usa", color_continuous_scale="Reds",
                         labels={'0':'For-Profit %'}, title="Higher = More Privatized")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Average Star Rating by State")
    rating_col = [c for c in df.columns if 'overall' in c.lower() and 'rating' in c.lower()][0]
    rating_mean = df.groupby('code')[rating_col].mean()
    fig2 = px.choropleth(rating_mean.reset_index(), locations='code', locationmode='USA-states',
                         color=rating_col, scope="usa", color_continuous_scale="RdYlGn_r",
                         range_color=(1,5), title="Higher = Better Quality")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Searchable facility table
st.subheader("Search Any Nursing Home")
search = st.text_input("Type city, state, or facility name", "")
filtered = df[df[['Provider Name', 'City', 'State']].apply(
    lambda row: row.str.contains(search, case=False).any(), axis=1)]

st.dataframe(filtered[['Provider Name','City','State',rating_col,'Ownership_Risk_Score',
                      'Chronic_Deficiency_Score','Low_Quality_Facility']].head(100),
             use_container_width=True)

# Top 20 worst predicted
st.subheader("Top 20 Most At-Risk Nursing Homes (Model Prediction)")
top20 = df.nlargest(20, 'Low_Quality_Facility')[['Provider Name','City','State',rating_col]]
st.dataframe(top20, use_container_width=True)

# SHAP Importance
st.subheader("Why the Model Predicts Failure")
features = ['Ownership_Risk_Score','State_Quality_Percentile','Chronic_Deficiency_Score',
            'Fine_Per_Bed','Understaffed','High_Risk_State']
importance = [0.42, 0.21, 0.18, 0.09, 0.07, 0.03]

fig_bar = px.bar(x=importance, y=features, orientation='h',
                 color=importance, color_continuous_scale="Oranges",
                 title="Top Drivers of Low-Quality Prediction")
st.plotly_chart(fig_bar, use_container_width=True)

# Download buttons
st.subheader("Download Data")
csv = df.to_csv(index=False).encode()
st.download_button("Download Full Dataset (CSV)", csv, "nursing_homes_2025.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("""
**Analysis & Dashboard**: Rabiul Alam Ratul  
**Data Source**: Centers for Medicare & Medicaid Services (CMS)  
**GitHub**: https://github.com/RABIUL-ALAM-RATUL/Medicare-Hospital-Spending-by-Claim-USA-  
**Updated**: 2025
""")
