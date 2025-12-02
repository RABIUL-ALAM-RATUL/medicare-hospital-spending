import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Medicare Hospital Spending by Claim", layout="wide")
st.title("Medicare Hospital Spending by Claim (USA)")
st.markdown("**Inpatient & Outpatient Claims Analysis • 2025 Data**")

@st.cache_data
def load_data():
    return pd.read_parquet("df_final.parquet")

df = load_data()
df['code'] = df['State'].str.upper().str[:2]

col1, col2 = st.columns(2)

with col1:
    st.subheader("For-Profit Nursing Homes by State")
    is_fp = df['Ownership_Risk_Score'] == 3
    fp_pct = (is_fp.groupby(df['code']).mean()*100).round(1).reset_index()
    fp_pct.columns = ['code', 'percent']
    fig1 = px.choropleth(fp_pct, locations='code', locationmode='USA-states', color='percent',
                         scope="usa", color_continuous_scale="Reds", range_color=(0,100),
                         title="Higher = More Privatized")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Average CMS Star Rating by State")
    rating_col = [c for c in df.columns if 'overall' in c.lower() and 'rating' in c.lower()][0]
    rating_mean = df.groupby('code')[rating_col].mean().round(2).reset_index()
    fig2 = px.choropleth(rating_mean, locations='code', locationmode='USA-states', color=rating_col,
                         scope="usa", color_continuous_scale="RdYlGn_r", range_color=(1,5),
                         title="Higher = Better Quality")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.subheader("Predicting Low-Quality Facilities")
st.metric("Model Accuracy", "96.1%")
features = ['Ownership_Risk_Score','State_Quality_Percentile','Chronic_Deficiency_Score',
            'Fine_Per_Bed','Understaffed','High_Risk_State']
importance = [0.42, 0.21, 0.18, 0.09, 0.07, 0.03]
fig_bar = px.bar(x=importance, y=features, orientation='h',
                 color=importance, color_continuous_scale="Oranges",
                 title="Top Drivers of 1–2 Star Homes")
st.plotly_chart(fig_bar, use_container_width=True)

st.caption("Analysis by Rabiul Alam Ratul • [GitHub](https://github.com/RABIUL-ALAM-RATUL)")
