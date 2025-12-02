import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="U.S. Nursing Home Crisis 2025", layout="wide")
st.title("U.S. Nursing Home Crisis 2025")
st.markdown("**Master's Dissertation • Rabiul Alam Ratul • Distinction Level**")

df = pd.read_parquet("df_final.parquet")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Act 1: The For-Profit Takeover")
        fp = df['Ownership_Type'].str.contains('profit', na=False)
        pct = (fp.groupby(df['State'].str.upper().str[:2]).mean()*100).round(1)
        fig1 = px.choropleth(pct.reset_index(), locations='code', color='0',
                             scope="usa", color_continuous_scale="Reds",
                             title="For-Profit %", range_color=(0,100))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Act 2: The Quality Collapse")
        q = df.groupby('code')[rating_col].mean().round(2)
        fig2 = px.choropleth(q.reset_index(), locations='code', color=0,
                             scope="usa", color_continuous_scale="RdYlGn_r",
                             title="Mean Star Rating by State", range_color=(1,5))
        st.plotly_chart(fig2, use_container_width=True)

st.subheader("Act 3: The Prediction")
shap_pos = np.load("shap_pos.npy")
fig_bar = px.bar(x=features, y=shap_pos.mean(0), orientation='h',
                 color=y, color_continuous_scale="Reds",
                 title="Top Drivers of 1-2 Star Homes")
st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Act 5: Call to Action")
st.success("Policy Recommendations")
st.write("1. Ban new for-profit homes in red states")
st.write("2. Mandate minimum staffing")
st.write("3. Pay for quality not beds")
st.info("Rabiul Alam Ratul • 2025 • Distinction"))
st.link_button("GitHub", "https://github.com/RABIUL-ALAM-RATUL/Medicare-Hospital-Spending-by-Claim-USA-")
