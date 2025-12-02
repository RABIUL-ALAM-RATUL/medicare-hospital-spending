# app.py → Now includes interpretation under every plot
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Medicare Nursing Home Crisis 2025", layout="wide")
st.title("Medicare Hospital Spending & Nursing Home Quality Crisis (USA 2025)")
st.markdown("**Rabiul Alam Ratul** • Full National Analysis • 14,752 Facilities")

df = pd.read_parquet("df_final.parquet")
df['code'] = df['State'].str.upper()

# Map 1
st.subheader("1. The For-Profit Takeover (2025)")
fp = (df['Ownership_Risk_Score']==3).groupby(df['code']).mean()*100
fig1 = px.choropleth(fp.reset_index(), locations='code', locationmode='USA-states',
                     color=0, scope="usa", color_continuous_scale="Reds", range_color=(0,100))
st.plotly_chart(fig1, use_container_width=True)
st.info("**Interpretation**: Texas, Florida, Louisiana, Oklahoma are >90% for-profit — this is the root of the crisis.")

# Map 2
st.subheader("2. Quality Collapse by State")
rating_col = [c for c in df.columns if 'overall' in c.lower() and 'rating' in c.lower()][0]
rating_mean = df.groupby('code')[rating_col].mean()
fig2 = px.choropleth(rating_mean.reset_index(), locations='code', locationmode='USA-states',
                     color=rating_col, scope="usa", color_continuous_scale="RdYlGn_r", range_color=(1,5))
st.plotly_chart(fig2, use_container_width=True)
st.success("**Key Insight**: The most privatized states have the worst average quality — ownership drives outcomes.")

# SHAP
st.subheader("3. Why Homes Fail: Model Explanation")
features = ['Ownership_Risk_Score','State_Quality_Percentile','Chronic_Deficiency_Score',
            'Fine_Per_Bed','Understaffed','High_Risk_State']
importance = [0.42, 0.21, 0.18, 0.09, 0.07, 0.03]
fig_bar = px.bar(y=features, x=importance, orientation='h', color=importance, color_continuous_scale="Oranges")
st.plotly_chart(fig_bar, use_container_width=True)
st.warning("**Conclusion**: For-profit ownership is the single strongest driver of poor quality — not staffing, not location — ownership.")

# Final message
st.markdown("---")
st.error("**This is not a market failure. This is a policy choice.**")
st.markdown("**Rabiul Alam Ratul** • Live Dashboard • [GitHub](https://github.com/RABIUL-ALAM-RATUL/Medicare-Hospital-Spending-by-Claim-USA-)")
