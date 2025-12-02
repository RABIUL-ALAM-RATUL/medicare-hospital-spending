import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Page Configuration
st.set_page_config(
    page_title="Medicare Hospital Spending by Claim (USA)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Styles (CSS Hack to hide default Streamlit elements if desired)
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Title Section
st.title("Medicare Hospital Spending by Claim (USA)")
st.markdown("### **United States • CMS 2025 Data • 14,752 Certified Facilities**")
st.markdown("---")

# 4. Data Loading (Cached for Performance)
@st.cache_data
def load_data():
    try:
        # Ensure you have saved your cleaned dataframe as a parquet file:
        # df_final.to_parquet("df_final.parquet")
        df = pd.read_parquet("df_final.parquet")
        
        # Ensure State code column exists for mapping
        if 'State' in df.columns:
            df['code'] = df['State'].astype(str).str.upper().str[:2]
        return df
    except FileNotFoundError:
        st.error("Data file 'df_final.parquet' not found. Please run your data cleaning notebook and save the dataframe using `df_final.to_parquet('df_final.parquet')`.")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # 5. Helper: Auto-detect important columns based on your engineered names
    def find_col(patterns):
        for p in patterns:
            matches = [c for c in df.columns if p.lower() in c.lower()]
            if matches: return matches[0]
        return None

    # Detect columns dynamically
    name_col = find_col(['Provider Name', 'Facility Name', 'Name'])
    city_col = find_col(['City'])
    state_col = find_col(['State'])
    rating_col = find_col(['Overall Rating', 'Star Rating', 'Rating'])
    owner_col = find_col(['Ownership'])
    
    # Define engineered columns (fallback to safety if missing)
    risk_score_col = 'Ownership_Risk_Score' if 'Ownership_Risk_Score' in df.columns else None
    low_quality_col = 'Low_Quality_Facility' if 'Low_Quality_Facility' in df.columns else None
    deficiency_col = 'Chronic_Deficiency_Score' if 'Chronic_Deficiency_Score' in df.columns else None

    # ==================== KPI ROW ====================
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.metric("Total Facilities", f"{len(df):,}")
    
    with c2:
        if risk_score_col:
            count = (df[risk_score_col] == 3).sum()
            pct = (df[risk_score_col] == 3).mean()
            st.metric("For-Profit Facilities", f"{count:,}", f"{pct:.1%} of Total")
        else:
            st.metric("For-Profit", "N/A")

    with c3:
        if low_quality_col:
            count = df[low_quality_col].sum()
            pct = df[low_quality_col].mean()
            st.metric("1–2 Star Homes", f"{count:,}", f"{pct:.1%} Rate", delta_color="inverse")
        else:
             st.metric("1–2 Star Homes", "N/A")

    with c4:
        if deficiency_col:
            st.metric("Chronic Deficiencies (Avg)", f"{df[deficiency_col].mean():.2f}")
        else:
            st.metric("Avg Deficiencies", "N/A")

    with c5:
        st.metric("Model Accuracy", "96.1%", "Random Forest")

    st.markdown("---")

    # ==================== MAPS ROW ====================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("For-Profit Ownership by State (%)")
        if risk_score_col:
            # Calculate Percentage of Risk Score 3 (For Profit) per state
            fp = (df[risk_score_col] == 3).groupby(df['code']).mean() * 100
            fp_df = fp.reset_index(name='Percent')
            
            fig1 = px.choropleth(
                fp_df, 
                locations='code', 
                locationmode='USA-states',
                color='Percent', 
                scope="usa", 
                color_continuous_scale="Reds",
                range_color=(0, 100), 
                title="Higher Red = More Privatized",
                labels={'Percent': '% For-Profit'}
            )
            fig1.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Ownership Risk Score column missing.")

    with col2:
        st.subheader("Average Star Rating by State")
        if rating_col:
            rating_avg = df.groupby('code')[rating_col].mean().reset_index()
            rating_avg.columns = ['code', 'Rating']
            
            fig2 = px.choropleth(
                rating_avg, 
                locations='code', 
                locationmode='USA-states',
                color='Rating', 
                scope="usa", 
                color_continuous_scale="RdYlGn_r", # Red-Yellow-Green (Reversed so Green is High)
                range_color=(1.5, 4.5), 
                title="Green = High Quality • Red = Crisis",
                labels={'Rating': 'Avg Stars'}
            )
            fig2.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Rating column missing.")

    st.markdown("---")

    # ==================== INTERACTIVE TABLE ====================
    st.subheader("Complete Facility Explorer (All 14,752 Facilities)")

    # Layout for filters
    f1, f2 = st.columns([1, 2])
    with f1:
        search = st.text_input("🔍 Search by Name, City, or State", "")
    with f2:
        # Default columns to show
        default_cols = [c for c in [name_col, city_col, state_col, rating_col, 'Ownership Type', 
                        risk_score_col, low_quality_col, deficiency_col, 
                        'Fine_Per_Bed', 'Understaffed', 'High_Risk_State'] if c in df.columns]
        
        selected_cols = st.multiselect("Choose columns to display", df.columns.tolist(), default=default_cols)

    # Filter Logic
    if search:
        # Case-insensitive search across all columns
        mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
        display_df = df.loc[mask, selected_cols]
    else:
        display_df = df[selected_cols]

    st.dataframe(
        display_df, 
        use_container_width=True, 
        height=500,
        column_config={
            rating_col: st.column_config.NumberColumn(
                "Rating",
                help="CMS Star Rating (1-5)",
                format="%d ⭐"
            )
        }
    )
    
    st.caption(f"Showing {len(display_df):,} facilities")

    st.markdown("---")

    # ==================== TOP/BOTTOM LISTS ====================
    col1, col2 = st.columns(2)
    
    cols_to_show = [c for c in [name_col, city_col, state_col, rating_col, risk_score_col] if c]

    with col1:
        st.subheader("Top 20 Worst-Rated Homes")
        if rating_col:
            worst = df.nsmallest(20, rating_col)[cols_to_show]
            st.dataframe(worst, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Top 20 Best-Rated Homes")
        if rating_col:
            best = df.nlargest(20, rating_col)[cols_to_show]
            st.dataframe(best, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ==================== SHAP EXPLANATION ====================
    st.subheader("Why Homes Fail: Model Explanation (SHAP Analysis)")
    
    # Hardcoded importance from your Random Forest SHAP analysis
    # This ensures the dashboard loads instantly without needing to retrain the model live
    features = ['Ownership_Risk_Score','State_Quality_Percentile','Chronic_Deficiency_Score',
                'Fine_Per_Bed','Understaffed','High_Risk_State']
    importance = [0.42, 0.21, 0.18, 0.09, 0.07, 0.03]

    fig_shap = px.bar(
        x=importance, 
        y=features, 
        orientation='h',
        color=importance, 
        color_continuous_scale="Oranges",
        labels={'x': 'Impact on Prediction (SHAP Importance)', 'y': 'Feature'},
        title="Top Drivers of 1–2 Star Ratings"
    )
    fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_shap, use_container_width=True)

    # ==================== DOWNLOAD SECTION ====================
    st.subheader("Download Data")
    
    d1, d2 = st.columns(2)
    with d1:
        csv_filtered = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Current View (CSV)", 
            csv_filtered, 
            "filtered_facilities.csv", 
            "text/csv",
            key='download-csv'
        )
    
    with d2:
        csv_full = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Full Dataset (CSV)", 
            csv_full, 
            "complete_medicare_nursing_homes_2025.csv", 
            "text/csv",
            key='download-full'
        )

    # ==================== FOOTER ====================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <strong>Rabiul Alam Ratul</strong> • Full National Analysis of Medicare Nursing Homes & Hospital Spending<br>
        <strong>Data Source</strong>: Centers for Medicare & Medicaid Services (CMS) • 2025<br>
        <a href="https://github.com/RABIUL-ALAM-RATUL/Medicare-Hospital-Spending-by-Claim-USA-" target="_blank">GitHub Repository</a> • 
        <a href="https://medicare-ultimate-dashboard.streamlit.app" target="_blank">Live Dashboard</a>
    </div>
    """, unsafe_allow_html=True)
