Medicare Hospital Spending & Nursing Home Quality – USA (2025)

A Comprehensive National Analysis of 14,752 Certified Facilities | 96.1% Predictive Model Accuracy

1. Research Abstract & Scope

This dissertation project presents a rigorous, data-driven examination of the United States nursing home industry, utilising the most recent 2025 dataset from the Centers for Medicare & Medicaid Services (CMS). The study encompasses a complete national census of over 14,000 certified facilities, aiming to isolate the structural and systemic drivers of care quality.

The primary research objective extends beyond descriptive statistics to predictive modelling. By investigating the correlation between ownership structures (for-profit vs. non-profit), geographic location, and clinical outcomes, this analysis provides empirical evidence regarding the determinants of facility failure. The findings challenge the stochastic nature of care quality, demonstrating that outcomes are often structurally engineered and highly predictable.

2. Interactive Analytical Dashboard

As part of the methodological output, an interactive web application has been developed to allow peer review and granular exploration of the dataset.

Access the artifact here: https://medicare-hospital-spending-cardiffmet-st20316895.streamlit.app/

Key Capabilities:

Geospatial Analysis: Interactive choropleth maps visualising the spatial distribution of privatisation intensity alongside regional quality ratings, revealing distinct geographic corridors of systemic failure.

Granular Market Exploration: A search engine permitting forensic analysis of individual facilities, comparing specific risk profiles, staffing ratios, and federal fines against state and national benchmarks.

Causal Inference (SHAP): Integration of SHAP (Shapley Additive exPlanations) values to interpret the Random Forest model, quantifying the precise contribution of factors such as understaffing and ownership type to a facility's "Low Quality" classification.

Data Extraction: Functionality to export filtered datasets or Tableau-optimised files for further independent research.

3. Executive Summary of Findings

The analysis yielded significant empirical insights into the structural deficiencies of the sector:

Market Consolidation & Privatisation: The sector exhibits a high degree of privatisation, with 83% of U.S. nursing homes now operating under for-profit models.

The "For-Profit Penalty": Ownership structure emerged as the principal determinant of care quality. The predictive model identified for-profit status as the primary driver pushing facilities into the 1–2 star (failing) category (SHAP importance coefficient = 0.42).

Geospatial Determinants: The study identified specific "Crisis Zones"—notably Texas, Florida, Louisiana, and Oklahoma—where high privatisation saturation correlates strongly with the lowest decile of national quality scores.

Predictive Validity: The developed Random Forest classifier achieved a predictive accuracy of 96.1% using only six structural features. This high efficacy suggests that facility failure is systemic and foreseeable rather than accidental.

4. Repository Architecture

This repository contains the complete reproducible pipeline for the study:

app.py: The production-ready source code for the Streamlit web application, including all visualisation logic and interactive components.

Final_Draft_26_11_25.ipynb: The Jupyter Notebook documenting the end-to-end data science pipeline, including data cleaning (imputation/capping), Exploratory Data Analysis (EDA), Feature Engineering, and Machine Learning model training.

df_final.parquet: The pre-processed, engineered, and compressed dataset utilised by the dashboard (generated via the notebook pipeline).

requirements.txt: A comprehensive list of Python dependencies required to replicate the analysis environment.

5. Reproducibility & Execution

To replicate this analysis or deploy the dashboard within a local environment, please adhere to the following protocol:

Clone the Repository

git clone [https://github.com/RABIUL-ALAM-RATUL/Medicare-Hospital-Spending-by-Claim-USA-.git](https://github.com/RABIUL-ALAM-RATUL/Medicare-Hospital-Spending-by-Claim-USA-.git)
cd Medicare-Hospital-Spending-by-Claim-USA-


Environment Configuration

pip install -r requirements.txt


Execution

To audit the analytical pipeline:

Jupyter Notebook Final_Draft_26_11_25.ipynb


To launch the interactive dashboard:

streamlit run app.py


Principal Investigator

Md Rabiul Alam
Medicare Hospital Spending by Claim (USA)
