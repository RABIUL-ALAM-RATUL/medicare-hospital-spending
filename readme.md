Medicare Hospital Spending by Claim (USA)

A National Analysis of 14,752 Certified Facilities | 96.1% Predictive Model Accuracy

1. Research Abstract & Scope

This project presents a detailed, data-based study of the nursing home industry in the United States, using the latest 2025 dataset from the Centers for Medicare & Medicaid Services (CMS). The research encompasses a comprehensive national census of over 14,000 certified facilities, aiming to identify the structural and systemic factors that impact care quality.

The main research goal goes beyond basic statistics to predictive modeling. In a time of increasing privatization in healthcare, this study examines the relationship between ownership types (for-profit versus non-profit), geographic location, and clinical outcomes. By applying machine learning techniques, this analysis provides evidence on what leads to facility failures. The results challenge the unpredictable nature of care quality, showing that outcomes can often be structured and predicted based on financial and operational incentives.

2. Interactive Analytical Dashboard (Methodological Output)

A key part of the output is an interactive web application developed with the Streamlit framework. This tool allows for peer review, detailed exploration of the dataset, and validation of findings through visualizations.

Access the application: Launch Interactive Dashboard

Key Analytical Features:

- Geospatial Analysis: Interactive maps show the spread of privatization along with regional quality ratings, revealing specific geographic areas where regulatory oversight may be lacking.
  
- Detailed Market Exploration: A custom search engine enables in-depth analysis of individual facilities. Users can compare specific risk profiles, staffing ratios, and federal fines against state and national averages to better understand local performance.

- Causal Inference (SHAP): The dashboard uses SHAP (SHapley Additive exPlanations) values to explain the Random Forest model. This measures the contributions of factors like understaffing, past fines, and ownership type to a facility's classification as "Low Quality."

- Data Extraction: Users can export filtered datasets or Tableau-compatible files, which support further independent research and ensure reproducibility.

3. Methodology & Analytical Framework

This research uses a systematic data science process to ensure the reliability of the results.

3.1 Data Preprocessing & Integrity

The raw CMS dataset has many variables and notable gaps. A careful cleaning process was applied:

- Imputation Strategy: Numerical variables that were right-skewed (like total fines) were adjusted using median imputation to maintain stability and reduce the effect of extreme outliers. Categorical variables used mode imputation.

- Outlier Management: To prevent the model from overfitting on extreme cases, Interquartile Range (IQR) capping was applied to financial penalty data, preserving the rank order of severity while normalizing variance.

3.2 Feature Engineering

To identify underlying patterns in the data, several important predictors were developed:

- Ownership_Risk_Score: A weighted categorical score based on the theoretical risks tied to profit motives (for-profit > government > non-profit).

- Chronic_Deficiency_Score: A combined measure of historical regulatory violations to highlight operational failures.

- Fine_Per_Bed: A standardized financial measure that allows for equitable comparisons of penalty intensity across facilities of different sizes.

3.3 Predictive Modeling

A Random Forest Classifier (with 600 trees) was trained to predict "Low Quality" facilities (those with 1–2 Star Ratings). The model reached an accuracy of 96.1% and an AUC-ROC of 0.987 on the test set, confirming that specific structural features mainly determine facility failure.

4. Empirical Findings & Discussion

The analysis provided valuable insights into the structural weaknesses of the sector, summarized as follows:

4.1 Market Consolidation & Privatisation

The industry has a high level of privatization, with 83% of U.S. nursing homes now run as for-profit entities. This consolidation has notably changed the care landscape, emphasizing efficiency metrics that often conflict with patient outcomes.

4.2 The "For-Profit Penalty"

Ownership structure was identified as the main factor influencing care quality. The predictive model showed that being for-profit is the leading reason facilities fall into the failing category. The SHAP analysis assigns an importance score of 0.42 to ownership type, far exceeding other aspects. This indicates a systemic "For-Profit Penalty," where the pursuit of profit relates to a decline in care standards.

4.3 Geospatial Determinants

The study pinpointed specific "Crisis Zones," particularly Texas, Florida, Louisiana, and Oklahoma. In these areas, a high concentration of for-profit ownership corresponds strongly with the lowest national quality scores. This pattern suggests that state-level regulations significantly affect the risks tied to privatization.

5. Policy Implications

Given the findings, this study suggests three structural changes:

- Moratorium on Licensure: A freeze on new for-profit licenses in states exceeding 80% privatization until quality benchmarks are consistent with national averages.

- Mandatory Staffing Ratios: Since understaffing was a major predictor of failure, federal minimums for nursing hours are warranted.

- Value-Based Reimbursement: Medicare payments should be separated from occupancy rates and directly linked to clinical outcomes (Star Ratings) to align financial incentives with patient care.

6. Repository Architecture

This repository holds the complete, reproducible process for the study:

- app.py: The source code for the Streamlit web application, including all visualization logic and interactive elements.

- Medicare Hospital Spending by Claim (USA).ipynb: The Jupyter Notebook documenting the entire data science process, including data cleaning, exploratory analysis, feature engineering, and model training.

- df_final.parquet: The pre-processed, engineered, and compressed dataset used by the dashboard (created through the notebook process).

- requirements.txt: A full list of Python dependencies needed to replicate the analysis environment.

7. Reproducibility & Execution

To replicate this analysis or run the dashboard locally, follow these steps:

Clone the Repository

git clone [https://github.com/RABIUL-ALAM-RATUL/Medicare-Hospital-Spending-by-Claim-USA-.git](https://github.com/RABIUL-ALAM-RATUL/Medicare-Hospital-Spending-by-Claim-USA-.git)  
cd Medicare-Hospital-Spending-by-Claim-USA-

Environment Configuration

pip install -r requirements.txt

Execution

To review the analytical process:

Jupyter Notebook "Medicare Hospital Spending by Claim (USA).ipynb"

To start the interactive dashboard:

streamlit run app.py

Principal Investigator

Md Rabiul Alam  
Medicare Hospital Spending by Claim (USA)
