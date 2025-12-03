# Medicare Hospital Spending & Nursing Home Quality – USA (2025)

**Full National Analysis | 14,752 Certified Facilities | 96.1% Predictive Accuracy**

[![Open Dashboard](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medicare-hospital-spending-cardiffmet-st20316895.streamlit.app/)  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RABIUL-ALAM-RATUL/Medicare-Hospital-Spending-by-Claim-USA-/blob/main/Final_Draft_26_11_25.ipynb)

### Live Interactive Dashboard (click above)
→ https://medicare-ultimate-dashboard.streamlit.app

One click → instantly explore:
- Two interactive U.S. maps (for-profit % + average star rating by state)  
- Search & filter all 14,752 nursing homes by name, city, county, ownership, fines, etc.  
- Top 20 worst & best facilities in the country  
- Full SHAP model explanation (why homes fail)  
- Download any view or the complete dataset  

### Key Findings
- 83% of U.S. nursing homes are for-profit
- For-profit ownership is the #1 driver of 1–2 star ratings (SHAP = 0.42)
- Texas, Florida, Louisiana, Oklahoma = national crisis zones
- Random Forest model predicts failing homes with **96.1% accuracy** using only 6 structural features**

### Repository Contents
### How to Run Locally
```bash
git clone https://github.com/RABIUL-ALAM-RATUL/Medicare-Hospital-Spending-by-Claim-USA-.git
cd Medicare-Hospital-Spending-by-Claim-USA-
pip install -r requirements.txt

# Option 1 – Run the notebook
jupyter notebook Final_Draft_26_11_25.ipynb

# Option 2 – Run the live dashboard
streamlit run app.py

# Option 2 – Run the live dashboard
streamlit run app.py
