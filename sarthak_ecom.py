import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Pricing Agent - The Hustler Store", layout="wide")
st.title("🤖 AI Pricing Agent: The Hustler Store")

# --- STEP 1: FILE UPLOAD ---
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your Pricing Data (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load the user's specific dataset
    df = pd.read_csv(uploaded_file)
    
    # --- STEP 2: DATA PRE-PROCESSING ---
    # 1. Rename complex columns to simple names for coding
    # Note: We match the specific headers found in your file
    df.rename(columns={
        'Product ID': 'Product_ID',
        'Cost Price (Ci\u200b) (\u20b9)': 'Cost',            # Matches "Cost Price (Ci) (₹)"
        'Inventory Level (Ii\u200b)': 'Inventory',           # Matches "Inventory Level (Ii)"
        'Competitor Price (Pc,i\u200b) (\u20b9)': 'Comp_Price', # Matches "Competitor Price"
        'Historical Demand (Units)': 'Hist_Demand',
        'Historical Avg Price (Pavg,i\u200b) (\u20b9)': 'Hist_Price'
    }, inplace=True)

    # 2. Map Product IDs to Categories & Elasticity
    # (Since these weren't in the file, we infer them for the AI logic)
    product_map = {
        'P1': {'Cat': 'Cricket',   'Elas': -1.5, 'Name': 'English Willow Bat'},
        'P2': {'Cat': 'Football',  'Elas': -2.0, 'Name': 'Pro Match Ball'},
        'P3': {'Cat': 'Swimming',  'Elas': -2.5, 'Name': 'Anti-Fog Goggles'},
        'P4': {'Cat': 'Badminton', 'Elas': -1.8, 'Name': 'Carbon Racket'},
        'P5': {'Cat': 'Fitness',   'Elas': -1.2, 'Name': 'Dumbbell Set'}
    }

    def get_meta(pid, field):
        return product_map.get(pid, {'Cat': 'General', 'Elas': -1.5, 'Name': pid})[field]

    df['Category'] = df['Product_ID'].apply(lambda x: get_meta(x, 'Cat'))
    df['Product_Name'] = df['Product_ID'].apply(lambda x: get_meta(x, 'Name'))
    df['Elasticity'] = df['Product_ID'].apply(lambda x: get_meta(x, 'Elas'))

    st.write("### Data Preview (Processed)")
    st.dataframe(df.head())

    # --- STEP 3: OPTIMIZATION ENGINE ---
    def optimize_price(row):
        cost = row['Cost']
        comp_price = row['Comp_Price']
        inventory = row['Inventory']
        hist_demand = row['Hist_Demand']
        hist_avg_price = row['Hist_Price']
        elasticity = row['Elasticity']

        # Objective: Maximize Profit
        def objective(price):
            p = price[0]
            # Demand Logic: 
            # We use Historical Demand as the baseline. 
            # If we price HIGHER than Historical Avg, demand drops based on elasticity.
            
            # Formula: New_Demand = Hist_Demand * (1 + Elasticity * %Change_in_Price)
            pct_change = (p - hist_avg_price) / hist_avg_price
            pred_demand = hist_demand * (1 + elasticity * pct_change)
            
            profit = (p - cost) * pred_demand
            return -profit # Negate for minimization

        # Constraints
        cons = [
            # 1. Floor: Price >= Cost + 10%
            {'type': 'ineq', 'fun': lambda p: p[0] - (cost * 1.10)},
            # 2. Ceiling: Price <= Competitor * 1.25 (Don't exceed rival by >25%)
            {'type': 'ineq', 'fun': lambda p: (comp_price * 1.25) - p[0]},
            # 3. Inventory: Sales <= Inventory
            {'type': 'ineq', 'fun': lambda p: inventory - (hist_demand * (1 + elasticity * ((p[0] - hist_avg_price) / hist_avg_price)))}
        ]

        # Run Solver (Start guess at Competitor Price)
        result = minimize(objective, [comp_price], constraints=cons, bounds=[(cost, comp_price*2)])
        
        opt_price = result.x[0]
        pct_change_opt = (opt_price - hist_avg_price) / hist_avg_price
        opt_demand = hist_demand * (1 + elasticity * pct_change_opt)
        opt_profit = (opt_price - cost) * opt_demand
        
        # Calculate Baseline (If we just kept pricing at Historical Avg)
        base_profit = (hist_avg_price - cost) * hist_demand

        return pd.Series([opt_price, opt_demand, opt_profit, base_profit], 
                         index=['Rec_Price', 'Pred_Sales', 'Proj_Profit', 'Base_Profit'])

    # Button to Trigger AI
    if st.button("🚀 Optimize Prices"):
        with st.spinner("AI is calculating optimal prices..."):
            results = df.apply(optimize_price, axis=1)
            final_df = pd.concat([df, results], axis=1)

            # --- STEP 4: DASHBOARD ---
            total_base = final_df['Base_Profit'].sum()
            total_proj = final_df['Proj_Profit'].sum()
            uplift = ((total_proj - total_base) / total_base) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Historical Profit Strategy", f"₹{total_base:,.0f}")
            col2.metric("AI Optimized Strategy", f"₹{total_proj:,.0f}")
            col3.metric("Profit Uplift", f"{uplift:.2f}%")

            # Chart 1: Price Comparison
            st.subheader("Pricing Strategy: Historical vs. AI vs. Competitor")
            # Let's filter for just one product to see the trend clearly (e.g. Cricket Bat P1)
            p1_data = final_df[final_df['Product_ID'] == 'P1']
            
            fig = px.line(p1_data, x='Day', y=['Hist_Price', 'Comp_Price', 'Rec_Price'],
                          title="Daily Pricing Adjustments for English Willow Bat (P1)",
                          labels={'value': 'Price (INR)', 'variable': 'Metric'})
            st.plotly_chart(fig, use_container_width=True)

            # Table Output
            st.subheader("Download Optimized Pricing Sheet")
            st.dataframe(final_df[['Day', 'Product_ID', 'Category', 'Cost', 'Comp_Price', 'Rec_Price', 'Pred_Sales', 'Proj_Profit']].style.format({
                'Cost': '₹{:.0f}', 'Comp_Price': '₹{:.0f}', 'Rec_Price': '₹{:.2f}', 'Pred_Sales': '{:.1f}', 'Proj_Profit': '₹{:.0f}'
            }))

else:
    st.info("👈 Please upload your CSV file in the sidebar to begin.")
