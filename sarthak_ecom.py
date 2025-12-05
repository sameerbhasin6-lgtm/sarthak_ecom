import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize  # <--- FIXED: Added missing import

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Pricing Agent - The Hustler Store", layout="wide")

st.title("🤖 AI Dynamic Pricing Agent")
st.markdown("""
**Objective:** Optimize daily product prices to maximize profit while adhering to inventory and competitive constraints.
""")

# --- PART 1: DATA GENERATION ---
@st.cache_data
def generate_data():
    # Create dates for 10 days
    dates = pd.date_range(start="2025-12-01", periods=10)
    
    products = [
        {'id': 'CRK001', 'cat': 'Cricket', 'name': 'English Willow Bat', 'cost': 4500, 'base_p': 6000, 'base_d': 20, 'elas': -1.5},
        {'id': 'FTB001', 'cat': 'Football', 'name': 'Pro Match Ball', 'cost': 1200, 'base_p': 1800, 'base_d': 35, 'elas': -2.0},
        {'id': 'FIT001', 'cat': 'Fitness', 'name': 'Hex Dumbbell Set', 'cost': 2000, 'base_p': 3200, 'base_d': 15, 'elas': -1.2},
        {'id': 'BAD001', 'cat': 'Badminton', 'name': 'Carbon Racket', 'cost': 1800, 'base_p': 2500, 'base_d': 25, 'elas': -1.8},
        {'id': 'SWM001', 'cat': 'Swimming', 'name': 'Anti-Fog Goggles', 'cost': 400, 'base_p': 750, 'base_d': 50, 'elas': -2.5}
    ]
    
    rows = []
    for date in dates:
        for p in products:
            # Simulate slight daily fluctuations
            fluctuation = np.random.uniform(0.95, 1.05)
            comp_price = int(p['base_p'] * fluctuation)
            inventory = int(np.random.randint(20, 100))
            
            rows.append({
                'Date': date,
                'Product_ID': p['id'],
                'Category': p['cat'],
                'Product_Name': p['name'],
                'Base_Cost': p['cost'],
                'Competitor_Price': comp_price,
                'Inventory': inventory,
                'Base_Demand': p['base_d'],
                'Elasticity': p['elas']
            })
    return pd.DataFrame(rows)

# Load Data
df = generate_data()

# --- PART 2: OPTIMIZATION ENGINE ---
def optimize_row(row):
    # Variables
    cost = row['Base_Cost']
    comp_price = row['Competitor_Price']
    inventory = row['Inventory']
    base_demand = row['Base_Demand']
    elasticity = row['Elasticity']
    
    # Objective: Maximize Profit => Minimize Negative Profit
    def objective_function(price):
        p = price[0]
        # Linear Demand Curve
        # Demand = Base_Demand * (1 + Elasticity * (% Change in Price))
        predicted_demand = base_demand * (1 + elasticity * ((p - comp_price) / comp_price))
        
        # Profit
        profit = (p - cost) * predicted_demand
        return -profit # Return negative for minimization

    # Constraints
    constraints = [
        # 1. Minimum Margin: Price >= Cost + 10%
        {'type': 'ineq', 'fun': lambda x: x[0] - (cost * 1.10)},
        
        # 2. Competitive Ceiling: Price <= Competitor * 1.25
        {'type': 'ineq', 'fun': lambda x: (comp_price * 1.25) - x[0]},
        
        # 3. Inventory Cap: Predicted Sales <= Inventory
        {'type': 'ineq', 'fun': lambda x: inventory - (base_demand * (1 + elasticity * ((x[0] - comp_price) / comp_price)))}
    ]
    
    # Run Solver
    # FIXED: Added float() casting to bounds and initial guess to prevent numpy errors
    result = minimize(
        objective_function, 
        [float(comp_price)], 
        constraints=constraints, 
        bounds=[(float(cost), float(comp_price)*2)]
    )
    
    opt_price = result.x[0]
    
    # Calculate Final Metrics
    opt_demand = base_demand * (1 + elasticity * ((opt_price - comp_price) / comp_price))
    opt_profit = (opt_price - cost) * opt_demand
    
    # Baseline Metrics (if we just matched competitor price)
    base_profit = (comp_price - cost) * base_demand
    
    return pd.Series([opt_price, opt_demand, opt_profit, base_profit], 
                     index=['Recommended_Price', 'Predicted_Sales', 'Projected_Profit', 'Baseline_Profit'])

# --- PART 3: DASHBOARD LAYOUT ---

if st.button('🚀 Run AI Pricing Optimization'):
    with st.spinner('AI is analyzing market data and calculating optimal prices...'):
        # Run optimization on every row
        results = df.apply(optimize_row, axis=1)
        df_final = pd.concat([df, results], axis=1)
        
        st.success("Optimization Complete!")
        
        # Summary Metrics
        total_base_profit = df_final['Baseline_Profit'].sum()
        total_opt_profit = df_final['Projected_Profit'].sum()
        uplift = ((total_opt_profit - total_base_profit) / total_base_profit) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline Profit (Match Competitor)", f"₹{total_base_profit:,.0f}")
        col2.metric("AI Optimized Profit", f"₹{total_opt_profit:,.0f}")
        col3.metric("Profit Uplift", f"🚀 {uplift:.1f}%")
        
        # Visualization 1: Profit Comparison by Category
        st.subheader("Profit Impact by Category")
        cat_group = df_final.groupby('Category')[['Baseline_Profit', 'Projected_Profit']].sum().reset_index()
        cat_group = cat_group.melt(id_vars='Category', var_name='Scenario', value_name='Total Profit')
        fig_bar = px.bar(cat_group, x='Category', y='Total Profit', color='Scenario', barmode='group',
                         color_discrete_map={'Baseline_Profit': 'gray', 'Projected_Profit': '#00CC96'})
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Visualization 2: Price Trends vs Competitor
        st.subheader("Pricing Strategy: Us vs. Them (Cricket Category)")
        cricket_data = df_final[df_final['Category'] == 'Cricket']
        fig_line = px.line(cricket_data, x='Date', y=['Competitor_Price', 'Recommended_Price'], 
                           markers=True, title="Competitor Price vs AI Recommended Price")
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Detailed Data Table
        st.subheader("Detailed Pricing Recommendations")
        st.dataframe(df_final[['Date', 'Category', 'Base_Cost', 'Competitor_Price', 'Recommended_Price', 'Predicted_Sales', 'Projected_Profit']].style.format({
            'Base_Cost': '₹{:.0f}',
            'Competitor_Price': '₹{:.0f}', 
            'Recommended_Price': '₹{:.2f}',
            'Predicted_Sales': '{:.1f}',
            'Projected_Profit': '₹{:.0f}'
        }))

else:
    st.info("Click the button above to generate data and run the AI agent.")
    st.dataframe(df.head())
