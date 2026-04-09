import pandas as pd

# Fix Market History
m_df = pd.read_csv('ml_training/datasets/market_history.csv')
if 'Price' in m_df.columns:
    m_df['Price_Per_Ton'] = m_df['Price_Per_Ton'].fillna(m_df['Price'])
    m_df['Demand_Level'] = m_df['Demand_Level'].fillna('High')
    m_df.drop('Price', axis=1, inplace=True)
m_df.to_csv('ml_training/datasets/market_history.csv', index=False)

# Fix Seed Rates
s_df = pd.read_csv('ml_training/datasets/seed_rates.csv')
if 'Seed_Rate_kg_per_acre' in s_df.columns:
    s_df['Seed_Rate_Per_Acre'] = s_df['Seed_Rate_Per_Acre'].fillna(s_df['Seed_Rate_kg_per_acre'])
if 'Cost_per_kg' in s_df.columns:
    s_df['Seed_Price_Per_Unit'] = s_df['Seed_Price_Per_Unit'].fillna(s_df['Cost_per_kg'])
    s_df['Seed_Unit'] = s_df['Seed_Unit'].fillna('kg')
    s_df.drop(['Seed_Rate_kg_per_acre', 'Cost_per_kg'], axis=1, errors='ignore', inplace=True)
s_df.to_csv('ml_training/datasets/seed_rates.csv', index=False)

print("Financial CSV headers successfully patched!")
