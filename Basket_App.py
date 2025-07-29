import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load data from both files
basket_data = pd.read_csv('LIVE_BASKETS.csv', thousands=',')
sp500_data = pd.read_csv('S&P 500.csv', thousands=',')

# Process S&P 500 data
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'], format='%m/%d/%Y')
sp500_data['Price'] = pd.to_numeric(sp500_data['Price'].astype(str).str.replace(',', ''), errors='coerce')
sp500_data['S&P 500'] = sp500_data['Price'].diff()
sp500_data = sp500_data[['Date', 'S&P 500']].rename(columns={'Date': 'VDATE'})

# Process basket data
basket_data['VDATE'] = pd.to_datetime(basket_data['VDATE'], format='%d-%m-%Y')
basket_data = basket_data.drop(columns=['THR'], errors='ignore')

# Merge datasets
data = pd.merge(basket_data, sp500_data, on='VDATE', how='left')
data.set_index('VDATE', inplace=True)

# Filter out baskets without recent data (last 20 days)
recent_cutoff = data.index.max() - pd.Timedelta(days=20)
active_baskets = [col for col in data.columns 
                 if ((data.index >= recent_cutoff) & (data[col] != 0)).any()]
data = data[active_baskets]

# Remove zero-only columns and _OVERLAY from names
data = data.loc[:, (data != 0).any(axis=0)]
data.columns = [col.replace('_OVERLAY', '') for col in data.columns]

# Compute correlation matrix
correlation_matrix = data.corr().fillna(0)

# Function to get top and least correlated pairs
def get_top_and_least_correlated(correlation_matrix, top_n=10, min_days=30):
    corr_pairs = (
        correlation_matrix.where(~np.eye(correlation_matrix.shape[0], dtype=bool))
        .stack()
        .reset_index()
        .rename(columns={0: 'Correlation', 'level_0': 'Basket 1', 'level_1': 'Basket 2'})
    )
    corr_pairs['Absolute Correlation'] = corr_pairs['Correlation'].abs()
    
    # Remove duplicates by sorting the basket names and keeping unique pairs
    corr_pairs['Pair'] = corr_pairs.apply(lambda row: tuple(sorted([row['Basket 1'], row['Basket 2']])), axis=1)
    unique_corr_pairs = corr_pairs.drop_duplicates(subset='Pair').reset_index(drop=True)

    # Calculate the number of common days between each basket pair
    unique_corr_pairs['Period (Days)'] = unique_corr_pairs.apply(
        lambda row: (
            (data[row['Basket 1']] != 0) & (data[row['Basket 2']] != 0)
        ).sum()
    , axis=1)

    # Filter out pairs with less than min_days of common data
    unique_corr_pairs = unique_corr_pairs[unique_corr_pairs['Period (Days)'] >= min_days]

    top_corr = unique_corr_pairs.nlargest(top_n, 'Absolute Correlation')
    least_corr = unique_corr_pairs.nsmallest(top_n, 'Absolute Correlation')
    return top_corr, least_corr

# Get top 10 and least 10 correlated pairs
top_corr_pairs, least_corr_pairs = get_top_and_least_correlated(correlation_matrix, min_days=10)

# Sort the least correlated pairs in ascending order of correlation
least_corr_pairs = least_corr_pairs.sort_values(by='Correlation', ascending=True)

# Reset index and add a rank column
top_corr_pairs.reset_index(drop=True, inplace=True)
top_corr_pairs['Rank'] = top_corr_pairs.index + 1

least_corr_pairs.reset_index(drop=True, inplace=True)
least_corr_pairs['Rank'] = least_corr_pairs.index + 1

# Set page configuration for wider layout
st.set_page_config(layout="wide")

# Get the date range from the data
start_date = data.index.min().strftime('%d-%m-%Y')
end_date = data.index.max().strftime('%d-%m-%Y')

# Streamlit app title with date range
# st.header(f"Basket Co-Relation Comparison [{start_date} to {end_date}]")
st.markdown(
    f"<h1 style='text-align: center;'>Basket Co-Relation Comparison [{start_date} to {end_date}]</h1>", 
    unsafe_allow_html=True
)
# Selection of baskets in the same row
basket_options = sorted(correlation_matrix.columns.tolist())
col1, col2 = st.columns(2)

with col1:
    basket1 = st.selectbox("Select Basket 1", basket_options, index=basket_options.index("S&P 500") if "S&P 500" in basket_options else 0)
with col2:
    basket2_options = ["ALL BASKETS"] + [b for b in basket_options if b != basket1]
    basket2 = st.selectbox("Select Basket 2", basket2_options)

# Clear any existing plots
plt.close('all')

# Display correlation based on selection
if basket2 == "ALL BASKETS":
    basket_correlations = correlation_matrix[basket1].drop(basket1)
    basket_correlations = basket_correlations.sort_values(ascending=False)

    # Get top 5 and least 5 correlated baskets
    top_5_corr_baskets = basket_correlations.nlargest(10).reset_index()
    top_5_corr_baskets.columns = ['Basket', 'Correlation']
    top_5_corr_baskets['Rank'] = range(1, 11)
    top_5_corr_baskets['Period (Days)'] = top_5_corr_baskets.apply(
        lambda row: (
            (data[basket1] != 0) & (data[row['Basket']] != 0)
        ).sum(), axis=1
    )

    least_5_corr_baskets = basket_correlations.nsmallest(10).reset_index()
    least_5_corr_baskets.columns = ['Basket', 'Correlation']
    least_5_corr_baskets['Rank'] = range(1, 11)
    least_5_corr_baskets['Period (Days)'] = least_5_corr_baskets.apply(
        lambda row: (
            (data[basket1] != 0) & (data[row['Basket']] != 0)
        ).sum(), axis=1
    )

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Top 10 Most Correlated Baskets with S&P")
        st.dataframe(top_5_corr_baskets[['Rank', 'Basket', 'Correlation', 'Period (Days)']].style.format({"Correlation": "{:.4f}"}), hide_index=True)

    with col2:
        st.subheader("Top 10 Least Correlated Baskets with S&P")
        st.dataframe(least_5_corr_baskets[['Rank', 'Basket', 'Correlation', 'Period (Days)']].style.format({"Correlation": "{:.4f}"}), hide_index=True)

    # Updated bar plot with hue parameter
    plt.figure(figsize=(15, 10))
    sns.barplot(x=basket_correlations.values, y=basket_correlations.index, 
                hue=basket_correlations.index, palette="coolwarm", legend=False)
    plt.title(f"Correlation of '{basket1}' with Active Baskets (in Last 20 Days)", pad=20)
    plt.xlabel("Correlation")
    plt.ylabel("Baskets")
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

else:
    correlation_value = correlation_matrix.loc[basket1, basket2]
    period_days = ((data[basket1] != 0) & (data[basket2] != 0)).sum()
    
    st.write(f"#### Correlation Value: {correlation_value:.4f}")
    st.write(f"#### Period (Days): {period_days}")

    plt.figure(figsize=(8, 8))
    sns.heatmap(correlation_matrix.loc[[basket1, basket2], [basket1, basket2]], 
                annot=True, fmt='.4f', cmap="RdBu_r", center=0, vmin=-1, vmax=1)
    plt.title(f"{basket1} vs {basket2}", pad=20)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

# Replace the last part of your code (the global comparison section) with this:

# st.header("Global Comparison (All Baskets)")
st.markdown("<h1 style='text-align: center;'>Global Comparison (All Baskets)</h1>", unsafe_allow_html=True)

# First expandable container for Most Correlated Pairs
with st.expander("Top 10 Most Correlated Basket Pairs", expanded=True):
    st.dataframe(
        top_corr_pairs.head(10)[['Rank', 'Basket 1', 'Basket 2', 'Correlation', 'Period (Days)']]
        .style.format({"Correlation": "{:.4f}"}),
        hide_index=True,
        use_container_width=True
    )

# Second expandable container for Least Correlated Pairs
with st.expander("Top 10 Least Correlated Basket Pairs", expanded=True):
    st.dataframe(
        least_corr_pairs.head(10)[['Rank', 'Basket 1', 'Basket 2', 'Correlation', 'Period (Days)']]
        .style.format({"Correlation": "{:.4f}"}),
        hide_index=True,
        use_container_width=True
    )
