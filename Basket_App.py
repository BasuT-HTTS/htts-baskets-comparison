import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv('LIVE_BASKETS.csv', thousands=',')
data['VDATE'] = pd.to_datetime(data['VDATE'], format='%d-%m-%Y')
data.set_index('VDATE', inplace=True)
data = data.drop(columns=['THR'], errors='ignore')  # Drop THR column if present
data = data.loc[:, (data != 0).any(axis=0)]  # Drop zero-only columns

# Remove _OVERLAY from column names
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
st.header(f"Basket Co-Relation Comparison [{start_date} to {end_date}]")

# Selection of baskets in the same row
basket_options = sorted(correlation_matrix.columns.tolist())  # Sort basket options alphabetically
col1, col2 = st.columns(2)

with col1:
    basket1 = st.selectbox("Select Basket 1", basket_options)

with col2:
    # Add "ALL BASKETS" as the first item in the selection
    basket2_options = ["ALL BASKETS"] + [b for b in basket_options if b != basket1]
    basket2 = st.selectbox("Select Basket 2", basket2_options)

# Clear any existing plots
plt.close('all')

# Display correlation based on selection
if basket2 == "ALL BASKETS":
    # Display all correlations for basket1 in descending order
    basket_correlations = correlation_matrix[basket1].drop(basket1)  # Exclude self-correlation
    basket_correlations = basket_correlations.sort_values(ascending=False)

    # Get top 5 and least 5 correlated baskets
    top_5_corr_baskets = basket_correlations.nlargest(5).reset_index()
    top_5_corr_baskets.columns = ['Basket', 'Correlation']
    top_5_corr_baskets['Rank'] = range(1, 6)  # Assign ranks
    top_5_corr_baskets['Period (Days)'] = top_5_corr_baskets.apply(
        lambda row: (
            (data[basket1] != 0) & (data[row['Basket']] != 0)
        ).sum(), axis=1
    )

    least_5_corr_baskets = basket_correlations.nsmallest(5).reset_index()
    least_5_corr_baskets.columns = ['Basket', 'Correlation']
    least_5_corr_baskets['Rank'] = range(1, 6)  # Assign ranks
    least_5_corr_baskets['Period (Days)'] = least_5_corr_baskets.apply(
        lambda row: (
            (data[basket1] != 0) & (data[row['Basket']] != 0)
        ).sum(), axis=1
    )

    col1, col2 = st.columns(2)
    
    with col1:
        # Display top 5 correlated baskets
        st.subheader(f"Top 5 Most Correlated Baskets")
        st.dataframe(top_5_corr_baskets[['Rank', 'Basket', 'Correlation', 'Period (Days)']].style.format({"Correlation": "{:.7f}"}), hide_index=True)

    with col2:
        # Display top 5 non-correlated baskets
        st.subheader("Top 5 Least Correlated Baskets")
        st.dataframe(least_5_corr_baskets[['Rank', 'Basket', 'Correlation', 'Period (Days)']].style.format({"Correlation": "{:.7f}"}), hide_index=True)

    # Plot a bar chart for clarity
    plt.figure(figsize=(15, 10))
    sns.barplot(x=basket_correlations.values, y=basket_correlations.index, palette="coolwarm")
    plt.title(f"Correlation of '{basket1}' with All Other Baskets", pad=20)
    plt.xlabel("Correlation")
    plt.ylabel("Baskets")
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

else:
    # Display correlation between basket1 and basket2
    correlation_value = correlation_matrix.loc[basket1, basket2]
    
    # Calculate the number of common days between basket1 and basket2
    period_days = (
        (data[basket1] != 0) & (data[basket2] != 0)
    ).sum()
    
    st.write(f"#### Correlation Value: {correlation_value:.7f}")
    st.write(f"#### Period (Days): {period_days}")

    # Display heatmap for the selected pair
    plt.figure(figsize=(8, 8))
    sns.heatmap(correlation_matrix.loc[[basket1, basket2], [basket1, basket2]], 
                annot=True, fmt='.7f', cmap="RdBu_r", center=0, vmin=-1, vmax=1)
    plt.title(f"{basket1} vs {basket2}", pad=20)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

st.header("Global Comparison (All Baskets)")

col1, col2 = st.columns(2)

with col1:
    # Display top 10 correlated pairs
    st.subheader("Top 10 Most Correlated Basket Pairs")
    st.dataframe(top_corr_pairs[['Rank', 'Basket 1', 'Basket 2', 'Correlation', 'Period (Days)']].style.format({"Correlation": "{:.7f}"}), hide_index=True)

with col2:
    # Display least 10 correlated pairs
    st.subheader("Top 10 Least Correlated Basket Pairs")
    st.dataframe(least_corr_pairs[['Rank', 'Basket 1', 'Basket 2', 'Correlation', 'Period (Days)']].style.format({"Correlation": "{:.7f}"}), hide_index=True)

# Display bar plots for top 10 and least 10 correlated pairs for a visual summary
def plot_corr_pairs(df, title):
    plt.figure(figsize=(15, 8))
    sns.barplot(x='Correlation', y=df['Basket 1'] + " vs " + df['Basket 2'], data=df, palette="coolwarm")
    plt.title(title, pad=20)
    plt.xlabel("Correlation")
    plt.ylabel("Basket Pairs")
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

plot_corr_pairs(top_corr_pairs, "Top 10 Most Correlated Basket Pairs")
plot_corr_pairs(least_corr_pairs, "Top 10 Least Correlated Basket Pairs")
