# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import dash_mantine_components as dmc
# pip install dash-mantine-components
from itertools import combinations

# Incorporate data
df = pd.read_csv('贺总.csv')


# Select only the relevant columns
df = df[['pairs', 'value', 'date']]

# Get unique pairs
unique_pairs = df['pairs'].unique()
outstr=''

# Generate all possible combinations of pairs
pair_combinations = list(combinations(unique_pairs, 2))
print(len(pair_combinations))
# Calculate correlation for each pair combination
for pair1, pair2 in pair_combinations:
    pair1_df = df[df['pairs'] == pair1].set_index('date')['value']
    pair2_df = df[df['pairs'] == pair2].set_index('date')['value']

    # Calculate the correlation between the pairs
    correlation = pair1_df.corr(pair2_df)

    out=f"Correlation between {pair1} and {pair2}: {correlation}"
    # print(out)
    # print(correlation > 0.5)
    # print(type(correlation))
    if correlation > 0.5:
        outstr=outstr+"\n"+out
    elif correlation < -0.5:
        outstr=outstr+"\n"+out
    else:
        pass   
print(outstr)                   
