# Given a PDF file, get the coordinates of a box 
# encompassing all the text except headers and footers.
# This script is work in progress
# It works for the provided example
# https://links.imagerelay.com/cdn/2958/ql/general-terms-and-conditions-sqbe-en
# For a more robust heuristic version, 
# you may want to play with the 3 variables below
# (in capital letters)

import fitz
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import HDBSCAN

# Path to the PDF file
pdf_path = 'pdf/SQE_Terms_and_Conditions.pdf'

# Load the document using PyMuPDF (fitz)
document = fitz.open(pdf_path)
# Count pages
n_pages = document.page_count

# Extract the coordinates of each block (paragraph)
coordinates = {'x0': [], 'y0': [], 'x1': [], 'y1': []}
for page in document:
	blocks = page.get_text('blocks')
	for block in blocks:
		coordinates['x0'].append(block[0])
		coordinates['y0'].append(block[1])
		coordinates['x1'].append(block[2])
		coordinates['y1'].append(block[3])

# Store the block coordinates in a dataframe
df = pd.DataFrame(coordinates)

# Header/footer threshold: Currently 15%
# You may want to modify this script to try different values
# and select the optimal one depending on cluster quality/differentiation
# you may also want to set independent thresholds for header and footer

# QUANTILE #
quantile = 0.15

# Calculate upper and lower quantiles
upper = np.floor(df['y0'].quantile(1 - quantile))
lower = np.ceil(df['y1'].quantile(quantile))
print('15% quantile thresholds')
print(lower, upper)

# Calculate box boundaries (including header and footer)
x_min = np.floor(df['x0'].min())
x_max = np.ceil(df['x1'].max())
y_min = np.floor(df['y0'].min())
y_max = np.ceil(df['y1'].max())
print('Text box boundaries including headers and footers')
print(x_min, x_max, y_min, y_max)

# Compute coordinate clusters
# We assume that header and/or footers are present in, at least, 80 % of the pages

# HEADER/FOOTER FREQUENCY #
hff = 0.8

hdbscan = HDBSCAN(min_cluster_size = int(np.floor(n_pages * hff)))
df['clusters'] = hdbscan.fit_predict(df)

# For each cluster, compute min, max and average
df_group = df.groupby('clusters').agg(avg_y0=('y0','mean'), avg_y1=('y1','mean'),
                       std_y0=('y0','std'), std_y1=('y1','std'),
                       max_y0=('y0','max'), max_y1=('y1','max'),
                       min_y0=('y0','min'), min_y1=('y1','min'),
                       cluster_size=('clusters','count'), avg_x0=('x0', 'mean')).reset_index()
                       
df_group = df_group.sort_values(['avg_y0', 'avg_y1'], ascending=[True, True])
                       
print(df_group)

# We assume that theaders and footers are located outside the 85 % quantiles
# and that they are located always at the same positions (standard deviation nearly zero)

# STANDARD DEVIATION #
std = 0 

# We also asume that the cluster size is equal or lower to the number of pages
# (this is not alway true)

footer = np.floor(df_group[(np.floor(df_group['std_y0']) == std) & (np.floor(df_group['std_y1']) == std) & (df_group['min_y0'] >= upper) & (df_group['cluster_size'] <= n_pages)]['min_y0'].min())
header = np.ceil(df_group[(np.floor(df_group['std_y0']) == std) & (np.floor(df_group['std_y1']) == std) & (df_group['min_y1'] <= lower) & (df_group['cluster_size'] <= n_pages)]['min_y1'].max())

# If there is a footer, exclude it
if not pd.isnull(footer):
	y_max = footer

# If there is a header, exclude it
if not pd.isnull(header):
	y_min = header

# Calculate box boundaries (excluding header and footer)
print('Text box boundaries excluding headers and footers')
print(x_min, x_max, y_min, y_max)

fig = px.scatter(df_group, x='avg_x0', y='avg_y0', color='clusters', size='cluster_size')
fig.show()
