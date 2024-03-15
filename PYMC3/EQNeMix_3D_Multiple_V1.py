#!/usr/bin/env python
# coding: utf-8

#Import libraries
import pandas as pd
import numpy as np
import pymc3 as pm
import theano
theano.config.exception_verbosity = 'high'
import theano.tensor as tt
import json
from obspy.core import UTCDateTime
import pyproj
import sys
import gc


# Read xlsx file with information provided by EQT and CNQ
df_filtered = pd.read_excel('/Users/cecilia/EQNeMix/PYMC3/Results/df_filtered.xlsx')


with open('countbase.txt') as fp:
    count = fp.read()

coun = int(count)

t_observed = df_filtered['t_observed'].iloc[coun]

weights_str = df_filtered['clusters_weight'].tolist()[coun]
clusters_weight_i = eval(weights_str)
w0 = clusters_weight_i[0]
w1 = clusters_weight_i[1]
w2 = clusters_weight_i[2]
w3 = clusters_weight_i[3]
w4 = clusters_weight_i[4]
w5 = clusters_weight_i[5]
weights = [w0, w1, w2, w3, w4, w5] 

# Select reference system: STA or TT
ref = 'TT'
# Choose dimensionality: 2D or 3D
dim = '3D'

# Upload json files with ellipse parameters information
ellipse_data = []
for i in range(6):
    file_path = f'/Users/cecilia/CONVN/data/6_clusters/csv_clusters/{dim}_{ref}/ellipse_parameters_{dim}_{ref}_{i}.json'
    with open(file_path, 'r') as file:
        data = json.load(file)
    ellipse_data.append(data)

# Extract covariance matrices information
cov_matrices = []
for i in range(6):
    cov_matrices.append(np.array(ellipse_data[i]['Covariance']))

# Extract ellipse means information
means = []
for i in range(6):
    means.append(np.array(ellipse_data[i]['Mean']))

# Define the function S_P_t (Theoretical traveltime function) [SECONDS]
def S_P_t(x, y, z):
    filename = '/Users/cecilia/EQNeMix/PYEIFMM/tsp.npy'
    tsp = np.load(filename)
    tsp2 = theano.shared(tsp) 
    X_rounded = tt.cast(tt.floor_div(x, 500) * 500, 'int64')
    Y_rounded = tt.cast(tt.floor_div(y, 500) * 500, 'int64')
    Z_rounded = tt.cast(tt.floor_div(z, 500) * 500, 'int64')
    # Find the corresponding indices in the tsp array
    x_index = X_rounded // 500
    y_index = Y_rounded // 500
    z_index = Z_rounded // 500
    tval = tsp2[x_index, y_index, z_index]
    return tval

# Define the Bayesian model
with pm.Model() as model:
    # Define the categories to choose the means
    category = pm.Categorical('category', p = weights)

    # Define the means corresponding to the categories
    mus = [pm.MvNormal(f'mu{i}', mu=means[i], cov=cov_matrices[i], shape=3) for i in range(len(weights))]

    # Select the averages corresponding to the selected category.
    x = pm.Deterministic('x', pm.math.switch(
        pm.math.eq(category, 0), mus[0][0],
        pm.math.switch(pm.math.eq(category, 1), mus[1][0],
        pm.math.switch(pm.math.eq(category, 2), mus[2][0],
        pm.math.switch(pm.math.eq(category, 3), mus[3][0],
        pm.math.switch(pm.math.eq(category, 4), mus[4][0], mus[5][0]))))))
    
    y = pm.Deterministic('y', pm.math.switch(
        pm.math.eq(category, 0), mus[0][1],
        pm.math.switch(pm.math.eq(category, 1), mus[1][1],
        pm.math.switch(pm.math.eq(category, 2), mus[2][1],
        pm.math.switch(pm.math.eq(category, 3), mus[3][1],
        pm.math.switch(pm.math.eq(category, 4), mus[4][1], mus[5][1]))))))

    z = pm.Deterministic('z', pm.math.switch(
        pm.math.eq(category, 0), mus[0][1],
        pm.math.switch(pm.math.eq(category, 1), mus[1][2],
        pm.math.switch(pm.math.eq(category, 2), mus[2][2],
        pm.math.switch(pm.math.eq(category, 3), mus[3][2],
        pm.math.switch(pm.math.eq(category, 4), mus[4][2], mus[5][2]))))))
    
    # Calculate t using the theoretical function
    t = S_P_t(x, y, z)

    # Likelihood of the observed data
    obs = pm.Normal('obs', mu = t, sigma = 0.1, observed = t_observed)

with model:
    trace = pm.sample(500, tune = 50, cores = 1)

# Trace summary
summary_df = pm.summary(trace)

# Convert summary to dataframe
summary_df = pd.DataFrame(summary_df)

# Specify the name and path for CSV file
summary_file = f'/Users/cecilia/EQNeMix/PYMC3/Results/summary_results_{coun}.csv'

# Save the DataFrame as a CSV file
summary_df.to_csv(summary_file, index=False)

# Print a message indicating that the file has been saved
print(f"==========> Saved summary result in: {summary_file}")

# Show DataFrame
#print(summary_df)

# Go to the next index
count = str(coun + 1)

with open('countbase.txt','wt') as fp:
    fp.write(count)

if coun == len(df_filtered) - 1:
    with open('countbase.txt', 'w') as fp:
        fp.write("EVENTS COMPLETED")

gc.collect()
