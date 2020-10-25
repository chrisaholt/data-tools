
#%%
import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas
import sys

sys.path.append('../src')
import groupops

%matplotlib inline

csv_filename = '../datasets/salmon.csv'
df = pandas.read_csv(csv_filename)

# %%
import importlib
importlib.reload(groupops)

grouped = df.groupby("Species")
groupops.quick_hist(grouped["Length"], title="Length")
groupops.quick_hist(grouped["Weight"], title="Weight")

# %%
