#!/usr/bin/env python3
import sys, os
import numpy as np
import pandas as pd
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# a function taking in a vector and returning its basic stats as a string
def basic_stats(v):
    try:
        info = {
            'ave': np.mean(v),
            'std': np.std(v),
            'med': np.median(v),
            'low': np.min(v),
            'hi': np.max(v),
        }
    except ValueError as e:
        info = {'value error': 0}
    s = ''
    for key, value in info.items():
        s = s + key + ': %.3f'% value + '\n'
    # return foramt: "str\nstr\nstr\nstr\n..."
    return s

filters={'NaN': 1.0,    # if NaN composes more than this ratio in a trait, the trait will not be plotted
        }

# load csv
traits = pd.read_csv('data/Soybean_Traits.csv', header=0, low_memory=False)

if not os.path.exists('plots'):
        os.makedirs('plots')

# there are different types of columns, and they will be processed differently
# currently we only focus on numerical ones
numerical_columns = [x for x in traits.select_dtypes(include=['float'])] # headers of numerical columns

# plotting numerical traits
for trait_name in numerical_columns:
    trait_vector = traits[trait_name]
    # applying filters
    nan_count = trait_vector.isna().sum()
    original_count = len(trait_vector)
    if (nan_count/original_count) >= filters['NaN']:
        continue
    # dropping NaN and keeping only valid numbers
    trait_vector = trait_vector.dropna()
    # generating kde plot
    f, ax = plt.subplots()
    sns.distplot(trait_vector, kde_kws={"shade": True,})
    # aesthetics
    plt.title(trait_name + '  (n={})'.format(original_count-nan_count))
    plt.legend([basic_stats(trait_vector) + 'NaN: {}\nTotal: {}'.format(nan_count, original_count)])
    # saving images
    plt.savefig('plots/{}.png'.format(trait_name))
    plt.close()

# add more sections accordingly if you want to plot other columns 
