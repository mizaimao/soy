#!/usr/bin/env python3
import sys, os
import subprocess
import numpy as np
import pandas as pd
import pickle
import tqdm


mapping = {'A':0,
        'C':1,
        'T':2,
        'G':3,
        'H':4,
        '-':5,
        '':6,
        }

# get samples that have valid oil measurement
traits = pd.read_csv('data/Soybean_Traits.csv', header=0, low_memory=False)
traits = traits[~traits['Oil'].isna()]

# dictionary to save oil per sample
oil_dic = {}
for index, row in traits.iterrows():
    sampleID=row['Plant ID'].replace(' ','') # formatting ID
    
    oil = str(row['Oil']) # read oil measurement
    if ';' in oil:  # use mean if multiple readings are provided
        tmp = [float(x) for x in oil.split(';')]
        oil = np.mean(tmp)
    oil_dic[sampleID] = float(oil) # save to dictionary

del traits # save memory 


# read SNP from genotype file according of above samples
snp_dic = {}
SNPs = list(pd.read_csv('data/soysnp50k_wm82.a2_41317.txt',nrows=0,sep='\t', header=0))
for sample in tqdm.tqdm(oil_dic):
    try:
        index = SNPs.index(sample)
        snp = subprocess.check_output('awk -F\'\t\' \'{print $%s}\' < data/soysnp50k_wm82.a2_41317.txt'%(index+1), shell=True)
        snp = snp.decode("utf-8") 
        snp = snp.split('\n')
        assert snp[0] == sample
        snp_dic[sample] = [mapping[x] for x in snp[1:]]
    except KeyError as e:
        print('{} not presented in file'.format(sample))
   
pickle.dump(snp_dic, open('snp.pickle','wb'))
