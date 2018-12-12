#!/usr/bin/env python3
import sys, os
import subprocess
import numpy as np
import pandas as pd
import pickle

mapping = {'A':0, # using integers to represent nucleotide types
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
pheo_names = [x.replace(' ','') for x in list(traits['Plant ID'])] # get sampleID from pheotype record

# read SNP sequences from genotype file of above samples
snp_dic = {} # keys are sampleID and values are their SNP sequences

# the following line is a naive solution to read large files as it takes a lot memory and time
# you may want to change it to a faster loading method
print('reading SNP file, this may take a long time...')
SNPdf = pd.read_csv('data/soysnp50k_wm82.a2_41317.txt', sep='\t', header=0)
snp_names = list(SNPdf)[5:] # get sampleID from snp record

sampleIDs = list(set(snp_names) & set(pheo_names)) # samples that have both pheotype record and oil value

# dictionary: keys are sampleID and values are cooresponding oil production
oil_dic = {}
for index, row in traits.iterrows():
    sampleID=row['Plant ID'].replace(' ','') # formatting ID to match the style in SNP file
    traits.loc[index, 'Plant ID'] = sampleID # change that in dataframe, too
    if sampleID not in sampleIDs:
        continue
    oil = str(row['Oil']) # read oil measurement
    if ';' in oil:  # use mean if multiple readings are provided
        tmp = [float(x) for x in oil.split(';')]
        oil = np.mean(tmp)
    oil_dic[sampleID] = float(oil) # save to dictionary

traits = traits[traits['Plant ID'].isin(sampleIDs)] # filter out samples from dataframe not presented in SNP records

for sampleID in sampleIDs:
    snp_dic[sampleID] = np.asarray([mapping[x] for x in SNPdf[sampleID]], dtype=np.int8) # using int8 saves more space

assert len(sampleIDs) == traits.shape[0] == len(oil_dic) == len(snp_dic)

preserved_col_names = list(SNPdf)[:5] + sampleIDs
SNPdf = SNPdf[preserved_col_names]
SNPdf.to_csv('data/soysnp50k_wm82.a2_41317_Cleaned.txt', sep='\t', index=False)
print('cleaned snp records saved')

traits.to_csv('data/Soybean_Traits_Cleaned.csv', index=False)
print('cleaned phenotype saved')

pickle.dump(snp_dic, open('data/snp.pickle','wb')) # save snp info to a dictionary 
print('snp pickle saved')

pickle.dump(oil_dic, open('data/oil.pickle','wb')) # save oil info
print('oil pickle saved')
