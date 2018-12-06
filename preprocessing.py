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

# dictionary: keys are sampleID and values are cooresponding oil production
oil_dic = {}
for index, row in traits.iterrows():
    sampleID=row['Plant ID'].replace(' ','') # formatting ID to match the style in SNP file

    oil = str(row['Oil']) # read oil measurement
    if ';' in oil:  # use mean if multiple readings are provided
        tmp = [float(x) for x in oil.split(';')]
        oil = np.mean(tmp)
    oil_dic[sampleID] = float(oil) # save to dictionary

pickle.dump(oil_dic, open('data/oil.pickle','wb')) # save oil info
print('oil infomation saved')
del traits # to save memory

# read SNP sequences from genotype file of above samples
snp_dic = {} # keys are sampleID and values are their SNP sequences
# the following line is a naive solution to read large files as it takes a lot memory and time
# you may want to change it to a faster loading method
print('reading SNP file, this may take a long time...')
SNPdf = pd.read_csv('data/soysnp50k_wm82.a2_41317.txt', sep='\t', header=0)
na_count = 0
for sampleID in oil_dic.keys():
    try:
        snp_dic[sampleID] = np.asarray([mapping[x] for x in SNPdf[sampleID]], dtype=np.int8) # using int8 saves more space
    except:
        print('{} not presented'.format(sampleID))
        na_count += 1
# you can print out na_count to see how many samples are missing from the SNP record
print('{} are missing from SNP record'.format(na_count))

pickle.dump(snp_dic, open('data/snp.pickle','wb')) # save snp info to a dictionary 
print('snp information saved')
