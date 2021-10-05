'''
This script gathers all of the calculated parameters of rendered images
into a pandas dataframe and saves it into an excel file.
The parameters are calculated using the EvalGlare command and saved
into .txt files across the folders.
'''
# %%
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
# %%
# Setting skies
keys = np.loadtxt('All_keys.txt', dtype=str)

# Changing the directory
os.chdir('../render/Octs')

# %%
# Creating the data set
dataset= np.zeros((keys.shape[0], 9))

for i, key in enumerate(tqdm(keys)):
    dataset[i] = np.loadtxt(f'{key}/{key}prmtr.txt')

np.savetxt('../../analysis/dataset.csv', dataset, delimiter=',')

dataframe = pd.DataFrame(data=dataset, index=keys, columns=['MAX', \
    'MEAN', 'av_lum', 'E_v', 'E_v_dir', 'lum_sources', 'omega_sources',\
    'med_lum', 'Ev_masked'])

dataframe[['MAX']] = dataframe[['MAX']] * 179
dataframe[['MEAN']] = dataframe[['MEAN']] * 179

writer = pd.ExcelWriter('extracted_prameters.xlsx')
dataframe.to_excel(writer)
writer.save()
