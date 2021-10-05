'''
This file was used to downsample 
the mask file from 800*800 to 200*200.
'''
# %%
import numpy as np
import io
import os
import matplotlib.pyplot as plt

# Set the PATH variables
os.environ['PATH'] += os.pathsep + ':/usr/local/radiance/bin'
os.environ['RAYPATH'] = '.:/usr/local/radiance/lib'

# To downsample the mask
os.system(f'pfilt -1 -x /4 -y /4 mask.HDR > small_mask.HDR')

# To make the mask matrix file
os.system('pvalue -h -d -H -b -o small_mask.HDR > mask.txt')
with io.open('mask.txt', encoding=None) as f:
    mask = np.loadtxt(f, delimiter='\n')

mask = np.where(mask > 0, 1, 0)

plt.imshow(mask.reshape(200, 200))
