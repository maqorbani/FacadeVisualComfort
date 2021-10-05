import numpy as np
import os
import concurrent.futures
import cv2

# Variables
nCPU = 64
dimension = 200  # Image dimensions

# ['av_lum', 'E_v', 'E_v_dir', 'lum_sources', 'omega_sources', 'med_lum']
params = [1, 2, 4, 9, 10, 19]
# End of Variables

# Set the PATH variables
os.environ['PATH'] += os.pathsep + ':/usr/local/radiance/bin'
os.environ['RAYPATH'] = '.:/usr/local/radiance/lib'

# Setting skies
skies = [i[:-6] for i in os.listdir('../render/skies/')]

divisionCPU = len(skies) // nCPU

skyDict = {}
for i in range(nCPU):
    skyDict['sky'+str(i)] = skies[i*divisionCPU:divisionCPU*(i+1)]

for i, sky in enumerate(skies[nCPU * divisionCPU:]):
    skyDict['sky'+str(i)].append(sky)
print(skyDict)

# Mask file preparation
mask = np.loadtxt('mask.txt')

# Changing the directory
os.chdir('../render/Octs')


def octMaker(sky, j):
    for i in sky:
        # Getting the HDR and applying the mask and saving the new HDR
        hdr = np.loadtxt(f'{i}/{i}.gz')
        hdr = np.where(mask != 0, hdr, 0)
        MAX = hdr.max()
        MEAN = hdr.sum()/mask.sum()
        basics = np.array([MAX, MEAN])

        cv2.imwrite(f'{i}/{i}_msk.HDR', hdr.reshape(dimension, dimension))

        # Glare evaluations
        os.system(f'evalglare -d -vth -vh 180 -vv 180 {i}/{i}_1.HDR > {i}/{i}egAll.txt')
        os.system(f'evalglare -d -vth -vh 180 -vv 180 {i}/{i}_msk.HDR > {i}/{i}eg.txt')

        # Evalglare parameter extraction
        with open(f'{i}/{i}egAll.txt') as f:
            egALL = f.readlines()
        prmtrs = np.array(egALL[-1].split(': ')[-1].split(' ')).astype(float)[params]

        with open(f'{i}/{i}eg.txt') as f:
            eg = f.readlines()
        Ev = np.array(eg[-1].split(': ')[-1].split(' ')).astype(float)[2]
        
        # Evalglare extracted parameter write
        # MAX, MEAN, av_lum, E_v, E_v_dir, lum_sources, omega_sources, med_lum, Ev_masked
        np.savetxt(f'{i}/{i}prmtr.txt', np.hstack((basics, prmtrs, Ev)))


# The parallel processing module
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = [executor.submit(octMaker, list(
        skyDict['sky'+str(i)]), i) for i in range(nCPU)]

    for f in concurrent.futures.as_completed(results):
        print(f.result())
