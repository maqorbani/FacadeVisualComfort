import os
import shutil
import concurrent.futures
import numpy as np
import io
import time

# Variables
ab = 4
nCPU = 64
All_skies = False
# End of Variables

# os.chdir('Desktop/TheRender/')
print('Current working dir is: ' + os.getcwd())

if not All_skies and os.path.exists('render_list.txt'):
    skies = np.loadtxt('render_list.txt', dtype=str)
    skies = [f'{i}__.sky' for i in skies]
else:
    skies = os.listdir('skies/')  # 4399
# skies = skies[:3]  # To test a few number

# Overview
print(f'Number of HDRs to be rendered: {len(skies)}')
print(f'Number of CPUs: {nCPU}')
print(f'Number of Ambient Bounces: {ab}')

if not os.path.exists('AB4'):
    os.mkdir('AB4')

# Omitting rendered skies from the list
dirs = []

for i in os.listdir('AB4/'):
    if os.path.isdir('AB4/'+i):
        if not os.path.exists(f'AB4/{i}/done'):
            dirs.append(i)
            # shutil.rmtree('AB4/'+i)
            print(i)
        else:
            try:
                skies.remove(f'{i}__.sky')
            except ValueError:
                print(f'{i}')

# print(skies)
divisionCPU = len(skies) // nCPU

skyDict = {}
for i in range(nCPU):
    skyDict['sky'+str(i)] = skies[i*divisionCPU:divisionCPU*(i+1)]

for i, sky in enumerate(skies[nCPU * divisionCPU:]):
    skyDict['sky'+str(i)].append(sky)
print(skyDict)
# print(len(list(skyDict.values())[5]))


# Changing the directory
os.chdir('AB4')

# Set the PATH variables
os.environ['PATH'] += os.pathsep + ':/usr/local/radiance/bin'
os.environ['RAYPATH'] = '.:/usr/local/radiance/lib'

# Setting empty list for bad HDRs
badDirs = []


def octMaker(sky, j):
    for i in sky:
        a = time.time()
        # make the oconv command
        oconv = f'oconv ../skies/{i} ../Geo.rad > {i[:-6]}/tower.oct'

        # make the rpict command
        render = f'rpict -t 60 -vth -vh 180 -vv 180 -x 800 -y 800 \
-ps 2 -pt 0.05 -pj 0.9 -dj 0.7 -ds 0.15 -dt 0.05 -dc 0.75 -dr 3 -dp 512 \
-st 0.15 -ab {ab} -ad 2048 -as 1024 -ar 512 -aa 0.1 -lr 8 -lw 0.005 \
-e {i[:-6]}/error.log {i[:-6]}/tower.oct > {i[:-6]}/{i[:-6]}.HDR'

        # Make the pfilt command
        pfilt = f'pfilt -x /4 -y /4 {i[:-6]}/{i[:-6]}.HDR > {i[:-6]}/{i[:-6]}_1.HDR'

        # Make the render matrix
        pvalue = f'pvalue -h -d -H -b -o {i[:-6]}/{i[:-6]}_1.HDR > {i[:-6]}/{i[:-6]}.txt'

        # Make the render directory
        os.mkdir(str(i[:-6]))

        # Make the octree file
        os.system(oconv)

        # renders using rpict & remove the octree afterwards
        os.system(f'{render}')
        os.remove(f'{i[:-6]}/tower.oct')

        # Downsampling rendered images
        os.system(pfilt)

        if os.system(pvalue):
            print(i[:-6], ' is bad image.')
            badDirs.append(i[:-6])
            os.remove(f'{i[:-6]}/{i[:-6]}.txt')
            continue
        else:
            print(i[:-6])
        with io.open(f'{i[:-6]}/{i[:-6]}.txt', 'r', encoding=None) as f:
            vector = np.loadtxt(f, delimiter='\n')
            MIN, MAX = vector.min(), vector.max()
            os.system(f'echo "{MIN},{MAX},{(time.time()-a)/60:.3f},{i[:-6]}" >> min-max{j}.txt')
            np.savetxt(f'{i[:-6]}/{i[:-6]}.gz', vector)
        os.remove(f'{i[:-6]}/{i[:-6]}.txt')

        # Finished statement
        os.system(f'touch {i[:-6]}/done')
        print(str(i[:-6] + ' done!'))

    return 'Process #' + str(j) + ' is done!'


# The parallel processing module
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = [executor.submit(octMaker, list(
        skyDict['sky'+str(i)]), i) for i in range(nCPU)]

    for f in concurrent.futures.as_completed(results):
        print(f.result())

with open('badDirs.txt', 'w') as f:
    # [f.write(i + ',') for i in badDirs]
    f.write(', '.join(badDirs))

#Concatenating all min-max files
os.system('cat min-max* > min-max.txt')
