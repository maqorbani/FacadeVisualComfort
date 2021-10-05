'''
The results.txt extracted from the k-means step, should be placed in the
data folder at the root of this directory representing keys of X samples.
'''
# %%
import os
from posix import listdir
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
alt = np.loadtxt('data/Altitude.txt')          # Sun altitude
azi = np.loadtxt('data/Azimuth.txt') - 180     # Sun azimuth
key = np.loadtxt('data/key.txt', dtype='str')  # Hour of year for each key

features = {
    "Alternative": 2,
}

View = features["Alternative"]
os.chdir(f'../Alt{View}/')
# %%


def sample_index(View):
    AB1 = [i[:-6] for i in os.listdir('skies')]
    AB4 = [i for i in os.listdir('AB4') if i in AB1]
    [AB1.remove(i) for i in AB4]
    return AB1, AB4


choice, selKeys = sample_index(View)
m, mTest = len(selKeys), len(choice)
selKeys = [list(key).index(i) for i in selKeys]

# %%
with open('AB1/min-max.txt') as f:
    ab1 = f.readlines()

ab1 = np.array([i.strip().split(',') for i in ab1])[:, 1].astype(float)

TheTuple =  [0.0, ab1.max(), 0.0, ab1.max()] 

# %%


def TensorMaker(indices, TheTuple, train_set):
    '''
    Creates the tensor ready for the DL model training and testing
    The images required for this function are 144 by 256 pixels
    This function assumes that the previous cells has have executed
    therefore alt,azi,dire,dif and key are available in the memory
    n is the number of samples in the desired tensor
    '''
    n = len(indices)  # if train set: n = m else, m = n of train set
    tnsr = np.zeros((n, 40000, 6))

    #                                                             # X, Y
    tnsr[:, :, 0] = np.array(list(range(200))*200)
    tnsr[:, :, 1] = np.array(list(range(200))*200).reshape(200, 200).T.ravel()

    for i, x in enumerate(tqdm(indices)):
        tnsr[i, :, 2] = alt[x]                                    # altitude
        tnsr[i, :, 3] = azi[x]                                    # azimuth

        #                                                         # AB0, Ab4
        tnsr[i, :, -2] = np.loadtxt(f'AB1/{key[x]}/{key[x]}.gz')
        tnsr[i, :, -1] = np.loadtxt(f'AB4/{key[x]}/{key[x]}.gz')

    tnsr = tnsr.astype('float32')

    tnsr[:, :, -2] = np.log10(tnsr[:, :, -2]+1e-1)                # Normalize
    tnsr[:, :, -1] = np.log10(tnsr[:, :, -1]+1e-1)                # Normalize

    Tuples = np.array(TheTuple)
    Tuples[:2] = np.log10(Tuples[:2]+1e-1)
    Tuples[2:] = np.log10(Tuples[2:]+1e-1)

    tnsr[:, :, -2:] = forceMinMax(tnsr[:, :, -2:], Tuples)        # AB0, AB4

    tnsr[:, :, :4] = minMaxScale(tnsr[:, :, :4 ], train_set, n)

    return tnsr


def minMaxScale(tnsr, train_set, n):
    if train_set:
        minMax = np.zeros((tnsr.shape[-1], 2))
        minMax[:, 0] = tnsr.min(axis=(0, 1))
        minMax[:, 1] = tnsr.max(axis=(0, 1))
        minMax[2, 0], minMax[2, 1] = alt.min(), alt.max()    # altitude
        minMax[3, 0], minMax[3, 1] = azi.min(), azi.max()    # azimuth

        np.save(
            f'data/-{n}-minMAX-key.npy',
            minMax)

    else:
        minMax = np.load(
            f'data/-{m}-minMAX-key.npy')

    for i in range(tnsr.shape[-1]):
        tnsr[:, :, i] = (tnsr[:, :, i]-minMax[i, 0]) / \
            (minMax[i, 1] - minMax[i, 0])

    return tnsr


def forceMinMax(tnsr, Tuples):
    ab0min, ab0MAX, ab4min, ab4MAX = Tuples

    tnsr[:, :, 0] = (tnsr[:, :, 0] - ab0min) / (ab0MAX - ab0min)
    tnsr[:, :, 1] = (tnsr[:, :, 1] - ab4min) / (ab4MAX - ab4min)
    return tnsr


def minMaxFinder():
    ab0MAX = 0
    ab0min = 10000
    for i in os.listdir('ab0'):
        file = np.loadtxt(f'ab0/{i}/{i}.gz')
        if file.max() > ab0MAX:
            ab0MAX = file.max()
        if file.min() < ab0min:
            ab0min = file.min()

    ab4MAX = 0
    ab4min = 10000
    for i in os.listdir('ab4'):
        file = np.loadtxt(f'ab4/{i}/{i}.gz')
        if file.max() > ab4MAX:
            ab4MAX = file.max()
        if file.min() < ab4min:
            ab4min = file.min()

        return ab0MAX, ab0min, ab4MAX, ab4min


# %%
train = TensorMaker(selKeys, TheTuple, True)

# %%
try:
    choice = np.loadtxt(f'data/test-set-{mTest}.txt', int)
except OSError:
    testList = list(range(4141))
    for i in selKeys:
        testList.remove(i)
    choice = np.random.choice(testList, mTest)
    np.savetxt(f'data/test-set-{mTest}.txt', choice, fmt='%s', delimiter='\n')

plt.scatter(azi, alt, c='grey', s=2)
plt.scatter(azi[choice], alt[choice], c='red', s=10)
plt.show()

# %%
test = TensorMaker(choice, TheTuple, False)

# %%
if os.path.exists(f'../V{View}DataAnalysis/data/data' + fileName + '.npz'):
    answer = input(
        "Are you sure that you want to overwrite the existing file? [yes/any]")
    if answer == 'yes':
        np.savez_compressed(f'../V{View}DataAnalysis/data/data' +
                            fileName + '.npz', train=train, test=test)
    else:
        print('The process was canceled.')
else:
    np.savez_compressed(f'../V{View}DataAnalysis/data/data' +
                        fileName + '.npz', train=train, test=test)

# %%
# TRANSFER LEARNING DATASET CREATOR!

if features["Transfer Learning mode"]:
    a, b, c = features["Transfer Learning"]
    TLsamples = np.arange(a, b, c)

    try:
        choice = np.loadtxt(f'data/test-set-{mTest}.txt', int)
    except OSError:
        testList = list(range(4141))
        for i in selKeys:
            testList.remove(i)
        choice = np.random.choice(testList, mTest)
        np.savetxt(f'data/test-set-{mTest}.txt',
                   choice, fmt='%s', delimiter='\n')

    for i in TLsamples:
        m = i  # m says which minMAX key, minMaxScale function shoud read
        selKeys = np.loadtxt(f'data/results{i}.txt')
        selKeys = [int(i) for i in selKeys]
        train = TensorMaker(selKeys, TheTuple, True)
        test = TensorMaker(choice, TheTuple, False)
        np.savez_compressed(f'../V{View}DataAnalysis/data/data' +
                            fileName+f'-{i}.npz', train=train, test=test)
