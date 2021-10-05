# %%
import time
import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.colors import LogNorm
import cv2
from piq import psnr, ssim
from tqdm import tqdm

# %%
Dictionary = {
    'epoch': 5,
    'batch': 8,
    'dataset': '-NM',
    'Model_Arch': 'UNet',
    'View #': 2,
    'All_data': True,
    'Loss': 'MSE',               # Loss: MSE or MSE+RER
    'avg_shuffle': False,        # Shuffle mode
    'avg_division': 50,          # For shuffle mode only
    'transfer learning': False,   # TL mode
    '# samples': 25,             # For transfer Learning only
    '# NeighborPx': 1,           # For model 3 and 4 px neighborhood
    'Bilinear': True             # Bilinear option for UNet
}

AB1 = [i[:-6] for i in os.listdir('skies')]
AB4 = [i for i in os.listdir('AB4') if i in AB1]

arch = Dictionary['Model_Arch']
theLoss = True if Dictionary['Loss'] == 'MSE+RER' else False
divAvg = Dictionary['avg_division']
pxNeighbor = Dictionary['# NeighborPx']
blnr = Dictionary['Bilinear']
alldata = Dictionary['All_data']

if arch == 'UNet':
    from unet import UNet as Model              # UNet model

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch = Dictionary['epoch']
batch = Dictionary['batch']

data_set = Dictionary['dataset']  # Data-sets
View = Dictionary['View #']

TLmode = Dictionary['transfer learning']
mTL = Dictionary['# samples']

x_train = np.load(f'data/data.npz')['train']
x_test = np.load(f'data/data.npz')['test']

n_features = x_train.shape[-1] - 1
m = x_train.shape[0]
mTest = x_test.shape[0]

modelArgs = [n_features, 1, device, blnr] if arch == 'UNet' \
    else [n_features, device]

print('Model summery:')
print(f'View point: {View}')
print(f'Model architecture: {arch}', f'\nData set: "{data_set}"')
print(f'Number of samples: {m}', f'\nNumber of test samples: {mTest}')
print(f'Transfer learning mode: {TLmode}')
print(f'Model arguments: {modelArgs}')

# %%
# Transforming the data into torch tensors.
x_train, y_train = torch.tensor(x_train[:, :, :n_features]), \
    torch.tensor(x_train[:, :, -1])
x_test = torch.tensor(x_test)

# relocation of the channel axes
x_train = np.transpose(x_train, [0, 2, 1]).reshape(-1, n_features, 200, 200)
x_test = np.transpose(x_test, [0, 2, 1]).reshape(-1, n_features, 200, 200)

# %%
# Load the model from PyTorchModel.py to the GPU
model = Model(*modelArgs)
model.to(device)

print(model)
print('\nNumber Of parameters:', sum(p.numel() for p in model.parameters()))

# For transfer learning model load
if TLmode:
    learnedView = 2
    model.load_state_dict(torch.load(
        f'../V{learnedView}DataAnalysis/models/' +
        f'ConvModel{data_set}-{arch}.pth'))

# %%
with torch.no_grad():
    out = model(x_train[3, :, :, :])
    plt.imshow(out.to("cpu").numpy().reshape(200, -1))
    plt.show()

torch.cuda.empty_cache()


# %%
def rer_loss(output, target):
    loss = torch.sqrt(torch.sum((output - target)**2) / torch.sum(target**2))
    return loss

# %%
# Post processing functions


def get_minMax(View):
    '''
    Returns min max values for ab0 and ab4 rendered images.
    '''
    with open(f'AB1/min-max.txt', 'r') as f:
        ab1 = f.readlines()
        ab1 = np.array([i.strip().split(',') for i in ab1])[:, 1].astype(float)
    return np.log10(np.array([ab1.min() + 1e-1, ab1.max()]))


def get_avg_minMax(View, m):
    '''
    Returns avg min max key for reverse normalization.
    ds stands for data-set name
    '''
    return np.load(
        f'data/-{m}-minMAX-key.npy')[6]


def revert_HDR(HDR, minMax):
    '''
    Reverts the predicted images' normalization
    back into original HDR images' scale.
    '''
    HDR = HDR * (minMax[1] - minMax[0]) + minMax[0]
    HDR = np.power(10, HDR)
    return HDR


def revert_avg(avg, m, View):
    keyMM = get_avg_minMax(View, m)
    avg = avg * (keyMM[1] - keyMM[0]) + keyMM[0]
    avg = revert_HDR(avg, get_minMax(View))
    return avg


def predict_HDR_write(x, View, m=None):
    '''
    Uses the trained model to predict the images on the given data set.
    Transforms the normalized predicted images back into HDR images.
    Writes the HDR images into their correspondingly folders.
    '''
    if m is None:
        m = x.shape[0]

    if len(x.shape) == 3:  # Check for one sample input
        x = x.unsqueeze(0)

    key = np.loadtxt('data/key.txt', dtype='str')  # Hour of year for each key
    selKeys = np.loadtxt(f'data/results{m}.txt')   # K-means selected keys
    selKeys = [int(i) for i in selKeys]

    minMax = get_minMax(View)

    with torch.no_grad():
        for i in range(x.shape[0]):
            out = model(x[i]).cpu().numpy().reshape(200, 200)
            out = revert_HDR(out, minMax)
            date_time = key[selKeys[i]]
            os.mkdir(f'E:/Rendered_bak/BaseModel/{date_time}/')
            cv2.imwrite(
             f'E:/Rendered_bak/BaseModel/{date_time}/{date_time}_2.HDR', out)
            print(date_time)


def get_date_time(index, m):
    key = np.loadtxt('data/key.txt', dtype='str')   # Hour of year for each key
    selKeys = np.loadtxt(f'data/results{m}.txt', dtype=int)  # K-means selected
    return key[selKeys[index]], selKeys[index]


# cv2.imwrite('out.HDR', revert_HDR(out, minMax))


# %%
criterion = nn.MSELoss()
epochLoss = []

epochLossBatch = []

if theLoss:
    def criterion(t, y): return nn.MSELoss()(t, y) + 10 * rer_loss(t, y)

# %%
optimizer = optim.Adam(model.parameters(), 0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 2,
                                                 verbose=True, threshold=1e-4,
                                                 cooldown=5)

# %%
# To change the learning rate
optimizer.param_groups[0]['lr'] = 0.0001

# %%
a = time.time()

# epochPercent = 0  # Dummy variable, just for printing purposes
model.train()

for i in tqdm(range(epoch*m)):
    target = y_train[i % m, :].reshape(-1, 1)  # avoiding 1D array
    x = x_train[i % m, :, :, :]
    output = model(x).cpu().reshape(-1, 1)
    loss = criterion(output, target)
    loss.backward()

    epochLoss.append(loss.item())
    epochLossBatch.append(epochLoss[-1])

    if i % batch == batch - 1:
        optimizer.step()
        model.zero_grad()

    # if (i + 1) * 10 // m == epochPercent + 1:
    #     print("#", end='')
    #     epochPercent += 1

    if i % m == m - 1:
        epochLossBatchAvg = sum(epochLossBatch)/m
        print('\n', "-->>Train>>", epochLossBatchAvg)
        epochLossBatch = []

        scheduler.step(epochLossBatchAvg)

print(f'\nIn {time.time() - a:.2f} Seconds')
# %%
plt.plot(np.log10(epochLoss))
# plt.plot(np.log10(testLoss))

a = np.array(epochLoss)
for i in range(int(len(epochLoss) / m)):
    a[i*m:((i+1)*m)] = a[i*m:((i+1)*m)].mean()
plt.plot(np.log10(a), lw=4)

plt.show()
# %%
# model.eval()
i = 1

# for i in range(12):
with torch.no_grad():
    out = revert_HDR(model(x_train[i, :, :]).to(
        "cpu").numpy().reshape(200, -1), get_minMax(View)) * 179
    T = revert_HDR(y_train[i, :].to("cpu").numpy().reshape(200, -1),
                get_minMax(View)) * 179
    out = np.where(y_train[i, :].to("cpu").numpy().reshape(200, -1)>0,
    out, 0)
    T = np.where(y_train[i, :].to("cpu").numpy().reshape(200, -1)>0,
    T, 0)

# Plotting both prediction and target images
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 25))
im = ax1.imshow(np.log10(out), cmap='jet', vmin=0, vmax=7)
ax1.title.set_text(f'{i}\n{AB4[i]}\nprediction')
ax2.imshow(np.log10(T), cmap='jet', vmin=0, vmax=7)
ax2.title.set_text('ground_truth')
ax3.imshow(np.log10(np.abs(out-T)), cmap='jet', vmin=0, vmax=7)
ax3.title.set_text('difference')
# fig.colorbar(im)

# plt.savefig(f'exports/{i}.png', dpi=150)
plt.show()

# %%
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(7, 10))
ax1.hist((out).ravel(), bins=100, range=(0, 1))
ax1.title.set_text('y_hat')
ax2.hist((T).ravel(), bins=100, range=(0, 1))
ax2.title.set_text('ground_truth')
ax3.hist(np.abs(out-T).ravel(), bins=100, range=(0, 1))
ax3.title.set_text('difference')

plt.show()
# %%
answer = input("Are you sure that you want to save? [yes/any]")

if answer == 'yes':
    torch.save(model.state_dict(), f'models/ConvModel-{arch}.pth')


# %%
model.load_state_dict(torch.load(f'models/ConvModel-{arch}.pth'))

# %%
# For transfer learning model load
learnedView = 2
model.load_state_dict(torch.load(
    f'../V{learnedView}DataAnalysis/models/ConvModel{data_set}-{arch}.pth'))

# %%
# Loss calculator over the train-test sets
a = time.time()

train_loss = []
test_loss = []

train_illum = []
test_illum = []

with torch.no_grad():
    for i in tqdm(range(m)):
        target = y_train[i, :].reshape(-1, 1)  # avoiding 1D array
        x = x_train[i, :, :, :]
        output = model(x).cpu().reshape(-1, 1)
        loss = criterion(output, target)
        train_loss.append(loss.item())
        train_illum.append(x[-1].mean().item())
    for i in tqdm(range(mTest)):
        target = y_test[i, :].reshape(-1, 1)  # avoiding 1D array
        x = x_test[i, :, :, :]
        output = model(x).cpu().reshape(-1, 1)
        loss = criterion(output, target)
        test_loss.append(loss.item())
        test_illum.append(x[-1].mean().item())

print('\n', 'MSE', sum(train_loss)/m, sep='\n')
print(sum(test_loss)/mTest)

print(f'\nIn {time.time() - a:.2f} Seconds')
# %%
a = time.time()

train_ssim = []
test_ssim = []
train_psnr = []
test_psnr = []

with torch.no_grad():
    for i in tqdm(range(m)):
        target = y_train[i, :].reshape(1, 1, 144, 256)
        x = x_train[i, :, :, :]
        output = model(x).cpu()
        loss = ssim(target, output)
        train_ssim.append(loss.item())
        loss = psnr(target, output)
        train_psnr.append(loss.item())
    for i in tqdm(range(mTest)):
        target = y_test[i, :].reshape(1, 1, 144, 256)
        x = x_test[i, :, :, :]
        output = model(x).cpu()
        loss = ssim(target, output)
        test_ssim.append(loss.item())
        loss = psnr(target, output)
        test_psnr.append(loss.item())

print('\nSSIM')
print(sum(train_ssim)/m)
print(sum(test_ssim)/mTest)
print('PSNR')
print(sum(train_psnr)/m)
print(sum(test_psnr)/mTest)

print(f'\nIn {time.time() - a:.2f} Seconds')

# %%

T1 = revert_HDR(T, get_minMax(5))
out1 = revert_HDR(out, get_minMax(5))

plt.scatter(T1.ravel()*179, out1.ravel()*179, s=1)
plt.plot([0, T1.max()*179], [0, T1.max()*179], color='red')
plt.xlabel('Ground truth luminance value')
plt.ylabel('Prediced luminance value')

# %%
fig, ax1 = plt.subplots(1, figsize=(10, 7))
plt.hist(revert_HDR(x_train[:, :, -1].ravel(),
                    minMax)/143, bins=200, color='black')

# %%
fig, ax1 = plt.subplots(1, figsize=(15, 3))
ax1.hist(revert_HDR(x_train[:, :, -1], get_minMax(2)
                    ).ravel()/143, bins=200, color='black')
ax1.set_xlim([0, 1])
fig.patch.set_visible(False)
ax1.axis('off')
plt.savefig('hist.png')

# %%
plt.style.use('seaborn')
plt.style.use('seaborn-talk')
fig, ax1 = plt.subplots(1, figsize=(15, 11))
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
    label.set_fontsize(16)
ax1.hist(y_test.numpy().ravel(), bins=500, color='black')
plt.xlabel('Pixel value', fontsize=25)
plt.ylabel('Number of pixels', fontsize=25)
# %%
plt.style.use('seaborn')
plt.style.use('seaborn-talk')
fig, ax1 = plt.subplots(1, figsize=(15, 11))
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
    label.set_fontsize(16)
ax1.hist(revert_HDR(y_test, get_minMax(2)).numpy().ravel()*179,
         bins=500, color='black')
plt.xlabel('Pixel value: $Cd/m^2$', fontsize=25)
plt.ylabel('Number of pixels', fontsize=25)
plt.savefig('../result/Normalization/3.png')

# %%
# a = np.loadtxt('data/results16.txt')
alt = np.loadtxt('data/Altitude.txt')
azi = np.loadtxt('data/Azimuth.txt') - 180
dire = np.loadtxt('data/dirRad.txt')
dif = np.loadtxt('data/difHorRad.txt')
key = np.loadtxt('data/key.txt', dtype='str')

with torch.no_grad():
    for nu in a:
        out = model(x_train[nu, :, :]).to(
            "cpu").numpy().reshape(144, -1)
        T = y_train[nu, :].to("cpu").numpy().reshape(144, -1)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(
            30, 10), gridspec_kw={'wspace': 0.08, 'hspace': 0})
        im = ax1.imshow(out, cmap='plasma', vmin=0, vmax=0.9)
        ax1.title.set_text(
            f'{key[nu][:-2]}:00\ndirect: {int(dire[nu])} wh/m2\ndiffuse:\
                 {int(dif[nu])} wh/m2\n')
        ax2.imshow(T, cmap='plasma', vmin=0, vmax=0.9)
        # ax2.title.set_text('ground_truth')
        ax3.imshow(np.abs(out-T), cmap='plasma', vmin=0, vmax=0.9)
        # ax3.title.set_text('difference')
        # plt.savefig(f'../result/False_color/{nu}.png', dpi=100)

# %%
a = range(5)  # [156, 556, 744, 1032, 2081, 3439, 3573, 3363]
date = get_date_time(a, m)[0]
hoy = get_date_time(a, m)[1]

alt = np.loadtxt('data/Altitude.txt')
azi = np.loadtxt('data/Azimuth.txt') - 180
dire = np.loadtxt('data/dirRad.txt')
dif = np.loadtxt('data/difHorRad.txt')

with torch.no_grad():
    for index, nu in enumerate(a):
        out = revert_HDR(model(x_train[nu, :, :]).to(
            "cpu").numpy().reshape(144, -1), get_minMax(View)) * 179
        T = revert_HDR(y_train[nu, :].to("cpu").numpy().reshape(144, -1),
                       get_minMax(View)) * 179

        print(criterion(torch.tensor(out), torch.tensor(T)))

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(
            30, 10), gridspec_kw={'wspace': 0.08, 'hspace': 0})
        im = ax1.imshow(np.log10(out), cmap='plasma', vmin=0, vmax=4)
        # ax1.title.set_text(
        #     f'{date[nu][:-2]}:00\ndirect: {int(dire[hoy[nu]])}' +
        #     f' wh/m2\ndiffuse: {int(dif[hoy[nu]])} wh/m2\n')
        ax2.imshow(np.log10(T), cmap='plasma', vmin=0, vmax=4)
        # ax2.title.set_text('ground_truth')
        ax3.imshow(np.log10(np.abs(out-T)), cmap='plasma', vmin=0, vmax=4)
        # ax3.title.set_text('difference')
        # fig.colorbar(im)
        # plt.savefig(f'../result/False_color/V2{hoy[index]}.png', dpi=300)
        print(T.max())
        plt.show()

# %%
# The average map plotter for TL datasets
avgMM = get_avg_minMax(View, m)[6]
avg = x_train[1, 6] * (avgMM[1] - avgMM[0]) + avgMM[0]
avg = revert_HDR(avg, get_minMax(5))*179
plt.imshow(np.log10(avg), vmin=0, vmax=4, cmap='plasma')
plt.savefig(f'../result/TLAVGmap/AVG_{mTL}.png', dpi=300)

# %%


def plot_results(inList, tnsr, m, save=False):
    '''
    Gets a list of numbers and a set of tensors then plots their predicted,
    ground truth, and error maps.
    tnsr is: (x_, y_)
    '''
    with torch.no_grad():
        for i, nu in enumerate(inList):
            # Get information
            date = get_date_time(nu, m)[0]
            hoy = get_date_time(nu, m)[1]

            # Prediction using the given tensors
            out = revert_HDR(model(tnsr[0][nu, :, :]).to(
                "cpu").numpy().reshape(144, -1), get_minMax(View)) * 179

            T = revert_HDR(tnsr[1][nu, :].to("cpu").numpy().reshape(144, -1),
                           get_minMax(View)) * 179

            # To print the loss and other information
            print(f'{criterion}'[:-3], ': ',
                  criterion(torch.tensor(out), torch.tensor(T)))
            print('Max Luminance:', T.max())
            print(date)

            # Results plotter
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(
                30, 10), gridspec_kw={'wspace': 0.08, 'hspace': 0})

            ax1.imshow(np.log10(out), cmap='plasma', vmin=0, vmax=4)
            ax2.imshow(np.log10(T), cmap='plasma', vmin=0, vmax=4)
            ax3.imshow(np.log10(np.abs(out-T)), cmap='plasma', vmin=0, vmax=4)

            # Save the results in the folder
            if save and not TLmode:
                plt.savefig(f'../result/False_color/TL/{hoy}.png', dpi=300)
            elif save and TLmode:
                plt.savefig(
                    f'../result/False_color/TL/{hoy}-{mTL}.png', dpi=300)

            plt.show()


def plot_avg(save=False):
    '''
    Plots the average maps of train and test sets and their error map.
    '''
    train = revert_avg(x_train[0, 6], mTL, View) * 179
    test = revert_avg(x_test[0, 6], m, View) * 179

    # Average plotter
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(
        30, 10), gridspec_kw={'wspace': 0.08, 'hspace': 0})

    ax1.imshow(np.log10(train), cmap='plasma', vmin=0, vmax=4)
    ax2.imshow(np.log10(test), cmap='plasma', vmin=0, vmax=4)
    ax3.imshow(np.log10(np.abs(train-test)), cmap='plasma', vmin=0, vmax=4)

    # Save the results in the folder
    plt.savefig(f'../result/False_color/TL/AVG-{mTL}.png', dpi=300)

    plt.show()


def tnsr_fixer(avgMap, x_test):
    tnsr = x_test[:, :, :, :].clone()
    tnsr[:, 6] = avgMap
    return tnsr


# %%
plot_results([9, 11], (tnsr_fixer(x_train[1, 6], x_test), y_test), 400, True)
