# %%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# %%
day = np.loadtxt('day.txt', dtype=int)
month = np.loadtxt('month.txt', dtype=int)
hour = np.loadtxt('hour.txt', dtype=int)
# %%
a = []
for i, _ in enumerate(day):
    a.append(f'{month[i]}_{day[i]}@{hour[i]}00')
# %%
df = pd.read_excel('Alt6_extrctd_prmtr.xlsx', engine='openpyxl')
print(df.head())
print(df.shape)
# %%
mm = 1  # 0: MAX, 1: MEAN
data = {i:[0, 0] for i in a}
for i, v in enumerate(df['Unnamed: 0']):
    data[v] = [df['MAX'][i], df['MEAN'][i]]

fig, ax = plt.subplots(1, figsize=(16, 8))
plt.imshow(np.log10(np.rot90(np.array(list(data.values()))[:, mm].reshape(-1, 24))+1), 
           aspect='auto', cmap='jet', interpolation='None', vmin=0)
plt.colorbar()
ms = [-0.5, 30.5, 58.5, 89.5, 119.5, 150.5, 180.5, 211.5, 242.5, 272.5, 303.5, 333.5] #, 364.5
# ms = [i + 15 for i in ms]
ax.set_xticks(ms)
ax.set_yticks([-0.5, 5.5, 11.5, 17.5, 23.5])
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=90, ha='left')
ax.set_yticklabels([0, 6, 12, 18, 24])

# Create offset transform by 5 points in y direction
dx = 28/72.; dy = 0/72. 
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
# apply offset transform to all x ticklabels.
for label in ax.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

ax.grid()
