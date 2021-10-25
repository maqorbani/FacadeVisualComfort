# import numpy as np
import os
import time

for i in sorted(os.listdir('skies')):
    os.system(f'oconv skies/{i} Geo.rad > 123.oct')
    time.sleep(0.1)
    print(i)
