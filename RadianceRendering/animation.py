# import numpy as np
import os
import time

for i in sorted(os.listdir('skies')):
    os.system(f'oconv skies/{i} Motahareh_ahmadi.rad > 123.oct')
    time.sleep(0.02)
    print(i)