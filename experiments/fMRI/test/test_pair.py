import os
import glob
import random
import math
from copy import deepcopy
import time
import numpy as np
import itertools
from PIL import Image
from psychopy.core import getTime

"""
Select ping pairs from ping pool, the pings in each pair should be different and 90 degree apart.
"""

def create_ping():
    rng = random.SystemRandom()
    ping_pairs = np.empty((0,2))
    print('Creating Ping pairs...')
    dist_pings = 90
    run = 0
    e = 0
    angles_pings = [int(i) for i in np.linspace(45, 360+45, 24, endpoint=False)]%np.array([360])
    while len(ping_pairs)*2 != len(angles_pings):
        if run > 0:
            print('Creation failed, trying to re-run it, run ', run)
        run += 1
        ping_pool = deepcopy(angles_pings)
        ping_pool_tmp = deepcopy(ping_pool)
        ping_pairs = np.empty((0,2))
        t0 = getTime()

        while len(ping_pool) > 0:
            ping1 = rng.choice(ping_pool_tmp)
            ping_pool_tmp = np.delete(ping_pool_tmp, ping_pool_tmp==ping1)
            ping2 = rng.choice(ping_pool_tmp)
            ping_pool_tmp = np.delete(ping_pool_tmp, ping_pool_tmp==ping2)
            
            if np.abs(ping1-ping2) >= dist_pings and np.abs(ping1-ping2) <= 360-dist_pings:
                ping_pairs = np.concatenate((ping_pairs, [[ping1, ping2]]))
                ping_pool = deepcopy(ping_pool_tmp)
            else:
                ping_pool_tmp = deepcopy(ping_pool)
            
            if len(ping_pool)==2 and (np.abs(ping_pool[0]-ping_pool[1]) <= dist_pings or 
                                        np.abs(ping_pool[0]-ping_pool[1]) >= 360-dist_pings):
                break

            if getTime() - t0 > 0.5:
                print('Time out, re-run it')
                print('ping_pool_tmp: ', ping_pool_tmp)
                print('ping_pool: ', ping_pool)
                print('ping_pairs: ', ping_pairs)
                e = 1
                break

    ping_pairs = ping_pairs.astype(int)
    print('Ping pairs created successfully')

    return ping_pairs, e

et = 0
for i in range(100):
    _, e = create_ping()
    print('i: ', i)
    et = et + e

print('et: ', et)