from scipy.stats import expon
import matplotlib.pyplot as plt
import numpy as np
import math
from nilearn.glm.first_level.hemodynamic_models import spm_hrf
import random
import time
import h5py
import os

def simulation_shuffle():
        # Create hrf curve
        hrf = spm_hrf(0.1, oversampling=1)
        hrf /= hrf.max()
        
        # 
        final_seq_seq = np.tile([*range(6)], 20)
        # Bulid ITI distribution
        ITI_expon = expon.rvs(scale=1,loc=2,size=len(final_seq_seq))
        ITI_expon = [math.floor(n*2)/2 for n in ITI_expon]
        
        # Total time of one trial
        trial_time = [0.4 + 0.4*4 + ITI_time for ITI_time in ITI_expon]
        trial_time_onset = np.cumsum(trial_time)
        trial_time_onset = [int(x*10) for x in trial_time_onset]
        all_time = np.arange(0, trial_time_onset[-1] + trial_time[-1], 0.1)
        seq_seq = final_seq_seq.copy()
        cov_cond_min = np.Inf

        # For loop simulation
        for i in range(1000):
            # Transfer sequence and onset to dm
            dm = np.zeros((all_time.shape[0]*10, max(final_seq_seq)))
            # Create the design matrix
            rng.shuffle(ITI_expon)
            rng.shuffle(seq_seq)
            event_times = np.random.choice(trial_time_onset, dm.shape)
            for i,x in enumerate(event_times.T):
                dm[x,i] = 1
            dm_c = np.array([np.convolve(d, hrf)[:d.shape[0]] for d in dm.T])
            cov_cond = np.sum(np.triu(np.cov(dm_c), k=1)**2)
            if cov_cond <= cov_cond_min:
                final_seq_seq = seq_seq.copy()
                final_ITI_expon = ITI_expon.copy()
            cov_cond_min = min(cov_cond, cov_cond_min)

        return final_seq_seq, final_ITI_expon, cov_cond_min

dir_exp = input('Input experiment directory: ')
path_h5file = os.path.join(dir_exp, 'stimuli', 'seq_1000.h5')
rng = random.SystemRandom()
sequence = []
ITI_expon = []

for i in range(1000):
    print(i)
    t = time.time()
    final_seq_seq, final_ITI_expon, cov_cond_min = simulation_shuffle()
    elapsed = time.time() - t
    print(elapsed)
    print(cov_cond_min)
    sequence.append(final_seq_seq)
    ITI_expon.append(final_ITI_expon)
    print('-----------------')

with h5py.File(path_h5file,'w') as h5f:
        h5f.create_dataset('sequence', data=sequence, dtype='i')
        h5f.create_dataset('ITI_expon', data=ITI_expon, dtype='f')