import scipy.stats as stats
import numpy as np
from prfpy.rf import vonMises1D as vm
from prfpy.rf import gauss1D_cart as g1d
import matplotlib.pyplot as plt

vm_params = np.zeros(20)
vm_params[12] = np.pi
print(vm_params)
print(vm(vm_params, 0.0, 5))
plt.plot(np.linspace(0, 2*np.pi, 20), vm(vm_params, 0, 1))
