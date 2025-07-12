
import sys
print(sys.executable)
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from datetime import datetime

from RGB_kernel_design import design_RGB_kernels

if __name__=="__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    imgs_output_dir = os.path.join('./outputs', timestamp)
    os.makedirs(imgs_output_dir, exist_ok=True)

    # The kernels below are what we want to inverse design with PSF engineering.
    np.random.seed(10)
    kernel = np.random.uniform(low=0,high=1.0,size=(3,10,10))
    
    kernel[0] = np.zeros_like(kernel[0])
    kernel[1] = np.zeros_like(kernel[1])
    
    design_RGB_kernels(kernel, imgs_output_dir, kernel_weight_RGB=[0.0, 0.0, 1.0])
