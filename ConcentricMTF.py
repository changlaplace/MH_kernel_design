
import sys
print(sys.executable)
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from datetime import datetime
from typing import List, Tuple
from scipy.ndimage import map_coordinates  

from RGB_kernel_design import design_RGB_kernels


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# imgs_output_dir = os.path.join('./outputs', timestamp)
imgs_output_dir = os.path.join('./outputs', 'MTF_trial')
os.makedirs(imgs_output_dir, exist_ok=True)

# Firstly generate a 1d array with shape [mtf_size//2,] as a radius slice of the 2d rotational symmetric MTF matrix

mtf_size :int = 48
gaussian_params :List[Tuple[float,float,float]] = [
        (0, 1, 1),
        (20, 1, 1),
        # (44, 1, 1)
    ]  #list of gaussians: [mu, sigma, amp]
x1d = np.arange(mtf_size//2)
signal = np.zeros(mtf_size//2)

assert max([mu for mu,sigma,amp in gaussian_params])<= mtf_size//2, \
    "This 1d array is a radius of the mtf so the size should <=mtf_size//2"
for gaussian_param in gaussian_params:
    mu, sigma, amp = gaussian_param
    signal += amp * np.exp(-0.5 * ((x1d - mu) / sigma) ** 2)
center = mtf_size // 2

# Then rotate the 1d array to get 2d mtf
y2d, x2d = np.ogrid[:mtf_size, :mtf_size]
radius = np.sqrt((x2d - center)**2 + (y2d - center)**2)
mtf_2d = map_coordinates(signal, [radius], order=1, mode='nearest')

# Visulize the MTFs
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(x1d, signal)
plt.title("Gaussian Peaks")
plt.xlabel("Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.subplot(1,2,2)
plt.imshow(mtf_2d, cmap='hot')
plt.title("Circular Symmetric mtf")
plt.colorbar()
plt.savefig(os.path.join(imgs_output_dir, 'rotational_sym_mtf.png'), dpi=300, bbox_inches='tight')
plt.close()

from OpticsUtils import mtf_to_psf_tf, psf_to_mtf_tf,GS_mtf2psf,GD_mtf2psf

mtf_2d = tf.convert_to_tensor(mtf_2d, dtype=tf.float64)
psf_2d = mtf_to_psf_tf(mtf_2d)
mtf_2d_p = psf_to_mtf_tf(psf_2d)
# Visulize the MTFs and PSFs couple
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(tf.abs(psf_2d), cmap='Blues')
plt.title("PSF")
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(tf.abs(mtf_2d), cmap='hot')
plt.title("MTF")
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(tf.abs(mtf_2d_p), cmap='hot')
plt.title("MTF2")
plt.colorbar()
plt.savefig(os.path.join(imgs_output_dir, 'mtf_and_psf.png'), dpi=300, bbox_inches='tight')
plt.close()

# Convert your input MTF to float32
mtf_abs = tf.cast(mtf_2d, dtype=tf.float32)

# Recover using Gerchberg–Saxton
psf_gs, mtf_gs = GS_mtf2psf(mtf_abs, num_iters=2000, tol=1e-7,verbose=True)
psf_gs_abs = tf.abs(psf_gs)
mtf_gs_abs = tf.abs(mtf_gs)
mtf_gs_phase = tf.math.angle(mtf_gs)

# Recover using Gradient Descent
psf_gd, mtf_gd = GD_mtf2psf(mtf_abs, num_iters=2000, lr=0.05, tol=1e-15, verbose=True)
psf_gd_abs = tf.abs(psf_gd)
mtf_gd_abs = tf.abs(mtf_gd)
mtf_gd_phase = tf.math.angle(mtf_gd)

# Visualize for conparison
center_row = mtf_abs.shape[0] // 2  # middle row for 1D slice
# Plot comparison with 5 columns
plt.figure(figsize=(25, 10))
# --- GS Method ---
plt.subplot(2, 5, 1)
plt.imshow(psf_gs_abs.numpy(), cmap='viridis')
plt.title("GS PSF (|PSF|)")
plt.colorbar()
plt.subplot(2, 5, 2)
plt.imshow(mtf_abs.numpy(), cmap='hot')
plt.title("Input MTF abs")
plt.colorbar()
plt.subplot(2, 5, 3)
plt.imshow(mtf_gs_abs.numpy(), cmap='hot')
plt.title("GS Recovered MTF abs")
plt.colorbar()
plt.subplot(2, 5, 4)
plt.imshow(mtf_gs_phase.numpy(), cmap='twilight')
plt.title("GS MTF Phase")
plt.colorbar()
plt.subplot(2, 5, 5)
plt.plot(mtf_abs[center_row].numpy(), label="Input", linestyle='--')
plt.plot(mtf_gs_abs[center_row].numpy(), label="GS Recovered", alpha=0.5)
plt.title("GS MTF abs (Slice)")
plt.legend()
# --- GD Method ---
plt.subplot(2, 5, 6)
plt.imshow(psf_gd_abs.numpy(), cmap='viridis')
plt.title("GD PSF (|PSF|)")
plt.colorbar()
plt.subplot(2, 5, 7)
plt.imshow(mtf_abs.numpy(), cmap='hot')
plt.title("Input MTF abs")
plt.colorbar()
plt.subplot(2, 5, 8)
plt.imshow(mtf_gd_abs.numpy(), cmap='hot')
plt.title("GD Recovered MTF abs")
plt.colorbar()
plt.subplot(2, 5, 9)
plt.imshow(mtf_gd_phase.numpy(), cmap='twilight')
plt.title("GD MTF Phase")
plt.colorbar()
plt.subplot(2, 5, 10)
plt.plot(mtf_abs[center_row].numpy(), label="Input", linestyle='--')
plt.plot(mtf_gd_abs[center_row].numpy(), label="GD Recovered", alpha=0.5)
plt.title("GD MTF abs (Slice)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(imgs_output_dir, 'MTF_to_PSF_gs_vs_gd.png'), dpi=300, bbox_inches='tight')
plt.close()

# The kernels below are what we want to inverse design with PSF engineering.
np.random.seed(10)
kernel = np.random.uniform(low=0,high=1.0,size=(3,mtf_size,mtf_size))
kernel[0] = np.zeros_like(kernel[0])
kernel[1] = psf_gs_abs.numpy().copy()
kernel[2] = np.zeros_like(kernel[2])

# We now only design green PSF so needs to set the corresponding value to 1.0 while others 0.0
design_RGB_kernels(kernel, imgs_output_dir, kernel_weight_RGB=[0.0, 1.0, 0.0])

