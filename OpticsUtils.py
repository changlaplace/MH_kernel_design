import numpy as np
import tensorflow as tf
from MathUtils import *
# use angular spectrum propagation to check
# Propagate a constant light field though the meta-optic
def asprop(e, z, λ, d):
    """
    Angular Spectrum method of propagation on unshifted complex field (unitless)
    INPUTS:
        e: [complexe array] amplitude field; e.g.: E*exp(j*phi)
        z: distance to propagation plane; e.g.: 1e3
        λ: [float] wavelength > 0
        d: [tuple] size of meta-cell (nx>0, ny>0); size of pixels
    OUTPUT:
        stack of complex field in propagation plane(s)
    """
    # invert field phase for back-propagation
    if np.sign(z) < 0:
        e = np.conjugate(e)

    # compute angular spectrum
    E = np.fft.fft2(e)

    # extract grid parameters
    nx, ny = e.shape
    dx, dy = d

    # get k-grid (spatial frequency); real mode:propagating; complex mode: evanescent
    u = np.fft.fftfreq(n=nx, d=dx) # compute the spatial frequencies directly, saves time and power
    v = np.fft.fftfreq(n=ny, d=dy)
    V, U = np.meshgrid(v, u)
    W = np.sqrt(0j + 1/λ ** 2 - U ** 2 - V ** 2).real # the pure imaginary values are evanescent waves which are not to be taken into account into the propagation; so by adding the .real, you convert pure imaginary to 0 values

    # calculate diffraction plane angular spectrum
    Ez = E * np.exp(1j * 2 * np.pi * W * np.abs(z))

    # retrieve diffraction-plane real-space field
#     ez = np.fft.ifft2(np.fft.ifftshift(Ez))
    ez = np.fft.ifft2(Ez)

    return ez

def discretize_phase_mask(phase_mask_wrapped, num_of_bins):
    """
    input:
            an 2D numpy wrapped phase mask in radians
            desired number of bins
    return:
            2D numpy discretized phase array
    """
    phase_mask = phase_mask_wrapped
    bins = np.linspace(0, 1, num_of_bins + 1)
    index_array = np.digitize(phase_mask, bins, right=True)
    discretized_phase_array = index_array/num_of_bins
    return discretized_phase_array

from tqdm import tqdm

def generate_radius_array(r2p_lookup_table, phase_mask):
    """
    takes in the radius (μm) to phase lookup table, and a 2D phase mask (wrapped in units of 2Pi),
    return: a 2D (numpy) radius array (cartesian).

    road map:
        unique_phase ->
        iterate over each unique phase ->
        find the closest radius on the r2p_lookup_table ->
        return radius array
    """
    radius_table, phase_table = r2p_lookup_table
    radius_array = np.zeros(np.shape(phase_mask))
    for n in tqdm(range(0,len(phase_mask))): #tdqm just adds a progress bar; not essential to running the code
        for m in range(0,len(phase_mask[1])):
            distance = phase_mask[n,m] - phase_table
            phase_distance = np.sin(distance/2)**2
            smallest_difference_index = phase_distance.argmin()
            closest_radius = radius_table[smallest_difference_index]
            radius_array[n,m] = closest_radius
    return radius_array


def generate_radius_array_TF(r2p_lookup_table, phase_mask):
    """
    takes in the radius (μm) to phase lookup table, and a 2D phase mask (wrapped in units of 2Pi),
    return: a 2D (tensorflow) radius array (cartesian).

    road map:
        unique_phase ->
        iterate over each unique phase ->
        find the closest radius on the r2p_lookup_table ->
        return radius array
    """
    radius_table, phase_table = r2p_lookup_table

    radius_array = tf.zeros(tf.shape(phase_mask), dtype=tf.float32)

    for n in tqdm(range(0, tf.shape(phase_mask)[0])):
        for m in range(0, tf.shape(phase_mask)[1]):
            distance = phase_mask[n, m] - phase_table
            phase_distance = tf.math.square(tf.math.sin(distance / 2.0))
            smallest_difference_index = tf.argmin(phase_distance, axis=0)
            closest_radius = tf.gather(radius_table, smallest_difference_index)
            radius_array = tf.tensor_scatter_nd_update(radius_array, [(n, m)], [closest_radius])

    return radius_array

def generate_radius_array_TF_optimized(r2p_lookup_table, phase_mask):
    """
    Vectorized TensorFlow implementation for computing radius array from a wrapped phase mask.
    """
    radius_table, phase_table = r2p_lookup_table
    radius_table = tf.convert_to_tensor(radius_table, dtype=tf.float32)
    phase_table = tf.convert_to_tensor(phase_table, dtype=tf.float32)

    phase_mask_flat = tf.reshape(phase_mask, [-1, 1])  # (N, 1)
    phase_diff = phase_mask_flat - phase_table  # (N, T)
    phase_dist = tf.math.square(tf.math.sin(phase_diff / 2.0))  # (N, T)
    indices = tf.argmin(phase_dist, axis=1, output_type=tf.int32)  # (N,)
    radius_vals = tf.gather(radius_table, indices)  # (N,)
    radius_array = tf.reshape(radius_vals, tf.shape(phase_mask))  # (H, W)
    return radius_array

def generate_phase_array(r2p_lookup_table1,r2p_lookup_table2, r2p_lookup_table3, radius_mask):
    """
    takes in the radius (μm) to phase lookup table, and a 2D phase mask (wrapped in units of 2Pi),
    return: a 2D (numpy) radius array (cartesian).

    road map:
        unique_phase ->
        iterate over each unique phase ->
        find the closest radius on the r2p_lookup_table ->
        return radius array
    """
    radius_table, phase_table1 = r2p_lookup_table1
    phase_array1 = np.zeros(np.shape(radius_mask))
    radius_table, phase_table2 = r2p_lookup_table2
    phase_array2 = np.zeros(np.shape(radius_mask))
    radius_table, phase_table3 = r2p_lookup_table3
    phase_array3 = np.zeros(np.shape(radius_mask))
    for n in tqdm(range(0,len(radius_mask[:]))): #tdqm just adds a progress bar; not essential to running the code
        for m in range(0,len(radius_mask[1,:])):
            distance = radius_mask[n,m] - radius_table
            radius_distance = np.sin(distance/2)**2
            smallest_difference_index = radius_distance.argmin()
            closest_phase1 = phase_table1[smallest_difference_index]
            closest_phase2 = phase_table2[smallest_difference_index]
            closest_phase3 = phase_table3[smallest_difference_index]
            phase_array1[n,m] = closest_phase1
            phase_array2[n,m] = closest_phase2
            phase_array3[n,m] = closest_phase3
    return phase_array1, phase_array2, phase_array3

# define the propagation functions
def _pad(field, pad_factor=1.):
    n_x, n_y = np.shape(field)
    pad_x = int(n_x * pad_factor / 2)
    pad_y = int(n_y * pad_factor / 2)
    return tf.pad(field, paddings=tf.constant([[pad_x, pad_x], [pad_y, pad_y]]))

def _unpad(field, pad_factor=1.):
    if pad_factor == 0.:
        return field
    else:
        n_x, n_y = np.shape(field)
        pad_x = int(n_x * pad_factor / (2 + 2 * pad_factor))
        pad_y = int(n_y * pad_factor / (2 + 2 * pad_factor))
        return field[pad_x:-pad_x, pad_y:-pad_y]

def get_frequencies(nx, ny, dx, dy, float_type):
    k_x = tf.constant(np.fft.fftfreq(n=nx, d=dx) * 2 * np.pi, dtype=float_type)
    k_y = tf.constant(np.fft.fftfreq(n=ny, d=dy) * 2 * np.pi, dtype=float_type)
    k_Y, k_X = tf.meshgrid(k_y, k_x, indexing='xy')
    return k_X, k_Y

def propagate_padded(propagator, field, k, z_list, dx, dy, pad_factor=1.):
    padded_field = _pad(field, pad_factor)
    padded_propagated_field = propagator(padded_field, k, z_list, dx, dy,)
    propagated_field = _unpad(padded_propagated_field, pad_factor=pad_factor)
    return propagated_field

def propagate_angular_bw_limited(field, k, z_list, dx, dy,):
    """Uses an angular propagation method to propagates a field to multiple planes

     Parameters:
         field (2D array): Complex field array to be propagated.
         k (complex): The wavenumber of the propagation medium. Can be calculated using 2*pi*n/wavelength
         z_list (1D array): Distances that the field should be propagated
         dx (float): physical size of each pixel in the x direction
         dy (float): physical size of each pixel in the y direction

     Returns:
         E_prop (3D array complex): Array of the propagated fields.
             The 1st, 2nd and 3rd dimensions correspond to the x, y, and z dimensions.
             The third dimension is the same as the length of z_list
     """

    if field.dtype == tf.complex128:
        float_type = tf.float64 #float64 previous
    else:
        float_type = tf.float32
    # pad
    pad_factor = 1.0
    n_x, n_y = field.shape
    pad_x = int(n_x * pad_factor / 2)
    pad_y = int(n_y * pad_factor / 2)
    field =  tf.pad(field, paddings=tf.constant([[pad_x, pad_x], [pad_y, pad_y]]))
    z_list = tf.constant(z_list, dtype=float_type)

    # shape of the input field

    n_x, n_y = field.shape
    k_X, k_Y = get_frequencies(n_x, n_y, dx, dy, float_type)

    # define wavenumber for each wavevector in the direction of propagation
    k_tensor = k * tf.ones(dtype=float_type, shape=field.shape)
    kz_squared = tf.cast(k_tensor ** 2 - k_X ** 2 - k_Y ** 2, dtype=field.dtype)
    k_z = tf.sqrt(kz_squared)

    U = tf.signal.fft2d(field)

    # broadcast k_z into 3rd dimension
    k_Z = tf.broadcast_to(k_z, shape=(len(z_list), *k_z.shape))

    d_Z = tf.broadcast_to(
        tf.expand_dims(
            tf.expand_dims(
                z_list,
                axis=-1
            ),
            axis=-1
        ),
        k_Z.shape
    )

    phase_real = tf.math.real(k_Z) * d_Z
    phase_imag = tf.math.imag(k_Z) * d_Z

    H_real = tf.cos(phase_real) * tf.exp(-phase_imag)
    H_imag = tf.sin(phase_real) * tf.exp(-phase_imag)

    # Start: apply antialias filter to H
    # See paper 'Band-Limited Angular Spectrum Method for Numerical SImulation of Free-Space Propagation in Far and near fields'
    del_f_x = 1. / (2. * n_x * dx)
    del_f_y = 1. / (2. * n_y * dy)

    k_x_limit = k / tf.sqrt((2 * del_f_x * z_list) ** 2 + 1)
    k_y_limit = k / tf.sqrt((2 * del_f_y * z_list) ** 2 + 1)

    k_x_limit = tf.cast(k_x_limit, dtype=float_type)
    k_y_limit = tf.cast(k_y_limit, dtype=float_type)

    k_X_limit = tf.transpose(
        tf.broadcast_to(k_x_limit, shape=(*k_z.shape, len(z_list),)),
        perm=[2, 0, 1],
    )
    k_Y_limit = tf.transpose(
        tf.broadcast_to(k_y_limit, shape=(*k_z.shape, len(z_list),)),
        perm=[2, 0, 1],
    )

    k_XX = tf.broadcast_to(k_X, shape=(len(z_list), *k_X.shape))
    k_YY = tf.broadcast_to(k_Y, shape=(len(z_list), *k_Y.shape))

    kx_mask = (k_XX / k_X_limit) ** 2 + (k_YY / k) ** 2 <= 1.
    ky_mask = (k_YY / k_Y_limit) ** 2 + (k_XX / k) ** 2 <= 1.
    comb_mask = tf.logical_and(kx_mask, ky_mask)
    filter = tf.cast(comb_mask, dtype=float_type)

    H_real, H_imag = complex_mul(H_real, H_imag, filter, 0.)
    # Finish: apply antialias filter to H

    E_k_prop_real, E_k_prop_imag = complex_mul(
        *split_complex(U),
        H_real,
        H_imag,
    )
    # import matplotlib.pyplot as plt; plt.imshow(tf.math.abs(tf.complex(E_k_prop_real, E_k_prop_imag))[0, :, :]); plt.show()
    result = tf.signal.ifft2d(tf.complex(E_k_prop_real, E_k_prop_imag))[0,:,:]

    # unpad

    n_x, n_y = result.shape
    pad_x = int(n_x * pad_factor / (2 + 2 * pad_factor))
    pad_y = int(n_y * pad_factor / (2 + 2 * pad_factor))
    result = result[pad_x:-pad_x, pad_y:-pad_y]

    return result



# A class representing a optics field in a plane
class OneDPhasorField(tf.Module):
    '''
    Class representing a optics field in a plane, with a tf non-trainable variable representing the field values.
    '''
    def __init__(self, array_size, dtype):
        #assert isinstance(pixel_width, int)
        #assert isinstance(pixel_height, int)
        #assert isinstance(pixel_spacing, int)
        assert isinstance(array_size,int)
        assert dtype == tf.complex64 or dtype == tf.complex128, "Dtype must be a complex datatype"

        self.dtype = dtype
        self.field = tf.Variable(tf.ones(shape=(array_size,array_size), dtype=dtype), trainable=False)  # Variable supports slice assign

    @tf.function
    def __call__(self):
        return self.field
    

def psf_to_mtf_tf(psf: tf.Tensor) -> tf.Tensor:
    """
    Convert center-aligned PSF to center-aligned MTF (complex).
    """
    psf = tf.cast(psf, tf.complex64)
    
    # Step 1: shift center to corner (align with FFT definition)
    psf_shifted = tf.signal.ifftshift(psf, axes=(0, 1))
    
    # Step 2: fft2d
    otf = tf.signal.fft2d(psf_shifted)
    
    # Step 3: shift back to center-aligned frequency domain
    mtf = tf.signal.fftshift(otf, axes=(0, 1))
    
    return mtf


def mtf_to_psf_tf(mtf: tf.Tensor) -> tf.Tensor:
    """
    Convert center-aligned MTF (complex) to center-aligned PSF (complex).
    """
    mtf = tf.cast(mtf, tf.complex64)
    
    # Step 1: shift center to corner
    otf = tf.signal.ifftshift(mtf, axes=(0, 1))
    
    # Step 2: ifft2d
    psf_shifted = tf.signal.ifft2d(otf)
    
    # Step 3: shift back to center-aligned spatial domain
    psf = tf.signal.fftshift(psf_shifted, axes=(0, 1))
    
    return psf


def GS_mtf2psf(mtf_abs: tf.Tensor, num_iters: int = 2000, tol: float = 1e-12, verbose=True):
    """
    Gerchberg-Saxton algorithm to estimate PSF and complex MTF from a center-aligned MTF magnitude.

    Args:
        mtf_abs: tf.Tensor [H, W], float32 or float64, center-aligned MTF magnitude
        num_iters: int, max iterations
        tol: float, convergence threshold (mean abs difference between MTFs)
        verbose: bool, whether to print convergence log

    Returns:
        psf: tf.Tensor [H, W], complex64, center-aligned
        mtf: tf.Tensor [H, W], complex64, center-aligned
    """
    mtf_abs = tf.convert_to_tensor(mtf_abs, dtype=tf.float32)
    H, W = mtf_abs.shape

    # Step 1: use near-zero phase as initial (more regular)
    initial_phase = tf.random.uniform((H, W), minval=-0.0, maxval=0.0)  # smoother start
    j = tf.complex(0.0, 1.0)
    mtf = tf.cast(mtf_abs, tf.complex64) * tf.exp(j * tf.cast(initial_phase, tf.complex64))

    for i in range(num_iters):
        mtf_prev = mtf

        # Step 2: spatial domain estimate (PSF)
        psf = mtf_to_psf_tf(mtf)

        # Step 3: apply spatial constraint (real, non-negative)
        psf_real = tf.nn.relu(tf.math.real(psf))

        # Step 4: frequency domain transform (new MTF)
        mtf = psf_to_mtf_tf(psf_real)

        # Step 5: keep phase, enforce given amplitude
        mtf = tf.cast(mtf_abs, tf.complex64) * tf.exp(j * tf.cast(tf.math.angle(mtf), tf.complex64))

        # Step 6: check convergence
        diff = tf.reduce_mean(tf.abs(mtf - mtf_prev))
        if verbose:
            print(f"Iter {i+1:02d} | MTF diff: {diff.numpy():.6f}")

        if diff < tol:
            if verbose:
                print(f"✅ Converged at iteration {i+1} with diff {diff.numpy():.6e}")
            break

    psf_final = mtf_to_psf_tf(mtf)
    psf_final = tf.math.real(psf_final)
    mtf_final = psf_to_mtf_tf(psf_final)
    return psf_final, mtf_final

def GD_mtf2psf(
    mtf_abs: tf.Tensor,
    num_iters: int = 2000,
    lr: float = 0.1,
    tol: float = 1e-14,
    verbose: bool = True
):
    """
    Gradient descent method to recover non-negative PSF from given absolute MTF.

    Args:
        mtf_abs: tf.Tensor [H, W], float32, center-aligned MTF magnitude
        num_iters: int, max optimization steps
        lr: float, learning rate
        tol: float, convergence threshold on MTF magnitude error
        verbose: bool, print loss per 50 iterations

    Returns:
        psf: tf.Tensor [H, W], float32, non-negative, center-aligned PSF
        mtf: tf.Tensor [H, W], complex64, corresponding complex MTF
    """
    mtf_abs = tf.convert_to_tensor(mtf_abs, dtype=tf.float32)
    H, W = mtf_abs.shape

    # Initialize trainable PSF variable (unconstrained at first)
    psf_real = tf.Variable(tf.random.normal((H, W), mean=1, stddev=0.1), dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for i in range(num_iters):
        with tf.GradientTape() as tape:
            # Apply soft non-negativity
  
            mtf = psf_to_mtf_tf(psf_real)
            mtf_mag = tf.abs(mtf)

            # L2 loss on magnitude
            loss = tf.reduce_mean((mtf_mag - mtf_abs) ** 2)

        grads = tape.gradient(loss, [psf_real])
        optimizer.apply_gradients(zip(grads, [psf_real]))


        if verbose and (i % 1 == 0 or i == num_iters - 1):
            print(f"Iter {i:04d} | Loss: {loss.numpy():.6e}")

        if loss < tol:
            if verbose:
                print(f"✅ Converged at iteration {i} with loss {loss.numpy():.6e}")
            break

    psf_final = tf.nn.relu(psf_real)
    mtf_final = psf_to_mtf_tf(psf_final)
    return psf_final, mtf_final
