import sys
print(sys.executable)
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from datetime import datetime

from configs import sim_args_R, sim_args_G, sim_args_B, field_args_R, field_args_G, field_args_B, dtype
from OpticsUtils import asprop
from OpticsUtils import OneDPhasorField
from MathUtils import split_complex, complex_mul, complex_initializer_random
from OpticsUtils import discretize_phase_mask
from OpticsUtils import generate_radius_array_TF_optimized
from OpticsUtils import generate_phase_array
from OpticsUtils import propagate_angular_bw_limited


def design_RGB_kernels(kernel :np.ndarray, imgs_output_dir, kernel_weight_RGB=[1.0, 1.0, 1.0], scalefactor = 2):
    '''
        Design RGB meta-optics kernels
        Args:
            kernel: [3,H,W], RGB color, H and W equal kernel size
            imgs_output_dir: Directory where output designed parameters and images
            scalefactor: scalefactor
    '''
    assert np.all(kernel >= -1e-6), "Assertion failed: kernel contains negative values."
    
    kernel_neg = kernel.copy()
    # Expand and then pad to define the desired psf
    scale = scalefactor;    # Scale factor setting
    numpx = scale * 10 # number of simulation pixels for each kernel pixel / previously 20 for MNIST
    kernel_neg_expanded_R = np.zeros((numpx*len(kernel[0,:]),numpx*len(kernel[0,0,:])))

    kernel_neg_expanded_G = np.zeros((numpx*len(kernel[0,:]),numpx*len(kernel[0,0,:])))

    kernel_neg_expanded_B = np.zeros((numpx*len(kernel[0,:]),numpx*len(kernel[0,0,:])))

    for i in range(0,len(kernel[0,:])):
        for j in range(0,len(kernel[0,0,:])):
            kernel_neg_expanded_R[i*numpx:(i+1)*numpx,j*numpx:(j+1)*numpx] = kernel_neg[0,i,j]

            kernel_neg_expanded_G[i*numpx:(i+1)*numpx,j*numpx:(j+1)*numpx] = kernel_neg[1,i,j]

            kernel_neg_expanded_B[i*numpx:(i+1)*numpx,j*numpx:(j+1)*numpx] = kernel_neg[2,i,j]
    
    # pad to 1600 by 1600 array, originally numpad=800-scale*35
    numpad = int(sim_args_R['numpx']//2)-scale*5*kernel.shape[-1] #35 came from 7 * 10 /2
    kernel_neg_expanded_R = (np.pad(kernel_neg_expanded_R,(numpad,numpad),'constant',constant_values=(0,0)))

    kernel_neg_expanded_B = (np.pad(kernel_neg_expanded_B,(numpad,numpad),'constant',constant_values=(0,0)))
   
    kernel_neg_expanded_G = (np.pad(kernel_neg_expanded_G,(numpad,numpad),'constant',constant_values=(0,0)))

    plt.subplot(1,3,1)
    plt.imshow(kernel_neg_expanded_R,cmap = 'Reds')
    plt.title("Red")
    plt.subplot(1,3,2)
    plt.imshow(kernel_neg_expanded_G,cmap = 'Greens')
    plt.title("Green")
    plt.subplot(1,3,3)
    plt.imshow(kernel_neg_expanded_B,cmap = 'Blues')
    plt.title("Blue")
    plt.savefig(os.path.join(imgs_output_dir, 'kernel_expanded_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()


    # Use Angular Spectrum Propagation but with negative z value to get the initial guess of the phase profile
    plt.subplot(1,3,1)
    backprop_R = asprop(kernel_neg_expanded_R,-sim_args_R['spacing_ms1_to_detector'],sim_args_R['wavelength'],(sim_args_R['dd'],sim_args_R['dd']))
    backphase_R = np.angle(np.real(backprop_R) + 1j * np.imag(backprop_R))
    #backphase_R = tf.cast(backphase_R, dtype=tf.float32)
    plt.imshow(backphase_R) # extent[xmin,xmax,ymin,ymax]
    plt.title("Red")
    plt.subplot(1,3,2)
    backprop_G = asprop(kernel_neg_expanded_G,-sim_args_G['spacing_ms1_to_detector'],sim_args_G['wavelength'],(sim_args_G['dd'],sim_args_G['dd']))
    backphase_G = np.angle(np.real(backprop_G) + 1j * np.imag(backprop_G))
    #backphase_G = tf.cast(backphase_G, dtype=tf.float32)
    plt.imshow(backphase_G) # extent[xmin,xmax,ymin,ymax]
    plt.title("Green")
    plt.subplot(1,3,3)
    backprop_B = asprop(kernel_neg_expanded_B,-sim_args_B['spacing_ms1_to_detector'],sim_args_B['wavelength'],(sim_args_B['dd'],sim_args_B['dd']))
    backphase_B = np.angle(np.real(backprop_B) + 1j * np.imag(backprop_B))
    #backphase_B = tf.cast(backphase_B, dtype=tf.float32)
    plt.imshow(backphase_B) # extent[xmin,xmax,ymin,ymax]
    plt.title("Blue")
    plt.savefig(os.path.join(imgs_output_dir, 'backprop_phase.png'), dpi=300, bbox_inches='tight')
    plt.close()


    # Create the initial fields with OneDPhasorField class. They are all initialized to tf.ones() matrixes.
    field_generator_R = OneDPhasorField(**field_args_R)
    field_generator_G = OneDPhasorField(**field_args_G)
    field_generator_B = OneDPhasorField(**field_args_B)
    # Plot the initial fields
    field_set_R = field_generator_R()
    field_set_G = field_generator_G()
    field_set_B = field_generator_B()
    plot_field_set_R = tf.abs(field_set_R).numpy()
    plot_field_set_G = tf.abs(field_set_G).numpy()
    plot_field_set_B = tf.abs(field_set_B).numpy()



    # Set the phase to optimize and the desired PSFs
    metasurface1_phase_R = tf.Variable(backphase_R,trainable=False,dtype=dtype['real'])
    metasurface1_phase_G = tf.Variable(backphase_G,trainable=False,dtype=dtype['real'])
    metasurface1_phase_B = tf.Variable(backphase_B,trainable=False,dtype=dtype['real'])
    desired_psf_R = tf.convert_to_tensor(kernel_neg_expanded_R, dtype = tf.float32)
    desired_psf_G = tf.convert_to_tensor(kernel_neg_expanded_G, dtype = tf.float32)
    desired_psf_B = tf.convert_to_tensor(kernel_neg_expanded_B, dtype = tf.float32)
    desired_psf = (desired_psf_R,desired_psf_G,desired_psf_B)

    # Plot the RCWA phase-cycle curves
    rcwa_R = np.loadtxt('./lookuptbs/SiN0.75_SiO2_p0.293_lam0.635um.txt')
    data_hf_list_R=rcwa_R[:,0]
    data_phase_R=rcwa_R[:,1]
    data_amp_R = rcwa_R[:,2]
    rcwa_G = np.loadtxt('./lookuptbs/SiN0.75_SiO2_p0.293_lam0.532um.txt')
    data_hf_list_G=rcwa_G[:,0]
    data_phase_G=rcwa_G[:,1]
    data_amp_G = rcwa_G[:,2]
    rcwa_B = np.loadtxt('./lookuptbs/SiN0.75_SiO2_p0.293_lam0.45um.txt')
    data_hf_list_B=rcwa_B[:,0]
    data_phase_B=rcwa_B[:,1]
    data_amp_B = rcwa_B[:,2]

    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.plot(data_hf_list_R,data_amp_R,'blue')
    plt.plot(data_hf_list_R,data_phase_R,'red')
    plt.xlabel("Feature Size (microns)")
    plt.ylabel("Transmission / Phase (units of 2Pi)")
    plt.title("Red")
    plt.subplot(1,3,2)
    plt.plot(data_hf_list_G,data_amp_G,'blue')
    plt.plot(data_hf_list_G,data_phase_G,'red')
    plt.xlabel("Feature Size (microns)")
    plt.ylabel("Transmission / Phase (units of 2Pi)")
    plt.title("Green")
    plt.subplot(1,3,3)
    plt.plot(data_hf_list_B,data_amp_B,'blue')
    plt.plot(data_hf_list_B,data_phase_B,'red')
    plt.xlabel("Feature Size (microns)")
    plt.ylabel("Transmission / Phase (units of 2Pi)")
    plt.title("Blue")
    plt.savefig(os.path.join(imgs_output_dir, 'rcwa_phase_cycle_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Discretize the Phase (from Huang's code)

    data_phase_discrete_R = discretize_phase_mask(data_phase_R,10)
    data_phase_discrete_G = discretize_phase_mask(data_phase_G,10)
    data_phase_discrete_B = discretize_phase_mask(data_phase_B,10)
    select = np.array([0])
    start = [int(1.6*22)]
    stop = [int(1.6*80)]
    for i in range(0,len(start)):
        select = np.concatenate((select,np.arange(start[i],stop[i]+1,1)))
    data_phase_discrete2_R = data_phase_discrete_R[select]
    data_hf_list2_R = data_hf_list_R[select]
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.plot(data_hf_list_R,data_amp_R,'blue')
    plt.plot(data_hf_list_R,data_phase_R,'red')
    plt.scatter(data_hf_list2_R,data_phase_discrete2_R,color='r')
    plt.xlabel("Feature Size (um)")
    plt.ylabel("Transmission / Phase (units of 2Pi)")
    plt.legend(["Transmission","Phase (units of 2Pi)","Discretized Phase"])
    plt.title("Red")
    data_phase_discrete2_G = data_phase_discrete_G[select]
    data_hf_list2_G = data_hf_list_G[select]
    plt.subplot(1,3,2)
    plt.plot(data_hf_list_G,data_amp_G,'blue')
    plt.plot(data_hf_list_G,data_phase_G,'red')
    plt.scatter(data_hf_list2_G,data_phase_discrete2_G,color='r')
    plt.xlabel("Feature Size (um)")
    plt.ylabel("Transmission / Phase (units of 2Pi)")
    plt.legend(["Transmission","Phase (units of 2Pi)","Discretized Phase"])
    plt.title("Green")
    data_phase_discrete2_B = data_phase_discrete_B[select]
    data_hf_list2_B = data_hf_list_B[select]
    plt.subplot(1,3,3)
    plt.plot(data_hf_list_B,data_amp_B,'blue')
    plt.plot(data_hf_list_B,data_phase_B,'red')
    plt.scatter(data_hf_list2_B,data_phase_discrete2_B,color='r')
    plt.xlabel("Feature Size (um)")
    plt.ylabel("Transmission / Phase (units of 2Pi)")
    plt.legend(["Transmission","Phase (units of 2Pi)","Discretized Phase"])
    plt.title("Blue")
    plt.savefig(os.path.join(imgs_output_dir, 'rcwa_phase_cycle_curves_discretized.png'), dpi=300, bbox_inches='tight')
    plt.close()


    # Generate the radius array from the backprop phase and the lookup table, using the Red channel as reference.

    phase_mask_wrapped_R = ((metasurface1_phase_R)%(2*np.pi))/(2*np.pi)
    lookup_table_R = tf.convert_to_tensor([data_hf_list2_R, data_phase_R[select]],dtype=tf.float32)
    lookup_table_G = tf.convert_to_tensor([data_hf_list2_G, data_phase_G[select]],dtype=tf.float32)
    lookup_table_B = tf.convert_to_tensor([data_hf_list2_B, data_phase_B[select]],dtype=tf.float32)
    radius_array_R = generate_radius_array_TF_optimized(lookup_table_R,phase_mask_wrapped_R) # This generates the array of pillar sizes
    # Make the radius array trainable
    radius_array_R = tf.Variable(radius_array_R,trainable=True,dtype=tf.float32)


    pitch = sim_args_R['dd']*1e6/2 #micron
    # Here defines a function to generate the phase with a fitted exponential function
    def generate_phase_array_TF(radius_mask):
        radius_mask = radius_mask % pitch
        phase_array_R = tf.zeros_like(radius_mask, dtype=tf.float32)
        phase_array_G = tf.zeros_like(radius_mask, dtype=tf.float32)
        phase_array_B = tf.zeros_like(radius_mask, dtype=tf.float32)

        phase_array_R = -5.03593*(tf.exp((radius_mask*radius_mask-2*0.0122*radius_mask)/(-0.28081))-1)
        phase_array_G = -2.28293*(tf.exp((radius_mask*radius_mask-2*0.01878*radius_mask)/(-0.07134))-1)
        phase_array_B = -1.87336*(tf.exp((radius_mask*radius_mask-2*0.0237*radius_mask)/(-0.03191))-1)
        return phase_array_R, phase_array_G, phase_array_B

    radius = np.linspace(0, pitch, 100)
    phase_R, phase_G, phase_B = generate_phase_array_TF(radius)
    phase_R = phase_R % 1
    phase_G = phase_G % 1
    phase_B = phase_B % 1
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(radius, phase_R, color='red')
    plt.xlabel('Radius (μm)')
    plt.ylabel('Phase (rad)')
    plt.title(f'Radius-to-Phase Mapping: Red')
    plt.subplot(1, 3, 2)
    plt.plot(radius, phase_G, color='green')
    plt.xlabel('Radius (μm)')
    plt.ylabel('Phase (rad)')
    plt.title(f'Radius-to-Phase Mapping: Green')
    plt.subplot(1, 3, 3)
    plt.plot(radius, phase_B, color='blue')
    plt.xlabel('Radius (μm)')
    plt.ylabel('Phase (rad)')
    plt.title(f'Radius-to-Phase Mapping: Blue') 
    plt.tight_layout()
    plt.savefig(os.path.join(imgs_output_dir, 'TF_fitted_phase_gen'), dpi=300)
    plt.close()


    # We use above radius array to further generate RGB phase profiles, here wrapped means the phase is in the range of [0, 2π).
    phase_R, phase_G, phase_B = generate_phase_array_TF(radius_array_R)
    backphase_R = phase_R * 2 * tf.constant(np.pi, dtype=tf.float32)
    backphase_G = phase_G * 2 * tf.constant(np.pi, dtype=tf.float32)
    backphase_B = phase_B * 2 * tf.constant(np.pi, dtype=tf.float32)
    metasurface_phase_R = tf.cast(backphase_R, dtype=tf.float32)
    metasurface_phase_G = tf.cast(backphase_G, dtype=tf.float32)
    metasurface_phase_B = tf.cast(backphase_B, dtype=tf.float32)
    phase_mask_wrapped_R = ((metasurface_phase_R) % (2 * tf.constant(np.pi, dtype=tf.float32))) / (2 * tf.constant(np.pi, dtype=tf.float32))
    phase_mask_wrapped_G = ((metasurface_phase_G) % (2 * tf.constant(np.pi, dtype=tf.float32))) / (2 * tf.constant(np.pi, dtype=tf.float32))
    phase_mask_wrapped_B = ((metasurface_phase_B) % (2 * tf.constant(np.pi, dtype=tf.float32))) / (2 * tf.constant(np.pi, dtype=tf.float32))


    # Define the tf function mapping the radius to the output sensor plane E field
    @tf.function
    def forward(radius_array_R):
        '''
        Rather straightforward forward function, which generates the phase profile, applies it to the field, and propagates it to the detector plane.
        '''
        phase_R, phase_G, phase_B = generate_phase_array_TF(radius_array_R)
        metasurface_phase_R = tf.cast(phase_R * 2 * tf.constant(np.pi),dtype=tf.float32)
        metasurface_phase_G = tf.cast(phase_G * 2 * tf.constant(np.pi),dtype=tf.float32)
        metasurface_phase_B = tf.cast(phase_B * 2 * tf.constant(np.pi),dtype=tf.float32)
        field_set_R = field_generator_R()
        field_set_G = field_generator_G()
        field_set_B = field_generator_B()
        def prop_R(field, distance):
            return propagate_angular_bw_limited(field = field, k=sim_args_R['k'], z_list=[distance, ], dx=sim_args_R['dd'], dy=sim_args_R['dd'],)#[0, :, :]
        def prop_G(field, distance):
            return propagate_angular_bw_limited(field = field, k=sim_args_G['k'], z_list=[distance, ], dx=sim_args_G['dd'], dy=sim_args_G['dd'],)#[0, :, :]
        def prop_B(field, distance):
            return propagate_angular_bw_limited(field = field, k=sim_args_B['k'], z_list=[distance, ], dx=sim_args_B['dd'], dy=sim_args_B['dd'],)#[0, :, :]
        def prop_1d_to_ms1_R(field):
            return prop_R(field,sim_args_R['spacing_start_to_ms1'])
        def prop_1d_to_ms1_G(field):
            return prop_G(field,sim_args_G['spacing_start_to_ms1'])
        def prop_1d_to_ms1_B(field):
            return prop_B(field,sim_args_B['spacing_start_to_ms1'])
        def prop_ms1_to_detector_R(field):
            return prop_R(field,sim_args_R['spacing_ms1_to_detector'])
        def prop_ms1_to_detector_G(field):
            return prop_G(field,sim_args_G['spacing_ms1_to_detector'])
        def prop_ms1_to_detector_B(field):
            return prop_B(field,sim_args_B['spacing_ms1_to_detector'])
        def meta1_R(A):
            field_after = tf.complex(
                *complex_mul(
                    *split_complex(A),
                    tf.math.cos(metasurface_phase_R), tf.math.sin(metasurface_phase_R),
                )
            )
            return field_after
        def meta1_G(A):
            field_after = tf.complex(
                *complex_mul(
                    *split_complex(A),
                    tf.math.cos(metasurface_phase_G), tf.math.sin(metasurface_phase_G),
                )
            )
            return field_after
        def meta1_B(A):
            field_after = tf.complex(
                *complex_mul(
                    *split_complex(A),
                    tf.math.cos(metasurface_phase_B), tf.math.sin(metasurface_phase_B),
                )
            )
            return field_after
        field_set_R = meta1_R(field_set_R)
        field_set_G = meta1_G(field_set_G)
        field_set_B = meta1_B(field_set_B)
        field_set_R = prop_ms1_to_detector_R(field_set_R)
        field_set_G = prop_ms1_to_detector_G(field_set_G)
        field_set_B = prop_ms1_to_detector_B(field_set_B)
        return field_set_R, field_set_G, field_set_B


    propagated_fields_R, propagated_fields_G, propagated_fields_B = forward(radius_array_R)
    propagated_intensities_R = tf.cast(tf.math.abs(propagated_fields_R) ** 2 ,dtype=tf.float32)
    propagated_intensities_G = tf.cast(tf.math.abs(propagated_fields_G) ** 2 ,dtype=tf.float32)
    propagated_intensities_B = tf.cast(tf.math.abs(propagated_fields_B) ** 2 ,dtype=tf.float32)
    normval_R = tf.math.reduce_max(propagated_intensities_R)
    normval_G = tf.math.reduce_max(propagated_intensities_G)
    normval_B = tf.math.reduce_max(propagated_intensities_B)
    propagated_intensities_R = tf.math.divide(propagated_intensities_R,normval_R)
    propagated_intensities_G = tf.math.divide(propagated_intensities_G,normval_G)
    propagated_intensities_B = tf.math.divide(propagated_intensities_B,normval_B)


    # Define loss functions and training functions
    @tf.function
    def loss(radius_array_R,desired_psf_R,desired_psf_G,desired_psf_B):
        """Loss function to map power from each input mode to a bin in the output field"""
        # Project values to the valid range after each optimization step
        radius_array_R = radius_array_R % pitch
        radius_array_R = tf.clip_by_value(radius_array_R, radius_array_R_min, radius_array_R_max)
        propagated_fields_R, propagated_fields_G, propagated_fields_B  = forward(radius_array_R)#, metasurface2_phase
        propagated_intensities_R = tf.cast(tf.math.abs(propagated_fields_R) ** 2 ,dtype=tf.float32)
        propagated_intensities_G = tf.cast(tf.math.abs(propagated_fields_G) ** 2 ,dtype=tf.float32)
        propagated_intensities_B = tf.cast(tf.math.abs(propagated_fields_B) ** 2 ,dtype=tf.float32)
        desired_intensities_R = tf.cast(tf.math.abs(desired_psf_R), dtype=tf.float32)
        desired_intensities_G = tf.cast(tf.math.abs(desired_psf_G), dtype=tf.float32)
        desired_intensities_B = tf.cast(tf.math.abs(desired_psf_B), dtype=tf.float32)
        propagated_intensities_R = tf.math.l2_normalize(propagated_intensities_R)
        propagated_intensities_G = tf.math.l2_normalize(propagated_intensities_G)
        propagated_intensities_B = tf.math.l2_normalize(propagated_intensities_B)
        desired_intensities_R = tf.math.l2_normalize(desired_intensities_R)
        desired_intensities_G = tf.math.l2_normalize(desired_intensities_G)
        desired_intensities_B = tf.math.l2_normalize(desired_intensities_B)
        diff_matrix_R = tf.math.subtract(propagated_intensities_R,desired_intensities_R)
        diff_matrix_G = tf.math.subtract(propagated_intensities_G,desired_intensities_G)
        diff_matrix_B = tf.math.subtract(propagated_intensities_B,desired_intensities_B)
        diff_matrix_R_squared = tf.square(diff_matrix_R)
        diff_matrix_G_squared = tf.square(diff_matrix_G)
        diff_matrix_B_squared = tf.square(diff_matrix_B)
        reduce_sum_R_squared = tf.reduce_sum(diff_matrix_R_squared)
        reduce_sum_G_squared = tf.reduce_sum(diff_matrix_G_squared)
        reduce_sum_B_squared = tf.reduce_sum(diff_matrix_B_squared)
        # Apply weight to make this function able to design mono color
        reduce_sum_squared = (reduce_sum_R_squared*reduce_sum_R_squared*kernel_weight_RGB[0] 
                              + reduce_sum_G_squared*reduce_sum_G_squared*kernel_weight_RGB[1] 
                              + reduce_sum_B_squared*reduce_sum_B_squared*kernel_weight_RGB[2])/3
        reduce_sum = tf.sqrt(reduce_sum_squared)
        return reduce_sum

    n_bins = 10
    holdsums = tf.Variable(tf.zeros(shape=(n_bins*n_bins), dtype=tf.float32), trainable=False)
    @tf.function
    def loss_binned(radius_array_R, holdsums):
        #Loss function to map power from each input mode to a bin in the output field
        # Change bins to 6x6 and shrink
        propagated_fields_R, propagated_fields_G, propagated_fields_B  = forward(radius_array_R)#, metasurface2_phase
        propagated_intensities_R = tf.cast(tf.math.abs(propagated_fields_R) ** 2 ,dtype=tf.float32)
        propagated_intensities_G = tf.cast(tf.math.abs(propagated_fields_G) ** 2 ,dtype=tf.float32)
        propagated_intensities_B = tf.cast(tf.math.abs(propagated_fields_B) ** 2 ,dtype=tf.float32)
        desired_intensities_R = tf.cast(tf.math.abs(desired_psf_R), dtype=tf.float32)
        desired_intensities_G = tf.cast(tf.math.abs(desired_psf_G), dtype=tf.float32)
        desired_intensities_B = tf.cast(tf.math.abs(desired_psf_B), dtype=tf.float32)
        # normalize
        propagated_intensities_R = tf.math.l2_normalize(propagated_intensities_R)
        propagated_intensities_G = tf.math.l2_normalize(propagated_intensities_G)
        propagated_intensities_B = tf.math.l2_normalize(propagated_intensities_B)
        desired_intensities_R = tf.math.l2_normalize(desired_intensities_R)
        desired_intensities_G = tf.math.l2_normalize(desired_intensities_G)
        desired_intensities_B = tf.math.l2_normalize(desired_intensities_B)
        diff_matrix_R = tf.math.subtract(propagated_intensities_R,desired_intensities_R)
        diff_matrix_G = tf.math.subtract(propagated_intensities_G,desired_intensities_G)
        diff_matrix_B = tf.math.subtract(propagated_intensities_B,desired_intensities_B)
        diff_matrix_R = tf.math.abs(diff_matrix_R)
        diff_matrix_G = tf.math.abs(diff_matrix_G)
        diff_matrix_B = tf.math.abs(diff_matrix_B)
        # Apply weight here to make it adaptive to mono color
        diff_matrix = tf.math.add(diff_matrix_R*kernel_weight_RGB[0],diff_matrix_G*kernel_weight_RGB[1])
        diff_matrix = tf.math.add(diff_matrix,diff_matrix_B*kernel_weight_RGB[2])
        # do the binning
        bin_size = sim_args_R['numpx'] // n_bins
        i = tf.range(n_bins * n_bins, dtype=tf.int64)
        def sums(i):
            ibin = i // n_bins
            jbin = i % n_bins
            return tf.reduce_sum(diff_matrix[ibin * bin_size:(ibin + 1) * bin_size, jbin * bin_size:(jbin + 1) * bin_size])
        new_holdsums = tf.reduce_sum(tf.map_fn(sums, i, dtype=tf.float32))
        return new_holdsums

    @tf.function
    def train_step(radius_array_R, optimizer):
        """
        Train step function to optimize the radius array.
        """
        train_vars = [radius_array_R]
        with tf.GradientTape() as tape:
            current_loss = loss(radius_array_R, desired_psf_R,desired_psf_G,desired_psf_B)
        grads = tape.gradient(target=current_loss, sources=train_vars)
        optimizer.apply_gradients(zip(grads, train_vars))
        return tf.reduce_mean(current_loss)


    # Initialize the radius array with random values within the specified range
    radius_array_R_min = 0.0  # minimum value
    radius_array_R_max = 0.240  # maximum value
    initial_radius_array_R = np.random.uniform(radius_array_R_min, radius_array_R_max, size=np.shape(radius_array_R))
    radius_array_R = tf.Variable(initial_radius_array_R, trainable=True, dtype=tf.float32)

    # Main Training loop
    iterations = 300
    n_update = 5
    t = time.time()
    errorplot = np.zeros(iterations)
    optimizer = optimizers.Adam(learning_rate=0.005)#(learning_rate=0.1)

    for i in range(iterations):
        error = train_step(radius_array_R, optimizer)
        errorplot[i] = error

        if i % n_update == 0:
            t_now = time.time()
            print("Loss: {}\tTimePerUpdate(s): {}\t {}/{}".format(error, t_now - t, i + 1, iterations))
            t = t_now
    plt.plot(errorplot)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(os.path.join(imgs_output_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot and save the optimized radius array
    radius_array_R = radius_array_R % pitch
    radius_array_R = tf.clip_by_value(radius_array_R, radius_array_R_min, radius_array_R_max)
    plt.imshow(radius_array_R)
    plt.colorbar()
    plt.title("Optimized Radius Array")
    plt.savefig(os.path.join(imgs_output_dir, 'optimized_radius_array.png'), dpi=300, bbox_inches='tight')
    plt.close()
    np.save(os.path.join(imgs_output_dir, f'radius_array_SF{scale*10}.npy'), radius_array_R)


    # Recalculate the propagated fields and intensities with the optimized radius array, and compare their difference
    propagated_fields_R, propagated_fields_G, propagated_fields_B  = forward(radius_array_R)#, metasurface2_phase
    propagated_intensities_R = tf.cast(tf.math.abs(propagated_fields_R) ** 2 ,dtype=tf.float32)
    propagated_intensities_G = tf.cast(tf.math.abs(propagated_fields_G) ** 2 ,dtype=tf.float32)
    propagated_intensities_B = tf.cast(tf.math.abs(propagated_fields_B) ** 2 ,dtype=tf.float32)
    desired_intensities_R = tf.cast(tf.math.abs(desired_psf_R), dtype=tf.float32)
    desired_intensities_G = tf.cast(tf.math.abs(desired_psf_G), dtype=tf.float32)
    desired_intensities_B = tf.cast(tf.math.abs(desired_psf_B), dtype=tf.float32)
    propagated_intensities_R = tf.math.l2_normalize(propagated_intensities_R)
    propagated_intensities_G = tf.math.l2_normalize(propagated_intensities_G)
    propagated_intensities_B = tf.math.l2_normalize(propagated_intensities_B)
    desired_intensities_R = tf.math.l2_normalize(desired_intensities_R)
    desired_intensities_G = tf.math.l2_normalize(desired_intensities_G)
    desired_intensities_B = tf.math.l2_normalize(desired_intensities_B)
    diff_matrix_R = tf.math.subtract(propagated_intensities_R,desired_intensities_R)
    diff_matrix_G = tf.math.subtract(propagated_intensities_G,desired_intensities_G)
    diff_matrix_B = tf.math.subtract(propagated_intensities_B,desired_intensities_B)
    diff_matrix_R_squared = tf.square(diff_matrix_R)
    diff_matrix_G_squared = tf.square(diff_matrix_G)
    diff_matrix_B_squared = tf.square(diff_matrix_B)
    reduce_sum_R_squared = tf.reduce_sum(diff_matrix_R_squared)
    reduce_sum_G_squared = tf.reduce_sum(diff_matrix_G_squared)
    reduce_sum_B_squared = tf.reduce_sum(diff_matrix_B_squared)
    error_R = np.array([tf.sqrt(reduce_sum_R_squared).numpy()], dtype=np.float32)
    error_G = np.array([tf.sqrt(reduce_sum_G_squared).numpy()], dtype=np.float32)
    error_B = np.array([tf.sqrt(reduce_sum_B_squared).numpy()], dtype=np.float32)
    error_RGB = np.array([error_R,error_G,error_B])
    print(tf.sqrt(reduce_sum_R_squared),tf.sqrt(reduce_sum_G_squared),tf.sqrt(reduce_sum_B_squared))
    plt.figure(figsize=(30, 20))
    plt.subplot(3,3,1)
    plt.imshow(desired_intensities_R,cmap = 'Reds')
    plt.colorbar()
    plt.title("Red Desired Intensities")
    plt.subplot(3,3,2)
    plt.imshow(desired_intensities_G,cmap = 'Greens')
    plt.colorbar()
    plt.title("Green Desired Intensities")
    plt.subplot(3,3,3)
    plt.imshow(desired_intensities_B,cmap = 'Blues')
    plt.colorbar()
    plt.title("Blue Desired Intensities")
    plt.subplot(3,3,4)
    plt.imshow(propagated_intensities_R,cmap = 'Reds')
    plt.colorbar()
    plt.title("Red Propagated Intensities")
    plt.subplot(3,3,5)
    plt.imshow(propagated_intensities_G,cmap = 'Greens')
    plt.colorbar()
    plt.title("Green Propagated Intensities")
    plt.subplot(3,3,6)
    plt.imshow(propagated_intensities_B,cmap = 'Blues')
    plt.colorbar()
    plt.title("Blue Propagated Intensities")
    plt.subplot(3,3,7)
    plt.imshow(diff_matrix_R,cmap = 'Reds')
    plt.colorbar()
    plt.title("Red Difference Matrix")
    plt.subplot(3,3,8)
    plt.imshow(diff_matrix_G,cmap = 'Greens')
    plt.colorbar()
    plt.title("Green Difference Matrix")
    plt.subplot(3,3,9)
    plt.imshow(diff_matrix_B,cmap = 'Blues')
    plt.colorbar()
    plt.title("Blue Difference Matrix")
    plt.savefig(os.path.join(imgs_output_dir, 'propagated_vs_desired_intensities.png'), dpi=300, bbox_inches='tight')
    plt.close()


    # Another time comparison but look closer to the kernel
    plt.figure(figsize=(12, 12))
    vmin=0
    vmax_R = np.max(np.abs(desired_intensities_R))
    vmax_G = np.max(np.abs(desired_intensities_G))
    vmax_B = np.max(np.abs(desired_intensities_B))
    plt.subplot(3,3,1)
    desired_PSF_R = plt.imshow(desired_intensities_R[numpad-1:numpad-1 + numpx*len(kernel[0,:]),numpad-1:numpad-1 + numpx*len(kernel[0,:])], vmin=vmin, vmax=vmax_R, cmap = 'Reds')
    plt.title("Red")
    plt.xticks([0,scale*30,scale*60])  # Remove x-axis ticks
    plt.yticks([0,scale*30,scale*60])  # Remove y-axis ticks
    cbar = plt.colorbar(desired_PSF_R,shrink=0.8)
    plt.subplot(3,3,2)
    desired_PSF_G = plt.imshow(desired_intensities_G[numpad-1:numpad-1 + numpx*len(kernel[0,:]),numpad-1:numpad-1 + numpx*len(kernel[0,:])], vmin=vmin, vmax=vmax_G,cmap = 'Greens')
    plt.title("Green")
    plt.xticks([0,scale*30,scale*60])  # Remove x-axis ticks
    plt.yticks([0,scale*30,scale*60])  # Remove y-axis ticks
    cbar = plt.colorbar(desired_PSF_G,shrink=0.8)
    plt.subplot(3,3,3)
    desired_PSF_B = plt.imshow(desired_intensities_B[numpad-1:numpad-1 + numpx*len(kernel[0,:]),numpad-1:numpad-1 + numpx*len(kernel[0,:])], vmin=vmin, vmax=vmax_B,cmap = 'Blues')
    plt.title("Blue")
    plt.xticks([0,scale*30,scale*60])  # Remove x-axis ticks
    plt.yticks([0,scale*30,scale*60])  # Remove y-axis ticks
    cbar = plt.colorbar(desired_PSF_B,shrink=0.8)
    plt.subplot(3,3,4)
    simulated_PSF_R = plt.imshow(propagated_intensities_R[numpad-1:numpad-1 + numpx*len(kernel[0,:]),numpad-1:numpad-1 + numpx*len(kernel[0,:])], vmin=vmin, vmax=vmax_R,cmap = 'Reds')
    cbar = plt.colorbar(simulated_PSF_R,shrink=0.8)
    plt.xticks([0,scale*30,scale*60])  # Remove x-axis ticks
    plt.yticks([0,scale*30,scale*60])  # Remove y-axis ticks
    plt.subplot(3,3,5)
    simulated_PSF_G = plt.imshow(propagated_intensities_G[numpad-1:numpad-1 + numpx*len(kernel[0,:]),numpad-1:numpad-1 + numpx*len(kernel[0,:])], vmin=vmin, vmax=vmax_G,cmap = 'Greens')
    cbar = plt.colorbar(simulated_PSF_G,shrink=0.8)
    plt.xticks([0,scale*30,scale*60])  # Remove x-axis ticks
    plt.yticks([0,scale*30,scale*60])  # Remove y-axis ticks
    plt.subplot(3,3,6)
    simulated_PSF_B = plt.imshow(propagated_intensities_B[numpad-1:numpad-1 + numpx*len(kernel[0,:]),numpad-1:numpad-1 + numpx*len(kernel[0,:])], vmin=vmin, vmax=vmax_B,cmap = 'Blues')
    cbar = plt.colorbar(simulated_PSF_B,shrink=0.8)
    plt.xticks([0,scale*30,scale*60])  # Remove x-axis ticks
    plt.yticks([0,scale*30,scale*60])  # Remove y-axis ticks
    plt.subplot(3,3,7)
    abs_error_R = plt.imshow(diff_matrix_R[numpad-1:numpad-1 + numpx*len(kernel[0,:]),numpad-1:numpad-1 + numpx*len(kernel[0,:])], vmin=vmin, vmax=vmax_R,cmap = 'Reds')
    cbar = plt.colorbar(abs_error_R,shrink=0.8)
    plt.xticks([0,scale*30,scale*60])  # Remove x-axis ticks
    plt.yticks([0,scale*30,scale*60])  # Remove y-axis ticks
    plt.subplot(3,3,8)
    abs_error_G = plt.imshow(diff_matrix_G[numpad-1:numpad-1 + numpx*len(kernel[0,:]),numpad-1:numpad-1 + numpx*len(kernel[0,:])], vmin=vmin, vmax=vmax_G,cmap = 'Greens')
    cbar = plt.colorbar(abs_error_G,shrink=0.8)
    plt.xticks([0,scale*30,scale*60])  # Remove x-axis ticks
    plt.yticks([0,scale*30,scale*60])  # Remove y-axis ticks
    plt.subplot(3,3,9)
    abs_error_B = plt.imshow(diff_matrix_B[numpad-1:numpad-1 + numpx*len(kernel[0,:]),numpad-1:numpad-1 + numpx*len(kernel[0,:])], vmin=vmin, vmax=vmax_B,cmap = 'Blues')
    cbar = plt.colorbar(abs_error_B,shrink=0.8)
    plt.xticks([0,scale*30,scale*60])  # Remove x-axis ticks
    plt.yticks([0,scale*30,scale*60])  # Remove y-axis ticks
    plt.savefig(os.path.join(imgs_output_dir, 'propagated_vs_desired_intensities_closer.png'), dpi=300, bbox_inches='tight')
    plt.close()


    # Do a further clipping of the radius array to ensure it is within the valid range
    Radius_array = radius_array_R.numpy()
    xsize_ref = len(Radius_array[:,:])
    ysize_ref = len(Radius_array[0,:])
    for x in range(0,xsize_ref):
        for y in range(0,ysize_ref):
            if Radius_array[x,y] < 0.060:
                Radius_array[x,y] = 0 # nm
            if Radius_array[x,y] > 0.233:
                Radius_array[x,y] = 0.233 # nm
                
    # Starting to draw intensity slices between Metasurface1 and Detector, calculating needed parameters            
    radius_array_R = Radius_array
    phase_array_R, phase_array_G, phase_array_B = generate_phase_array_TF(Radius_array)
    array_R = np.where(phase_array_R > 1, phase_array_R - 1, phase_array_R)
    array_G = np.where(phase_array_G > 1, phase_array_G - 1, phase_array_G)
    array_B = np.where(phase_array_B > 1, phase_array_B - 1, phase_array_B)
    wavelength_R = 0.635*10**(-6) #R: 635, G: 532, B: 450 # can be R/G/B
    wavelength_G = 0.532*10**(-6) #R: 635, G: 532, B: 450 # can be R/G/B
    wavelength_B = 0.450*10**(-6) #R: 635, G: 532, B: 450 # can be R/G/B
    k_R = 2*np.pi/wavelength_R
    k_G = 2*np.pi/wavelength_G
    k_B = 2*np.pi/wavelength_B
    period = sim_args_R['dd']#5*10**(-6) # periodicity of the metasurface IN MICRONS
    ddet_R = sim_args_R['spacing_ms1_to_detector'] # distance from MS2 to the detector
    ddet_G = sim_args_G['spacing_ms1_to_detector'] # distance from MS2 to the detector
    ddet_B = sim_args_B['spacing_ms1_to_detector'] # distance from MS2 to the detector
    numpad = 400
    phase_R,phase_G,phase_B = generate_phase_array_TF(radius_array_R)
    metasurface1_phase_R = phase_R * 2*np.pi
    metasurface1_phase_G = phase_G * 2*np.pi
    metasurface1_phase_B = phase_B * 2*np.pi
    metasurface1_phase_R = np.pad(metasurface1_phase_R,(numpad,numpad),'constant',constant_values=(0,0))
    metasurface1_phase_G = np.pad(metasurface1_phase_G,(numpad,numpad),'constant',constant_values=(0,0))
    metasurface1_phase_B = np.pad(metasurface1_phase_B,(numpad,numpad),'constant',constant_values=(0,0))
    resamplefactor = 1.
    crop = int(numpad*resamplefactor)

    # Calculte fields at different distances for Red Channel
    efield_R = plot_field_set_R
    efield_R = (np.pad(efield_R,(numpad,numpad),'constant',constant_values=(0,0)))
    propfield1_R = efield_R
    propfield1_R = propfield1_R*np.exp(1j*(metasurface1_phase_R)) #can be R/G/B
    plt.figure(figsize=(30, 6))
    propfieldMS_R = asprop(propfield1_R,ddet_R/4,wavelength_R,(period/resamplefactor,period/resamplefactor)) #can be R/G/B
    plt.subplot(1,5,1)
    plt.imshow((np.abs(propfieldMS_R)**2)[crop:len(efield_R)-crop, crop:len(efield_R)-crop], vmin=0, vmax=np.max(np.abs(propfieldMS_R)**2))
    plt.colorbar()
    plt.title('1/4 d')
    propfieldMS_R = asprop(propfield1_R,ddet_R/2,wavelength_R,(period/resamplefactor,period/resamplefactor))
    plt.subplot(1,5,2)
    plt.imshow((np.abs(propfieldMS_R)**2)[crop:len(efield_R)-crop, crop:len(efield_R)-crop], vmin=0, vmax=np.max(np.abs(propfieldMS_R)**2))
    plt.colorbar()
    plt.title('Halfway there')
    propfieldMS_R = asprop(propfield1_R,ddet_R*(9/10),wavelength_R,(period/resamplefactor,period/resamplefactor))
    plt.subplot(1,5,3)
    plt.imshow((np.abs(propfieldMS_R)**2)[crop:len(efield_R)-crop, crop:len(efield_R)-crop], vmin=0, vmax=np.max(np.abs(propfieldMS_R)**2))
    plt.colorbar()
    plt.title('9/10 d')
    propfieldMS = asprop(propfield1_R,ddet_R,wavelength_R,(period/resamplefactor,period/resamplefactor))
    plt.subplot(1,5,4)
    plt.imshow((np.abs(propfieldMS)**2)[crop:len(efield_R)-crop,crop:len(efield_R)-crop], vmin=0, vmax=np.max(np.abs(propfieldMS)**2)-45, cmap="Reds")#,vmin=0,vmax=20.)
    plt.colorbar()
    plt.title('At detector')
    plt.subplot(1,5,5)
    plt.imshow(kernel_neg_expanded_R, cmap="Reds") # can be R/G/B
    plt.colorbar()
    plt.title('Desired Result')
    plt.savefig(os.path.join(imgs_output_dir, 'intensity_slices_R.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate fields at different distances for Green Channel
    efield_G = plot_field_set_G
    efield_G = (np.pad(efield_G, (numpad, numpad), 'constant', constant_values=(0, 0)))
    propfield1_G = efield_G
    propfield1_G = propfield1_G * np.exp(1j * metasurface1_phase_G)
    plt.figure(figsize=(30, 6))
    propfieldMS_G = asprop(propfield1_G, ddet_G / 4, wavelength_G, (period / resamplefactor, period / resamplefactor))
    plt.subplot(1, 5, 1)
    plt.imshow((np.abs(propfieldMS_G) ** 2)[crop:len(efield_G) - crop, crop:len(efield_G) - crop], vmin=0, vmax=np.max(np.abs(propfieldMS_G) ** 2))
    plt.colorbar()
    plt.title('1/4 d')
    propfieldMS_G = asprop(propfield1_G, ddet_G / 2, wavelength_G, (period / resamplefactor, period / resamplefactor))
    plt.subplot(1, 5, 2)
    plt.imshow((np.abs(propfieldMS_G) ** 2)[crop:len(efield_G) - crop, crop:len(efield_G) - crop], vmin=0, vmax=np.max(np.abs(propfieldMS_G) ** 2))
    plt.colorbar()
    plt.title('Halfway there')
    propfieldMS_G = asprop(propfield1_G, ddet_G * (9 / 10), wavelength_G, (period / resamplefactor, period / resamplefactor))
    plt.subplot(1, 5, 3)
    plt.imshow((np.abs(propfieldMS_G) ** 2)[crop:len(efield_G) - crop, crop:len(efield_G) - crop], vmin=0, vmax=np.max(np.abs(propfieldMS_G) ** 2))
    plt.colorbar()
    plt.title('9/10 d')
    propfieldMS_G = asprop(propfield1_G, ddet_G, wavelength_G, (period / resamplefactor, period / resamplefactor))
    plt.subplot(1, 5, 4)
    plt.imshow((np.abs(propfieldMS_G) ** 2)[crop:len(efield_G) - crop, crop:len(efield_G) - crop], vmin=0, vmax=np.max(np.abs(propfieldMS_G) ** 2) - 45, cmap="Greens")
    plt.colorbar()
    plt.title('At detector')
    plt.subplot(1, 5, 5)
    plt.imshow(kernel_neg_expanded_G, cmap="Greens")
    plt.colorbar()
    plt.title('Desired Result')
    plt.savefig(os.path.join(imgs_output_dir, 'intensity_slices_G.png'), dpi=300, bbox_inches='tight')
    plt.close()


    # Calculate fields at different distances for Blue Channel
    efield_B = plot_field_set_B
    efield_B = (np.pad(efield_B, (numpad, numpad), 'constant', constant_values=(0, 0)))
    propfield1_B = efield_B
    propfield1_B = propfield1_B * np.exp(1j * metasurface1_phase_B)
    plt.figure(figsize=(30, 6))
    propfieldMS_B = asprop(propfield1_B, ddet_B / 4, wavelength_B, (period / resamplefactor, period / resamplefactor))
    plt.subplot(1, 5, 1)
    plt.imshow((np.abs(propfieldMS_B) ** 2)[crop:len(efield_B) - crop, crop:len(efield_B) - crop], vmin=0, vmax=np.max(np.abs(propfieldMS_B) ** 2))
    plt.colorbar()
    plt.title('1/4 d')
    propfieldMS_B = asprop(propfield1_B, ddet_B / 2, wavelength_B, (period / resamplefactor, period / resamplefactor))
    plt.subplot(1, 5, 2)
    plt.imshow((np.abs(propfieldMS_B) ** 2)[crop:len(efield_B) - crop, crop:len(efield_B) - crop], vmin=0, vmax=np.max(np.abs(propfieldMS_B) ** 2))
    plt.colorbar()
    plt.title('Halfway there')
    propfieldMS_B = asprop(propfield1_B, ddet_B * (9 / 10), wavelength_B, (period / resamplefactor, period / resamplefactor))
    plt.subplot(1, 5, 3)
    plt.imshow((np.abs(propfieldMS_B) ** 2)[crop:len(efield_B) - crop, crop:len(efield_B) - crop], vmin=0, vmax=np.max(np.abs(propfieldMS_B) ** 2))
    plt.colorbar()
    plt.title('9/10 d')
    propfieldMS_B = asprop(propfield1_B, ddet_B, wavelength_B, (period / resamplefactor, period / resamplefactor))
    plt.subplot(1, 5, 4)
    plt.imshow((np.abs(propfieldMS_B) ** 2)[crop:len(efield_B) - crop, crop:len(efield_B) - crop], vmin=0, vmax=np.max(np.abs(propfieldMS_B) ** 2) - 45, cmap="Blues")
    plt.colorbar()
    plt.title('At detector')
    plt.subplot(1, 5, 5)
    plt.imshow(kernel_neg_expanded_B, cmap="Blues")
    plt.colorbar()
    plt.title('Desired Result')
    plt.savefig(os.path.join(imgs_output_dir, 'intensity_slices_B.png'), dpi=300, bbox_inches='tight')
    plt.close()



if __name__=="__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    imgs_output_dir = os.path.join('./outputs', timestamp)
    os.makedirs(imgs_output_dir, exist_ok=True)

    # The kernels below are what we want to inverse design with PSF engineering.
    np.random.seed(0)
    kernels = np.random.uniform(low=-1.0,high=1.0,size=(16,3,7,7))
    kernel_index = 9 # Load the 9th kernel to test with
    kernel = kernels[kernel_index,:]
    kernel_neg = kernel*0
    kernel_pos = kernel*0
    plotcount = 0
    for n in range(0, len(kernel)): #n = 0,1,2 = Color R,G,B
        for i in range(0, len(kernel[0,:])): #i = 0,1,2 = X
            for j in range(0, len(kernel[0,0,:])): #j = 0,1,2 = Y
                if kernel[n,i,j] < 0:
                    kernel_neg[n,i,j] = np.abs(kernel[n,i,j])
                else:
                    kernel_pos[n,i,j] = kernel[n,i,j]
        # Normalize
        max_neg = np.amax(kernel_neg[n,:])
        max_pos = np.amax(kernel_pos[n,:])
        max = np.max([max_neg, max_pos])
        kernel_neg[n] = kernel_neg[n]/max
        kernel_pos[n] = kernel_pos[n]/max
        # Plot
        if n==0:
            plotcount = plotcount + 1
            plt.subplot(2,len(kernel),plotcount)
            plt.imshow(kernel_pos[n],cmap = 'Reds')
            plt.colorbar()
            plt.title("Positive")
            plt.subplot(2,len(kernel),plotcount+3)
            plt.imshow(kernel_neg[n],cmap = 'Reds')
            plt.colorbar()
            plt.title("Negative")
        if n==1:
            plotcount = plotcount + 1
            plt.subplot(2,len(kernel),plotcount)
            plt.imshow(kernel_pos[n],cmap = 'Greens')
            plt.colorbar()
            plt.title("Positive")
            plt.subplot(2,len(kernel),plotcount+3)
            plt.imshow(kernel_neg[n],cmap = 'Greens')
            plt.colorbar()
            plt.title("Negative")
        if n==2:
            plotcount = plotcount + 1
            plt.subplot(2,len(kernel),plotcount)
            plt.imshow(kernel_pos[n],cmap = 'Blues')
            plt.colorbar()
            plt.title("Positive")
            plt.subplot(2,len(kernel),plotcount+3)
            plt.imshow(kernel_neg[n],cmap = 'Blues')
            plt.colorbar()
            plt.title("Negative")
    plt.savefig(os.path.join(imgs_output_dir, 'kernel_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    design_RGB_kernels(kernel_neg, imgs_output_dir)


