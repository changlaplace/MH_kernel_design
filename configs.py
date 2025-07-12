import numpy as np
import tensorflow as tf

# Set the simulation arguments
dtype = {'real': tf.float32, 'comp': tf.complex64, }
sim_args_R = {
    'wavelength': 635e-9, #R: 635, G: 532, B: 450
    'dd': 586e-9, # simulation grid size
    'numpx': 1600,  # number of pixels in array, without padding
    } # size of lens
sim_args_R = {
    **sim_args_R,
    'k': 2. * np.pi / sim_args_R['wavelength'],
    'spacing_start_to_ms1': 0.5e-3, #2.0e-3,
    'spacing_ms1_to_detector': 2.4e-3, #20e-3,
    'array_shape': (sim_args_R['numpx'], sim_args_R['numpx'],),
    'lens_aperture': sim_args_R['numpx']*sim_args_R['dd'],
    'dtype': dtype['comp'],
    }

sim_args_G = {
    'wavelength': 532e-9, #R: 635, G: 532, B: 450
    'dd': 586e-9, # simulation grid size
    'numpx': 1600,  # number of pixels in array, without padding
    } # size of lens
sim_args_G = {
    **sim_args_G,
    'k': 2. * np.pi / sim_args_G['wavelength'],
    'spacing_start_to_ms1': 0.5e-3, #2.0e-3,
    'spacing_ms1_to_detector': 2.4e-3, #20e-3,
    'array_shape': (sim_args_G['numpx'], sim_args_G['numpx'],),
    'lens_aperture': sim_args_G['numpx']*sim_args_G['dd'],
    'dtype': dtype['comp'],
    }

sim_args_B = {
    'wavelength': 450e-9, #R: 635, G: 532, B: 450
    'dd': 586e-9, # simulation grid size
    'numpx': 1600,  # number of pixels in array, without padding
    } # size of lens
sim_args_B = {
    **sim_args_B,
    'k': 2. * np.pi / sim_args_B['wavelength'],
    'spacing_start_to_ms1': 0.5e-3, #2.0e-3,
    'spacing_ms1_to_detector': 2.4e-3, #20e-3,
    'array_shape': (sim_args_B['numpx'], sim_args_B['numpx'],),
    'lens_aperture': sim_args_B['numpx']*sim_args_B['dd'],
    'dtype': dtype['comp'],
    }

field_args_R = {'array_size' : sim_args_R['numpx'], 'dtype' : dtype['comp']}
field_args_G = {'array_size' : sim_args_G['numpx'], 'dtype' : dtype['comp']}
field_args_B = {'array_size' : sim_args_B['numpx'], 'dtype' : dtype['comp']}
