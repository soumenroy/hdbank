#from __future__ import division
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

import lal
import lalsimulation
from lalinspiral.sbank.psds import get_PSD, get_ASD


__author__ = "Soumen Roy <soumen.roy@ligo.org>"


def get_lalsim_psd_list():
    """
    Return a list of available reference PSD functions from LALSimulation.
    
    Ref: pycbc psd module
    """
    _name_prefix = 'SimNoisePSD'
    _name_suffix = 'Ptr'
    _name_blacklist = ('FromFile', 'MirrorTherm', 'Quantum', 'Seismic', 'Shot', 'SuspTherm')
    _psd_list = []
    for _name in lalsimulation.__dict__:
        if _name != _name_prefix and _name.startswith(_name_prefix) and not _name.endswith(_name_suffix):
            _name = _name[len(_name_prefix):]
            if _name not in _name_blacklist:
                _psd_list.append(_name)
    _psd_list = sorted(_psd_list)
    return _psd_list


def lalsim_psd_generator(psd_name):
    """
    Return a PSD generator for LALSimulation PSD specified by name.
    
    Parameters
    ----------
    psd_name : string
        PSD name as found in LALSimulation, minus the SimNoisePSD prefix.
        
    Returns
    -------
    noise_generator : generator
        Generator for specified PSD.
    """
    _name_prefix = 'SimNoisePSD'
    _name_suffix = 'Ptr'
    # check if valid PSD model
    if psd_name not in get_lalsim_psd_list():
        raise ValueError(psd_name + ' not found among analytical '
                         'PSD functions.')
    # if PSD model is in LALSimulation
    try:
        noise_generator = lalsimulation.__dict__[_name_prefix + psd_name]
    except:
        noise_generator = lalsimulation.__dict__[_name_prefix + psd_name + _name_suffix]
        
    numpy_noise_generator = np.vectorize(lambda f: noise_generator(f))
    return numpy_noise_generator




def psd_generator_from_ASCII(filename, delta_f, low_freq_cutoff, is_asd_file=False):
    """
    Read an ASCII file containing one-sided ASD or PSD  data and generate
    a frequency series with the corresponding PSD. The ASD or PSD data is
    interpolated in order to match the desired resolution of the
    generated frequency series.
    
    Ref: pycbc psd module
    Parameters
    ----------
    filename : string
        Path to a two-column ASCII file. The first column must contain
        the frequency (positive frequencies only) and the second column
        must contain the amplitude density OR power spectral density.
    delta_f : float
        Frequency resolution of the frequency series in Herz.
    low_freq_cutoff : float
        Frequencies below this value are set to zero.
    is_asd_file : Boolean
        If false assume that the second column holds power spectral density.
        If true assume that the second column holds amplitude spectral density.
        Default: True
    Returns
    -------
    psd : FrequencySeries
        The generated frequency series.
    Raises
    ------
    ValueError
        If the ASCII file contains negative, infinite or NaN frequencies
        or amplitude densities.
    """
    file_data = np.loadtxt(filename)
    if (file_data < 0).any() or \
                            np.logical_not(np.isfinite(file_data)).any():
        raise ValueError('Invalid data in ' + filename)

    freq_data = file_data[:, 0]
    noise_data = file_data[:, 1]
    
    # Only include points above the low frequency cutoff
    if freq_data[0] > low_freq_cutoff:
        raise ValueError('Lowest frequency in input file ' + filename + \
          ' is higher than requested low-frequency cutoff ' + str(low_freq_cutoff))
    
    if is_asd_file is True:
        noise_data = np.square(noise_data)
    
    f_max_orig = max(freq_data)
    interpolator = interp1d(np.log(freq_data), np.log(noise_data))
    noise_generator = lambda g: np.where(g < f_max_orig, np.exp(interpolator(np.log(g))), np.inf)
    
    return noise_generator


def get_psd_from_arguments(args):
    """
    This function verify the psd related arguments and returns 
    both PSD and ASD for a specified noise model or given psd/asd file.
    
    """
    if args.flow > args.fhigh:
        raise ValueError("Lowest cutoff frequency must be less than highest cuttoff frequency.")
    elif args.df > args.fhigh:
        raise valueError("Frequency sampling rate is larger then highest cuttoff frequency.")
    else:
        pass
    
    if args.psd_model is None and args.asd_file is None and args.psd_file is None:
        raise ValueError("psd-model or psd-model or asd-file  must be specified.")
    elif args.psd_model is not None and args.asd_file is not None:
        raise ValueError("Both the psd-model and asd-file can not be specified together.")
    elif args.psd_model is not None and args.psd_file is not None:
        raise ValueError("Both the psd-model and psd-file can not be specified together.")
    elif args.psd_file is not None and args.asd_file is not None:
        raise ValueError("Both the psd-file and asd-file can not be specified together.")
        
    elif args.psd_model is not None and (args.asd_file is None and args.psd_file is None):
        
        # Obtain psd from available lalsimulation noise model
        if args.psd_model in get_lalsim_psd_list():
            noise_generator = lalsim_psd_generator(args.psd_model)
        else:
            raise ValueError("Specified psd-model is not available. \n Choose from "+str(get_lalsim_psd_list()))
    else:
        if args.asd_file is not None:
            # Obtain psd from provided psd file
            noise_generator = psd_generator_from_ASCII(args.asd_file, args.df, args.flow, is_asd_file=True)
            
        elif args.psd_file is not None:
            # Obtain psd from provided psd file
            noise_generator = psd_generator_from_ASCII(args.psd_file, args.df, args.flow, is_asd_file=False)
        else:
            raise TypeError("provide the psd generation argument.")
                                       
    PSD = get_PSD(args.df, args.flow, args.fhigh, noise_generator)
    ASD = get_ASD(args.df, args.flow, args.fhigh, noise_generator)
                                       
    return PSD, ASD, noise_generator
