import numpy as np
import lalsimulation
from hdbank import metric

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

_name_prefix = 'SimNoisePSD'
_name_suffix = 'Ptr'

psd_name = "aLIGOZeroDetHighPower"
noise_generator = lalsimulation.__dict__[_name_prefix + psd_name]
generator = np.vectorize(lambda f: noise_generator(f))

flow = 20.0
fref = 20.0
fhigh = 1024.0
df = 0.1


length = int(fhigh/df + 1)

psd = generator( np.arange(length)*df )                                  
psd[0] = np.inf
ASD = np.sqrt(psd)


params_dic = {'mtotal': 20.0, 'eta': 0.2, 'chi': 0.5}


print('Metric on dimensionless coordinate system \n')
print('Reference point at %s \n '%str(params_dic))
gm = metric.ComputeNumericalIMRMetric(fref, flow, fhigh, df, ASD, approximant='SEOBNRv4_ROM')
print('SEOBNRv4ROM \n', gm.Theta0Theta3Theta3s(**params_dic) )

gm = metric.ComputeNumericalIMRMetric(fref, flow, fhigh, df, ASD, approximant='IMRPhenomD')
print('IMRPhenomD \n', gm.Theta0Theta3Theta3s(**params_dic) )

gm = metric.ComputeNumericalIMRMetric(fref, flow, fhigh, df, ASD, approximant='IMRPhenomXAS')
print('IMRPhenomXAS \n', gm.Theta0Theta3Theta3s(**params_dic) )

gm = metric.ComputeNumericalIMRMetric(fref, flow, fhigh, df, ASD, approximant='TaylorF2')
print('TaylorF2 \n', gm.Theta0Theta3Theta3s(**params_dic) )
