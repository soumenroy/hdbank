from __future__ import division
import numpy as np

import lal
from lal import PI, MSUN_SI, MTSUN_SI, PC_SI, CreateCOMPLEX8FrequencySeries, CreateREAL8Vector, ResizeCOMPLEX16FrequencySeries
from lalinspiral import CreateSBankWorkspaceCache, InspiralSBankComputeMatch
import lalsimulation as lalsim
from lalinspiral.sbank.psds import noise_models, get_PSD
from pycbc.psd.read import from_txt


global mm, fhigh, df, fref, phi0, distance, ASD, PI_square, PI_quant, one_by_three, PI_p5, alpha, trunc_oct_ngb

mm = 0.965
DYN_flow= 15.0
fhigh = 1024.0
df = 0.1

fref=15.0

phi0 = 0.0
distance = 1000000 * PC_SI
dur_lim = 0.15


nsbh_bound = 2.0

PI_square = PI * PI
PI_quant = PI_square * PI_square * PI
one_by_three = 1.0/3.0
PI_p5 = PI_quant


ind_max = int(fhigh/df)

length = ind_max+1
psd_file = "/home/soumen.roy/HybriBankCode/final_for_num_v4/H1L1-AVERAGE_PSD-1163174417-604800.txt.gz"
pycbc_psd = from_txt(psd_file, length, df, DYN_flow, is_asd_file=False)
psd = pycbc_psd.data[:] 



psd[:150] = np.inf
ASD = np.sqrt(psd[:ind_max])



def m1m2_to_mtotaleta(m1, m2):
    mtotal = m1 + m2
    eta = m1*m2/(m1+m2)**2.0
    return mtotal, eta

def mtotaleta_to_m1m2(mtotal, eta):
    m1 = mtotal*(1.0 + np.sqrt(1.0-4.0*eta))/2.0
    m2 = mtotal -m1
    return m1, m2

def _get_dur(m1, m2, spin1z, spin2z, flow):
    dur = lalsim.SimIMRSEOBNRv4ROMTimeOfFrequency(flow, m1*MSUN_SI, m2*MSUN_SI,\
                                                  spin1z, spin2z)
    # Allow a 10% margin of error
    return dur * 1.1


def optimize_flow(m1, m2, spin1z, spin2z, sigma_frac=0.995):
    """Set the template's flow as high as possible but still recovering
    at least the given fraction of the template's sigma when calculated
    from the minimum allowed flow. This avoids having unnecessarily long
    templates.
    """
    flow = 15.0
    fref = 15.0
    # compute the whitened waveform
    asd = ASD[:ind_max]
    wf = lalsim.SimIMRSEOBNRv4ROMAmpPhs(phi0, df, flow, fhigh, fref,\
                                        distance, 0.0, m1 * MSUN_SI, m2 * MSUN_SI, spin1z, \
                                        spin2z, -1)[0].data.data[:][:ind_max].real
    wwf = wf / asd
    # sum the power cumulatively from high to low frequencies
    integral = np.cumsum(np.flipud(wwf * wwf))
    ref_sigmasq = integral[-1]
    # find the frequency bin corresponding to the target loss
    i = np.searchsorted(integral, ref_sigmasq * sigma_frac ** 2)
    opt_flow = (len(integral) - i) * df
    return opt_flow





def optimize_flow_compute_dur(m1, m2, spin1z, spin2z):
    opt_flow = optimize_flow(m1, m2, spin1z, spin2z)
    dur =  _get_dur(m1, m2, spin1z, spin2z, opt_flow)
    return opt_flow, dur
    

    
data = np.loadtxt("results.dat")
indices = []
for i in range(len(data)):
    mass_spin = data[i][9:17]
    if mass_spin[0]>=5 and  mass_spin[1] >= 5.0:
        opt_flow, dur = optimize_flow_compute_dur(mass_spin[0], mass_spin[1], mass_spin[4], mass_spin[7])
        if dur > 0.15:
            indices.append(i)
    print i
        
np.savetxt("fresults.dat", data[indices])

