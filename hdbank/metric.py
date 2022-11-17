from __future__ import division
from collections import defaultdict
import numpy as np
import sys

import lal
from lal import PI, PC_SI, MSUN_SI, MTSUN_SI, CreateREAL8Vector, CreateREAL8FrequencySeries, HertzUnit
import lalsimulation as lalsim
from lalinspiral.sbank.tau0tau3 import mtotal_eta_to_mass1_mass2



__author__ = "Soumen Roy <soumen.roy@ligo.org>"


def projection(gamma, dim):
    """
    Projection of metric using Schur Complement.
    """
    gamma1 = gamma[:dim, :dim]
    gamma2 = gamma[dim:, :dim]
    gamma4 = gamma[dim:, dim:]
    return gamma1 - np.dot(gamma2.T, np.dot(np.linalg.inv(gamma4), gamma2))

def reshape_metric(arr1D, n=3):
    """
    Resize of 1D array into a symmetric 2D array 
    
    Parameters
    ----------
    arr1D : numpy.array or list or tuple of order 1x(n*(n-1)/2)
        Independent element of a symmetric matrix of order for
        3D case: [g00, g01, g02, g11, g12, g22]
    n ; int
        Dimension of the matrix.
    """
    out = np.array([[arr1D[0], arr1D[1], arr1D[2]],\
                    [arr1D[1], arr1D[3], arr1D[4]],\
                    [arr1D[2], arr1D[4], arr1D[5]]])
    return out


one_by_three = 1.0/3.0
PI_square = PI * PI
PI_quant = PI_square * PI_square * PI

class ComputeNumericalMetric(object):
    """
    This class computes the numerical metric using first order 
    finite difference method. Metric for BBH system is determined 
    assuming equal aligned spin binary (spin1z = spin2z) and for 
    NSBH system, the neutron star are assumed to be non-spinning.
    
    Ref: Sec. II of https://arxiv.org/pdf/1711.08743.pdf
    
    Parameters
    ----------
    phi0 : float
        Phase at coalescence.
    fref : float
        Reference frequency. Unit: Hz.
    flow : float 
        Lower cutoff frequency. Unit: Hz.
    fhigh : float
        Higher cutoff frequency. Unit: Hz.
    df : float
        Frequency sampling. Unit: Hz.
    ASD : numpy array of order 1xn
        Amplitude spectral density of detector noise. Unit: sqrt(Hz).
    delta_param : float, optional
        Step size to calculate the partial derivative.
    nsbh_bound : float
        NSBH boundary mass.
    approximant : str
        Approximant of the waveform model.
    delta_options : callable
        Returns values \f$(\delta mtotal, \delta eta, \delta chi)\f$
        for a chosen parameter.
    etaLimit : float
        Limit of eta where the cenetral finite difference method
        are allowed to calulate the derivative, if the eta exceeds
        the limit the the derivative would be calculated using 
        backward finite difference method.
    sigmasq : float
        Norm of whitened h(f).
    """
    
    def __init__(self, phi0, fref, flow, fhigh, df, ASD, delta_param = 0.0005,\
                                     nsbh_bound=2.0, approximant='IMRPhenomD', optimize_flow =None):
        self._phi0 = phi0
        self._fref = fref
        self._flow = flow
        self._fhigh = fhigh
        self._df = df
        self._ind_min = int(self._flow/self._df)
        self._ind_max = int(self._fhigh/self._df + 1)
        self._freqSeries = np.linspace(flow, fhigh, self._ind_max-self._ind_min)
        
        self._optimize_flow=optimize_flow
        self._ASD = ASD
        self._delta_param = delta_param
        self._nsbh_bound = nsbh_bound
        self._approximant = approximant
        self._delta_options = defaultdict(lambda: None, {'mtotal': [delta_param, 0.0, 0.0],\
                                                         'eta':    [0.0, delta_param, 0.0],\
                                                         'chi':    [0.0, 0.0, delta_param]})
        self._sigmasq = 0.0
        
        self._mass1 = 0.0
        self._mass2 = 0.0
        self._chi = 0.0
        
        self._distance = 1000000*PC_SI
        

    def _amplitude_phase(self, mtotal, eta, spin1z, spin2z):
        """
        Return two COMPLEX8FrequencySeries of the amplitude and phase of the specified approximant.
        
        Parameters
        ----------
        mtotal: float
            Total mass of the system. Unit: Solar mass
        eta: float
            Symmetric mass ratio of the system.
        spin1z: float
            Spin of the first object.
        spin2z: float
            Spin of the second object.
        """
        
        mass1, mass2 = mtotal_eta_to_mass1_mass2(mtotal, eta)
        
        # Set the value of spin1z and spin2z for specified system type.
        if self._mass2 <= self._nsbh_bound and self._mass1 >= self._nsbh_bound:
            spin2z = 0.0
            
        if self._approximant == 'IMRPhenomD':
            amp_phs = lalsim.SimIMRPhenomDGenerateFDAmpPhs(self._phi0, self._fref, self._df, \
                                   mass1 * MSUN_SI, mass2 * MSUN_SI, spin1z, spin2z, \
                                   self._flow, self._fhigh, self._distance, None, lalsim.NoNRT_V)
            
        elif self._approximant == 'SEOBNRv4_ROM':
            amp_phs = lalsim.SimIMRSEOBNRv4ROMAmpPhs(self._phi0, 
                                 self._df, self._flow, self._fhigh, self._fref, self._distance, 0.0, \
                                 mass1 * MSUN_SI, mass2 * MSUN_SI, spin1z, spin2z, -1, None, lalsim.NoNRT_V)
            
        elif self._approximant == 'IMRPhenomXAS':
            amp_phs = lalsim.SimIMRPhenomXASGenerateFDAmpPhs(mass1 * MSUN_SI, mass2 * MSUN_SI, spin1z, spin2z, \
                                 self._distance, self._flow, self._fhigh, self._df, self._phi0, self._fref, None)
            
            
        
        elif self._approximant == 'TaylorF2':
            amp_phs = lalsim.SimInspiralTaylorF2AmpPhs(self._phi0, self._df, mass1 * MSUN_SI, mass2 * MSUN_SI, spin1z, spin2z, \
                                 self._flow, self._fhigh, self._fref, self._distance, None)
            
        else: 
            raise ValueError("Numerical metric not available for given approximant.")
    
        return amp_phs
    
    
    
    
    def _amplitude_phase_derivative(self, mtotal, eta, chi, der_opt=None):
        """
        Compute the partial derivative of amplitude and phase w. r. t mtotal, eta or chi
        using first order finite difference method.
        
        Parametrs
        ---------
        mtotal : float
            Total mass of the system. Unit: Solar mass
        eta : float
            Symmetric mass ratio of the system.
        chi : float
            Spin of the black hole.
        Der : str
            Derivative with respect to Der
            - 'mtotal' partial derivate w. r. t total mass.
            - 'eta' partial derivative w. r. t symmetric mass ratio.
            - 'chi' partial derivative w. r. t spin of black hole.
        
        """
        # Set the step size for given argument Der.
        delta_mtotal, delta_eta, delta_chi = self._delta_options[der_opt]
        
        # Use centeral finite difference method when eta <= etaLimit
        # else use the backward or forward difference method.
        
        if eta<0.2495 and abs(chi)<0.9995 and eta > 0.0105:
        
            amp_plus, phs_plus= self._amplitude_phase(mtotal+delta_mtotal, eta+delta_eta, chi+delta_chi, chi+delta_chi)
            amp_plus = amp_plus.data.data[self._ind_min:self._ind_max]
            phs_plus = phs_plus.data.data[self._ind_min:self._ind_max]

            amp_minus, phs_minus = self._amplitude_phase(mtotal-delta_mtotal, eta-delta_eta, chi-delta_chi, chi-delta_chi)
            amp_minus = amp_minus.data.data[self._ind_min:self._ind_max]
            phs_minus = phs_minus.data.data[self._ind_min:self._ind_max]
            
            diff_amp = (amp_plus - amp_minus)/(2*self._delta_param)
            diff_phs = (phs_plus - phs_minus)/(2*self._delta_param)
        
        elif eta > 0.0105:
            amp, phs = self._amplitude_phase(mtotal, eta, chi, chi)
            amp = amp.data.data[self._ind_min:self._ind_max]
            phs = phs.data.data[self._ind_min:self._ind_max]
        
            amp_minus, phs_minus = self._amplitude_phase(mtotal-delta_mtotal, eta-delta_eta, chi-delta_chi, chi-delta_chi)
            amp_minus = amp_minus.data.data[self._ind_min:self._ind_max]
            phs_minus = phs_minus.data.data[self._ind_min:self._ind_max]
        
            diff_amp = (amp - amp_minus)/self._delta_param
            diff_phs = (phs - phs_minus)/self._delta_param
        
        else:
            amp_plus, phs_plus = self._amplitude_phase(mtotal+delta_mtotal, eta+delta_eta, chi+delta_chi, chi+delta_chi)
            amp_plus = amp_plus.data.data[self._ind_min:self._ind_max]
            phs_plus = phs_plus.data.data[self._ind_min:self._ind_max]
        
            amp, phs = self._amplitude_phase(mtotal, eta, chi, chi)
            amp = amp.data.data[self._ind_min:self._ind_max]
            phs = phs.data.data[self._ind_min:self._ind_max]
        
            diff_amp = (amp_plus - amp)/self._delta_param
            diff_phs = (phs_plus - phs)/self._delta_param
        
        return diff_amp, diff_phs
        
    
    
    
    def _compute_metric_in_mtotal_eta_chi_t0_phi0(self, mtotal ,eta, chi):
        """
        Compute the Fischer information matrix in (Mtotal, Eta, Chi, T0, Phi0)
        coordinate sytstem from the derivative of phase and amplitude.
        
        Ref: Eq. 2.8 of the article https://arxiv.org/pdf/1711.08743.pdf
        
        Parameters
        ----------
        mtotal: float
            Total mass of the system.
        eta: float
            Symmetric mass ratio of the sytem.
        chi: float
            Effective spin of the binary system, treat equal-aligned-spin binary
            when both the individual masses are greater then NSBH boundary mass
            otherwise treat NSBH system with BH is spinning only.
            
        Return
        ------
        gamma : matrix of order 5x5
            Fischer information matrix. 
        """
        amp = self._amplitude_phase(mtotal, eta, chi, chi)[0].data.data[self._ind_min:self._ind_max]
        amp_by_ASD = amp/self._ASD[self._ind_min:self._ind_max]
        
        # Compute the derivatives of amplitude and phase w.r.t mtotal, eta, chi, t0 and phi0.
        # To calulate the inner product, divide them by sqrt(S_h(f))
        
        DAmpDM, DPsiDM = self._amplitude_phase_derivative(mtotal, eta, chi, der_opt="mtotal")
        DAmpDM /= self._ASD[self._ind_min:self._ind_max]
        DPsiDM *= amp_by_ASD
        
        
        DAmpDEta, DPsiDEta = self._amplitude_phase_derivative(mtotal, eta, chi, der_opt="eta")
        DAmpDEta /= self._ASD[self._ind_min:self._ind_max]
        DPsiDEta *= amp_by_ASD
        
        DAmpDChi, DPsiDChi = self._amplitude_phase_derivative(mtotal, eta, chi, der_opt="chi")
        DAmpDChi /= self._ASD[self._ind_min:self._ind_max]
        DPsiDChi *= amp_by_ASD
    
        DPsiDTc = amp_by_ASD * 2.0 * PI * self._freqSeries
        DPsiDPhi0 = amp_by_ASD
    
        gamma_MM    = np.vdot(DPsiDM, DPsiDM) +  np.vdot(DAmpDM, DAmpDM)
        gamma_MEta  = np.vdot(DPsiDM, DPsiDEta) + np.vdot(DAmpDM, DAmpDEta)
        gamma_MChi = np.vdot(DPsiDM, DPsiDChi) + np.vdot(DAmpDM, DAmpDChi)
        gamma_MTc   = np.vdot(DPsiDM, DPsiDTc)
        gamma_MPhi0 = np.vdot(DPsiDM, DPsiDPhi0)

        gamma_EtaEta  = np.vdot(DPsiDEta, DPsiDEta) + np.vdot(DAmpDEta, DAmpDEta)
        gamma_EtaChi = np.vdot(DPsiDEta, DPsiDChi) + np.vdot(DAmpDEta, DAmpDChi)
        gamma_EtaTc   = np.vdot(DPsiDEta, DPsiDTc)
        gamma_EtaPhi0 = np.vdot(DPsiDEta, DPsiDPhi0)

        gamma_ChiChi = np.vdot(DPsiDChi, DPsiDChi) + np.vdot(DAmpDChi, DAmpDChi)
        gamma_ChiTc   = np.vdot(DPsiDChi, DPsiDTc)
        gamma_ChiPhi0 = np.vdot(DPsiDChi, DPsiDPhi0)

        gamma_TcTc   = np.vdot(DPsiDTc, DPsiDTc)
        gamma_TcPhi0 = np.vdot(DPsiDTc, DPsiDPhi0)

        gamma_Phi0Phi0 = np.vdot(DPsiDPhi0, DPsiDPhi0)

        # Norm of whitened h(f)
        hsquare = 2.0 * gamma_Phi0Phi0
        self._sigmasq = 2.0 * hsquare * self._df
        
        # Fill the numpy matrix contains the Fisher matrix in (mtotal, eta, chi, t0, phi0)
        gamma = np.zeros((5,5))
        
        gamma[0, 0] = gamma_MM / hsquare
        gamma[0, 1] = gamma[1, 0] = gamma_MEta / hsquare
        gamma[0, 2] = gamma[2, 0] = gamma_MChi / hsquare
        gamma[0, 3] = gamma[3, 0] = gamma_MTc / hsquare
        gamma[0, 4] = gamma[4, 0] = gamma_MPhi0 / hsquare
    

        gamma[1, 1] = gamma_EtaEta / hsquare
        gamma[1, 2] = gamma[2, 1] = gamma_EtaChi / hsquare
        gamma[1, 3] = gamma[3, 1] = gamma_EtaTc / hsquare
        gamma[1, 4] = gamma[4, 1] = gamma_EtaPhi0 / hsquare

        gamma[2, 2] = gamma_ChiChi / hsquare
        gamma[2, 3] = gamma[3, 2] = gamma_ChiTc / hsquare
        gamma[2, 4] = gamma[4, 2] = gamma_ChiPhi0 / hsquare


        gamma[3, 3] = gamma_TcTc / hsquare
        gamma[3, 4] = gamma[4, 3] = gamma_TcPhi0 / hsquare
    
        gamma[4, 4] = gamma_Phi0Phi0 / hsquare
        
        return gamma
    
    
    
    def _Jacobian_transformation(self, gamma, mtotal, eta, chi, theta0, theta3, theta3s):
        """
        Jacobian transformation of metric from (Mtotal, Eta, Chi, T0, Phi0) 
        coordinate to (theta_0, theta_3, theta_3s, T0, Phi0). 
        
        Ref: Eq.(3.7) from the article https://arxiv.org/pdf/1210.6666.pdf
        
        Prameters:
        ----------
        gamma : numpy matrix of order 5x5
            Fischer information matrix in (Mtotal, Eta, Chi, T0, Phi0) coordinate.
        mtotal : float
            Total mass of the system.
        eta : float
            Symmetric mass ratio of the sytem.
        chi : float
            Effective spin of the binary system, treat equal-aligned-spin binary
            when both the individual masses are greater then NSBH boundary mass
            otherwise treat NSBH system with BH is spinning only.
        theta0 : float
            0 PN dimensionless chirp time.
        theta3 : float
            1.5 PN dimensionless chirp time (spin independent).
        theta3s : float
            1.5 PN dimensionless chirp time (spin dependent).
            
        Return:
        ------
        gamma_theta: numpy matrix of order 5x5
            Fischer information matrix in (theta0, theta3, theta3s, T0, Phi0) coordinate.
        """

        jaco = np.zeros((5,5))
        
        # Coordinate Derivatives for Jacobian matrix from (Mtotal, Eta, Chi, T0, Phi0) coordinate
        # to (Mtotal, Eta, Chi_red, T0, Phi0) coordinate.
        
        if self._mass2 >= self._nsbh_bound or self._mass1 <= self._nsbh_bound:
            dChidEta  = 76.0*chi/113.0  / (1.0 - 76.0*eta/113.0)
            dChidChi_r = 1.0/(1.0 - 76.0*eta/113.0)
        else:
            delta = (1.0-4.0*eta)**0.5
            factor = (1.0 + delta - 76.0*eta/113.0)
            dChidEta  =  chi*(2.0/delta + 76.0/113.0) / factor
            dChidChi_r = 2.0/factor
            
        jaco[0, 0] = 1.0
        jaco[1, 1] = 1.0
        jaco[1, 2] = dChidEta
        jaco[2, 2] = dChidChi_r
        jaco[3, 3] = 1.0
        jaco[4, 4] = 1.0
        
        g = np.dot(jaco, np.dot(gamma, jaco.T))
        
        if self._optimize_flow != None:
            flow = self._fref
        else:
            flow = self._flow
        
        # Coordinate Derivatives for Jacobian matrix from (Mtotal, Eta, Chi_red, T0, Phi0) coordinate
        # to (theta_0, theta_3, theta_3s, T0, Phi0) coordinate.
        theta3_p2 = theta3*theta3
        theta0_p2 = theta0*theta0
        theta3_p2by3 = theta3_p2**one_by_three
        dMdTheta0 = (-0.015831434944115277*theta3)/(flow*theta0_p2)
        dMdTheta3 = 0.015831434944115277/(flow*theta0)
        dMdTheta3S = 0.0
        dEtadTheta0 = 3.8715528021485643/(theta0**one_by_three * theta3*theta3_p2by3)
        dEtadTheta3 = -9.678882005371412 * theta0_p2**one_by_three/(theta3_p2*theta3_p2by3)
        dEtadTheta3S = 0.0
        dChirdTheta0 = 0.0
        dChirdTheta3 = -48*PI*theta3s/(113*theta3*theta3)
        dChirdTheta3S = 48*PI/(113 * theta3)

        jaco = np.zeros((5, 5))
        jaco[0,0] = dMdTheta0/MTSUN_SI
        jaco[0,1] = dEtadTheta0
        jaco[0,2] = dChirdTheta0
        jaco[1,0] = dMdTheta3/MTSUN_SI
        jaco[1,1] = dEtadTheta3
        jaco[1,2] = dChirdTheta3
        jaco[2,0] = dMdTheta3S/MTSUN_SI
        jaco[2,1] = dEtadTheta3S
        jaco[2,2] = dChirdTheta3S
        jaco[3,3] = 1.0
        jaco[4,4] = 1.0

        gamma_theta = np.dot(jaco, np.dot(g, jaco.T))
        
        return gamma_theta
    
    def MetricInTheta0Theta3Theta3s(self, center, mis_match=1.0):
        """
        This function returns the Fischer information matrix in (theta0, theta3, theta3s, t0, phi0)
        and a spatial metric in (theta0, theta3, theta3s) scaled by mismatch.
        
        Parameters:
        -----------
        center : list of order 1x8
                It has eight elements corresponds to the template point.
                    theta0 : float
                        0 PN dimensionless chirp time.
                    theta3 : float
                        1.5 PN dimensionless chirp time (spin independent).
                    theta3s : float
                        1.5 PN dimensionless chirp time (spin dependent).
                    mtotal : float
                        Total mass of the binary system. Unit: Solar mass 
                    eta : float
                        Symmetric mass ratio.
                    chi : float
                        Effective spin of the binary system, treat equal-aligned-spin binary
                        when both the individual masses are greater then NSBH boundary mass
                        otherwise treat NSBH system with BH is spinning only.
                    opt_floe : float
                        Optimized lower frequency.
                    dur : float
                        Template duration.
                    
        mis_match : float
            Mismatch between center and minimal matched surface. 
        
        Returns
        -------
        Pgamma : list of order 1x6
            Containing upper triangular metric elements scaled by mismatch (theta_0, theta_3, theta_3s) 
            gamma1 = [g00, g01, g02, g11, g12, g22]
        Fgamma : numpy matrix of order 5x5
            Fischer information matrix in (theta0, theta3, theta3s, t0, phi0)
        """

        theta0, theta3, theta3s, mtotal, eta, chi, _,  opt_flow, dur = center
        self._mass1, self._mass2 = mtotal_eta_to_mass1_mass2(mtotal, eta)
        
        if opt_flow != self._flow:
            self._ind_min = int(opt_flow/self._df)
            self._freqSeries = np.linspace(opt_flow, self._fhigh, self._ind_max-self._ind_min)
        
        # Compute the metric over (mtotal, eta, chi, t0, phi0) coordinates.
        g_ij = self._compute_metric_in_mtotal_eta_chi_t0_phi0(mtotal ,eta, chi)
        
        # Transform the metric g_ij
        # from (mtotal, eta, chi, t0, phi0) to (theta_0, theta_3, theta_3s, t0, phi0) 
        FGamma = self._Jacobian_transformation(g_ij, mtotal, eta, chi, theta0, theta3, theta3s)
        
        # Project FGamma along t0 and phi0 to get g(theta_0, theta_3, theta_3s) and scale
        Pgamma = projection(FGamma, 3)/mis_match
        
        return Pgamma, [FGamma, self._sigmasq]

             
        

class ObtainAnalyticalMetric(object):
    """
    Obtain the analytical metric in theta_0, theta_3, theta_3
    coordinate system from lalsimulation. This metric is available 
    only for TaylorF2RedSpin and IMRPhenomB waveform model.
    
    Parameters
    ----------
    phi0 : float
        Phase at coalescence.
    fRef : float
        Reference frequency. Unit: Hz.
    fLow : float 
        Lower cutoff frequency. Unit: Hz.
    fHigh : float
        Higher cutoff frequency. Unit: Hz.
    df : float
        Frequency sampling. Unit: Hz.
    ASD : numpy array of order 1xn
        Amplitude spectral density of detector noise. Unit: sqrt(Hz).
    nsbh_bound : float, optional
        NSBH boundary mass. Required for NSBH+BBH system.
    approximant : str
        Approximant of the waveform model.
    """
    
    def __init__(self, phi0, fRef, fLow, fHigh, df, ASD, delta_param = 0.0005,\
                 sys_type='BBH', nsbh_bound=None, approximant='IMRPhenomB'):
        self._phi0 = phi0
        self._fRef = fRef
        self._fLow = fLow
        self._fHigh = fHigh
        self._df = df
        self.freq_idex_low = int(fLow/df)
        self.freq_idex_high = int(fHigh/df)+1
        self._ASD = ASD[ int(fLow/df) : int(fHigh/df)+1 ]
        self.nsbh_bound = nsbh_bound
        self._approximant = approximant
        self.one_by_three = 1.0/3.0
        
        if approximant == "IMRPhenomB":
            dummy = lalsim.SimInspiralTaylorF2ReducedSpin(0, 1, 10 * MSUN_SI, 10 * MSUN_SI, \
                                                           0, 40, 41, 1e6 * PC_SI, 0, 0)
            Sh = CreateREAL8FrequencySeries("S_h(f)", dummy.epoch, 0., df, HertzUnit, len(self._ASD)) 
            Sh.data.data = np.square(self._ASD)
            self.Sh = Sh
            
        elif approximant == "TaylorF2RedSpin":
            R8V = CreateREAL8Vector(len(ASD))
            R8V.data = [np.square(ASD)[i] for i in range(len(R8V.data))]
            self._moments = [CreateREAL8Vector(int((fHigh - fLow)/df)) for _ in range(29)]
            lalsim.SimInspiralTaylorF2RedSpinComputeNoiseMoments(*(self._moments + [R8V, self._fLow, self._df]))
        else:
            raise ValueError("Analytical metric is not available for given approximant.")
        
    def MetricInTheta0Theta3Theta3s(self, theta0, theta3, theta3s, coord_utils, mis_match=1.0):
        """
        This function returns the Fischer information matrix in (theta_0, theta_3, theta_3s, t0, phi0)
        and spatial metric in (theta_0, theta_3, theta_3s) scaled by mismatch.
        
        Parameters:
        -----------
        theta0: float
            0 PN dimensionless chirp time.
        theta3: float
            1.5 PN dimensionless chirp time (spin independent).
        theta3s: float
            1.5 PN dimensionless chirp time (spin dependent) for
            TaylorF2RedSpin approximant, for IMRPhenomB approximant
            Ref: Eq.(3.5) of https://arxiv.org/pdf/1501.04418.pdf
        mis_match: float
            Mismatch 
        
        Returns
        -------
        gamma1 : list of order 1x6
            Containing upper triangular metric elements scaled by mismatch (theta_0, theta_3, theta_3s) 
            gamma1 = [g00, g01, g02, g11, g12, g22]
        gamma2: numpy matrix of order 5x5
            Fischer information matrix in (theta0, theta3, theta3s, t0, phi0)
        """
    
        if self._approximant == "IMRPhenomB":
            mtotal, eta, m1, m2, chieff, chieff = \
                                coord_utils.compute_mass_spin_from_dimensionless_chirptimes(theta0, theta3, theta3s)
            gamma = lalsim.SimIMRPhenomBMetricInTheta0Theta3Theta3S(m1*MSUN_SI, m2*MSUN_SI, chieff, self._fLow, self.Sh)
            
        elif self._approximant == "TaylorF2RedSpin":
            gamma = lalsim.SimInspiralTaylorF2RedSpinMetricChirpTimes(*([theta0, theta3, theta3s, \
                                                                         self._fLow, self._df]+self._moments))
        else:
            raise ValueError("Requested approximant's metric is not available")
        
        # FIXME: Currenetly the analytical metric are not available 
        # in (theta_0, theta_3, theta_3s, t0, phi0).
        gamma2 = []
        return reshape_metric([x/mis_match  for x in gamma]), gamma2
            
    
        
    

waveform_dict = {
                "SEOBNRv4_ROM" : ComputeNumericalMetric,
                "IMRPhenomD" : ComputeNumericalMetric,
                "IMRPhenomXAS" : ComputeNumericalMetric,
                "TaylorF2" : ComputeNumericalMetric,
                "IMRPhenomB" : ObtainAnalyticalMetric,
                "TaylorF2RedSpin" : ObtainAnalyticalMetric
                }





class ComputeNumericalIMRMetric(ComputeNumericalMetric):
    """
    A sub-class to compute the numerical metric for IMR waaveform falies on
    three dimensional parameter space of (mtotal, eta, chi) or (theta0, theta3, theta3s).
    Metric is caluculated assuming single effective spin of the binary system. 
    Assume equal-aligned-spin binary when both the individual masses are greater 
    then NSBH boundary mass or both the masses are below that boundary, 
    chieff = chi. For NSBH systems, we assume BH is spinning only, 
    chi_eff = m1*chi/(m1+m2).
    
    
    Parameters
    ----------
    fref : float
        Reference frequency. Unit : Hz
    flow : float 
        Lower cutoff frequency. Unit: Hz.
    fhigh : float
        Higher cutoff frequency. Unit: Hz.
    df : float
        Frequency sampling. Unit: Hz.
    ASD : numpy array of order 1xn
        Amplitude spectral density of detector noise. Unit: sqrt(Hz).
    nsbh_bound : float
        NSBH bounday mass. Unit: Solar mass 
    approximant : str
        Approximant of the waveform model.
    
    
    Examples
    --------
    >>> import numpy as np
    >>> from psds import lalsim_psd_generator
    >>> from metric import ComputeNumericalIMRMetric
    >>> 
    >>> flow = 20.0
    >>> fref = 20.0
    >>> fhigh = 1024.0
    >>> df = 0.1
    >>> 
    >>> psd_name = "aLIGOZeroDetHighPower"
    >>> 
    >>> length = int(fhigh/df + 1)
    >>> 
    >>> psd_func = lalsim_psd_generator(psd_name)
    >>> psd = psd_func(np.arange(length)*df)
    >>> psd[0] = np.inf
    >>> ASD = np.sqrt(psd)
    >>> 
    >>> gm = ComputeNumericalIMRMetric(fref, flow, fhigh, df, ASD, approximant='SEOBNRv4_ROM')
    >>> print(gm.Theta0Theta3Theta3s(20.0, 0.2, 0.5) )
        
    """
    
    def __init__(self, fref, flow, fhigh, df, ASD, nsbh_bound=2.0, \
                 delta_param = 0.0005, approximant='SEOBNRv4_ROM'):
        
        ComputeNumericalMetric.__init__(self, 0.0, fref, flow, fhigh, df, ASD, delta_param = delta_param,\
                                     nsbh_bound=nsbh_bound, approximant=approximant)
    
    def _compute_dimensionless_chirptimes_from_mass_spin(self, mtotal, eta, chi):
        """
        Compute the mass and spin from dimensionless chirptimes.
        
        Parameters
        ----------
        mtotal : float
            Total mass of the binary system. Unit: Solar mass 
        eta : float
            Symmetric mass ratio.
        chi: float
             Effective spin of the binary system, treat equal-aligned-spin binary
             when both the individual masses are greater then NSBH boundary 
             otherwise treat NSBH system with BH is spinning only.
        
        Returns
        -------
        theta0: float
            0PN dimensionless chirptime
        theta3: float
            1.5PN dimensionless chirptime (spin independent term only)
        theta3s:
            1.5PN dimnesionless chirptime (spin dependent term only)
        """
        
        M = mtotal*MTSUN_SI
        
        if self._mass2 >= self._nsbh_bound or self._mass1 <= self._nsbh_bound:
            chir = (1.0 - 76.0*eta/113.0)*chi
        else:
            chir = (1.0 + (1.0 -4*eta)**0.5 - 76.0*eta/113.0)*chi*0.5
        
        v0 = (PI * M * self._flow)**one_by_three
        v0_square = v0*v0
        v0_quant = v0_square * v0_square * v0
        theta0 = 5.0/(128.0*eta*v0_quant)
        theta3 = PI/(4.0*eta*v0_square)
        theta3s = 113*chir*theta3/(48*PI)
        
        return theta0, theta3, theta3s 

    
    
    def Theta0Theta3Theta3sT0Phi0(self, mtotal, eta, chi):
        """
        Compute PhenomD metric in (theta0, theta3, theta3s, t0, phi0) coordinate.
        
        Parameters
        ----------
        mtotal : float
            Total mass of the binary system. Unit: Solar mass 
        eta : float
            Symmetric mass ratio.
        chi: float
             Effective spin of the binary system, treat equal-aligned-spin binary
             when both the individual masses are greater then NSBH boundary 
             otherwise treat NSBH system with BH is spinning only.
             
        Return
        ------
        FGamma : numpy.array of order 5x5
            Fischer information matrix.
        """
        self._mass1, self._mass2 = mtotal_eta_to_mass1_mass2(mtotal, eta)
        theta0, theta3, theta3s = self._compute_dimensionless_chirptimes_from_mass_spin(mtotal, eta, chi)
        
        g_ij = self._compute_metric_in_mtotal_eta_chi_t0_phi0(mtotal ,eta, chi)
        
        # Transform g_ij(mtotal, eta, chi, t0, phi0) --> g(theta_0, theta_3, theta_3s, t0, phi0) 
        FGamma = self._Jacobian_transformation(g_ij, mtotal, eta, chi, theta0, theta3, theta3s)
        return FGamma
    
    def Theta0Theta3Theta3s(self, mtotal, eta, chi):
        """
        Compute PhenomD metric in (theta0, theta3, theta3s) coordinate.
        
        Parameters
        ----------
        mtotal : float
            Total mass of the binary system. Unit: Solar mass 
        eta : float
            Symmetric mass ratio.
        chi: float
             Effective spin of the binary system, treat equal-aligned-spin binary
             when both the individual masses are greater then NSBH boundary mass
             otherwise treat NSBH system with BH is spinning only.
             
        Return
        ------
        PGamma : numpy.array of order 3x3
            Spatial metric on (theta0, theta3, theta3s) coordinate.
        """
        self._mass1, self._mass2 = mtotal_eta_to_mass1_mass2(mtotal, eta)
        theta0, theta3, theta3s = self._compute_dimensionless_chirptimes_from_mass_spin(mtotal, eta, chi)
        
        # 
        g_ij = self._compute_metric_in_mtotal_eta_chi_t0_phi0(mtotal ,eta, chi)
        
        # Transform g_ij(mtotal, eta, chi, t0, phi0) --> g(theta_0, theta_3, theta_3s, t0, phi0) 
        FGamma = self._Jacobian_transformation(g_ij, mtotal, eta, chi, theta0, theta3, theta3s)
        
        # Projection over t0 and phi0
        PGamma = projection(FGamma, 3)
        return PGamma
    
    
    
