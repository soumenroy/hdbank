from __future__ import division
import sys
import random
import numpy as np
import bisect
from scipy import spatial
from multiprocessing import Process, Queue, Manager, Array, RawArray
import ctypes
import logging

import lal
import lalsimulation as lalsim
from lalinspiral.sbank.tau0tau3 import urand_tau0tau3_generator, set_default_constraints, mtotal_eta_to_mass1_mass2
from lalinspiral import CreateSBankWorkspaceCache, InspiralSBankComputeMatch
from .utils import eta_to_mratio, compute_metric_match, find_neighbors, mass1_mass2_to_mtotal_eta
from .utils import compute_delta_theta0_theta3_of_bounding_box, find_neighbors


import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

__author__ = "Soumen Roy <soumen.roy@ligo.org>"



def GenrandomPointsOverTheta0Theta3Theta3S(args, coord_utils):
    """
    This function generates the uniform random points in dimensionless
    chirptime coordinates.
    
    Parameters
    ----------
    args : argparse.Values instance
        Result of parsing the input options with OptionParser
    coord_utils : callable object
        Contain the class CoordinateTransformUtils to convert the 
        dimensionless chirptimes to mass spin and vice-versa and
        also depending on the approximant.
    
    Return
    ------
    rList : list of order nx3
        List of uniform random points over dimensionless
        chirptime coordinates.
    """
    constraints = {"mass1": (args.min_mass1, args.max_mass1), \
                   "mass2": (args.min_mass2, args.max_mass2)}
    if args.min_total_mass != None:
        constraints["mtotal"] = (args.min_total_mass, args.max_total_mass)
    if args.min_chirp_mass != None:
        constraints["mchirp"] = (args.min_chirp_mass, args.max_chirp_mass)
    if args.min_eta != None:
        mratio_min = eta_to_mratio(args.min_eta)
        constraints["mratio"] = (eta_to_mratio(args.max_eta), mratio_min)
    constraints = set_default_constraints(constraints)
    
    massVals = urand_tau0tau3_generator(args.flow, **constraints)
    mass = [massVals.__next__() for _ in range(args.random_list_size)]
    rList = []
    for i in range(args.random_list_size):
        
        if args.approximant in ["SEOBNRv4_ROM", "IMRPhenomD"]:
            # Apply the appropriate spin bounds
            min_spin, max_spin = coord_utils._spin_bounds(mass[i][0], mass[i][1])
            mtotal = mass[i][0] + mass[i][1]
            eta = mass[i][0]*mass[i][1]/mtotal**2.0
            
            if mass[i][1] <= args.nsbh_boundary_mass and mass[i][0] >= args.nsbh_boundary_mass:
                chir_max = lalsim.SimInspiralTaylorF2ReducedSpinComputeChi(mass[i][0], mass[i][1], max_spin, 0.0)
                chir_min = lalsim.SimInspiralTaylorF2ReducedSpinComputeChi(mass[i][0], mass[i][1], min_spin, 0.0)
                    
            else:
                chir_max = lalsim.SimInspiralTaylorF2ReducedSpinComputeChi(mass[i][0], mass[i][1], max_spin, max_spin)
                chir_min = lalsim.SimInspiralTaylorF2ReducedSpinComputeChi(mass[i][0], mass[i][1], min_spin, min_spin)
                
            chi_r = np.random.mtrand.uniform(chir_min, chir_max)
            spin1z, spin2z = coord_utils._compute_spin_from_chired_eta(mass[i][0], mass[i][1], eta, chi_r)   
            
            rList.append(coord_utils._compute_dimensionless_chirptimes_from_mass_spin(mass[i][0], mass[i][1], spin1z, spin2z))
            
        elif args.approximant == "TaylorF2RedSpin":
            # Apply the appropriate spin bounds
            spin1z_min, spin1z_max, spin2z_min, spin2z_max = coord_utils._spin_bounds[args.system_type](mass[i][0], mass[i][1])
            
            mtot = mass[i][0] + mass[i][1]
            chis_min = (mass[i][0]*spin1z_min + mass[i][1]*spin2z_min)/mtot
            chis_max = (mass[i][0]*spin1z_max + mass[i][1]*spin2z_max)/mtot
            chis = np.random.mtrand.uniform(chis_min, chis_max)
            
            s2min = max(spin2z_min, (mtot*chis - mass[i][0]*spin1z_max)/mass[i][1])
            s2max = min(spin2z_max, (mtot*chis - mass[i][0]*spin1z_min)/mass[i][1])
            
            spin2 = np.random.mtrand.uniform(s2min, s2max)
            spin1 = (chis*mtot - mass[i][1]*spin2)/mass[i][0]
            rList.append(coord_utils.compute_dimensionless_chirptimes_from_mass_spin(mass[i][0], mass[i][1], spin1, spin2))
            
        elif args.approximant == "IMRPhenomB":
            # Apply the appropriate spin bounds
            min_spin, max_spin = coord_utils.chi_limit[args.system_type](mass[i][0], mass[i][1])
            
            spin1 = np.random.mtrand.uniform(min_spin, max_spin)
            spin2 = np.random.mtrand.uniform(min_spin, max_spin)
            rList.append(coord_utils.compute_dimensionless_chirptimes_from_mass_spin(mass[i][0], mass[i][1], spin1, spin2))
            
        else: 
            raise ValueError("")
    
    # Add seed of random points list to rList
    # any random point outside the parameter space can not be a part of template bank
    # ASCII files are allowed with columns mass1, mass2, spin1z, spin2z
    
    if len(args.rlist_seed)>0:
        
        for file in args.rlist_seed:
            mass_spin = np.loadtxt(file)
            for i in range(len(mass_spin)):
                rList.append(coord_utils._compute_dimensionless_chirptimes_from_mass_spin(mass_spin[i, 0], \
                                                         mass_spin[i, 1], mass_spin[i,2], mass_spin[i,3]))
        
    return rList


class DeleteRandomPoints(object):
    """
    This class holds the uniformly generated random points. After each iteration
    of geometric placement, deletes the random points which are inside the 
    minimal match surfaces of templates. 
    
    Parameters
    ----------
    fLow : float
        Lower cutoff frequency. Unit: Hz
    rList : list of order nx3
        List of the uniform random points over dimensionless
        PN chirptimes coordinate. Assumed to be sorted along theta_0.
    tList : list of order nx3
        List of the template points over dimensionless
        PN chirptimes coordinate. Assumed to be sorted along theta_0.
    temp_bin_size : int, optional
        Number of templtes per each bin. Used to delete the rList using
        KDTree data structure.
    neighbourhood_size : float, optional
        Neighbors (along tau_0) are must be within neighbourhood_size. 
    """    
    def __init__(self, fLow, rList, neighbourhood_size=0.5, npts=1000):
        self.fLow = fLow
        self.init_randpts_size = len(rList)
        self._rList = np.array(rList)
        self._tList = np.array([])
        self._mList = np.array([])
        self.temp_bin_size = npts
        self.nhood_size = 2.0 * lal.PI * fLow *neighbourhood_size
        
    @property   
    def __len__(self):
        return len(self._rList)
    
    def _main_delete_function(self, tList, mList):
        """
        Main function to delete the rList points which are inside
        the minimal-matched surfaces of the templates.
        
        Parameters
        ----------
        tList : list of order nx3
            Templtes list generated from last the iteration of
            geometric placement. Assumed to be sorted along theta0.
        mList : list of order nx6
            Metric lists of the templates tList.
        """
        self._tList = np.array(tList)[:, :3]
        self._mList = np.array(mList)
        
        if len(self._rList) > self.init_randpts_size/20:
            self._delete_rlist_using_KDTree()
        else:
            self._simple_delete_rlist()
            
    
    def _simple_delete_rlist(self):
        """
        This function used to delete rList when the size of rList reduces to 20 times.
        """
        for k in range(len(self._tList)):
            if len(self._rList) < 200:
                ds_sq = np.array([compute_metric_match(self._rList[i], self._tList[k], self._mList[k])\
                                                               for i in range(self._rList.shape[0])])
                index = np.where(ds_sq <= 1.0)[0]
            else:
                low, high = find_neighbors(self._rList[:,0].tolist(), self._tList[k, 0], self.nhood_size)
                index = []
                for i in range(low, high):
                    if compute_metric_match(self._rList[i], self._tList[k], self._mList[k]) <= 1.0:
                        index.append(i)
            self._rList = np.delete(self._rList, index, axis=0)
        
    
    def _delete_rlist_using_KDTree(self):
        """
        This function deletes the random points \"rlist\" which are
        are inside the minimal match surfaces of templates \"tlist\"
        after binning of templates. Each bin contains \"npts\" points.
        
        Ref. https://arxiv.org/pdf/1702.06771.pdf (Sec III.A.2 and III.A.3)
        """
        
        # Decide the number of templates in each bin.
        if len(self._tList) > self.temp_bin_size:
            nsteps = int(len(self._tList)/self.temp_bin_size) + 2
        else:
            nsteps = 2
        
        # Start and end indices of bins.
        # FIXME: dtype 
        step_index = np.array(np.linspace(0, len(self._tList) - 1, nsteps, endpoint=True), dtype=int)
        
        for j in range(nsteps-1):
            indices = []
            
            # Value of (theta0 -/+ nhood_size) where theta is corresponding to
            # the min/max indices of templates in the j-th bin.
            theta0_min = self._tList[step_index[j]][0] - self.nhood_size
            theta0_max = self._tList[step_index[j+1]][0] + self.nhood_size
            
            # Indices of random points list corresponding corresponding 
            # to (theta0 -/+ nhood_size).
            rlow = bisect.bisect_left(self._rList[:,0].tolist(), theta0_min)
            rhigh = bisect.bisect_right(self._rList[:,0].tolist(), theta0_max)
            
            if rhigh-rlow>0:
                # Take conformal coordinate transformation over the templates 
                # and random points by using the metric of mid template point of the j-th bin.
                rlist_zeta, tlist_zeta, metric_zeta = \
                            self._conformal_coordinate_transformer(step_index[j], step_index[j+1], rlow, rhigh)

                # Construct KDTree to make faster the deletion process.
                rTree = spatial.cKDTree(zip(rlist_zeta[:,0], rlist_zeta[:,1], rlist_zeta[:,2]))

                for i in range(len(tlist_zeta)):

                    # Calculate semi-axes of the minimal-matched ellipsoid.
                    eigval, eigvec = np.linalg.eig(metric_zeta[i])
                    semiaxes = 1.0/np.sqrt(eigval)

                    # Find subset of rList which are within the distace max(semi-axes).
                    index = rTree.query_ball_point(tlist_zeta[i], max(semiaxes))

                    # Compute metric match with points in the subset.
                    delta_lambda = rlist_zeta[index]-tlist_zeta[i]
                    ds_sq = np.array([np.dot(delta_lambda[_], np.dot(metric_zeta[i], delta_lambda[_].T)) \
                                      for _ in range(len(index))])

                    # Find the points inside the minimal-matched surface.
                    indelp = np.where(ds_sq <= 1.0)[0]
                    indices.extend(np.array(index)[indelp]+rlow)

                self._rList =  np.delete(self._rList, list(set(indices)), axis=0)
            
            

     
    def _conformal_coordinate_transformer(self, temp_low, temp_high, rand_low, rand_high):
        """
        This function returns the matrices corresponding to conformal coordinate transformation
        for a given metric. Transform the metric at the temp_list into a unit matrix.

        Coordinate transformation:
        \f$ \theta \to \zeta : \zeta = S * R^{T} * \theta \f$
        
        Metric transformation:
        \f$ \gamma \to \gamma^' : \gamma^' = S^{-1}R^{T} \gamma R S^{-1} \f$
        
        \f$ S \f$ is scaling matrix: \f$ S = \sqrt{\lambda_i} * I_{ij} \f$
        
        Where, \f$ \lambda_i \f$ is i-th eigenvalues of $ \gamma $ and $I_{ij}$ is unit matrix.
        \f$ R \f$ is rotation matrix: $R_{ij}$ is i-th component of j-th eigenvector.
        
        Reference
        ----------
        See Eq.(9) and (10) from https://arxiv.org/pdf/1702.06771.pdf

        """
        if temp_high + 1 == len(self._tList):
            temp_high += 1
        gamma = self._mList[ int((temp_low + temp_high)/2) ]
        eigval, eigvec = np.linalg.eig(gamma)
        S = np.diag(np.sqrt(eigval)); R = eigvec
        # coordinate transformer
        scl_rot_trans = np.dot(S, R.T)

        # metric transformers
        scl_inv_rot_trans = np.dot( np.linalg.inv(S), R.T)
        rot_scl_inv = np.dot(R, np.linalg.inv(S)) 
        
        #rlist_zeta = np.dot(scl_rot_trans, self._rList[rand_low:rand_high].T).T
        #tlist_zeta = np.dot(scl_rot_trans, self._tList[temp_low:temp_high].T).T
        
        rlist_zeta = np.array([np.dot(scl_rot_trans, self._rList[_]) \
                               for _ in range(rand_low, rand_high)])
        
        tlist_zeta = np.array([np.dot(scl_rot_trans, self._tList[_]) \
                               for _ in range(temp_low, temp_high)])
        
        metric_zeta = np.array([np.dot(scl_inv_rot_trans, np.dot(self._mList[_], rot_scl_inv)) \
                                for _ in range(temp_low, temp_high)])
        return rlist_zeta, tlist_zeta, metric_zeta


    
    

# Create workspace for match calculation
workspace_cache = CreateSBankWorkspaceCache()

def _create_multiprocessing_array(arr):
    if len(arr.shape)== 1:
        mp_arr = Array(ctypes.c_double, arr)
        np_arr = np.frombuffer(mp_arr.get_obj())
        
    
    elif len(arr.shape)== 2:
        mp_arr = Array(ctypes.c_double, arr.flatten())
        np_arr = np.frombuffer(mp_arr.get_obj())
        np_arr = np_arr.reshape(arr.shape)
        
    return np_arr


    
class ExactMatchUtilsToDeleteRandomPoints(object):
    """
    This class holds the uniformly generated random points. After each iteration
    of geometric placement, deletes the random points which are at a distance of 
    less than (1.0 - match)^(1/2) from the templates. The match between the templates
    and the random points are computed using exact match function. The computation
    of matches are parallalized.
    
    Parameters
    ----------
    fLow : float
        Lower cutoff frequency. Unit: Hz
    min_match : float
        Minimal match of the bank.
    nProcess : int
        Number of process.
    rList : numpy array of order nx3
        List of the uniform random points over dimensionless
        PN chirptimes coordinate. Assumed to be sorted along theta_0.
    rList_theta0 : list of order nx1
        List of the first column of rList.
    tList : numpy array of order nx3
        List of the template points over dimensionless
        PN chirptimes coordinate. Assumed to be sorted along theta_0.
    mList : numpy array of order nx3x3
        List of metrices of the templates in tList
    
    
    * Provide exact_match_utils which has a function to compute
      whitened waveform in Fourier domain.
      
    * Provide coord_utils which has function to calculate mass 
      and spin values from dimensionless chirptimes.
    """    
    
    def __init__(self, args, exact_match_utils, coord_utils, rList):
        self.fLow = args.flow
        self._min_match = args.min_match
        self.nProcess = int(max(args.number_threads, 4))
        self._rList = _create_multiprocessing_array( np.array(rList) )
        self._rList_theta0 = _create_multiprocessing_array( np.array(rList)[:,0] )
        self._out_arr = []
        self._compute_whiten_normalized_fd_waveform = exact_match_utils._compute_whiten_normalized_fd_waveform
        self._compute_mass_spin_from_dimensionless_chirptimes = coord_utils._compute_mass_spin_from_dimensionless_chirptimes
        
    @property   
    def __len__(self):
        return len(self._rList)
    
    
    def _find_rList_index_near_tList(self, tList, mList):
        """
        This function computes the match between template and random 
        points which are inside the bounding box of the template.
        
        ind_start : int
            Starting index of the tList
        ind_end : int
            Ending index of the tList
        """
        length = len(tList)
        for i in range(length):
            g_ij = mList[i]/6.0
            
            tmp = tList[i]
            tmp_mtotal, tmp_eta, tmp_spin1z, tmp_spin2z, tmp_opt_flow, tmp_dur = tmp[3:]
            tmp_mass1, tmp_mass2 = mtotal_eta_to_mass1_mass2(tmp_mtotal, tmp_eta)
            
            # Compute whitened template waveform
            whiten_tmp_wf = self._compute_whiten_normalized_fd_waveform( \
               tmp_mass1, tmp_mass2, tmp_spin1z, tmp_spin2z, opt_flow=tmp_opt_flow)
             
            # Compute the width of the bounding box along theta0
            delta_theta0 = compute_delta_theta0_theta3_of_bounding_box(g_ij)
            
            low, high = find_neighbors(self._rList_theta0, tmp[0], delta_theta0)

            if high-low > 0:
                delta_theta = self._rList[low:high] - tmp[:3]
                ds_sq = np.sum(delta_theta.T*np.dot(g_ij,delta_theta.T), axis=0)
                indices = np.where(ds_sq < 1)[0] + low
            else: indices = []
            
            for j in indices:
                if self._out_arr[j] == 0:
                    proposal = self._rList[j]
                    prop_mtotal, prop_eta, prop_mass1, prop_mass2, prop_spin1z, prop_spin2z= \
                          self._compute_mass_spin_from_dimensionless_chirptimes( \
                                                     proposal[0], proposal[1], proposal[2])
                
                    try:
                        # Try to generate the whitened waveform of the proposal.
                        # It may fail when the lower cutoff frequency less than the 
                        # ringdown frequnecy.
                        whiten_prop_wf = self._compute_whiten_normalized_fd_waveform( \
                                               prop_mass1, prop_mass2, prop_spin1z, prop_spin2z)
                        
                        overlap = InspiralSBankComputeMatch(whiten_prop_wf, whiten_tmp_wf, workspace_cache)
                        
                    except: overlap = 1.0

                    if overlap >= self._min_match:
                        self._out_arr[j] = 1

        
    def _main_delete_function(self, tList, mList):
        
        self.nProcess = min(self.nProcess, len(tList))
        
        jobIds = np.arange( 0, len(tList),  self.nProcess)
        if jobIds[-1] != len(tList):
            jobIds = np.append( jobIds, len(tList) )

        
                
        for ii in range( len(jobIds)-1 ):
            
            self._out_arr = Array(ctypes.c_int, len(self._rList) )
            
            # Setup the input arguments of parallel processes helper function
            args = [( tList[i:i+1], mList[i:i+1] ) for i in range(jobIds[ii], jobIds[ii+1])]

            # Setup a list of processes
            jobs = [Process(target=self._find_rList_index_near_tList,  args=args[i])  \
                    for i in range(jobIds[ii+1]- jobIds[ii])]

            # Run processes
            for job in jobs:
                job.start()

            # Exit the completed processes
            for job in jobs:
                job.join()

            # Get process results from the output Array
            final_indices = np.nonzero( np.array(self._out_arr.get_obj()[:]) )[0]
            
            # Delete the random points which has match greater equal to minimal match.
            self._rList = np.delete(self._rList, final_indices, axis=0)
            self._rList_theta0 = np.delete(self._rList_theta0, final_indices, axis=0)
            
            string = "Removing rList, Length of rList: %d Number of iterations: %d, Done: %d"%( \
                                                                     len(self._rList), len(jobIds)-1, ii)
            logging.info(string)
        
    
    
    
    
