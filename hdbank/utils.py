from __future__ import division
import sys
from collections import defaultdict
import numpy as np
import scipy
import bisect
import operator
import logging
from time import strftime
from multiprocessing import Process, Queue, Manager

import lalsimulation as lalsim
from lalinspiral.sbank.tau0tau3 import mtotal_eta_to_mass1_mass2
from lalinspiral.sbank.tau0tau3 import mchirp_eta_to_mass1_mass2
from lalinspiral.sbank.tau0tau3 import mchirpm1_to_m2
from lalinspiral.sbank.tau0tau3 import m1m2_to_mchirp
from lalinspiral.sbank.tau0tau3 import tau0tau3_to_m1m2
from lalinspiral.sbank.tau0tau3 import m1m2_to_tau0tau3
from lalinspiral import CreateSBankWorkspaceCache, InspiralSBankComputeMatch

from lal import PI, MSUN_SI, MTSUN_SI, PC_SI, TWOPI, GAMMA, CreateCOMPLEX8FrequencySeries



from ligo.lw import ligolw
from ligo.lw import lsctables
from ligo.lw import utils as ligolw_utils
from ligo.lw.utils import process as ligolw_process
from ligo.lw.ligolw import LIGOLWContentHandler

# from lalmetaio import SnglInspiralTable
from glue.ligolw.lsctables import SnglInspiralTable as gluesit

# Add some small code to initialize columns to 0 or ''
_sit_cols = gluesit.validcolumns
class SnglInspiralTable(gluesit):
    def __init__(self, *args, **kwargs):
        gluesit.__init__(self, *args, **kwargs)
        for entry in _sit_cols.keys():
            if not(hasattr(self,entry)):
                if _sit_cols[entry] in ['real_4','real_8']:
                    setattr(self,entry,0.)
                elif _sit_cols[entry] == 'int_4s':
                    setattr(self,entry,0)
                elif _sit_cols[entry] == 'lstring':
                    setattr(self,entry,'')
                elif _sit_cols[entry] == 'ilwd:char':
                    setattr(self,entry,'')
            else:
                print("Column %s not recognized" %(entry), file=sys.stderr)
                raise ValueError




import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="invalid value encountered in power")
warnings.filterwarnings("ignore", message="invalid value encountered in less_equal")
warnings.filterwarnings("ignore", message="invalid value encountered in greater_equal")


__author__ = "Soumen Roy <soumen.roy@ligo.org>"




PI_square = PI*PI
PI_quant = PI_square * PI_square * PI
one_by_three = 1.0/3.0

def eta_to_mratio(eta):
    """
    Compute mass ratio from symmetric mass ratio.
    
    Parameter
    ---------
    eta : float
        Symmetric mass ratio of the system.
    
    Return
    ------
    mratio : float
        Mass ratio of the system.
    """
    mratio = 0.5*(1.0 - 2.0*eta + np.sqrt(1.0 - 4.0*eta))/eta
    return mratio

def mass1_mass2_to_mtotal_eta(mass1, mass2):
    """
    Compute total mass and symmetric mass ration from 
    individual masses.
    
    Parameters
    ----------
    mass1 : float
        Mass of the first object. Unit : Solar mass
    mass2 : float
        Mass of the second object. Unit : Solar mass
    
    Returns
    -------
    mtotal : float
        Total mass of the system. Unit : Solar mass
    eta : float
        Symmetric mass ratio of the system.
    """
    mtotal = mass1 + mass2
    eta = mass1*mass2/(mtotal*mtotal)
    return mtotal, eta


def mass_from_knownmass_eta(eta, known_mass, known_is_secondary=False,
                            force_real=True):
    """
    Returns the other component mass given one of the component masses
    and the symmetric mass ratio.

    This requires finding the roots of the quadratic equation:

    .. math::
        \f$ \eta m_2^2 + (2\eta - 1)m_1 m_2 + \eta m_1^2 = 0. \f$

    This has two solutions which correspond to :math:\f$ m_1 \f$ being the heavier
    mass or it being the lighter mass. By default, "known_mass" is assumed to
    be the heavier (primary) mass, and the smaller solution is returned. Use
    the "other_is_secondary" to invert.
    
    Ref: pycbc conversions module

    Parameters
    ----------
    known_mass : float
        The known component mass.
    eta : float
        The symmetric mass ratio.
    known_is_secondary : {False, bool}
        Whether the known component mass is the primary or the secondary. If
        True, "known_mass" is assumed to be the secondary (lighter) mass and
        the larger solution is returned. Otherwise, the smaller solution is
        returned. Default is False.
    force_real : {True, bool}
        Force the returned mass to be real.

    Returns
    -------
    float
        The other component mass.
    """
    roots = np.roots([eta, (2*eta - 1)*known_mass, eta*known_mass**2.])
    if force_real:
        roots = np.real(roots)
    if known_is_secondary:
        return roots[roots.argmax()]
    else:
        return roots[roots.argmin()]


def projection(gamma, dim):
    """
    Projection of metric using Schur Complement.
    """
    gamma1 = gamma[:dim, :dim]
    gamma2 = gamma[dim:, :dim]
    gamma4 = gamma[dim:, dim:]
    return gamma1 - np.dot(gamma2.T, np.dot(np.linalg.inv(gamma4), gamma2))






def verify_argparse_options(opts, parser, nonSpin=False):
    """
    Verify the provided mass spin ranges from parser input argument.
    Verification of mass ranges are modified from pycbc function 
    "verify_mass_range_options"

    Parameters
    ----------
    opts : argparse.Values instance
        Result of parsing the input options with OptionParser
    parser : object
        The OptionParser instance.
    nonSpin : boolean, optional (default=False)
        If this is provided the spin-related options will not be checked.
    """
    
    if opts.fref == None:
        opts.fref = opts.flow

    if nonSpin == True:
        warning.warn("Current version is not available for non-spinning system", DeprecationWarning)
        
    # Mass1 must be the heavier!
    if opts.min_mass1 < opts.min_mass2:
        parser.error("min-mass1 cannot be less than min-mass2!")
    if opts.max_mass1 < opts.max_mass2:
        parser.error("max-mass1 cannot be less than max-mass2!")
    # If given are min/max total mass/chirp mass possible?
    if opts.min_total_mass \
            and (opts.min_total_mass > opts.max_mass1 + opts.max_mass2):
        err_msg = "Supplied minimum total mass %f " %(opts.min_total_mass,)
        err_msg += "greater than the sum of the two max component masses "
        err_msg += " %f and %f." %(opts.max_mass1,opts.max_mass2)
        parser.error(err_msg)
    if opts.max_total_mass \
            and (opts.max_total_mass < opts.min_mass1 + opts.min_mass2):
        err_msg = "Supplied maximum total mass %f " %(opts.max_total_mass,)
        err_msg += "smaller than the sum of the two min component masses "
        err_msg += " %f and %f." %(opts.min_mass1,opts.min_mass2)
        parser.error(err_msg)
    if opts.max_total_mass and opts.min_total_mass \
            and (opts.max_total_mass < opts.min_total_mass):
        parser.error("Min total mass must be larger than max total mass.")
   

    # Assign min/max total mass from mass1, mass2 if not specified
    if (not opts.min_total_mass) or \
            ((opts.min_mass1 + opts.min_mass2) > opts.min_total_mass):
        opts.min_total_mass = opts.min_mass1 + opts.min_mass2
    if (not opts.max_total_mass) or \
            ((opts.max_mass1 + opts.max_mass2) < opts.max_total_mass):
        opts.max_total_mass = opts.max_mass1 + opts.max_mass2

    # It is vital that min and max total mass be set correctly.
    # This is becasue the heavily-used function get_random_mass will place
    # points first in total mass (to some power), and then in eta. If the total
    # mass limits are not well known ahead of time it will place unphysical
    # points and fail.
    # This test is a bit convoluted as we identify the maximum and minimum
    # possible total mass from chirp mass and/or eta restrictions.
    if opts.min_chirp_mass is not None:
        # Need to get the smallest possible min_tot_mass from this chirp mass
        # There are 4 possibilities for where the min_tot_mass is found on the
        # line of min_chirp_mass that interacts with the component mass limits.
        # Either it is found at max_m2, or at min_m1, or it starts on the equal
        # mass line within the parameter space, or it doesn't intersect
        # at all.
        # First let's get the masses at both of these possible points
        m1_at_max_m2 = mchirpm1_to_m2(opts.min_chirp_mass,
                                                     opts.max_mass2)
        if m1_at_max_m2 < opts.max_mass2:
            # Unphysical, remove
            m1_at_max_m2 = -1
        m2_at_min_m1 = mchirpm1_to_m2(opts.min_chirp_mass,
                                                      opts.min_mass1)
        if m2_at_min_m1 > opts.min_mass1:
            # Unphysical, remove
            m2_at_min_m1 = -1
        # Get the values on the equal mass line
        m1_at_equal_mass, m2_at_equal_mass = mchirp_eta_to_mass1_mass2(
                                                     opts.min_chirp_mass, 0.25)

        # Are any of these possible?
        if m1_at_max_m2 <= opts.max_mass1 and m1_at_max_m2 >= opts.min_mass1:
            min_tot_mass = opts.max_mass2 + m1_at_max_m2
        elif m2_at_min_m1 <= opts.max_mass2 and m2_at_min_m1 >= opts.min_mass2:
            min_tot_mass = opts.min_mass1 + m2_at_min_m1
        elif m1_at_equal_mass <= opts.max_mass1 and \
                 m1_at_equal_mass >= opts.min_mass1 and \
                 m2_at_equal_mass <= opts.max_mass2 and \
                 m2_at_equal_mass >= opts.min_mass2:
            min_tot_mass = m1_at_equal_mass + m2_at_equal_mass
        # So either the restriction is low enough to be redundant, or is
        # removing all the parameter space
        elif m2_at_min_m1 < opts.min_mass2:
            # This is the redundant case, ignore
            min_tot_mass = opts.min_total_mass
        else:
            # And this is the bad case
            err_msg = "The minimum chirp mass provided is not possible given "
            err_msg += "restrictions on component masses."
            raise ValueError(err_msg)
        # Is there also an eta restriction?
        if opts.max_eta:
            # Get the value of m1,m2 at max_eta, min_chirp_mass
            max_eta_m1, max_eta_m2 = mchirp_eta_to_mass1_mass2(
                                         opts.min_chirp_mass, opts.max_eta)
            max_eta_min_tot_mass = max_eta_m1 + max_eta_m2
            if max_eta_min_tot_mass > min_tot_mass:
                # Okay, eta does restrict this further. Still physical?
                min_tot_mass = max_eta_min_tot_mass
                if max_eta_m1 > opts.max_mass1:
                    err_msg = "The combination of component mass, chirp "
                    err_msg += "mass, eta and (possibly) total mass limits "
                    err_msg += "have precluded all systems."
                    raise ValueError(err_msg)
        # Update min_tot_mass if needed
        if min_tot_mass > opts.min_total_mass:
            opts.min_total_mass = float(min_tot_mass)

    # Then need to do max_chirp_mass and min_eta
    if opts.max_chirp_mass is not None:
        # Need to get the largest possible maxn_tot_mass from this chirp mass
        # There are 3 possibilities for where the max_tot_mass is found on the
        # line of max_chirp_mass that interacts with the component mass limits.
        # Either it is found at min_m2, or at max_m1, or it doesn't intersect
        # at all.
        # First let's get the masses at both of these possible points
        m1_at_min_m2 = mchirpm1_to_m2(opts.max_chirp_mass,
                                                     opts.min_mass2)
        m2_at_max_m1 = mchirpm1_to_m2(opts.max_chirp_mass,
                                                      opts.max_mass1)
        # Are either of these possible?
        if m1_at_min_m2 <= opts.max_mass1 and m1_at_min_m2 >= opts.min_mass1:
            max_tot_mass = opts.min_mass2 + m1_at_min_m2
        elif m2_at_max_m1 <= opts.max_mass2 and m2_at_max_m1 >= opts.min_mass2:
            max_tot_mass = opts.max_mass1 + m2_at_max_m1
        # So either the restriction is low enough to be redundant, or is
        # removing all the paramter space
        elif m2_at_max_m1 > opts.max_mass2:
            # This is the redundant case, ignore
            max_tot_mass = opts.max_total_mass
        else:
            # And this is the bad case
            err_msg = "The maximum chirp mass provided is not possible given "
            err_msg += "restrictions on component masses."
            raise ValueError(err_msg)
        # Is there also an eta restriction?
        if opts.min_eta:
            # Get the value of m1,m2 at max_eta, min_chirp_mass
            min_eta_m1, min_eta_m2 = mchirp_eta_to_mass1_mass2(
                                         opts.max_chirp_mass, opts.min_eta)
            min_eta_max_tot_mass = min_eta_m1 + min_eta_m2
            if min_eta_max_tot_mass < max_tot_mass:
                # Okay, eta does restrict this further. Still physical?
                max_tot_mass = min_eta_max_tot_mass
                if min_eta_m1 < opts.min_mass1:
                    err_msg = "The combination of component mass, chirp "
                    err_msg += "mass, eta and (possibly) total mass limits "
                    err_msg += "have precluded all systems."
                    raise ValueError(err_msg)
        # Update min_tot_mass if needed
        if max_tot_mass < opts.max_total_mass:
            opts.max_total_mass = float(max_tot_mass)

    # Need to check max_eta alone for minimum and maximum mass
    if opts.max_eta:
        # Similar to above except this can affect both the minimum and maximum
        # total mass. Need to identify where the line of max_eta intersects
        # the parameter space, and if it affects mass restrictions.
        m1_at_min_m2 = mass_from_knownmass_eta(opts.max_eta, opts.min_mass2,
                                               known_is_secondary=True, force_real=True)
                                                      
        m2_at_min_m1 = mass_from_knownmass_eta(opts.max_eta, opts.min_mass1,
                                                  known_is_secondary=False, force_real=True)
                                                     
        m1_at_max_m2 = mass_from_knownmass_eta(opts.max_eta, opts.max_mass2,
                                                  known_is_secondary=True, force_real=True)
                                                      
        m2_at_max_m1 = mass_from_knownmass_eta(opts.max_eta, opts.max_mass1,
                                                  known_is_secondary=False, force_real=True)
                                                      
        # Check for restrictions on the minimum total mass
        # Are either of these possible?
        if m1_at_min_m2 <= opts.max_mass1 and m1_at_min_m2 >= opts.min_mass1:
            min_tot_mass = opts.min_mass2 + m1_at_min_m2
        elif m2_at_min_m1 <= opts.max_mass2 and m2_at_min_m1 >= opts.min_mass2:
            # This case doesn't change the minimal total mass
            min_tot_mass = opts.min_total_mass
        # So either the restriction is low enough to be redundant, or is
        # removing all the paramter space
        elif m2_at_min_m1 > opts.max_mass2:
            # This is the redundant case, ignore
            min_tot_mass = opts.min_total_mass
        elif opts.max_eta == 0.25 and (m1_at_min_m2 < opts.min_mass2 or \
                                                m2_at_min_m1 > opts.min_mass1): 
            # This just catches potential roundoff issues in the case that
            # max-eta is not used
            min_tot_mass = opts.min_total_mass
        else:
            # And this is the bad case
            err_msg = "The maximum eta provided is not possible given "
            err_msg += "restrictions on component masses."
            print(m1_at_min_m2, m2_at_min_m1, m1_at_max_m2, m2_at_max_m1)
            print(opts.min_mass1, opts.max_mass1, opts.min_mass2, opts.max_mass2)
            raise ValueError(err_msg)
        # Update min_tot_mass if needed
        if min_tot_mass > opts.min_total_mass:
            opts.min_total_mass = float(min_tot_mass)

        # Check for restrictions on the maximum total mass
        # Are either of these possible?
        if m2_at_max_m1 <= opts.max_mass2 and m2_at_max_m1 >= opts.min_mass2:
            max_tot_mass = opts.max_mass1 + m2_at_max_m1
        elif m1_at_max_m2 <= opts.max_mass1 and m1_at_max_m2 >= opts.min_mass1:
            # This case doesn't change the maximal total mass
            max_tot_mass = opts.max_total_mass
        # So either the restriction is low enough to be redundant, or is
        # removing all the paramter space, the latter case is already tested
        else:
            # This is the redundant case, ignore
            max_tot_mass = opts.max_total_mass
        if max_tot_mass < opts.max_total_mass:
            opts.max_total_mass = float(max_tot_mass)

    # Need to check min_eta alone for maximum and minimum total mass
    if opts.min_eta:
        # Same as max_eta.
        # Need to identify where the line of max_eta intersects
        # the parameter space, and if it affects mass restrictions.
        m1_at_min_m2 = mass_from_knownmass_eta(opts.min_eta, opts.min_mass2,
                                               known_is_secondary=True, force_real=True)
                                                      
        m2_at_min_m1 = mass_from_knownmass_eta(opts.min_eta, opts.min_mass1,
                                               known_is_secondary=False, force_real=True)
                                                     
        m1_at_max_m2 = mass_from_knownmass_eta(opts.min_eta, opts.max_mass2,
                                               known_is_secondary=True, force_real=True)
                                                      
        m2_at_max_m1 = mass_from_knownmass_eta(opts.min_eta, opts.max_mass1,
                                               known_is_secondary=False, force_real=True)
                                                      

        # Check for restrictions on the maximum total mass
        # Are either of these possible?
        if m1_at_max_m2 <= opts.max_mass1 and m1_at_max_m2 >= opts.min_mass1:
            max_tot_mass = opts.max_mass2 + m1_at_max_m2

        elif m2_at_max_m1 <= opts.max_mass2 and m2_at_max_m1 >= opts.min_mass2:
            # This case doesn't affect the maximum total mass
            max_tot_mass = opts.max_total_mass
        # So either the restriction is low enough to be redundant, or is
        # removing all the paramter space
        elif m2_at_max_m1 < opts.min_mass2:
            # This is the redundant case, ignore
            max_tot_mass = opts.max_total_mass
        else:
            # And this is the bad case
            err_msg = "The minimum eta provided is not possible given "
            err_msg += "restrictions on component masses."
            raise ValueError(err_msg)
        # Update min_tot_mass if needed
        if max_tot_mass < opts.max_total_mass:
            opts.max_total_mass = float(max_tot_mass)

        # Check for restrictions on the minimum total mass
        # Are either of these possible?
        if m2_at_min_m1 <= opts.max_mass2 and m2_at_min_m1 >= opts.min_mass2:
            min_tot_mass = opts.min_mass1 + m2_at_min_m1
        elif m1_at_min_m2 <= opts.max_mass1 and m1_at_min_m2 >= opts.min_mass1:
            # This case doesn't change the maximal total mass
            min_tot_mass = opts.min_total_mass
        # So either the restriction is low enough to be redundant, or is
        # removing all the paramter space, which is tested above
        else:
            # This is the redundant case, ignore
            min_tot_mass = opts.min_total_mass
        if min_tot_mass > opts.min_total_mass:
            opts.min_total_mass = float(min_tot_mass)

    if opts.max_total_mass < opts.min_total_mass:
        err_msg = "After including restrictions on chirp mass, component mass, "
        err_msg += "eta and total mass, no physical systems are possible."
        raise ValueError(err_msg)

    if opts.max_eta and opts.min_eta and (opts.max_eta < opts.min_eta):
        parser.error("--max-eta must be larger than --min-eta.")
    if nonSpin:
        return
    
    
    # Warning or error message for IMRPhenomB approximant.
    message = """IMRPhenomB model is reliable within spin limit
                   -0.85 <= chi_eff <= 0.85 for mass ratio q <= 4,
                   and -0.5 <= chi_eff <= 0.75 for mass ratio 4 < q <= 10."""
    
    
    # Verify the provided spin ranges for SEOBNRv4_ROM, IMRPhenomD or IMRPhenomB approximant.
    if opts.approximant in ("SEOBNRv4_ROM", "IMRPhenomD", "IMRPhenomB"):
        if opts.min_spin1 != None or opts.max_spin1 != None \
                or opts.min_spin2 != None or opts.max_spin2 != None:
            parser.error("--min/max-spin1/2 are not allowed for %s approximant." % opts.approximant)
            
        if (opts.min_ns_spin != None and opts.max_ns_spin == None) or (opts.min_ns_spin == None and opts.max_ns_spin != None):
            parser.error("Must supply both the --min/max-bh-spin.")
                
        if (opts.min_bh_spin != None and opts.max_bh_spin == None) or(opts.min_bh_spin == None and opts.max_bh_spin != None):
            parser.error("Must supply both the --min/max-bh-spin .")
        
        
    # Verify the provided spin ranges for IMRPhenomB approximant.
    elif opts.approximant == "IMRPhenomB" and opts.PhenomB_spin_truncate:
        
        if opts.min_eta != None and opts.max_mtotal == None:
            q = eta_to_mratio(opts.min_eta)
            
            # PhenomB is not reliable for Mass ratio > 10
            if q > 10.0:
                parser.error(message)
                
        elif opts.max_mtotal != None:
            min_mass2 = max(opts.min_mass2, opts.min_mtotal-opts.min_mass1)
            max_mass1 = min(opts.max_mass1, opts.max_mtotal-opts.min_mass2)
            q = max_mass1/min_mass2
            if opts.min_eta != None:
                q = min(q, eta_to_mratio(opts.min_eta))
        else:
            q = opts.max_mass1/opts.min_mass2
        
        if q > 10:
            parser.error(message)
        
        elif q <= 10.0 :
            if (opts.min_bh_spin != None and opts.min_bh_spin < -0.85) or \
               (opts.max_bh_spin != None and opts.max_bh_spin > 0.85)  or \
               (opts.min_ns_spin != None and opts.min_ns_spin < -0.85) or \
               (opts.max_ns_spin != None and opts.max_ns_spin > 0.85):
                    parser.error(message)
                    
        elif 10 >= q >= 4.0:
            if (opts.min_bh_spin != None and opts.min_bh_spin < -0.5) or \
               (opts.max_bh_spin != None and opts.max_bh_spin > 0.75)  or \
               (opts.min_ns_spin != None and opts.min_ns_spin < -0.5) or \
               (opts.max_ns_spin != None and opts.max_ns_spin > 0.75):
                    warning.warn(message + "So we will silently truncate over spin bounds.")
                    
    elif opts.approximant == "IMRPhenomB" and opts.PhenomB_spin_truncate != True:
        warning.warn(message + "So bank performance may not be adequate.")
    
    elif opts.approximant != "IMRPhenomB" and opts.PhenomB_spin_truncate:
        parser.error("--PhenomB-spin-truncate not allowed for %s approximant." % opts.approximant)
              
    
    # Verify the provided spin ranges for TaylorF2RedSpin approximant.
    elif opts.approximant == "TaylorF2RedSpin":
        if (opts.min_spin1 == None or opts.max_spin1 == None 
            or opts.min_spin2 == None or opts.max_spin2 == None):
            parser.error("Must supply --min/max-spin1/2 values for %s approximant." % opts.approximant)
            
    else:
        parser.error("Current version does not allowed %s approximant." % opts.approximant)
        
                

def set_boundary_constraints(args):
    """
    This function returns the constraints for boundary conditions
    from the argument of parser (args).
    """
    constraints = {"mass1": (args.min_mass1, args.max_mass1), \
                   "mass2": (args.min_mass2, args.max_mass2)}
    if args.min_total_mass != None:
        constraints["mtotal"] = (args.min_total_mass, args.max_total_mass)
    else:
        constraints["mtotal"] = (args.max_mass1+args.max_mass2, args.min_mass1+args.min_mass2)
    if args.min_chirp_mass != None:
        constraints["mchirp"] = (args.min_chirp_mass, args.max_chirp_mass)
    if args.min_eta != None:
        constraints["eta"] = (args.min_eta, args.max_eta)
    else:
        constraints["eta"] = (mass1_mass2_to_mtotal_eta(args.max_mass1, args.min_mass2)[1], 0.25)
    if args.min_duration != None:
        if args.max_duration != None:
            max_dur = args.max_duration
        else:
            max_dur = np.inf
        constraints["duration"] = (args.min_duration, max_dur)
    return constraints



def set_starting_position(args, rList, coord_utils):
    """
    Set the starting point of the template placement.
    
    Parameters
    ----------
    args : argparse.Values instance
        Result of parsing the input options with OptionParser.
    rList : list of order nx3
        Uniformly generated random points in dimensionless 
        PN chirtime coordinate.
    bndry_utils : object
        Contain the class CheckBoundary to access the boundary conditions
        and also depending on the approximant.
    coord_utils : object
        Contain the class CoordinateTransformUtils to convert the 
        dimensionless chirptimes to mass spin and vice-versa and
        also depending on the approximant.
    
    """
    if len(args.starting_point) == 2:
        mass1, mass2 = args.starting_point
        mtotal, eta = mass1_mass2_to_mtotal_eta(mass1, mass2)
    
    else:
        # Find almost ceneter points of rList over theta_0 and theta_3
        theta0_min, theta0_max = min(rList[:,0]), max(rList[:,0])
        theta0_mid = (theta0_min + theta0_max)/2.5
        if len(rList) >= 1000:
            delta_theta0 = (theta0_max-theta0_min)/100
        else:
            delta_theta0 = (theta0_max-theta0_min)/10
    
        ind = np.where((rList[:,0]>(theta0_mid-delta_theta0)) & (rList[:,0]<(theta0_mid+delta_theta0)))[0]
        ind2 = bisect.bisect_left(rList[:,1][ind].tolist(), np.mean(rList[:,1][ind]))
        theta0 = rList[:,0][ind[ind2]]
        theta3 = rList[:,1][ind[ind2]]
    
        # Convert into masses
        mtotal, eta = coord_utils._theta0_theta3_to_mtotal_eta(theta0, theta3)
        mass1, mass2 = mtotal_eta_to_mass1_mass2(mtotal, eta)
    
        
        
    if args.approximant == "TaylorF2RedSpin":
        spin1z_min, spin1z_max, spin2z_min, spin2z_max = coord_utils._spin_bounds(mass1, mass2)
        spin1z = 0.5*(spin1z_min + spin1z_max)
        spin2z = 0.5*(spin2z_min + spin2z_max)
        chi = (mass1*spin1z + mass2*spin2z)/(mass1+mass2)
    else:
        # Average spin
        chi_min, chi_max = coord_utils._spin_bounds(mass1, mass2)
        chi = (chi_min + chi_max)/2.0
        if mass2 >= args.nsbh_boundary_mass or mass1 <= args.nsbh_boundary_mass:
            spin1z, spin2z = chi, chi
        else:
            spin1z = chi
            spin2z = 0.0
        
    # Transform into dimensionless PN chirptimes
    theta0, theta3, theta3s = coord_utils._compute_dimensionless_chirptimes_from_mass_spin(mass1, mass2, spin1z, spin2z)
    
    if args.optimize_flow != None:
            opt_flow = coord_utils._compute_optimize_flow(mass1, mass2, spin1z, spin2z)
    else:
        opt_flow = args.flow
        
    if args.min_duration != None:    
        dur = coord_utils._compute_dur(mass1, mass2, spin1z, spin2z, opt_flow)
    else:
        dur = 0.0
        
    return [theta0, theta3, theta3s, mtotal, eta, spin1z, spin2z, opt_flow, dur] 
    
    


def compute_metric_match(other, center, gamma):
    """
    Return geometrical match between two points using metric connection.
    
    Parameters
    ----------
    other: list of order 3x1
        Coordinate of an arbitrary point.
    center: list of order 3x1
        Coordinate of the center point.
    gamma: list of order 6x1 or numpy.ndarray of order 3x3
        Metric components of center point.      
    """
    
    g00, g01, g02, g11, g12, g22 = gamma[0,0], gamma[0,1], gamma[0,2], gamma[1,1], gamma[1,2], gamma[2,2]
    dx0 = other[0] - center[0]
    dx1 = other[1] - center[1]
    dx2 = other[2] - center[2]
    
    return g00*dx0*dx0 + 2*g01*dx0*dx1 + 2*g02*dx0*dx2 + g11*dx1*dx1 + 2*g12*dx1*dx2 + g22*dx2*dx2


def pairwise_dist(proposal, tlist, mlist, dim=3):
    lgth = len(tlist)
    prop_lst = np.array(proposal[:3]*lgth)
    tmp_lst = np.array(tlist)[:, :3].reshape((3*lgth))
    deltaL = prop_lst-tmp_lst
    delta = deltaL.reshape(lgth, 3)
    
    gamma_array = mlist.reshape(3*lgth, 3)
    delta_replica = np.zeros((3*lgth, 3))

    for i in range(3):
        delta_replica[i::3] = delta

    delta_gamma = gamma_array*delta_replica
    deltaR = np.sum(delta_gamma, axis=1)
    dist = deltaL*deltaR
    dist_square = np.sum(dist.reshape(lgth, 3), axis=1)
    
    return dist_square



def find_neighbors(tmplates, proposal, bin_size):
    """
    Return the min and max indices of templates that cover the given
    template at proposal within a parameter difference of bin_size (seconds).
    tmpltes should be a sequence of neighborhood values in sorted order.
    
    """
    low_ind = bisect.bisect_left(tmplates, proposal - bin_size)
    high_ind = bisect.bisect_right(tmplates, proposal + bin_size)
    return low_ind, high_ind



def insert_proposal(tList, theta0_list, proposal, metric_list=None, gamma_proposal=None):
    """
    Insert the first element of proposal in theta0_list without violating
    the ascending order. Assumes theta0_list is a sorted list.
    Insert theta in tlist at the same location assuming the first column
    of np.array(tlist) and theta0_list are same.
    If gamma_proposal is available then insert gamma in metric_list at the same
    location.

    Parameters
    ------------
    tList: list of order nx3
        List of template points.
    theta0_list: list of order nx1
        List of theta0 values of template points.
    proposal: list of order 1x3
        New born template point.
    metric_list: list of order nx6
        List of metric of template points "tlist".
    gamma_proposal: list of order 1x6
        Metric of new born template point.
        
    Returns
    ---------
    tList, theta0_list, metric_list after insertion.
    """
    
    left_index = bisect.bisect_left(theta0_list, proposal[0])
    theta0_list.insert(left_index, proposal[0])
    tList.insert(left_index, proposal)
    if gamma_proposal is not None:
        metric_list.insert(left_index, gamma_proposal)
        return tList, theta0_list, metric_list
    else: 
        return tList, theta0_list





def find_neighbors_distance(slist, theta0_list, smlist, proposal, bin_size, norm_ds_sq=1.0):
    """
    Given a proposal this function check if it is far from  all the existant templates 
    within theta0 +/- bin_size.
    """
    # Set a fix value of norm_ds_sq, condiders the propsal has a mismatch < (1-minimal match)/2
    

    low, high = find_neighbors(theta0_list, proposal[0], bin_size)
    tlist = slist[low:high]
    mlist = smlist[low:high]
    t0list = theta0_list[low:high]
    abst0_list = [abs(t0list[i] - proposal[0]) for i in range(len(t0list))]
    ind = [ i for (i,j) in sorted(enumerate(abst0_list), key=operator.itemgetter(1))]
    for i in ind:
            ds_sq = compute_metric_match(proposal, tlist[i], np.array(mlist[i]))      
            if ds_sq <= norm_ds_sq: return False
    return True
                           

def find_neighbors_distance2(slist, theta0_list, smlist, proposal, bin_size):
    """
    Given a proposal this function check if it is far from  all the existant templates 
    within theta0 +/- bin_size.
    """
    
    low, high = find_neighbors(theta0_list, proposal[0], bin_size)
    tlist = slist[low:high]
    mlist = smlist[low:high]
    ds_sq = pairwise_dist(proposal, tlist, np.array(mlist))   
    
    if min(ds_sq) <= 1.0:
        cond = False
    else:
        cond = True
    
    return cond   
                            

def eulerAnglesToRotationMatrix(theta) :
   
    R_x = np.array([[1,  0,                  0              ],
                    [0,  np.cos(theta[0]), -np.sin(theta[0])],
                    [0,  np.sin(theta[0]),  np.cos(theta[0])]
                    ])         
                     
    R_y = np.array([[np.cos(theta[1]),  0,  np.sin(theta[1])],
                    [0,                 1,  0               ],
                    [-np.sin(theta[1]), 0,  np.cos(theta[1])]
                    ])
                 
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]),  0],
                    [np.sin(theta[2]),  np.cos(theta[2]),  0],
                    [0,                 0,                 1]
                    ])
 
    return np.dot(R_z, np.dot( R_y, R_x ))


# A3-star lattice neighbour for unit radius sphere when 
A3_star_neighbor_pos = 2.0/np.sqrt(5.0)* \
        np.array([[ 1.0,  1.0,  1.0],
                  [-1.0, -1.0, -1.0],
                  [ 1.0, -1.0,  1.0],
                  [-1.0,  1.0, -1.0],
                  [ 0.0,  2.0,  0.0],
                  [ 0.0, -2.0,  0.0],
                  [ 1.0, -1.0, -1.0],
                  [-1.0,  1.0,  1.0],
                  [ 2.0,  0.0,  0.0],
                  [-2.0,  0.0,  0.0],
                  [ 1.0,  1.0, -1.0],
                  [-1.0, -1.0,  1.0],
                  [ 0.0,  0.0,  2.0],
                  [ 0.0,  0.0, -2.0]])
    
# Geometrical properties of lattice is invariant under similarity transformation.
# Rotate around a y-axis to align most of the neighbor on theta0-theta3 plane

def A3_star_lattice_neighbor(center, gamma, neighbors_pos):
    """
    Compute the A3-star lattice neighbors from metric.
    
    Parameters
    ----------
    center : numpy.array of order 1x3
        Center point.
    gamma : numpy.array of order 3x3
        Metreic at the center point.
        Assumed to be scaled by (1-mismatch).
    """
    center = center[:3]
    eigval, eigvec = np.linalg.eig(gamma)
    semiaxis = 1.0/np.sqrt(eigval)
    return np.array([center + np.dot(eigvec, semiaxis*neighbors_pos[i]) for i in range(14)])



    
class PrallelProcessingUtilsForA3StarOrientation(object):
    """
    Base class to handles the parallel processing 
    """
    
    def __init__(self, args):
        self._threads = int(min(args.number_threads, 20))
        self._max_mismatch = 1.0 - args.min_match
        self._rndnpts = 2000
        self._angnpts = 50000
        
   
    def _classify_indices_from_number_threads(self, length):
        return np.linspace(0, length-1, int(self._threads +1), endpoint=True).astype(int)
    
    def _valid_random_point_and_compute_metric(self, _utils, dic, key):
        """
        This function validates the random points and computes their metric.
        It also computes the semi-axes and the eignvector of the minimal match 
        surface of the valid random points.
        """
        coord_utils, bndry_utils, metric_utils, rpts = _utils
        
        # define empty list for valid random points
        rand_pts, gamma_list = [], []
        for i in range(len(rpts)):
            # Compute component masses and spin
            mtotal, eta, mass1, mass2, spin1z, spin2z = \
                        coord_utils._compute_mass_spin_from_dimensionless_chirptimes(rpts[i,0], rpts[i,1], rpts[i,2])
            
            # Compute starting frequency if require and check template duration
            opt_flow, dur = bndry_utils._verify_dur_optimize_flow(mass1, mass2, spin1z, spin2z, coord_utils)
            
            if dur is not False:
                center = [rpts[i,0], rpts[i,1], rpts[i,2], mtotal, eta, spin1z, spin2z, opt_flow, dur]
                pGamma, fGamma = metric_utils.MetricInTheta0Theta3Theta3s(center, mis_match=self._max_mismatch)
                eigval, eigvec = np.linalg.eig(pGamma)
                semiaxis = 1.0/np.sqrt(eigval)
                gamma_list.append([semiaxis, eigvec])
                rand_pts.append(rpts[i])
        
        # Append the random points list, semi-axes list and eigenvector list in dictonary manager   
        dic[key] = [rand_pts, gamma_list]
    
    def _find_possible_neighbors_after_rotation(self, _utils, dic, key):
        """
        This function finds the average number of possible neighbors of the
        random points after a rotation of the Eucledian-A3-star lattices.
        """
        coord_utils, bndry_utils, randpts_list, randpts_metric_list, angle_list = _utils
        avg_num_ngb = []
        
        for angle in angle_list:
            # Compute Euler rotation matrix
            rotation_matrix = eulerAnglesToRotationMatrix(angle)
            
            # Rotate the neighbors location in Eucledian space
            neighbors_pos = np.array([np.dot(rotation_matrix, A3_star_neighbor_pos[k]) for k in range(14)])
            num_ngb = []
            
            for j in range(len(randpts_list)):
                neighbors = np.array([randpts_list[j] + np.dot(randpts_metric_list[j][1],\
                                randpts_metric_list[j][0]*neighbors_pos[k]) for k in range(14)])
                
                # Check boundary condition
                possible_neighbors = bndry_utils.pts_inside_bndry(neighbors, coord_utils)
                num_ngb.append(len(possible_neighbors))
                
            # store average number of possible neighbors
            avg_num_ngb.append(np.mean(num_ngb))
        # Append the average number of possible neighbors in the dictonary manager.
        dic[key] = avg_num_ngb
        
    
    def _set_A3Star_orientation(self, args, coord_utils, bndry_utils, randpts_utils, metric_utils):
        """
        Set the neighbors location of A3-star lattice in Eucledian space by optimizing
        the orinetation of the lattice.
        """
        if len(args.A3_star_orientation) != 3 and args.optimize_A3star_lattice_orientation is True:
            pre_rnd_size = args.random_list_size
            args.random_list_size = int(2*self._rndnpts)
            rand_pts_list = np.array(randpts_utils.GenrandomPointsOverTheta0Theta3Theta3S(args, coord_utils))

            mgr = Manager()
            dic = mgr.dict()
            index_list = self._classify_indices_from_number_threads(len(rand_pts_list))
            jobs = [Process(target=self._valid_random_point_and_compute_metric, \
                    args=([coord_utils, bndry_utils, metric_utils, rand_pts_list[index_list[k]:index_list[k+1]]], dic, k)) \
                    for k in range(self._threads)]
            for job in jobs: job.start()
            for job in jobs: job.join()  
            # collect the results from manager dict object
            randpts_list = []
            randpts_metric_list = []
            for k in range(int(self._threads)):
                if len(randpts_list)<self._rndnpts:
                    randpts_list.extend(dic[k][0])
                    randpts_metric_list.extend(dic[k][1])

            # Generate a random vector of angles
            angle_list = np.random.uniform(-PI, PI, size=(self._angnpts, 3)).tolist()
            mgr = Manager()
            dic = mgr.dict()
            index_list = self._classify_indices_from_number_threads(self._angnpts)
            jobs = [Process(target=self._find_possible_neighbors_after_rotation, \
                            args=([coord_utils, bndry_utils, randpts_list, randpts_metric_list,\
                           angle_list[index_list[k]:index_list[k+1]]], dic, k)) for k in range(self._threads)]
            for job in jobs: job.start()
            for job in jobs: job.join()
            # collect the results from manager dict object
            avg_num_ngb_list = []
            for k in range(int(self._threads)):
                avg_num_ngb_list.extend(dic[k])

            maximize_angle = angle_list[avg_num_ngb_list.index(max(avg_num_ngb_list))]
            args.random_list_size = pre_rnd_size
            
        elif len(args.A3_star_orientation) == 3:
            maximize_angle = args.A3_star_orientation
        else:
            maximize_angle = [0.0, 0.0, 0.0]
            
        coord_utils._A3StarAng = maximize_angle
        rotation_matrix = eulerAnglesToRotationMatrix(maximize_angle)
        coord_utils._neighbors_pos = np.array([np.dot(rotation_matrix, A3_star_neighbor_pos[i]) for i in range(14)])
        
        

PI_square = PI*PI
PI_quant = PI_square*PI_square*PI
one_by_three = 1.0/3.0

class CoordinateTransformUtils(object):
    """
    Base class that handles the coordinate transfrom from mass parameters
    to dimesionless PN chirptimes (0PN and spin independent 1.5PN), and
    also corresponding inverse transformation.
    
    Parameters
    ----------
    fLow : float
        Lower cutoff frequency. Unit: HZ.
    sys_type : str
        Type of the system.
    nsbh_bound : float
        NSBH boundary mass. Unit: Solar mass.
    """
    def __init__(self, args, PSD, ASD, noise_generator):
        self._flow = args.flow
        self._df = args.df
        self._fref = args.fref
        self._fhigh = args.fhigh
        
        self._optimize_flow = args.optimize_flow
        if self._optimize_flow is not None:
            self._flow = self._fref
            self._sigma_frac = self._optimize_flow
        
        self._ind_min = int(self._flow/self._df)
        self._ind_max = int(self._fhigh/self._df)
        
        self._nsbh_bound = args.nsbh_boundary_mass
        self._approximant = args.approximant
        self._distance = 1e6*PC_SI
        
        self._args = args
        self._ASD = ASD
        
        self._A3StarAng = []
        self._neighbors_pos = []
        
        
    def _theta0_theta3_to_mtotal_eta(self, theta0, theta3):
        """
        Compute total mass and symmetric mass ratio from 0PN and
        1.5PN (spin-independent) dimensionless chirptimes.
        
        Parameters
        ----------
        theta0 : float
            0PN dimensionless chirptime.
        theta3 : float
            1.5PN dimensionless chirptime.
            
        Returns
        -------
        mtotal : float
            Total mass of the binary.
        eta : float
            Symmetric mass ratio of the binary.
        """
        mtotal = 5.0/(32.0*PI_square*self._flow) * theta3/theta0/MTSUN_SI
        eta = (16*PI_quant/25)**one_by_three * theta0**(2.0/3.0) / theta3**(5.0/3.0)
        return mtotal, eta
    
    def _mtotal_eta_to_theta0_theta3(self, mtotal, eta):
        """
        Compute 0PN and 1.5PN (spin-independent) dimensionless chirptimes
        from total mass and symmetric mass ratio of the system.
        
        Parameters
        ----------
        mtotal : float
            Total mass of the binary.
        eta : float
            Symmetric mass ratio of the binary.
            
        Returns
        -------
        theta0 : float
            0PN dimensionless chirptime.
        theta3 : float
            1.5PN dimensionless chirptime.
        """
        mtotal *= MTSUN_SI
        v0 = (PI * mtotal * self._flow)**one_by_three
        v0_square = v0*v0
        v0_quant = v0_square * v0_square * v0
        theta0 = 5.0/(128.0*eta*v0_quant)
        theta3 = PI/(4.0*eta*v0_square)
        return theta0, theta3
    
    def _get_optimize_flow_from_amp_ASD(self, amp):
        # compute the whitened amplitude
        asd = self._ASD[:self._ind_max]
        amp = amp[:self._ind_max]
        wamp = amp / asd
        # sum the power cumulatively from high to low frequencies
        integral = np.cumsum(np.flipud(wamp * wamp))
        ref_sigmasq = integral[-1]
        # find the frequency bin corresponding to the target loss
        i = np.searchsorted(integral, ref_sigmasq * self._sigma_frac ** 2)
        opt_flow = (len(integral) - i) * self._df
        return opt_flow

    
    
class SingleSpinAlignedCoordTransform(CoordinateTransformUtils):
    """
    A sub-class of CoordinateTransformUtils for coordinate transfrom from mass-spin
    to dimesionless PN chirp times and also inverse.
    Single spin parameter is construted by consedering equal-aligned-spin when 
    mass of the second object is greater then NSBH boundary mass (BBH system) or mass of the
    first object is less then NSBH boundary mass (BNS system).
    Otherwise the single spin parameter is constructed by considering BH is spinning only
    for NSBH system.
    
    Reduced Spin:
    
    \f$ \chi_{r} = \frac{1}{2}\left(\chi_1 + \chi_2\right) +
                       \frac{1}{2}*\sqrt{1 - 4\eta}\left(\chi_1 - \chi_2\right)
                                    - \frac{76\eta}{226}\left(\chi_1 + \chi_2\right) \f$
    Parameters
    ----------
    Chi : callable
        Returns the value of spin for  given values of reduced
        and symmetric mass ratio and also depending on the type of
        system.
    """
    def __init__(self, args, PSD, ASD, noise_generator):
        CoordinateTransformUtils.__init__(self, args, PSD, ASD, noise_generator)
        
    def _spin_bounds(self, mass1, mass2):
        if mass1 >= self._nsbh_bound:
            bound = (self._args.min_bh_spin, self._args.max_bh_spin)
        else:
            bound = (self._args.min_ns_spin, self._args.max_ns_spin)
        return bound
    
    def _compute_chired_from_eta_chi(self, mass1, mass2, spin1z, spin2z):
        """
        Compute reduced spin from individual masses and a single spin.
        """
        chi_r = lalsim.SimInspiralTaylorF2ReducedSpinComputeChi(mass1, mass2, spin1z, spin2z)
        
        if mass2 < self._nsbh_bound and mass1 > self._nsbh_bound:
            if abs(chi_r)>1.0:
                chi_r = lalsim.SimInspiralTaylorF2ReducedSpinComputeChi(mass1, mass2, spin1z, 0.0)
            
        return chi_r
    
    def _compute_spin_from_chired_eta(self, mass1, mass2, eta, chi_r):
        """
        Compute single spin from reduced spin and symmetric mass ratio.
        """
        if mass2 >= self._nsbh_bound or mass1 <= self._nsbh_bound:
            chi = chi_r/(1.0 - 76*eta/113.0)
            spin1z, spin2z = chi, chi
        else:
            chi = 2.0*chi_r/(1.0 + (1.0 -4*eta)**0.5 - 76.0*eta/113)
            spin1z, spin2z = chi, 0.0 
        return spin1z, spin2z
        
    
    def _compute_dimensionless_chirptimes_from_mass_spin(self, mass1, mass2, spin1z, spin2z):
        """
        Compute the dimensionless chirptimes from mass and spin.
        
        Parameters
        ----------
        mass1 : float
            Mass of the first object. Unit: Solar mass
        mass2 : float
             Mass of the second object. Unit: Solar mass
        chi : float
             Spin of the equaled spin binary system when the flag is BBH or BNS
             or spin of the black hole only when the flag is NSBH.
        
        Returns
        -------
        theta0: float
            0PN dimensionless chirptime
        theta3: float
            1.5PN dimensionless chirptime (spin independent term only)
        theta3s:
            1.5PN dimnesionless chirptime (spin dependent term only)
        """
        mtotal, eta = mass1_mass2_to_mtotal_eta(mass1, mass2)
        theta0, theta3 = self._mtotal_eta_to_theta0_theta3(mtotal, eta)
        chi_r = self._compute_chired_from_eta_chi(mass1, mass2, spin1z, spin2z)
        theta3s = 113*chi_r*theta3/(48*PI)
        return theta0, theta3, theta3s
    
    def _compute_mass_spin_from_dimensionless_chirptimes(self, theta0, theta3, theta3s):
        """
        Compute the mass and spin from dimensionless chirptimes.
        
        Parameters
        ----------
        theta0: float
            0PN dimensionless chirptime
        theta3: float
            1.5PN dimensionless chirptime (spin independent term only)
        theta3s:
            1.5PN dimnesionless chirptime (spin dependent term only)
        
        Returns
        -------
        mtotal : float
            Total mass of the binary. Unit: Solar mass 
        eta : float
            Symmetric mass ratio.
        m1: float
            Mass of the first object. Unit: Solar mass
        m2: float
             Mass of the second object. Unit: Solar mass
        chi: float
             Spin of the equaled spin binary system when the flag is BBH or BNS
             or spin of the black hole only when the flag is NSBH.
        """
        mtotal, eta = self._theta0_theta3_to_mtotal_eta(theta0, theta3)
        mass1, mass2 = mtotal_eta_to_mass1_mass2(mtotal, eta)
        chi_r = 48*PI*theta3s/(113*theta3)
        spin1z, spin2z = self._compute_spin_from_chired_eta(mass1, mass2, eta, chi_r)
        return mtotal, eta, mass1, mass2, spin1z, spin2z


class SEOBNRv4ROMCoordTransform(SingleSpinAlignedCoordTransform):
    
    def __init__(self, args, PSD, ASD, noise_generator):
        SingleSpinAlignedCoordTransform.__init__(self, args, PSD, ASD, noise_generator)
        
    def _compute_optimize_flow(self, mass1, mass2, spin1z, spin2z):
        amp = lalsim.SimIMRSEOBNRv4ROMAmpPhs(0.0, self._df, self._flow, self._fhigh, \
                                 self._fref, self._distance, 0.0, mass1 * MSUN_SI, mass2 * MSUN_SI, \
                                 spin1z, spin2z, -1, None, 0)[0].data.data[:][:self._ind_max]
        
        
        return self._get_optimize_flow_from_amp_ASD(amp)
    
    def _compute_dur(self, mass1, mass2, spin1z, spin2z, flow):
        dur = lalsim.SimIMRSEOBNRv4ROMTimeOfFrequency(flow,
                                mass1*MSUN_SI, mass2*MSUN_SI, spin1z, spin2z)
        # Allow a 10% margin of error
        return dur * 1.1
        
        

class IMRPhenomDCoordTransform(SingleSpinAlignedCoordTransform):
    
    def __init__(self, args, PSD, ASD, noise_generator):
        SingleSpinAlignedCoordTransform.__init__(self, args, PSD, ASD, noise_generator)
        
    def _compute_optimize_flow(self, mass1, mass2, spin1z, spin2z):
        amp = lalsim.SimIMRPhenomDGenerateFDAmpPhs(self._phi0, self._fref, self._df, \
                                   mass1 * MSUN_SI, mass2 * MSUN_SI, spin1z, spin2z, \
                                   self._ref, self._fhigh, self._distance, None)[0].data.data[:][:self._ind_max]
        return self._get_optimize_flow_from_amp_ASD(amp)
    
    
    def _compute_dur(self, mass1, mass2, spin1z, spin2z, flow):
        dur = lalsim.SimIMRPhenomDChirpTime(mass1*MSUN_SI, mass2*MSUN_SI, spin1z, spin2z, flow)
        # Allow a 10% margin of error
        return dur * 1.1
        
    

class IMRPhenomBCoordTransform(CoordinateTransformUtils):
    """
    A sub-class of CoordinateTransformUtils for coordinate transfrom from mass-spin
    to dimesionless PN chirp times and vice-versa for IMRPhenomB approximant.
    
    Effective Spin:
    
    \f$ \chi_{\epsilon} = \frac{m_1\chi_1 + m_2\chi_2}
                                {m_1 + m_2}\f$
    
    
    Parameters
    ----------
    _spin_bounds : callable
        This function includes restrictions on q and chi based
        on IMRPhenomB's range of believability. We will silently
        truncate if --PhenomB-spin-truncate flag is true.
        
        For \f$ 1 \leq m_1/m_2 \leq 4.0 \f$, effective spin limit
        \f$ -0.85 \leq \chi_{e} 0.85\f$ and when $ m_1/m_2 > 4$ 
        then effective spin limit
        \f$ -0.5 \leq \chi_{e} 0.75\f$
        
    chi_limit : callable
        IMRPhenomB depends on single effective spin, for BBH and
        NSBH system we assume minimum and maximum spins are 
        --min/max-bh-spin
        
        for BNS system we assume minimum and maximum spins are 
        --min/max-ns-spin
        
    """
    def __init__(self, args):
        CoordinateTransformUtils.__init__(self, args)
        self._PhenomB_spin_truncate = args.PhenomB_spin_truncate
    
    def _spin_bounds(self, mass1, mass2):
        if mass1 >= self._nsbh_bound:
            spin_min, spin_max = self._args.min_bh_spin, self._args.max_bh_spin
        else:
            spin_min, spin_max = self._args.min_ns_spin, self._args.max_ns_spin
            
        if 1.0 <= mass1/mass2 <= 4.0:
            bounds = (max(-0.85, spin_min), min(0.85, spin_max))
        else:
            bounds = (max(-0.5, spin_min), min(0.75, spin_max))
        return bounds
    
    def _compute_optimize_flow(self, mass1, mass2, spin1z, spin2z):
        chieff = lalsim.SimIMRPhenomBComputeChi(mass1, mass2, spin1z, spin2z)
        wf = lalsim.SimIMRPhenomBGenerateFD(0, self._df,
                               mass1 * MSUN_SI, mass2 * MSUN_SI,
                               chieff, self._fref, self._fhigh, 1000000 * PC_SI)
        amp = np.abs(wf.data.data[:])
        return self._get_optimize_flow_from_amp_ASD(amp)
    
    
    def _compute_dur(self, mass1, mass2, spin1z, spin2z, flow):
        chieff = lalsim.SimIMRPhenomBComputeChi(mass1, mass2, spin1z, spin2z)
        dur = lalsim.SimInspiralTaylorF2ReducedSpinChirpTime(flow,
                             mass1*MSUN_SI, mass2*MSUN_SI, chieff, 7)
        return 1.1*dur + 1.0
    
    
    def _compute_dimensionless_chirptimes_from_mass_spin(self, mass1, mass2, chi1, chi2):
        """
        Compute the dimensionless chirptimes from mass and individual dimensioless spins.
        
        Parameters
        ----------
        mass1 : float
            Mass of the first object. Unit: Solar mass
        mass2 : float
             Mass of the second object. Unit: Solar mass
        chi1 : float
             Spin of the first object.
        chi2 : float
            Spin of the second object.
        
        Returns
        -------
        theta0 : float
            0PN dimensionless chirptime.
        theta3: float
            1.5PN dimensionless chirptime (spin independent term only)
        theta3s: float
            1.5PN dimnesionless chirptime (comes from IMRPhenomB model)
        
        """
        mtotal, eta = mass1_mass2_to_mtotal_eta(mass1, mass2)
        theta0, theta3 = self._mtotal_eta_to_theta0_theta3(mtotal, eta)
        chi_eff = lalsim.SimIMRPhenomBComputeChi(mass1, mass2, chi1, chi2)
        theta3s = (17022. - 9565.9*chi_eff)*eta*theta3
        return theta0, theta3, theta3s
    
    def _compute_mass_spin_from_dimensionless_chirptimes(self, theta0, theta3, theta3s):
        """
        Compute the mass and spin from dimensionless chirptimes.
        
        Parameters
        ----------
        theta0: float
            0PN dimensionless chirptime
        theta3: float
            1.5PN dimensionless chirptime (spin independent term only)
        theta3s:
            1.5PN dimnesionless chirptime (comes from IMRPhenomB model)
        
        Returns
        -------
        mtotal : float
            Total mass of the binary. Unit: Solar mass 
        eta : float
            Symmetric mass ratio.
        m1: float
            Mass of the first object. Unit: Solar mass
        m2: float
             Mass of the second object. Unit: Solar mass
        chi_eff : float
             Effective spin of the system.
        """
        mtotal, eta = self._theta0_theta3_to_mtotal_eta(theta0, theta3)
        mass1, mass2 = mtotal_eta_to_mass1_mass2(mtotal, eta)
        chi_eff = (17022.0 - theta3s / eta / theta3) / 9565.9
        return mtotal, eta, mass1, mass2, chi_eff

        


class TaylorF2RedSpinCoordTransform(CoordinateTransformUtils):
    """
    A sub-class of CoordinateTransformUtils for coordinate transfrom from mass-spin
    to dimesionless PN chirp times and vice-versa for TaylorF2RedSpin approximant.
    
    Reduced Spin:
    
    \f$ \chi_{r} = \frac{1}{2}\left(\chi_1 + \chi_2\right) +
                       \frac{1}{2}*\sqrt{1 - 4\eta}\left(\chi_1 - \chi_2\right)
                                    - \frac{76\eta}{226}\left(\chi_1 + \chi_2\right) \f$
    Parameters
    ----------
    spin_bounds : callable
        Returns the value of spin limits from given values of 
        individual masses and system type.
    """
    def __init__(self, args):
        CoordinateTransformUtils.__init__(self, args)
        
    def _spin_bounds(self, mass1, mass2, args):
        if mass2 >= slef._nsbh_bound:
            bounds = (args.min_spin1, args.max_spin1, args.min_spin1, args.max_spin1)
        else:
            bounds = (args.min_spin1, args.max_spin1, args.min_spin2, args.max_spin2)
        return bounds
    
    def _compute_dur(self, mass1, mass2, spin1z, spin2z, flow):
        chieff = lalsim.SimIMRPhenomBComputeChi(mass1, mass2, spin1z, spin2z)
        dur = lalsim.SimInspiralTaylorF2ReducedSpinChirpTime(flow,
                             mass1*MSUN_SI, mass2*MSUN_SI, chieff, 7)
        return dur
    
    def _compute_optimize_flow(self, mass1, mass2, spin1z, spin2z):
        approx = lalsim.GetApproximantFromString( self.approximant )

        hplus_fd, hcross_fd = lalsim.SimInspiralChooseFDWaveform(
                mass1 * MSUN_SI, mass2 * MSUN_SI,
                0., 0., spin1z, 0., 0., spin2z,
                1e6*PC_SI, 0., 0.,
                0., 0., 0.,
                df, self._fref, self._fhigh, self._fref, None, approx)
       
        amp = np.abs(hplus_fd.data.data[:])
        return self._get_optimize_flow_from_amp_ASD(amp)
    
    
    def _compute_dimensionless_chirptimes_from_mass_spin(self, mass1, mass2, chi1, chi2):
        """
        Compute the dimensionless chirptimes from mass and individual dimensioless spins.
        
        Parameters
        ----------
        m1 : float
            Mass of the first object. Unit: Solar mass
        m2 : float
             Mass of the second object. Unit: Solar mass
        chi1 : float
             Spin of the first object.
        chi2 : float
            Spin of the second object.
        
        Returns
        -------
        theta0 : float
            0PN dimensionless chirptime.
        theta3: float
            1.5PN dimensionless chirptime (spin independent term only)
        theta3s: float
            1.5PN dimnesionless chirptime (spin dependent term only)
        
        """
        mtotal, eta = mass1_mass2_to_mtotal_eta(mass1, mass2)
        theta0, theta3 = self._mtotal_eta_to_theta0_theta3(mtotal, eta)
        chi_r = lalsim.SimInspiralTaylorF2ReducedSpinComputeChi(mass1, mass2, chi, chi)
        theta3s = 113*chi_r*theta3/(48*PI)
        return theta0, theta3, theta3s
    
    def _compute_mass_spin_from_dimensionless_chirptimes(self, theta0, theta3, theta3s):
        """
        Compute the mass and spin values from dimensionless chirptimes.
        
        Parameters
        ----------
        theta0: float
            0PN dimensionless chirptime
        theta3: float
            1.5PN dimensionless chirptime (spin independent term only)
        theta3s: float
            1.5PN dimnesionless chirptime (spin dependent term only)
        
        Returns
        -------
        mtotal : float
            Total mass of the system. Unit : Solar mass.
        eta : float
            Symmetric mass ratio of the system.
        mass1 : float
            Mass of the first object. Unit : Solar mass.
        mass2 : float
            Mass of the second object. Unit : Solar mass.
       spin1z : float
           Dimensionless spin of the first object.
       spin2z : float
           Dimensionless spin of the second object.
        """
        mtotal, eta = self._theta0_theta3_to_mtotal_eta(theta0, theta3)
        mass1, mass2 = mtotal_eta_to_mass1_mass2(mtotal, eta)
        delta_eta = (1.0 - 4.0*eta)**0.5
        chi_r = 48*PI*theta3s/(113*theta3)
        spin1z_min, spin1z_max, spin2z_min, spin2z_max = self._spin_bounds(mass1, mass2, self._args)
        
        while True:
            # Generate a random number between the limits of the second object's spin.
            # Considering the random number as second object's spin, evalute the first objects
            # spin from the value of reduced spin.
            spin2z = np.random.mtrand.uniform(spin2z_min, spin2z_max)
            spin1z = (2.0*chi_r - (1.0 - 76*eta/113 - delta_eta)*spin2z)/(1.0 - 76*eta/113 + delta_eta)
            
            
            # Accept the spin values if the first object's spin belong inside spin limits.
            if spin1z_max >= spin1z >= spin1z_min:
                return mtotal, eta, mass1, mass2, spin1z, spin2z

    
    
        

coord_transform_utils = {
    "SEOBNRv4_ROM" : SEOBNRv4ROMCoordTransform,
    "IMRPhenomD" : IMRPhenomDCoordTransform,
    "IMRPhenomB" : IMRPhenomBCoordTransform,
    "TaylorF2RedSpin" : TaylorF2RedSpinCoordTransform
    }


class CheckBoundary(object):
    """
    Base class that handles the boundary condition from given argparse arguments.

    Parameters
    ----------
    _mass_constraints : dict
        Contains the constraints related to mass ranges.
    mass1_min : float
        Minimum mass of the frist object. Unit: Solar mass.
    mass1_max : float
        Maximum mass of the frist object. Unit: Solar mass.
    mass2_min : float
        Minimum mass of the second object. Unit: Solar mass.
    mass2_max : float
        Maximum mass of the second object. Unit: Solar mass.
    mtotal_min : float
        Minimum total mass of the system. Unit: Solar mass. 
    mtotal_max : float
        Maximum total mass of the system. Unit: Solar mass.
    eta_min : float
        Minimum symmetric mass ratio.
    eta_max : float
        Maximum symmetric mass ratio.
    sys_type : str
        Type of the system.
    nsbh_bound : float
        NSBH boundary mass. Unit: Solar mass.
    """
    
    def __init__(self, args):
        self._args = args
        self._mass_constraints = set_boundary_constraints(args)
        self.mass1_min, self.mass1_max = self._mass_constraints['mass1']
        self.mass2_min, self.mass2_max = self._mass_constraints['mass2']
        self.mtotal_min, self.mtotal_max = self._mass_constraints['mtotal']
        self.eta_min, self.eta_max = self._mass_constraints['eta']
        self._nsbh_bound = args.nsbh_boundary_mass
        
    def _verify_mass_limit(self, theta0, theta3, coord_utils):
        """
        Verify mass limits of A3-star lattice neighbors. Return 
        indices of the valid neighbors, total mass, symmetric mass 
        ratio, mass of the first object and mass of the second object
        of all the neighbors.
        
        Parameters
        ----------
        theta0 : numpy.array of order 1xn
            0PN dimensionless chirptimes.
        theta3 : numpy.array of order 1xn
            1.5PN dimensionless chirptimes (spin-independent).
            
        Returns
        -------
        ind : numpy.array
            Indices of the valid neighbors.
        mtotal : numpy.array of order 1xn
            Total mass of the neighbors. Unit : Solar mass
        eta : numpy.array of order 1xn
            Symmetric mass ratio of the neighbors.
        mass1 : numpy.array of order 1xn
            Mass of the first object of the neighbors.
        mass2 : numpy.array of order 1xn
            Mass of the second object of the neighbors.
        
        """
        mtotal, eta = coord_utils._theta0_theta3_to_mtotal_eta(theta0, theta3)
        mass1, mass2 = mtotal_eta_to_mass1_mass2(mtotal, eta)
        ind = np.where((self.mtotal_min <= mtotal) & (mtotal <= self.mtotal_max) &
                       (self.mass1_min <= mass1) & (mass1 <= self.mass1_max) &
                       (self.mass2_min <= mass2) & (mass2 <= self.mass2_max) &
                       (self.eta_min <= eta)& (eta <= self.eta_max))[0]
        
        if self._args.min_chirp_mass is not None:
            mchirp = mtotal[ind]*eta[ind]**(3.0/5.0)
            valid_ind = np.where((mchirp >= self._mass_constraints["mchirp"][0]) & \
                                 (mchirp <= self._mass_constraints["mchirp"][1]))[0]
            ind = ind[valid_ind]
        return ind, mtotal, eta, mass1, mass2
    
    def _verify_dur_optimize_flow(self, mass1, mass2, spin1z, spin2z, coord_utils):
        
        if self._args.optimize_flow != None:
            opt_flow = coord_utils._compute_optimize_flow(mass1, mass2, spin1z, spin2z)
        else:
            opt_flow = self._args.flow
            
        if self._args.min_duration != None:
            dur = coord_utils._compute_dur(mass1, mass2, spin1z, spin2z, opt_flow)
            if dur < self._mass_constraints["duration"][0] \
                or dur > self._mass_constraints["duration"][1]:
                dur = False
        else:
            dur = 0.0
        
        return opt_flow, dur
                

      
    
class SingleSpinAlignedIMRChkBoundary(CheckBoundary):
    """
    A sub-class of CheckBoundary to check the boundary condition parameters
    for SEOBNRv4_ROM and IMRPhenomD approximant.
    """
    def __init__(self, args):
        CheckBoundary.__init__(self, args)
        
    def pts_inside_bndry(self, tList, coord_utils, chk_dur_flow=False):
        """
        Return the points which are inside the specified parameter space.
        
        Parameters
        ----------
        tList : numpy.ndarray of order nx3
            Array of 3D points in dimensionless chirptime coordinates.
            
        coord_utils : callable object
            Contain the class CoordinateTransformUtils to convert the 
            tList values in mass spin coordinate system.
        """
        
        ind, mtotal, eta, mass1, mass2 = self._verify_mass_limit(tList[:,0], tList[:,1], coord_utils)
        tlist_inside_bnd = []
        for i in ind:
            # Calculate reduced spin
            chi_r = 48*PI*tList[i,2]/(113*tList[i,1])
            
            # Calculate single-spin from reduced spin.
            spin1z, spin2z = coord_utils._compute_spin_from_chired_eta(mass1[i], mass2[i], eta[i], chi_r)
            
            # Find the spin limits.
            chi_min, chi_max = coord_utils._spin_bounds(mass1[i], mass2[i])
            
            if chi_min <= spin1z <= chi_max:
                    
                if chk_dur_flow == True:
                    opt_flow, dur = self._verify_dur_optimize_flow(mass1[i], mass2[i], spin1z, spin2z, coord_utils)
                    
                    if dur is not False:
                        tlist_inside_bnd.append([tList[i,0], tList[i,1], tList[i,2], mtotal[i], eta[i], \
                                                      spin1z, spin2z, opt_flow, dur])
                    else: pass
                else:
                    tlist_inside_bnd.append([tList[i,0], tList[i,1], tList[i,2], mtotal[i], eta[i], \
                                                      spin1z, spin2z])
            else:
                pass
     
        return tlist_inside_bnd

    
class TaylorF2RedSpinChkBoundary(CheckBoundary):
    """
    A sub-class of CheckBoundary to check the boundary condition parameters
    for TaylorF2RedSpin approximant.
    """
    def __init__(self, args):
        CheckBoundary.__init__(self, args)
        
    def pts_inside_bndry(self, tList, coord_utils):
        """
        Return the points which are inside the specified parameter space.
        
        Parameters
        ----------
        tList : numpy.ndarray of order nx3
            Array of 3D points in dimensionless chirptime coordinates.
            
        coord_utils : callable object
            Contain the class CoordinateTransformUtils to convert the 
            tList values in mass spin coordinate system.
        """
        ind, mtotal, eta, mass1, mass2 = self._verify_mass_limit(tList[:,0], tList[:,1], coord_utils)
        index = []
        
        for i in ind:
            # Calculate assymetric mass ratio.
            delta_eta = (1.0 - 4.0*eta[i])**0.5

            # Calculate reduced spin.
            chi_r = 48*PI*tList[i,2]/(113*tList[i,1])
            
            # Find individual spin bound
            spin1z_min, spin1z_max, spin2z_min, spin2z_max = coord_utils._spin_bounds(mass1[i], mass2[i])
            
            factorm = (1.0 - 76.0*eta[i]/113.0 - delta_eta)
            factorp = (1.0 - 76.0*eta[i]/113.0 + delta_eta)
                
            # Calculate possible minimum and maximum values of first object
            # from limits of second objects spin.
            s1z_max = (2.0*chi_r - factorm*spin2z_min)/factorp
            s1z_min = (2.0*chi_r - factorm*spin2z_max)/factorp

            # Accept if first object spin belong inside limits.
            if (spin1z_max >= s1z_max >= spin1z_min or spin1z_max >= s1z_min >= spin1z_min) and abs(chi_r)<=1.0:
                index.append(i)
                    
            else:
                pass
        return tList[index]

    
class IMRPhenomBChkBoundary(CheckBoundary):
    """
    A sub-class of CheckBoundary to check the boundary condition parameters
    for IMRPhenomB approximant.
    """
    def __init__(self, args):
        CheckBoundary.__init__(self, args)                         
        
    def pts_inside_bndry(self, tList, coord_utils):
        """
        Return the points which are inside the specfied parameter space.
        
        Parameters
        ----------
        tList : numpy.ndarray of order nx3
            Array of 3D points in dimensionless chirptime coordinates.
            
        coord_utils : callable object
            Contain the class CoordinateTransformUtils to convert the 
            tList values in mass spin coordinate system.
        """
        
        ind, mtotal, eta, mass1, mass2 = self._verify_mass_limit(tList[:,0], tList[:,1], coord_utils)
        index = []
        for i in ind:
            # Calculate effective spin
            chi_eff = (17022.0 - tList[i,2] / eta[i] / tList[i,1]) / 9565.9
            
            # Find effective spin limits.
            chi_min, chi_max = coord_utils._spin_bounds(mass1[i], mass2[i])
            if chi_min <= chi_eff <= chi_max:
                index.append(i)
            else:
                pass
            
        return tList[index]

        
        
chk_bndry_utils = {
    "SEOBNRv4_ROM" : SingleSpinAlignedIMRChkBoundary,
    "IMRPhenomD" : SingleSpinAlignedIMRChkBoundary,
    "IMRPhenomB" : IMRPhenomBChkBoundary,
    "TaylorF2RedSpin" : TaylorF2RedSpinChkBoundary
    }       


def compute_delta_theta0_theta3_of_bounding_box(gamma):
    delta_theta0 = 2.0*np.sqrt(gamma[1,1]*(gamma[1,2]**2.0 - gamma[1,1]*gamma[2,2]) \
                               /((gamma[0,1]*gamma[1,2] - gamma[1,1]*gamma[0,2])**2.0 \
                               - (gamma[1,2]**2.0 - gamma[1,1]*gamma[2,2]) \
                               * (gamma[0,1]**2 - gamma[0,0]*gamma[1,1])))

    return delta_theta0


def NextPow2(number):
    return int(2**(np.ceil(np.log2( number ))))

def resize_ASD(ASD, length):
    if length > len(ASD):
        ASD2 = np.ones(length) * np.inf
        ASD2[ :len(ASD) ] = ASD.copy()
    else:
        ASD2 = ASD[ :length ]
    return ASD2

# Create workspace for match calculation
workspace_cache = CreateSBankWorkspaceCache()

class ExactMatchUtilsToFindValidNeighbors(object):
    """
    Class to incorporate the exact match function to calculate the match between the
    A3 star lattice neighbors and the existant templates. 
    """
    
    def __init__(self, args, ASD, coord_utils, bndry_utils, metric):
        self._phi0 = 0.0
        self._fref = args.fref
        self._flow = args.flow
        self._fhigh = args.fhigh
        self._df = args.df
        self._approx = lalsim.GetApproximantFromString( args.approximant )
        self._nhood_size_theta0 = 2.0 * PI * args.flow * args.neighborhood_size
        self._min_match = args.min_match
        self._max_mismatch = 1.0 - args.min_match
        self._length = int(NextPow2(args.fhigh/args.df) + 1)
        self._ASD = resize_ASD(ASD, self._length)
        self.coord_utils = coord_utils
        self._verify_dur_optimize_flow = bndry_utils._verify_dur_optimize_flow
        self.MetricInTheta0Theta3Theta3s = metric.MetricInTheta0Theta3Theta3s
        
    def _compute_whiten_normalized_fd_waveform(self, mass1, mass2, spin1z, spin2z, opt_flow=None):
        """
        Return a COMPLEX8FrequencySeries of the waveform, whitened by the
        given ASD and normalized. The ASD and waveform are not zero-padded 
        after fhigh.
        """
        
        _ASD_ = self._ASD.copy()
        if opt_flow is None:
            fLow = self._flow
        else:
            fLow = opt_flow
        

        wf = lalsim.SimInspiralChooseFDWaveform( mass1 * MSUN_SI, mass2 * MSUN_SI,
                0., 0., spin1z, 0., 0., spin2z, 1e6*PC_SI, 0., 0., 0., 0., 0.,
                self._df, self._flow, self._fhigh, self._fref, None, self._approx)[0]
        
        if wf.data.length == self._length:
            _ASD_ =  self._ASD.copy()
        else:
            _ASD_ = resize_ASD(self._ASD, wf.data.length)
        _ASD_[ :int(fLow/self._df) ] = np.inf
        
        whiten_wf = CreateCOMPLEX8FrequencySeries(wf.name, wf.epoch, wf.f0, wf.deltaF,\
                                            wf.sampleUnits, wf.data.length)
        whiten_wf.data.data[:] = wf.data.data[:]/_ASD_
        
        sigmasq = float(np.vdot(whiten_wf.data.data, whiten_wf.data.data).real * 4 * self._df)
        whiten_wf.data.data /= sigmasq**0.5
        
        return whiten_wf
    
    def _template_index_inside_bounding_box(self, proposal, g_ij, std_tlist, std_theta0_list):
        
        delta_theta0 = compute_delta_theta0_theta3_of_bounding_box(g_ij)
                
        low, high = find_neighbors(std_theta0_list, proposal[0], delta_theta0)
        tlist = std_tlist[low:high]
        t0list = std_theta0_list[low:high]
        abst0_list = [abs(t0list[i] - proposal[0]) for i in range(len(t0list))]
        index = [ i for (i,j) in sorted(enumerate(abst0_list), key=operator.itemgetter(1))]
                
        if 15000 > len(index) > 0:
            tlist = np.array(tlist)[index]
            delta_theta = np.array(tlist[:,:3] - proposal[:3])
            ds_sq = np.sum(delta_theta.T*np.dot(g_ij,delta_theta.T), axis=0)
            index = np.where(ds_sq < 1)[0] 
        else:
            index = []
            
        return index, tlist
                
        
    
    
    def _compute_max_match_against_bank( self, proposal, std_tList, std_theta0_list, std_mList, norm_dsq):
        
        output = [False, None]
        
        out = find_neighbors_distance(std_tList, std_theta0_list, std_mList, proposal, \
                                      self._nhood_size_theta0, norm_ds_sq=norm_dsq)
        
        if out != False:
            
            prop_mtotal, prop_eta, prop_spin1z, prop_spin2z = proposal[3:]
            prop_mass1, prop_mass2 = mtotal_eta_to_mass1_mass2(prop_mtotal, prop_eta)
            
            prop_opt_flow, prop_dur = self._verify_dur_optimize_flow( \
                                            prop_mass1, prop_mass2, prop_spin1z, prop_spin2z, self.coord_utils)
            
            if prop_dur is not False:
                
                proposal.extend([prop_opt_flow, prop_dur])
                Pgamma =  self.MetricInTheta0Theta3Theta3s(proposal, mis_match=self._max_mismatch)[0]
                # FIXME: Choice of mtotal is not optimal
                g_ij = Pgamma/5.0
                index, ftList = self._template_index_inside_bounding_box(proposal, g_ij, std_tList, std_theta0_list)
                whiten_prop_wf = self._compute_whiten_normalized_fd_waveform(prop_mass1, prop_mass2, \
                                                                 prop_spin1z, prop_spin2z)
                
                
            
                for ind in index:
                    tmp = ftList[ind]
                    tmp_mtotal, tmp_eta, tmp_spin1z, tmp_spin2z, tmp_opt_flow, tmp_dur = tmp[3:]
                    tmp_mass1, tmp_mass2 = mtotal_eta_to_mass1_mass2(tmp_mtotal, tmp_eta)
                    whiten_tmp_wf = self._compute_whiten_normalized_fd_waveform(tmp_mass1, tmp_mass2, tmp_spin1z, tmp_spin2z, opt_flow=tmp_opt_flow)
                
                    overlap = InspiralSBankComputeMatch(whiten_prop_wf, whiten_tmp_wf, workspace_cache)
                    
                    if overlap >= self._min_match:
                        return output
                output = [True, [proposal, Pgamma, whiten_prop_wf]]
        return output
    
    def _find_valid_neighbors_from_possible_neighbors(self, proposals, \
                            std_tList, std_theta0_list, std_mList, norm_dsq=0.99):
        full_output = []
        
        for proposal in proposals:
            out = self._compute_max_match_against_bank(proposal, std_tList, std_theta0_list, std_mList, norm_dsq)
            if out[0] == True:
                full_output.append(out[1])
            else: pass
            
        if len(full_output) > 1:
            
            valid_neighbors = [full_output[0]]
            for i in range(1, len(full_output)):
                
                wf1 = full_output[i][-1]
                overlap = 0.0; j = 0
                while overlap < self._min_match and j < len(valid_neighbors):
                    
                    wf2 = valid_neighbors[j][-1]
                    overlap = InspiralSBankComputeMatch(wf1, wf2, workspace_cache)
                    j += 1
            
                if overlap < self._min_match:
                    valid_neighbors.append(full_output[i])
                    
        else:
            valid_neighbors = full_output
               
        return valid_neighbors

            
        
def output_sngl_inspiral_table(args, tList, mList, coord_utils,
                               programName="", outdoc=None, proc_start_time=None, **kwargs):
    """
    Function that converts the information produced by the various pyCBC bank
    generation codes into a valid LIGOLW xml file containing a sngl_inspiral
    table and outputs to file.
    """
    
    optDict = args.__dict__
    if outdoc is None:
        outdoc = ligolw.Document()
        outdoc.appendChild(ligolw.LIGO_LW())
    
    lsctables.SnglInspiralTable.RowType = SnglInspiralTable
    tbl = lsctables.New(lsctables.SnglInspiralTable)
    outdoc.childNodes[-1].appendChild(tbl)

    tbl = lsctables.SnglInspiralTable.get_table(outdoc)

    process = ligolw_process.register_to_xmldoc(outdoc, programName, optDict, \
             cvs_repository="hdbank", cvs_entry_time=strftime('%Y-%m-%d %H:%M:%S +0000'), **kwargs)
    
       
    if proc_start_time is not None:
        process.start_time = proc_start_time
    proc_id = process.process_id
    
    for i in range(len(tList)):
        theta0, theta3, theta3s, mtotal, eta, spin1z, spin2z, flow, dur = tList[i]
        mass1, mass2 = mtotal_eta_to_mass1_mass2(mtotal, eta)
        values = [theta0, theta3, theta3s, mtotal, eta, mass1, mass2, spin1z, spin2z, flow, dur]
        col_names = ['tau0', 'tau3', 'tau4', 'mtotal', 'eta', 'mass1','mass2','spin1z','spin2z', \
                     'alpha6', 'template_duration']
        row = SnglInspiralTable()
        for colname, value in zip(col_names, values):
            setattr(row, colname, value)
        
        
        # Insert Gamma components if needed
        if args.write_metric is True and args.approximant in ["SEOBNRv4_ROM", "IMRPhenomD"]:
            gamma = mList[i]*(1.0 - args.min_match )
            GammaVals = [gamma[0,0], gamma[0, 1], gamma[0, 2], gamma[1, 1],\
                         gamma[1, 2], gamma[2, 2]]
            
            # assign the highest cutoff frequency, sigma_square and Gamma0-5 values

            row.f_final = args.fhigh
            row.sigmasq = mList[i][1][1]
            for i in range(len(GammaVals)):
                setattr(row, "Gamma"+str(i), GammaVals[i])
                
        row.process_id = proc_id
        row.event_id = tbl.get_next_id()
        tbl.append(row)
        
    # write the xml doc to disk     
    process.set_end_time_now()
    ligolw_utils.write_filename(outdoc, args.output_filename)

    