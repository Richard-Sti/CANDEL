from constants import *
import numpy as np
import catalog_io as cio
from lflike3 import LF_SCHECHTER_FUNCTION, LF_DOUBLE_POWER_LAW_FUNCTION, LF_TRIPLE_POWER_LAW_FUNCTION, LF_DOUBLE_SCHECHTER_FUNCTION, USE_NUMBER_SCHEME, USE_LUMINOSITY_SCHEME

NUMBER_WEIGHING=USE_NUMBER_SCHEME
LUMINOSITY_WEIGHING=USE_LUMINOSITY_SCHEME

__all__ = ['compute_surface', 'recompute_fit', 'compute_2mpp_weights',
           'LF_SCHECHTER_FUNCTION', 'LF_DOUBLE_POWER_LAW_FUNCTION', 
           'LF_TRIPLE_POWER_LAW_FUNCTION', 'LF_DOUBLE_SCHECHTER_FUNCTION',
           'NUMBER_WEIGHING', 'LUMINOSITY_WEIGHING','TMPP_Likelihood']


def generateSubsetForSchechter(catalog,dmin,dmax,m11_5,m12_5,subset=0,global_mlim=cut_faint):
    if subset == 0:
        n1 = 'c1_all'
        n2 = 'c2_all'
    elif subset == 1:
        n1 = 'c1_6df'
        n2 = 'c2_6df'
    elif subset == 2:
        n1 = 'c1_sdss'
        n2 = 'c2_sdss'
    elif subset == 3:
        n1 = 'c_2mrs'
        n2 = None
    else:
        raise ValueError("invalid subset")
 
    # Do the photometry cut, (K < 12.5 or 11.5 depending on region + good completeness)
    condition = catalog['flag_vcmb_mask']* \
        (catalog['flag_2mrs_mask_final'] + \
        catalog['flag_6df_mask_final'] +  \
        catalog['flag_sdss_mask_final'])
    # Only accept decent objects with a positive distance and not zoa
    condition *= (catalog['flag_zoa']==0)

    if (n2 == None):
        # Do extra photometry cut
        condition *= (catalog['K2MRS']<=cut_bright)*(catalog[n1]>=0.5)
    else:
        condition *= (catalog['K2MRS']<=cut_bright)*(catalog[n1]>0.5)+(catalog[n2]>0.5)*(catalog['K2MRS']<=cut_faint)*(catalog['K2MRS']>cut_bright)

    condition *= catalog['K2MRS']<global_mlim

    # Select by distance
    condition *= (catalog['distance']>=dmin) * (catalog['distance']<=dmax)

    c11_5 = np.where((m11_5 >= 0.5) * (m12_5 < 0.5))[0]
    c12_5 = np.where((m11_5 >= 0.5) * (m12_5 >= 0.5))[0]

    catalog = catalog[np.where(condition)]


    mag=catalog['K2MRS'][:]
    maglimb=np.empty(mag.size,dtype=float)
    maglimf=np.empty(mag.size,dtype=float)
    maglimb[:] = min(cut_bright,global_mlim)
    maglimf[:] = min(cut_faint,global_mlim)
    dists=catalog['distance'].copy()
    cb=catalog[n1].copy()
    # All object with low completess at K<11.5 were removed
    cb[np.where(cb<0.5)] = 0
    if (n2 != None):
	    cf=catalog[n2].copy()
	    # Put the completeness of the faint end to zero as they have
	    # been taken away
	    cf[np.where(cf<0.5)] = 0
    else:
	    # Put the completeness to zero. There is no object in the faint magnitude band
	    cf=np.zeros(cb.size)

    cf[np.where(np.isnan(cf))] = 0
    cb[np.where(np.isnan(cb))] = 0

    assert len(np.where(mag > maglimf)[0]) == 0

    return catalog,mag,maglimb,maglimf,cb,cf,dists,c11_5,c12_5


def compute_surface(lh,m11_5,m12_5,
                    dmin,dmax,c11_5,c12_5,J3=40.,
                    absmagmin=None,absmagmax=None,global_mlim=cut_faint):
    if J3 != 0.:
       raise RuntimeError("Not supported J3 != 0")

    else:
       m_faint = min(global_mlim,cut_faint)
       m_bright = min(global_mlim,cut_bright)
       volb = lh.computePhiStarNormalisation(m_faint, m_bright, [0.0], [1.0], dmin, dmax, 0.0, 1e-3)
       volf = lh.computePhiStarNormalisation(m_faint, m_bright, [1.0], [0.0], dmin, dmax, 0.0, 1e-3)

       effective_vol_b = sum(volb * m11_5[c11_5]) + sum(volb*m11_5[c12_5]);
       effective_vol_f = sum(volf * m12_5[c12_5])

       print 'effective_vol_b = ', effective_vol_b
       print 'effective_vol_f = ', effective_vol_f
       effective_vol = effective_vol_b + effective_vol_f

    effective_vol *= 4*np.pi/(len(m11_5))

    print "Effective volume = ", effective_vol, " (Mpc/h)^3"
    print "Claimed volume = ", 4*np.pi*(dmax**3-dmin**3), " (Mpc/h)^3"

    return effective_vol

def recompute_fit(catalog,dmax=8000.,dmin=1000.,
                  do_PhiStar=True,do_Exact=True,J3=0.,
                  subset=0,absmagmin=-25,absmagmax=-20.9,functype=LF_SCHECHTER_FUNCTION,global_mlim=cut_faint):
    """
    Attempt to compute Schechter luminosity function parameters and 
    normalization. The subset on which to do the adjustment is determined
    by dmin, dmax (apparent velocity min&max), J3 (the parameter in Davis&Huchra, only "0" is supported
    in the current implementation), and subset (0 for all, 1 for 6dF, 2 for SDSS, 3 for 2MRS)
    functype:
       * LF_SCHECHTER_FUNCTION
       * LF_DOUBLE_POWER_LAW_FUNCTION
       * LF_TRIPLE_POWER_LAW_FUNCTION
       * LF_DOUBLE_SCHECHTER_FUNCTION
    """

    import lflike3 as ll
    import gzip

    m11_5 = np.load(gzip.GzipFile(filename=mask_11_5))
    m12_5 = np.load(gzip.GzipFile(filename=mask_12_5))

    if subset == 0:
      m11_5 = m11_5[3]
      m12_5 = m12_5[3]
    elif subset == 1:
      m11_5 = m11_5[1]
      m12_5 = m12_5[1]
    elif subset == 2:
      m11_5 = m11_5[2]
      m12_5 = m12_5[2]
    elif subset == 3:
      m11_5 = m11_5[3]
      m12_5 = m12_5[3].copy()
      m12_5[:] = 0.
    else:
      raise ValueError("invalid subset");

    if global_mlim > cut_faint:
      raise ValueError("Unsupported global magnitude limit. Maximum is " + str(cut_faint))

    catalog,mag,maglimb,maglimf,cb,cf,dists,c11_5,c12_5 = \
	generateSubsetForSchechter(catalog,dmin,dmax,m11_5,m12_5,subset=subset,global_mlim=global_mlim)

    lh = ll.LuminosityFunctionHelper()
    cosmo = ll.Cosmology()
    cosmo.omega_m = 0.30
    cosmo.omega_de = 0.70
    cosmo.w = -1.0

    lh.setSunAbsoluteMagnitude(Msun)
    lh.setCosmology(cosmo)
    lh.set_absmag_range(absmagmin, absmagmax)

    dists = np.array(dists, dtype=np.double)/100.
    mag = np.array(mag, dtype=np.double)
    cb = np.array(cb, dtype=np.double)
    cf = np.array(cf, dtype=np.double)

    absmag = lh.computeAbsoluteMagnitudes(dists, mag)
    absmaglimb = lh.computeAbsoluteMagnitudes(dists, maglimb)
    absmaglimf = lh.computeAbsoluteMagnitudes(dists, maglimf)

    likelihood = ll.LuminosityFunctionLikelihood(absmag, absmaglimb, absmaglimf, cb, cf)

    lh.setLuminosityFunction(functype)

    if functype == LF_SCHECHTER_FUNCTION:
	    params = [-24.0, -0.9]
	    step=[0.01,0.01]
    elif functype == LF_DOUBLE_POWER_LAW_FUNCTION:
	    params = [-24.0,-0.9,-5.9]
	    step = [0.01,0.01,0.01]
    elif functype == LF_TRIPLE_POWER_LAW_FUNCTION:
	    params = [-22.0,-24,-0.9,-2.9,-5.9]
	    step = [0.01,0.01,0.01,0.01,0.01]
    elif functype == LF_DOUBLE_SCHECHTER_FUNCTION:
	    params = [-23.17, -0.72, -20, -1.18, 0.30]
	    step=[0.01, 0.01, 0.01, 0.01, 0.01]
            
    likelihood.findBestParameters(lh, params, step, 1e-3)
    nbar = likelihood.computeEffectiveNumber(lh, J3);

     # Compute real angular distance
    if do_PhiStar:
      effective_vol = compute_surface(lh,m11_5,m12_5,dmin,dmax,c11_5,c12_5,J3=J3,global_mlim=global_mlim)

      err_phi_star = np.sqrt(nbar)/nbar
      nbar /= effective_vol

      # Now we want to transform the mean density to the normalization constant
      # It is a matter of multiplying nbar by int_{M_min}^{M_max} Phi(M) dM, with 

      phistar = nbar / lh.integral_nw_lumfun_2m(absmagmax, absmagmin)
      err_phi_star *= phistar 

    return params,(phistar,err_phi_star)

def compute_2mpp_weights_bare(catalog,lh,global_mlim=cut_faint):
    """
    This function needs a catalog and already setup luminosity function helper (lhelper)
    """
    from helper import add_field

    mag=catalog['K2MRS']
    maglimb=np.empty(len(mag),dtype=float)
    maglimf=np.empty(len(mag),dtype=float)
    maglimb[:] = min(cut_bright,global_mlim)
    maglimf[:] = min(cut_faint,global_mlim)
    dists=catalog['distance'][:]
    cb=catalog['c1_all'][:]
    cf=catalog['c2_all'][:]
    
    w = np.zeros(mag.size)
    
    lh.computeGalaxyWeights(mag, maglimf, maglimb, cf, cb, dists, w)
    AM = np.array(lh.computeAbsoluteMagnitudes(dists/100., mag),dtype=np.float32)
    
    try:
        v = catalog['weight'][0]
    except ValueError:
        catalog = add_field(catalog,[('weight','f')])

    try:
        v = catalog['AbsMag'][0]
    except ValueError:
        catalog = add_field(catalog,[('AbsMag','f')])
        
    catalog['weight'] = np.array(w,dtype='float32')
    catalog['AbsMag'] = AM
    
    return catalog


def compute_2mpp_weights(catalog,functype=LF_SCHECHTER_FUNCTION,wtype=NUMBER_WEIGHING,funcparams=(),dmin=1000.,absmagmin=-28,absmagmax=-17,global_mlim=cut_faint):
    import lflike3 as ll

    lh =  ll.LuminosityFunctionHelper()
    lh.set_absmag_range(absmagmin, absmagmax)

    cosmo = ll.Cosmology()
    cosmo.omega_m = 0.30
    cosmo.omega_de = 0.70
    cosmo.w = -1.0

    lh.setSunAbsoluteMagnitude(Msun)
    lh.setCosmology(cosmo)
    lh.setLuminosityFunction(functype)
    lh.updateLuminosityFunctionParameters(funcparams)

    lh.chooseWeight(wtype)
    
    catalog = compute_2mpp_weights_bare(catalog,lh,global_mlim=global_mlim)

    return catalog



class TMPP_Likelihood(object):

    def __init__(self,catalog,dmin=1000.,dmax=8000.,
                 subset=0,absmagmin=-25,absmagmax=-20.9,
                 functype=LF_SCHECHTER_FUNCTION,global_mlim=cut_faint):
        import lflike3 as ll
        import gzip
    
        m11_5 = np.load(gzip.GzipFile(filename=mask_11_5))
        m12_5 = np.load(gzip.GzipFile(filename=mask_12_5))
        if subset == 0:
            m11_5 = m11_5[3]
            m12_5 = m12_5[3]
        elif subset == 1:
            m11_5 = m11_5[1]
            m12_5 = m12_5[1]
        elif subset == 2:
            m11_5 = m11_5[2]
            m12_5 = m12_5[2]
        elif subset == 3:
            m11_5 = m11_5[3]
            m12_5 = m12_5[3].copy()
            m12_5[:] = 0.
        else:
            raise ValueError("invalid subset");

        if global_mlim > cut_faint:
          raise ValueError("Unsupported global magnitude limit. Maximum is " + str(cut_faint))

        catalog,mag,maglimb,maglimf,cb,cf,dists,c11_5,c12_5 = \
        	generateSubsetForSchechter(catalog,dmin,dmax,m11_5,m12_5,subset=subset,global_mlim=global_mlim)

        self.catalog = catalog
        self.cb = cb
        self.cf = cf
        self.maglimb = maglimb
        self.maglimf = maglimf
        self.mag = mag

        self.lh = ll.LuminosityFunctionHelper()
        cosmo = ll.Cosmology()
        cosmo.omega_m = 0.30
        cosmo.omega_de = 0.70
        cosmo.w = -1.0

        self.lh.setSunAbsoluteMagnitude(Msun)
        self.lh.setCosmology(cosmo)
        self.lh.set_absmag_range(absmagmin, absmagmax)
        self.lh.setLuminosityFunction(functype)

        self.mag = np.array(mag, dtype=np.double)
        self.cb = np.array(cb, dtype=np.double)
        self.cf = np.array(cf, dtype=np.double)

    def update_distances(self,dists):

        dists = np.array(dists, dtype=np.double)/100.

        self.absmag = self.lh.computeAbsoluteMagnitudes(dists, self.mag)
        self.absmaglimb = self.lh.computeAbsoluteMagnitudes(dists, self.maglimb)
        self.absmaglimf = self.lh.computeAbsoluteMagnitudes(dists, self.maglimf)

    def get_catalog(self):
        return self.catalog

    def compute_loglike(self,funcparams=()):
        import lflike3 as ll

        likelihood = ll.LuminosityFunctionLikelihood(self.absmag, self.absmaglimb, self.absmaglimf, 
                                                    self.cb, self.cf)

        self.lh.updateLuminosityFunctionParameters(funcparams)

        return likelihood.computeLogLike(self.lh)

