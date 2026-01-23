from constants import *

__all__=['extractRedshiftSurvey','load_catalog']


def extractRedshiftSurvey(catalog,vmax=35000,include_all=False):

    condition = catalog['flag_vcmb_mask']*\
	(catalog['flag_2mrs_mask_final'] + \
	 catalog['flag_6df_mask_final'] + \
	 catalog['flag_sdss_mask_final'])

    try:
	condition *= (catalog['flag_zoa']==0)
    except ValueError:
	pass

    if (include_all):
        c0 = (catalog['K2MRS']<=cut_bright)*(catalog['flag_2mrs_mask_final'])*(catalog['flag_6df_mask_final']==0)*(catalog['flag_sdss_mask_final']==0)
        c0 += (catalog['K2MRS']<=cut_faint)*(catalog['flag_6df_mask_final']+catalog['flag_sdss_mask_final'])
	condition *= c0

    ## UPDATE THIS WITH DISTANCE
    condition *= catalog['velcmb']<vmax
    ###

    return condition


def load_catalog(fname,vmax=20000.):
    from zoa import generate_zoa
    import numpy as np
    from helper import grow_catalog

    c = np.load(fname)

    c = c[np.where(extractRedshiftSurvey(c,vmax=vmax))]
    idx = np.where(c['flag_2mrs_mask_final'])
    c['c2_all'][idx] = 0
    
    c_zoa = generate_zoa(c)

    return grow_catalog(c, c_zoa, marker_name='flag_zoa', marker1=0, marker2=1)


