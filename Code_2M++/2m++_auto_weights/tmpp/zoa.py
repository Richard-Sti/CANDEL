import numpy as np
from constants import *

__all__=['generate_zoa']

def subGenerationZoneOfAvoidance(catalog, threshold):
    from numpy import sin, pi, arcsin, where, empty
    from numpy.random import random
    
    sinb0 = sin(threshold*pi/180)
    sinb1 = 2*sin(threshold*pi/180)
    
    sinb = sin(catalog['gal_lat']*pi/180)
    
    idx0 = where((sinb>sinb0)*(sinb<sinb1))[0]
    idx1 =  where((sinb<-sinb0)*(sinb>-sinb1))[0]
    
    f0_sinb = sinb0-sinb[idx0]
    f1_sinb = -sinb0-sinb[idx1]
    
    folded_zoa = empty(shape=(idx0.shape[0]+idx1.shape[0]),dtype=catalog.dtype)
#	folded_zoa = empty(shape=(idx0.shape[0]),dtype=catalog.dtype)
    folded_zoa[0:idx0.shape[0]] = catalog[idx0]
    folded_zoa[idx0.shape[0]:] = catalog[idx1]
    
    folded_zoa['gal_lat'][0:idx0.shape[0]] = arcsin(f0_sinb)*180/pi
    folded_zoa['gal_lat'][idx0.shape[0]:] = arcsin(f1_sinb)*180/pi
    folded_zoa['flag_2mrs'] = 0
    folded_zoa['flag_6df'] = 0
    folded_zoa['flag_sdss'] = 0
    folded_zoa['flag_sdss_mask'] = 0
    folded_zoa['flag_6df_mask'] = 0
    folded_zoa['flag_6dfext'] = 0
    folded_zoa['flag_copied'] = 0
    folded_zoa['flag_vcmb'] = 0
    folded_zoa['flag_vcmb_mask'] = 0
    folded_zoa['flag_lga'] = 0
    folded_zoa['zref_code'] = ''
    try:
        folded_zoa['group_id'] = -1
    except ValueError:
        pass
    
#	folded_zoa = folded_zoa[where(random(folded_zoa.shape) < 0.5)]
    
    return folded_zoa

def generate_zoa(catalog):	

    from numpy import sin, pi, arcsin, where, empty
    from numpy.random import random
    from helper import grow_catalog

    lat_disk_threshold=5.
    lat_bulge_threshold=10.
    
    long_bulge=30.
    
    catbulge = catalog[np.where((catalog['gal_long']<long_bulge)+(catalog['gal_long']>(360-long_bulge)))]
    catdisk = catalog[np.where((catalog['gal_long']>long_bulge)*(catalog['gal_long']<(360-long_bulge)))]
    
    zoabulge=subGenerationZoneOfAvoidance(catbulge,lat_bulge_threshold)
    zoadisk=subGenerationZoneOfAvoidance(catdisk,lat_disk_threshold)
    
    return grow_catalog(zoabulge,zoadisk)
