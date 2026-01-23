from config import tmpp_dir

LIGHT_SPEED=299792.458 # in km/s

mask_11_5=tmpp_dir+"mask_clean_11_5.npy.gz"
mask_12_5=tmpp_dir+"mask_clean_12_5.npy.gz"
#mask_11_5=2mpp_dir+"masks_11_5.npy.gz"
#mask_12_5=2mpp_dir+"masks_12_5.npy.gz"

#alpha,Mstar,PhiStar,lbar
default_2mpp_schechter=\
(-0.7286211634758224,
 -23.172904033796893,
 0.0113246633636846,
 393109973.22508669)

absmagmin_catalog=-26
absmagmax_catalog=-17

cut_bright=11.5
cut_faint=12.5

apertureDistanceLimit = 200. # We stop increasing the selection window at 200 Mpc/h
maxAperture = 3. # Maximum 3 Mpc/h angular separation
Vlink=350.
densityGroupSelection = 80 # High density catalog, Ramella 89
minMemberInGroup = 2 # Minimal number of member to consider a group is formed

K20C_INVALID_THRESHOLD=0.22 # This is a 2-sigma deviation, it corresponds to an uncertainty of 20% on the luminosity

Msun = 3.29
CHOSEN_MAG='K2MRS'
