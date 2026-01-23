import numpy as np
import _lflike as ll

LIGHT_SPEED=299792.458 # in km/s

cat = np.load("fake_cat.npy")

dlist=cat[0,:]
mlist=cat[1,:]
cb=cat[2,:]
cf=cat[3,:]

# alpha=-1.02  (actually -1.05)
# Mstar=-23.27 (actually -23.29)

maglimb=11.5
maglimf=12.5

cosmo = ll.Cosmology()
cosmo.omega_m = 0.26
cosmo.omega_de = 0.74
cosmo.w = -1

dl = ll.LuminosityDistance(cosmo)
mu = np.empty(dlist.size)
for i in range(dlist.size):
    mu[i] = 5.0*np.log10(dl.dl(dlist[i]/LIGHT_SPEED)*1e5)

Mlist = mlist - mu
Mb_list = maglimb - mu
Mf_list = maglimf - mu


lf_helper = ll.LuminosityFunctionHelper()
lf_helper.setCosmology(cosmo)
lf_helper.chooseWeight(0)
lf_helper.setLuminosityFunction(0)
lf_helper.set_absmag_range(-27, -16)

idx = np.where((Mlist >= -27)*(Mlist < -16))[0]
Mlist = Mlist[idx]
Mb_list = Mb_list[idx]
Mf_list = Mf_list[idx]
cb = cb[idx]
cf = cf[idx]

lf_like = ll.LuminosityFunctionLikelihood(Mlist, Mb_list, Mf_list,
                                          cb, cf)

params = np.array([-24, -0.9]) #-23.295970513126907, -1.0489785674445669 ])
step_p = np.array([0.01,0.01])

lf_like.findBestParameters(lf_helper, params, step_p, 0.01)

print params
