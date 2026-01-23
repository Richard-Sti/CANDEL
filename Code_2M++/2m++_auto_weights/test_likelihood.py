import numpy as np
import tmpp
import gzip

c = tmpp.load_catalog(gzip.GzipFile("data/2m++.npy.gz"))
# print c['group_id']

likelihood = tmpp.TMPP_Likelihood(c, dmin=500,dmax=15000,absmagmin=-26,absmagmax=-20)

c_used = likelihood.get_catalog()

likelihood.update_distances(c_used['distance'])
for M in np.arange(-24.,-19.,0.1):
  print "%g %10.20g" % (M, likelihood.compute_loglike(funcparams=[M,-0.9]))
