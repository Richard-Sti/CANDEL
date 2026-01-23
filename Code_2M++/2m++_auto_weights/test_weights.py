import tmpp
import gzip

c = tmpp.load_catalog(gzip.GzipFile("data/2m++.npy.gz"))
# print c['group_id']
p0,p1 = tmpp.recompute_fit(c, dmin=500,dmax=15000., absmagmin=-26, absmagmax=-20)

c = tmpp.compute_2mpp_weights(c, funcparams=p0, wtype=tmpp.LUMINOSITY_WEIGHING, absmagmin=-26, absmagmax=-20)
