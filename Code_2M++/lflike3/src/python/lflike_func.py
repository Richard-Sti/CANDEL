#
# This is the LFLike wrapper module to make the interface more 
# user friendly and safe in python
#
"""This module provides the function luminosityFunctionLike and the
helper exception LuminosityFunctionError. 
Written by G. Lavaux, 2010, 2011
"""

LIGHT_SPEED=299792.458

print "This is Luminosity function fit package tool version 3.0-"

class LuminosityFunctionError(Exception):
	def __init__(self, what):
		self.what = what

	def __str__(self):
		return repr(self.what)

def proxy_integrand_number_galaxies(d0,lfobj,mstar,alpha,maglimb,maglimf,cb,cf,J3):
	return lfobj.integrand_number_galaxies(d0,mstar,alpha,maglimb,maglimf,cb,cf,J3)

class LF_compute:

	def __init__(self, absmagmin,absmagmax,appmaglim):
		self.absmagminLF=absmagmin
		self.absmagmaxLF=absmagmax
		self.appmaglimLF=appmaglim

	def computeNbar(self, phistar, mstar, alpha, absmagmax, absmagmin=None):
		import lflike._lflike as ll

		if (absmagmin == None):
			return phistar * ll.lflike.intschechnum2m(mstar, alpha, absmagmax, absmagmax, 1.0, 0.0)
		else:
			return phistar * ll.lflike.intschechnum2m(mstar, alpha, absmagmin, absmagmax, 0.0, 1.0)

	def computeNbarDistance(self, phistar, mstar, alpha, maglimb, maglimf, dist_min, dist_max):
		import lflike._lflike as ll
		import numpy as np

		absmagmax = self.absmagmaxLF
		absmagmin = self.absmagminLF

		return self.computeNbar(phistar, mstar, alpha, absmagmax, absmagmin)

	def integrand_number_galaxies(self,d0,mstar,alpha,maglimb,maglimf,cb,cf,J3):
		import lflike._lflike as ll
		import numpy as np

		dmodulus = 5.0*np.log10(d0*1e5)
		absmaglimb = maglimb - dmodulus
		absmaglimf = maglimf - dmodulus
                c = ll.lflike.intschechnum2m(mstar,alpha,absmaglimb,absmaglimf,cb,cf)
		
		return d0**2*c/(1+J3*c)

	def computePhiStarNormalisation(self,mstar,alpha,maglimb,maglimf,cb,cf,dmin,dmax, J3=40., epsrel=1e-3):
		import lflike._lflike as ll
		import numpy as np

                drealMin = ll.luminosity_distance.dl(dmin/LIGHT_SPEED)
                drealMax = ll.luminosity_distance.dl(dmax/LIGHT_SPEED)

		if (J3==0.):
		    epsabs = (drealMax**3-drealMin**3)/3. * epsrel 
		else:
		    epsabs = (drealMax**3-drealMin**3)/3. * epsrel / J3

#		thisabsmagmin = maglimb - 5.0*np.log10(drealMax) - 25
#		thisabsmagmax = maglimf - 5.0*np.log10(drealMin) - 25

	        absmagmin = self.absmagminLF; #max(thisabsmagmin, self.absmagminLF)
		absmagmax = self.absmagmaxLF; #min(thisabsmagmax, self.absmagmaxLF)

		num = ll.lf_norm.computeDavisHuchraNormalization_magrange(mstar, alpha, \
			       drealMin, drealMax, maglimb, maglimf, cb, cf, \
			       J3, epsabs, epsrel, \
			       absmagmin=absmagmin, absmagmax=absmagmax)

#		print num, epsabs, epsrel
#		print num,epsabs

		return num

        def predictNumber(self,mstar,alpha,maglimb,maglimf,cb,cf,dmin,dmax,J3=0.):
	        from scipy.integrate import quad
		import lflike._lflike as ll


		if (dmax <= dmin):
			return 0

                drealMin = ll.luminosity_distance.dl(dmin/LIGHT_SPEED)
                drealMax = ll.luminosity_distance.dl(dmax/LIGHT_SPEED)

                num,errnum=quad(proxy_integrand_number_galaxies,drealMin,drealMax,args=(self,mstar,alpha,maglimb,maglimf,cb,cf,J3)) 
		return num

	def luminosityFunctionLike(self,mag,maglimb,maglimf,cb,cf,distances,J3=40.):
		"""This function computes the maximum likelihood value of
		the luminosity function parameter alpha and mstar using the absolute
		magnitude of the objects "absmag", and the limiting absolute magnitude
		for each object "absmaglim". 

		"absmag" and "absmaglim" must be 1-d arrays and the same size.
		The return value is a tuple of two values: the computed alpha and mstar
		"""

		import _lflike as lf
		import numpy as np
	
		alpha=-0.9
		mstar=-24.

		if (len(mag.shape) != 1):
			raise LuminosityFunctionError("mag must be a 1D array");
		if (len(maglimb.shape) != 1 or len(maglimf.shape) != 1):
			raise LuminosityFunctionError("maglim must be a 1D array");
		if (len(maglimb) != len(mag) or len(maglimf) != len(mag)):
			raise LuminosityFunctionError("boggus maglim");
		if (len(cb.shape) != 1 or len(cf.shape) != 1):
			raise LuminosityFunctionError("completeness must be 1D-arrays");
		if (len(cb) != len(mag) or len(cf) != len(mag)):
			raise LuminosityFunctionError("the number of completenesses must be equal to the number of objects");
		
		
		if (alpha == -1.0 or alpha >= 0.0):
			raise LuminosityFunctionError("Alpha value is incorrect for convergence")
		
		a_array = np.array(alpha,dtype=np.double)
		m_array = np.array(mstar,dtype=np.double)
		
		if (a_array.shape != ()): 
			raise LuminosityFunctionError("alpha must be a scalar")
		if (m_array.shape != ()): 
			raise LuminosityFunctionError("mstar must be a scalar")
		
		lf.cosmo.setcosmo(0.30,0.70,-1)
		lf.luminosity_distance.Setup_Dl()
		
		lf.lflike.dminLF = min(distances)*0.9
		lf.lflike.dmaxLF = max(distances)*1.1
		lf.lflike.absmagmaxLF = self.absmagmaxLF
		lf.lflike.absmagminLF = self.absmagminLF
		lf.lflike.appmaglimLF = self.appmaglimLF

#		print 'dminLF=', lf.lflike.dminLF
		
		print 'getabsmags...'
		lf.lflike.getabsmags(distances,mag,maglimb,maglimf,mag.shape[0])
		
		print 'selectLFgals...'
		lf.lflike.selectLFgals(distances, cb, cf, mag.shape[0])
		
		print 'schechterML...'
		lf.lflike.SchechterML(m_array, a_array)
		
		print 'normalisation...'
		phi_star = lf.lflike.getphistarvol(m_array, a_array, J3)
		
		return (float(a_array),float(m_array),phi_star)
