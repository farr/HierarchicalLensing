from cosmocalc import cosmocalc
import numpy as np
import plotutils.parameterizations as par
import scipy.stats as ss

def NFW(r, M200, c200, zl, zs):
    #this code calculates the shear at a given distance r from an object at redshift zl of mass M200 and concentration c200 assuming a background distribution of zs
    
    #set cosmology
    h    = 0.7                          #hubble parameter
    Ho   = h*100.0                      #km s^-1 Mpc-1
    Om   = 0.28                         #cosmological parameter
    Ol   = 0.72                         #cosmological parameter
    c    = 299792.0                     #speed of light in km/s
    Mpc  = 3.0856e19                    #1Mpc in km
    cMpc = c/Mpc                        #speed of light in Mpc/s
    pc   = 3.0856e16                    #1pc in m
    Msun = 1.989e30                     #1Msolar in kg
    G    = 6.6726e-11                   #G in m^3 kg-1 s-2
    GG   = G*(pc**(-3.0))*(1e-18)*Msun    #G in Mpc^3 Msun^-1 s^-2
    Ez   = np.sqrt(Om*(1.0+zl)**3.0+Ol) #evolution
    pcrit= 3.0*((Ho*Ez/Mpc)**2.0)/(8.0*np.pi*GG)
    
    #calculate angular diameter distances in Mpc to lens and source
    dL  =   cosmocalc(zl,Ho,Om,Ol)['DA_Mpc']
    dS  =   cosmocalc(zs, Ho,Om, Ol)['DA_Mpc']
    dLS =   (dS*(1.0+zs)-dL*(1.0+zl))/(1.0+zs)
    
    sigmacr=cMpc**2.0*dS/(4.0*GG*np.pi*dL*dLS)    #mean critical surface mass density in units Mpc^-2 Msolar
    
    p200    =   200.0*pcrit                         #density that is 200 times critical
    p500    =   500.0*pcrit                         #density that is 500 times critical
    r200    =   (3.0*M200/(4.0*np.pi*p200))**(1.0/3.0)     #radius within which the density is 200 times critical
    
    rs      =   r200/c200               #characteristic radius in Mpc
    ps      =   (p200*c200**3.0)/(3.0*(np.log(1.0+c200)-c200/(1.0+c200)))   #core density in Msolar Mpc-3

    r = np.atleast_1d(r)

    term = np.where(r <= rs, \
                    2.0*np.arctanh(np.sqrt((1.0-(r/rs))/(1.0+(r/rs))))/(np.sqrt(1.0-np.square(r/rs))), \
                    2.0*np.arctan(np.sqrt(((r/rs)-1.0)/(1.0+(r/rs))))/(np.sqrt(np.square(r/rs)-1.0)))
    sigma = 2.0*rs*ps*(1.0-term)/((np.square(r/rs))-1.0)
    meansigma = 4.0*rs*ps*(term+np.log((r/rs)/2.0))/(np.square(r/rs))
    rhoNFW = ps/((r/rs)*np.square(1.0+r/rs))

    kappaNFW = sigma/sigmacr              #calculate convergence
    meankappaNFW = meansigma/sigmacr      #mean convergence
    gammaNFW = meankappaNFW-kappaNFW                   #calculate shear
    reducedshear = np.abs(gammaNFW)/(1.0-kappaNFW)        #calculate reduced shear

    return reducedshear

class WeakLensingLikelihood(object):
    """Likelihood function for weak lensing measurements of shear.  Our
    model is that the measured shear is normally distributed about the
    true shear with standard deviation equal to the measured
    uncertainty.  This is, of course, crap, since it predicts some
    negative data, but can be extended later.

    """
    
    def __init__(self, rs, shears, shear_uncert, zl, zs):
        """Initialise the likelihood with the observations of shear (and
        associated uncertainty) at the given radii.

        :param rs: The radii at which the shear has been measured (in
          Mpc).

        :param shears: The dimensionless shear measured at that
          radius.

        :param shear_uncert: The 1-sigma uncertainty on the shear.

        :param zl: The lens redshift.

        :param zs: The source redshift.

        """
        self._rs = rs
        self._shears = shears
        self._shear_uncert = shear_uncert
        self._zl = zl
        self._zs = zs

    @property
    def rs(self):
        return self._rs
    @property
    def shears(self):
        return self._shears
    @property
    def shear_uncert(self):
        return self._shear_uncert
    @property
    def zl(self):
        return self._zl
    @property
    def zs(self):
        return self._zs
    @property
    def clow(self):
        return 1.0
    @property
    def chigh(self):
        return 100.0
    @property
    def dtype(self):
        return np.dtype([('log_m200', np.float),
                         ('c', np.float)])

    def to_params(self, p):
        """Returns a view of the array ``p`` with named columns corresponding
        to the parameters of the lens model: ``log_m200`` and
        ``log_cm1`` (both natural logs).

        """
        return np.atleast_1d(p).view(self.dtype).squeeze()

    def log_likelihood(self, p):
        """Returns the log of the likelihood of the stored data, assuming that
        each measurement at a given radius is independent 

        """

        p = self.to_params(p)

        m200 = np.exp(p['log_m200'])
        c = p['c']

        if c < self.clow or c > self.chigh:
            return np.NINF

        shear_true = NFW(self.rs, m200, c, self.zl, self.zs)

        return np.sum(ss.norm.logpdf(self.shears, loc=shear_true, scale=self.shear_uncert))

    def __call__(self, p):
        """Synonym for ``log_likelihood``."""

        return self.log_likelihood(p)
