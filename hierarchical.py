import gaussian_likelihood as gl
import numpy as np
import plotutils.parameterizations as par
import wl_likelihood as wl

class HierarchicalWLPosterior(object):
    r"""A hierarchical model for a set of weak-lensing observations.  We
    model each lens with an NFW profile, whose mass and concentration
    parameter are drawn from a (correlated) log-normal distribution

    ..math::

      \ln M_{200}, \ln c \sim N\left[ \mu, \Sigma \right]

    where :math:`\mu` and :math:`\Sigma` are parameters describing the
    overall distribution of lens systems.

    We assume that there are a number of weak lensing shear
    measurements for each lens system.  The measurements are treated
    as independent, and their sampling distribution is assumed to be
    log-normal, with a peak at the NFW shear prediction and with
    standard deviation equal to the quoted observational uncertainty:

    ..math::
      \ln \kappa(r) \sim N\left[ \mu_{NFW}(r; M_{200}, c), \sigma(r; M_{200}, c) \right]

    Where :math:`\mu_{NFW}(r; M_{200}, c)` is the log of the predicted
    shear from an NFW profile with parameters :math:`M_{200}` and
    :math:`c`, and :math:`\sigma(r; M_{200}, c)` is related to the
    predicted shear and observational uncertainty by the constraint
    that the standard deviation of the log-normal distribution matches
    the quoted uncertainty.

    """

    def __init__(self, rs, kappas, sigma_kappas, zl, zs):
        """Initialise the posterior object with the given observations.

        :param rs: A list of arrays, each array giving the radii at
          which the corresponding shear observations and uncertainties
          for each lens were taken.

        :param kappas: A list of arrays, giving the shear measurements
          for each lens.

        :param sigma_kappas: A list of arrays, giving the
          observational uncertainties for the shear observations for
          each lens.

        :param zl: An array giving the lens redshift for each lens.

        :param zs: An array giving the source redshift for each lens.

        """

        self._rs = rs
        self._kappas = kappas
        self._sigma_kappas = sigma_kappas
        self._zl = zl
        self._zs = zs

        self._wl_likelihoods = [wl.WeakLensingLikelihood(r, s, su, zl, zs) for \
                                r, s, su, zl, zs in \
                                zip(self.rs, self.kappas, self.sigma_kappas, \
                                    self.zl, self.zs)]

        @property
        def rs(self):
            return self._rs
        @property
        def kappas(self):
            return self._kappas
        @property
        def sigma_kappas(self):
            return self._sigma_kappas
        @property
        def zl(self):
            return self._zl
        @property
        def zs(self):
            return self._zs
        @property
        def wl_likelihoods(self):
            return self._wl_likelihoods
        @property
        def nlens(self):
            return len(self.rs)
        @property
        def dtype(self):
            return np.dtype([('mu', np.float, 2),
                             ('sigma_params', np.float, 3),
                             ('lens_params', np.float, (self.nlens, 2))])
        @property
        def nparams(self):
            return 5 + 2*self.nlens

        def to_params(self, p):
            return np.atleast_1d(p).view(self.dtype).squeeze()

        def _covariance_matrix(self, p):
            p = self.to_params(p)

            return par.cov_matrix(p['sigma_params'])

        def __call__(self, p):
            p = self.to_params(p)

            log_like = 0.0

            pgaussian = np.concatenate((p['mu'], p['sigma_params']))
            gaussian_likelihood = gl.GaussianLikelihood(p['lens_params'])
            log_like += gaussian_likelihood.log_likelihood(pgaussian)

            for lens_params, wll in zip(p['lens_params'], self.wl_likelihoods):
                log_like += wll.log_likelihood(lens_params)

            return log_like
