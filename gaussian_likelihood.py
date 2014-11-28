import numpy as np
import plotutils.parameterizations as par
import scipy.stats as ss

class GaussianLikelihood(object):
    """Represents a corellated Gaussian in some number of dimensions.

    """

    def __init__(self, data):
        """Initialise the likelihood function with the given data.

        :param data: Should be a two-dimensional array shape ``(Ndata,
          Ndim)`` representing the data over which the likelihood is
          to be computed.

        """
        
        data = np.atleast_2d(data)

        self._dim = data.shape[1]
        self._N = data.shape[0]

        self._data = data

    @property
    def dim(self):
        return self._dim
    @property
    def N(self):
        return self._N
    @property
    def data(self):
        return self._data
    @property
    def dtype(self):
        """A data type suitable for parameters to this likelihood function.
        The type represents a named array with element names 

        ``mean``
          The ``Ndim`` array of mean values.

        ``cov_params`` 
          A parameterisation for the covariance matrix of the
          Gaussian.  See :func:`ss.cov_matrix` and related functions
          for the parameterisation used.

        """
        
        return np.dtype([('mean', np.float, self.dim),
                         ('cov_params', np.float, self.dim*(self.dim+1)/2)])
    @property
    def nparams(self):
        return self.dim + self.dim*(self.dim+1)/2

    def to_params(self, p):
        """Returns a view of the input array as parameters to the likelihood
        function.

        """

        return np.atleast_1d(p).view(self.dtype).squeeze()

    def log_likelihood(self, p):
        """Returns the log-likelihood of the given data at parameters ``p``.

        """
        
        p = self.to_params(p)

        mu = p['mean']
        cov = par.cov_matrix(p['cov_params'])

        return np.sum(ss.multivariate_normal.logpdf(self.data, mu, cov))

    def __call__(self, p):
        """Synonym for ``log_likelihood``."""

        return self.log_likelihood(p)
    
