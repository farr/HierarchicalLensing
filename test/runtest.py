#!/usr/bin/env python3

from argparse import ArgumentParser
import bz2
import os
import pickle
import pystan

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--niter', metavar='N', type=int, default=2000, help='number of iterations (default: %(default)s)')

    parser.add_argument('--nchain', metavar='N', type=int, default=4, help='number of chains (default: %(default)s)')

    args = parser.parse_args()

    with bz2.BZ2File('testdata.pkl.bz2', 'r') as inp:
        data = pickle.load(inp)

    desired_nout = 1000
    samples = args.niter // 2

    thin = samples // desired_nout
    if thin == 0:
        thin = 1

    fit = pystan.stan(file='nfw_hier.stan', data=data, iter=args.niter, thin=thin, chains=args.nchain, n_jobs=args.nchain)

    print(fit)

    with bz2.BZ2File('testchains.pkl.bz2.temp', 'w') as out:
        pickle.dump(fit.extract(permuted=True), out)
    with bz2.BZ2File('testmodel.pkl.bz2.temp', 'w') as out:
        pickle.dump(fit.get_stanmodel(), out)
    with bz2.BZ2File('testfit.pkl.bz2.temp', 'w') as out:
        pickle.dump(fit, out)

    # Almost atomic rename/commit updates
    os.rename('testchains.pkl.bz2.temp', 'testchains.pkl.bz2')
    os.rename('testmodel.pkl.bz2.temp', 'testmodel.pkl.bz2')
    os.rename('testfit.pkl.bz2.temp', 'testfit.pkl.bz2')
