# dscorr: Computing $\Delta \Sigma$ from weak lensing surveys

This is a small package that allows the user to compute $\Delta\Sigma$ using a lens sample with individual redshifts (e.g. a spectroscopic or good photo-z sample) and a source sample where only the redshift distribution $n(z)$ is known.

The code uses [Treecorr](https://rmjarvis.github.io/TreeCorr/_build/html/overview.html) to do the underlying correlation functions and has options for boost factors and random subtraction 

The Jackknife covariance produced by `dscorr` uses Treecorr's internal jackknife patching tools *applied to the lens sample*. If your lenses and sources do not have the same footprint, I recommend you cut the lens sample to the overlap region before running (if you want to use the jackknife covariance). 

credit: Jack Elvin-Poole, Pierre Burger 
