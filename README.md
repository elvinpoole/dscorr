# dscorr: Computing $\Delta \Sigma$ from weak lensing surveys

WARNING: This package is still in development!!! It has not yet been tested enough to use for science

This is a small package that allows the user to compute $\Delta\Sigma$ using a lens sample with individual redshifts (e.g. a spectroscopic or good photo-z sample) and a source sample where only the redshift distribution $n(z)$ is known.

The code uses [Treecorr](https://rmjarvis.github.io/TreeCorr/_build/html/overview.html) to do the underlying correlation functions and has options for boost factors and random subtraction 

The Jackknife covariance produced by `dscorr` uses Treecorr's internal jackknife patching tools *applied to the lens sample*. If your lenses and sources do not have the same footprint, I recommend you cut the lens sample to the overlap region before running (if you want to use the jackknife covariance). 

## Install

To install, pull the repo and run:
```
cd dscorr/src/
pip install -e .[dev]
```

## Example

We are still working on getting some examples up and running but the basic procedure is the following (assuming you have a lens catalog in a Table `table_l` and sources in `table_s`):

```python
#set up correlation function config
corr_config = dscorr.DeltaSigmaConfig(
    min_r = 0.09,             # R bin minimum
    max_r = 3.0,              # R bin minimum
    nbins_r = 10,             # R bin minimum
    bin_slop = 0.01,          # A Treecorr input: Accuracy parameter. 0.01 is usually plenty good enough
    bin_type = "Log",         # A Treecorr input: Should the R bins be log spaced ("Log") or Linear ("Lin")?
    var_method = "jackknife", # A Treecorr input: Should the errorbars use Jackknife resampling or just the shot noise 
    nprocess = 4,             # How many process to you want treecorr to use
    njk = 50,                 # If var_method="jackknife", how many patches shoudl you use (uses treecorr's internal k-means JK patching )
    sigcrit_weights = True,   # Should the lenses each get an additional 1/sigma_crit weight (inproves S/N)
    comoving = False,         # Are distances defined as cosmoving or not
    flip_e1 = False,          # Does your input catalog need e_1 -> -1*e_1
    flip_e2 = False,          # Does your input catalog need e_2 -> -1*e_2
    z_fid = 2.0,              # Fiducial value for the source redshifts before the n(z) correction is applied. Can be any number larger than your lens galaxies
    fbias_nsamples = 1000,    # How many samples of the n(z) should be used when correcting for photo-z spread (larger->slower but less noisy correction)
    use_boost = False,        # Do you want to apply the "boost factor" correction (requires randoms)
    use_random_sub = False,   # Do you want to apply the "random subtraction" correction (requires randoms)
)

#prepare the Delta Sigma calculation
corr = dscorr.DeltaSigma(
        lens_ra = table_l['RA'],
        lens_dec = table_l['Dec'],
        lens_z =  table_l['z'],
        lens_w = None,              #A weight for each lens galaxy (optional)
        source_ra = table_s['RA'],
        source_dec = table_s['Dec'],
        source_e1 = table_s['e1'],
        source_e2 = table_s['e2'],
        source_w =  table_s['w'],
        nz_z=nz_z,                 # An array of redshifts at which to specify your source n(z)
        source_nz=source_nz,       # The source redshift distribution (evaluated at nz_z)
        rand_ra=None,              # If you want to use boost factors or random subtraction specify your random catalog columns here
        rand_dec=None,
        rand_z=None,
        rand_w=None,
        config = corr_config, #pass the configuration of your correlation function here
        )

#run the calculation
r, ds, err = corr.process()
```

Please see the [Treecorr documentation](https://rmjarvis.github.io/TreeCorr/_build/html/index.html) for more details on the treecorr config parameters, and the [dsigma documentation](https://dsigma.readthedocs.io/en/latest/) for a great primer on the different correction applied to Delta Sigma calculations

credit: Jack Elvin-Poole, Pierre Burger 
