import numpy as np
import fitsio as fio
import treecorr
import matplotlib.pyplot as plt
import dataclasses
from typing import Callable, Optional
import pyccl as ccl
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import c, G
from astropy import units as u
from scipy.interpolate import interp1d

@dataclasses.dataclass
class DeltaSigmaConfig:
    treecorr_min_r: float = 0.09   #R bins
    treecorr_max_r: float = 11.    #R bins
    treecorr_nbins_r: int = 10     #R bins
    treecorr_bin_slop: float = 0.1
    treecorr_bin_type: str = "Log"
    treecorr_var_method: str = "jackknife"
    treecorr_nprocess: int = 2
    njk: int = 100
    use_kG: bool = True
    sigcrit_weights: bool = True
    covmoving: bool = True
    flip_e1: bool = False
    flip_e2: bool = False
    z_fid: float = 2.0
    fbias_nsamples: int = 10000
    use_boost: bool = False
    use_random_sub: bool = False

class DeltaSigma:
    def __init__(self, *,
                 lens_ra, lens_dec, lens_z, lens_w=None,
                 source_ra, source_dec, source_e1, source_e2, source_w=None,
                 rand_ra=None, rand_dec=None, rand_z=None,
                 source_nz=None,
                 config: Optional[DeltaSigmaConfig] = None,
                 **kwargs):
        
        if self.config.use_boost or self.config.use_random_sub:
            assert self.rand_ra is not None
            assert self.rand_dec is not None
            assert self.rand_z is not None
        
        # store arrays
        self.lens_ra, self.lens_dec, self.lens_z = map(np.asarray, (lens_ra, lens_dec, lens_z))
        self.source_ra, self.source_dec = map(np.asarray, (source_ra, source_dec))
        self.source_e1, self.source_e2 = map(np.asarray, (source_e1, source_e2))
        self.source_nz = source_nz
        if rand_ra:
            self.rand_ra, self.rand_dec, self.rand_z = map(np.asarray, (rand_ra, rand_dec, rand_z))

        # config + overrides
        self.config, self.config_specified = (config, True) if config else (DeltaSigmaConfig(), False)
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                if self.config_specified:
                    print(f'Overwriting config option {k} with {v} from kwargs')
                setattr(self.config, k, v)
            else:
                raise ValueError(f"Unknown config option {k}")
            
        self.lens_zmin = self.lens_z.min() if (self.rand_z is not None) else np.min([self.lens_z.min(), self.rand_z.min()])
        self.lens_zmax = self.lens_z.max() if (self.rand_z is not None) else np.max([self.lens_z.max(), self.rand_z.max()])

        self.results = {}

    def process(self):
        """Compute ΔΣ using TreeCorr"""

        lens_cat, source_cat = self._prepare_catalogs()

        #measure disgma with treecorr using the KG class
        kg = treecorr.KGCorrelation(
            min_sep=self.config.treecorr_min_r, max_sep=self.config.treecorr_max_r, nbins=self.config.treecorr_nbins_r,
            bin_slop=self.config.treecorr_bin_slop, bin_type=self.config.treecorr_bin_type, metric="Rlens",
            var_method=self.config.treecorr_var_method, cross_patch_weight='match', )
        kg.process(lens_cat, source_cat)

        self.results["dsigma_raw"] = kg.xi
        self.results["dsigma_raw_cov"] = kg.cov
        self.results["meanR"] = kg.meanr

        if self.config.use_boost:
            #TODO: add boosts
            pass

        if self.config.use_random_sub:
            #TODO: add random subtraction
            pass

        return 

    def _prepare_catalogs(self):
        fbias = self._calc_fbias(self.nz_zarray, self.nz, z_fid=self.config.z_fid, nsamples=self.config.fbias_samples)

        if self.config.comoving:
            chi_fid = cosmo.comoving_distance(self.config.z_fid).to(u.Mpc).value  
            chi_lens = cosmo.comoving_distance(self.lens_z).to(u.Mpc).value  
        else:
            chi_fid =  cosmo.angular_diameter_distance(self.config.z_fid).to(u.Mpc).value  
            chi_lens =  cosmo.angular_diameter_distance(self.lens_z).to(u.Mpc).value  

        sc_per_lens = DeltaSigma.sig_crit(self.lens_z, self.config.z_fid)
        fbias_per_lens = fbias(self.lens_z)

        if self.config.sigcrit_weights:
            print('Applying sig_crit^-2 weights to the lenses (assuming fixed z sources)')
            lens_weight = lens_cat.w*sc_per_lens**-2
        else:
            lens_weight = lens_cat.w

        # a 3D lens catalog
        lens_cat = treecorr.Catalog(k=sc_per_lens*fbias_per_lens, ra=lens_cat.ra, dec=lens_cat.dec, r=chi_lens, ra_units='deg', dec_units='deg',
                                w=lens_weight, npatch=self.config.njk)

        # a "3D" source catalog, where all the source redshifts are set to z_fid
        source_cat_zfid = treecorr.Catalog(
            ra=self.source_ra, dec=self.source_dec,
            ra_units='deg', dec_units='deg',
            g1=-1*self.source_e1 if self.config.flip_e1 else self.source_e1,
            g2=-1*self.source_e2 if self.config.flip_e2 else self.source_e2, 
            r = chi_fid*np.ones(len(self.source_ra)),
            w=self.source_w, 
            patch_centers=lens_cat.patch_centers,
        )

        #TODO: check if treecorr copies the coordinate data, if it does, I can clear some memory here
        #self.clear()

        return lens_cat, source_cat_zfid

    def _calc_fbias(self, nz_zarray, nz, z_fid=2.0, nsamples=1000):
        """
        Compute the redshift bias correction factor (f_bias) for ΔΣ measurements 
        when sources have been approximated as lying at a single fiducial redshift.

        The correction accounts for the mismatch between the true source redshift 
        distribution n(z) and the assumed fiducial redshift z_fid, by averaging 
        ratios of critical surface densities (Σ_crit) over Monte Carlo samples.

        Parameters
        ----------
        nz_source : tuple of (array_like, array_like)
            Source redshift distribution as (n(z), zgrid).
            - n(z) : array of relative source counts or probabilities
            - zgrid : corresponding redshift grid
        z_fid : float, default=2.0
            Fiducial redshift assigned to all sources in the measurement.
        nsamples : int, default=1000
            Number of Monte Carlo samples drawn from n(z) to estimate the correction.

        Returns
        -------
        fbias_func : callable
            Interpolated function f_bias(z_l) giving the multiplicative correction 
            factor for each lens redshift.

        Notes
        -----
        - Sampling from n(z) is performed via inverse-CDF sampling.
        - The bias factor is defined as:
        
        .. math::
            f_{bias}(z_l) = \\left[ \\frac{ 
                \\sum_i \\Sigma_{crit}^{-1}(z_l, z_{true,i}) \\Sigma_{crit}^{-1}(z_l, z_{fid}) }
                { \\sum_i \\Sigma_{crit}^{-2}(z_l, z_{fid}) }
            \\right]^{-1}

        - The returned function extrapolates beyond the input redshift range 
        using the values at the edges of the computed grid.
        """
        zphot_samples, ztrue_samples = self.generate_zsamples_from_nz(nz_zarray, nz, nsamples, z_fid)

        zl_grid = np.linspace(self.lens_zmin, self.lens_zmax, 100)
        finvbias_grid = np.zeros(len(zl_grid))
        #TODO: make this weighted?
        for i, zl in enumerate(zl_grid):
            finvbias_grid[i] = np.sum( (DeltaSigma._sig_crit(zl,ztrue_samples)**-1)*(DeltaSigma._sig_crit(zl,zphot_samples)**-1) ) / np.sum(DeltaSigma._sig_crit(zl,zphot_samples)**-2)

        fbias_func = interp1d(zl_grid, 1/finvbias_grid, bounds_error=False, fill_value=(1/finvbias_grid[0], 1/finvbias_grid[-1]))
        return fbias_func

    def _generate_zsamples_from_nz(self, nz_zarray, nz, nsamples, z_fid):
        #sample redshifts from the n(z)
        cdf = np.cumsum(nz)
        cdf /= cdf[-1]  # normalize to 1
        inv_cdf = interp1d(cdf, nz_zarray, bounds_error=False, fill_value=(nz_zarray[0], nz_zarray[-1]))
        u = np.random.rand(nsamples)  # uniform [0,1]
        ztrue_samples = inv_cdf(u)

        zphot_samples = z_fid*np.ones(nsamples)
        return zphot_samples, ztrue_samples

    @staticmethod
    def _ang_diam_dist(redshift1, redshift2=None, v=False):
        """
        Vectorized angular diameter distance between redshifts.

        Parameters
        ----------
        redshift1 : float or array-like
            Foreground redshift(s).
        redshift2 : float or array-like, optional
            Background redshift(s).
            - If None, returns D_A(0->z1).
            - If provided and >= z1, returns D_A(z1->z2).
            - If z2 < z1, returns 0 (and optionally prints a warning).
        v : bool, optional
            Print warnings when z2 < z1.

        Returns
        -------
        astropy.units.Quantity
            Angular diameter distances in pc, broadcast to input shape.
        """
        z1 = np.atleast_1d(redshift1)
        if redshift2 is None:
            return cosmo.angular_diameter_distance(z1)

        z2 = np.atleast_1d(redshift2)
        z1, z2 = np.broadcast_arrays(z1, z2)

        mask = z2 >= z1
        d = np.zeros_like(z1, dtype=float) * u.pc
        if np.any(mask):
            d[mask] = cosmo.angular_diameter_distance_z1z2(z1[mask], z2[mask])
        if v and np.any(~mask):
            print(f"{np.sum(~mask)} cases with z2 < z1")

        return d if (np.ndim(redshift1) or np.ndim(redshift2)) else d.item()

    @staticmethod
    def _sig_crit(zl, zs):
        """
        Vectorized critical surface mass density for gravitational lensing.

        Parameters
        ----------
        zl : float or array-like
            Lens redshift(s).
        zs : float or array-like
            Source redshift(s).

        Returns
        -------
        astropy.units.Quantity
            Critical surface density in Msun/pc^2, broadcast to input shape.
        """
        zl = np.atleast_1d(zl)
        zs = np.atleast_1d(zs)
        zl, zs = np.broadcast_arrays(zl, zs)

        pre = c**2. / (4.0 * np.pi * G)

        D_s  = DeltaSigma._ang_diam_dist(zs)
        D_l  = DeltaSigma._ang_diam_dist(zl)
        D_ls = DeltaSigma._ang_diam_dist(zl, zs)

        # avoid division by zero where zs <= zl
        mask = zs > zl
        sigma = np.full(zl.shape, np.inf) * (u.M_sun / u.pc**2)
        #sigma[mask] = pre * D_s[mask] / (D_l[mask] * D_ls[mask])
        if np.any(mask):
            ratio = D_s[mask] / (D_l[mask] * D_ls[mask])  # dimension: 1/Mpc
            sigma_si = pre * ratio                        # kg/m^2
            sigma[mask] = sigma_si.to(u.M_sun / u.pc**2)  # convert to Msun/pc^2

        return sigma if (np.ndim(zl) or np.ndim(zs)) else sigma.item()

    def save(self, filename="results_dsigma.npz"):
        # dump self.results + config
        pass

    @classmethod
    def load(cls, filename):
        # load config + results
        pass

    def clear(self):
        del self.lens_ra, self.lens_dec, self.lens_z 
        del self.source_ra, self.source_dec
        del self.source_e1, self.source_e2