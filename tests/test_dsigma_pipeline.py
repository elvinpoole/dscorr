# tests/test_dsigma_pipeline.py
import numpy as np
import pytest
from dscorr import DeltaSigmaConfig, DeltaSigma
from scipy.stats import norm

@pytest.fixture
def mock_catalogs():
    """Return a toy lens and source catalog for testing."""
    # Tiny "lens" catalog: 2 lenses at z ~ 0.3
    lenses = {
        "ra": np.array([150.0, 151.0]),
        "dec": np.array([2.0, 2.5]),
        "z": np.array([0.3, 0.3]),
    }

    randoms = {
        "ra": np.array([150.5, 151.5]),
        "dec": np.array([2.5, 3.0]),
        "z": np.array([0.35, 0.29]),
    }

    # Tiny "source" catalog: random scatter at z ~ 0.6
    rng = np.random.default_rng(42)
    n = 50
    sources = {
        "ra": 150.0 + rng.normal(scale=0.05, size=n),
        "dec": 2.0 + rng.normal(scale=0.05, size=n),
        "z": rng.uniform(0.5, 0.8, size=n),
        "e1": rng.normal(scale=0.2, size=n),
        "e2": rng.normal(scale=0.2, size=n),
        "weight": np.ones(n),
    }

    nz_z = np.linspace(0,2.0,100)
    nz = norm(0.5,0.1).pdf(nz_z)

    return lenses, sources, randoms, (nz, nz_z)


def test_pipeline_runs(mock_catalogs):
    """Ensure the ΔΣ pipeline runs end-to-end without crashing."""
    lenses, sources, randoms, (nz_z, nz) = mock_catalogs

    corr = DeltaSigma(
        lens_ra=lenses["ra"],
        lens_dec=lenses["dec"],
        lens_z=lenses["z"],
        source_ra=sources["ra"],
        source_dec=sources["dec"],
        source_e1=sources["e1"],
        source_e2=sources["e2"],
        nz_z=nz_z,
        source_nz=nz,
        rand_ra=randoms["ra"],
        rand_dec=randoms["dec"],
        rand_z=randoms["z"],
        fbias_nsamples=100,
    )
    r,ds,dserr = corr.process()

    # Check structure of results
    assert "meanr" in corr.results
    assert "dsigma" in corr.results
    assert "dsigma_cov" in corr.results
    assert len(corr.results["meanr"]) == len(corr.results["dsigma"])

