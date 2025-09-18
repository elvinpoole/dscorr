# tests/test_dsigma_pipeline.py
import numpy as np
import pytest
from dscorr import DeltaSigmaConfig, DeltaSigma
from scipy.stats import norm
import itertools

@pytest.fixture(scope="session") #session means we only run this part once
def mock_input():
    """Return toy lens, source and randoms catalogs for testing."""
    rng = np.random.default_rng(1)
    n=50

    # Tiny "lens" catalog:  uniformly sampled between z=0.5 and z=0.8
    lenses = {
        "ra": 150.0 + rng.normal(scale=0.05, size=n),
        "dec": 2.0 + rng.normal(scale=0.05, size=n),
        "z": rng.uniform(0.5, 0.8, size=n),
        "weight": 1.0 + rng.normal(scale=0.2, size=n),
    }

    # Tiny "random" catalog:
    randoms = {
        "ra": 150.0 + rng.normal(scale=0.05, size=n),
        "dec": 2.0 + rng.normal(scale=0.05, size=n),
        "z": rng.uniform(0.5, 0.8, size=n),
        "weight": 1.0 + rng.normal(scale=0.2, size=n)
    }

    # Tiny "source" catalog: gaussian distribution around z=0.8
    sources = {
        "ra": 150.0 + rng.normal(scale=0.05, size=n),
        "dec": 2.0 + rng.normal(scale=0.05, size=n),
        "z": rng.normal(0.8, 0.1, size=n),
        "e1": rng.normal(scale=0.2, size=n),
        "e2": rng.normal(scale=0.2, size=n),
        "weight": 1.0 + rng.normal(scale=0.2, size=n),
    }

    nz_z = np.linspace(0,2.0,100)
    nz = norm(0.8,0.1).pdf(nz_z)

    return lenses, randoms, sources, nz_z, nz

@pytest.fixture( params=list(itertools.product([False, True], repeat=3)) )
def mock_catalog_combo(request, mock_input):
    """Return catalog_inputs with each combination of weights speficied/not specified"""
    lenses, randoms, sources, nz_z, nz = mock_input

    catalog_inputs = {
        "lens_ra": lenses["ra"],
        "lens_dec": lenses["dec"],
        "lens_z": lenses["z"],
        "source_ra": sources["ra"],
        "source_dec": sources["dec"],
        "source_e1": sources["e1"],
        "source_e2": sources["e2"],
        "nz_z": nz_z,
        "source_nz": nz,
        "rand_ra": randoms["ra"],
        "rand_dec": randoms["dec"],
        "rand_z": randoms["z"],
    }

    use_source_w, use_lens_w, use_rand_w = request.param
    if use_source_w:
        catalog_inputs["source_w"] = sources["weight"]
    if use_lens_w:
        catalog_inputs["lens_w"] = lenses["weight"]
    if use_rand_w:
        catalog_inputs["rand_w"] = randoms["weight"]

    return catalog_inputs

#define some config combinations to test (add new configs here to test additional variants)
@pytest.fixture
def config_default():
    return DeltaSigmaConfig(fbias_nsamples=100, njk=10, min_r=0.09, max_r=20., nbins_r=5 )

@pytest.fixture
def config_shot():
    return DeltaSigmaConfig(var_method="shot", fbias_nsamples=100, min_r=0.09, max_r=20., nbins_r=5 )

@pytest.fixture
def config_boost_rs():
    return DeltaSigmaConfig(use_boost=True, use_random_sub=True, fbias_nsamples=100, njk=10, min_r=0.09, max_r=20., nbins_r=5 )

#fixture that returns one of the above configs (also add name of new configs here to test additional variants)
@pytest.fixture(params=["config_default", "config_shot", "config_boost_rs"])
def any_config(request):
    """Fixture that cycles through all configs."""
    return request.getfixturevalue(request.param)

##### Tests #####

def test_pipeline_run(mock_catalog_combo, any_config):
    """Ensure the ΔΣ pipeline runs end-to-end without crashing."""

    corr = DeltaSigma(**mock_catalog_combo, config=any_config )
    _ = corr.process()

    # Check structure of results
    assert "meanr" in corr.results
    assert "dsigma" in corr.results
    assert len(corr.results["meanr"]) == len(corr.results["dsigma"])