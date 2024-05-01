import numpy as np

import spurt

# Fix the seed for repeatability
np.random.seed(32)


def gen_data():
    """Generate test data."""

    n_sar = 20
    rng = 900e3
    wvl = 0.056
    sinlook = np.sin(40 * np.pi / 180.0)

    # In meters
    bperp = np.random.randn(n_sar) * 600

    # In mm/yr
    vel_range = slice(-100, 100.1, 5)

    # In meters
    demerr_range = slice(-5, 5.5, 0.5)

    # Use a Hop3 graph as an example
    g_time = spurt.graph.Hop3Graph(n_sar)
    nifgs = len(g_time.links)

    amat = np.zeros((nifgs, 2))
    amat[:, 0] = (
        (g_time.links[:, 1] - g_time.links[:, 0]) * 4 * np.pi / (wvl * 360.0 * 1000.0)
    )
    amat[:, 1] = (
        (bperp[g_time.links[:, 1]] - bperp[g_time.links[:, 0]])
        * 4
        * np.pi
        / (wvl * rng * sinlook)
    )

    return amat, vel_range, demerr_range


def test_grid_estimate():
    """Test estimate_model."""
    amat, vel_range, demerr_range = gen_data()

    vels = np.ogrid[vel_range]
    demerrs = np.ogrid[demerr_range]

    # Create model
    model = spurt.links.GridSearchLinearModel(
        matrix=amat, ranges=(vel_range, demerr_range)
    )

    for vel in vels[5:-5:2]:
        for demerr in demerrs[5:-5:2]:

            vv = vel + 2 * np.random.randn()
            dd = demerr + np.random.randn()
            # Estimate the forward model
            fwd = np.exp(
                1j * (model.fwd_model([vv, dd]) + 0.01 * np.random.randn(model.nobs))
            )

            # Estimat the parameters back
            param, coh = model.estimate_model(np.angle(fwd))

            assert np.abs(vv - param[0]) < 15.0
            assert np.abs(dd - param[1]) < 0.5
            assert coh > 0.99


def test_grid_estimate_many():
    """Test estimate_many."""
    amat, vel_range, demerr_range = gen_data()

    vels = np.ogrid[vel_range]
    demerrs = np.ogrid[demerr_range]

    # Create model
    model = spurt.links.GridSearchLinearModel(
        matrix=amat, ranges=(vel_range, demerr_range)
    )

    vv, dd = np.meshgrid(vels, demerrs)
    vv += 2 * np.random.randn(*vv.shape)
    dd += np.random.randn(*dd.shape)
    true_model = np.zeros((amat.shape[1], vv.size))
    true_model[0, :] = vv.flatten()
    true_model[1, :] = dd.flatten()
    nruns = vv.size

    fwd = np.exp(
        1j * (model.fwd_model(true_model) + 0.01 * np.random.randn(model.nobs, nruns))
    )

    # Estimate the parameters back
    param, coh = model.estimate_model_many(np.angle(fwd))

    assert np.allclose(param[0, :], true_model[0, :], atol=15.0)
    assert np.allclose(param[1, :], true_model[1, :], atol=0.5)
    assert np.all(coh > 0.99)
