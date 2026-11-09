# tests/unit/test_core.py
import numpy as np
import pytest

from replay_trajectory_classification.core import (
    _acausal_decode,
    _causal_decode,
    atleast_2d,
    check_converged,
    get_centers,
    mask,
    normalize_to_probability,
    scaled_likelihood,
)

# ---------- small utilities ----------


def stochastic(n):
    """Return a simple valid transition matrix with equal rows."""
    A = np.ones((n, n), dtype=float)
    A /= A.sum(axis=1, keepdims=True)
    return A


# ---------- atleast_2d ----------


def test_atleast_2d_columnize():
    x = np.arange(5)
    y = atleast_2d(x)
    assert y.shape == (5, 1)
    # If already 2D, ensure it's returned as-is (not transposed)
    z = atleast_2d(np.arange(6).reshape(2, 3))
    assert z.shape == (2, 3)


# ---------- get_centers ----------


def test_get_centers_simple():
    edges = np.array([0.0, 1.0, 3.0, 6.0])
    centers = get_centers(edges)
    np.testing.assert_allclose(centers, [0.5, 2.0, 4.5])


# ---------- normalize_to_probability ----------


def test_normalize_to_probability_sums_to_one():
    dist = np.array([0.2, 0.3, 0.5])
    out = normalize_to_probability(dist)
    assert out.shape == dist.shape
    np.testing.assert_allclose(np.nansum(out), 1.0, rtol=0, atol=1e-12)


def test_normalize_to_probability_handles_nans():
    dist = np.array([0.2, np.nan, 0.8])
    out = normalize_to_probability(dist)
    # NaN stays NaN in its position; the rest should sum to 1
    assert np.isnan(out[1])
    np.testing.assert_allclose(np.nansum(out), 1.0, rtol=0, atol=1e-12)


# ---------- scaled_likelihood ----------


def test_scaled_likelihood_extremes_float64():
    ll = np.array([[-np.inf, -1e300, -100.0, 0.0, np.nan]], dtype=np.float64)
    out = scaled_likelihood(ll, axis=1)
    assert out.shape == ll.shape
    # Strictly positive where finite, not inf anywhere
    finite_mask = np.isfinite(ll)
    assert np.all(out[finite_mask] > 0.0)
    assert np.all(np.isfinite(out[np.isfinite(out)]))


def test_scaled_likelihood_extremes_float32():
    ll = np.array([[-np.inf, -100.0, 0.0]], dtype=np.float32)
    out = scaled_likelihood(ll, axis=1)
    assert out.dtype == np.float32
    assert np.all(out[np.isfinite(out)] > 0.0)


# ---------- causal decode (forward filter) ----------


def test_causal_decode_shapes_and_normalization():
    # Two-bin toy problem, constant likelihoods
    n_bins = 2
    T = 5
    init = np.array([0.8, 0.2], dtype=float)
    A = stochastic(n_bins)
    lik = np.ones((T, n_bins), dtype=float)  # constant evidence
    post, logp = _causal_decode(init, A, lik)

    assert post.shape == (T, n_bins)
    # Each time slice sums to ~1
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-12)
    # Probabilities are non-negative
    assert np.all(post >= 0.0)
    # Log evidence is finite
    assert np.isfinite(logp)


def test_causal_decode_identity_transition_keeps_distribution():
    n_bins = 3
    T = 4
    init = np.array([0.5, 0.25, 0.25], dtype=float)
    A = np.eye(n_bins, dtype=float)
    lik = np.ones((T, n_bins), dtype=float)
    post, _ = _causal_decode(init, A, lik)
    # With identity transition and flat likelihoods, posterior stays constant
    for t in range(T):
        np.testing.assert_allclose(post[t], init, atol=1e-12)


# ---------- acausal decode (forward-backward smoother) ----------


def test_acausal_decode_matches_causal_with_identity_transition():
    # In a trivial setting (identity transition), acausal == causal
    n_bins = 3
    T = 6
    init = np.array([0.2, 0.5, 0.3], dtype=float)
    A = np.eye(n_bins, dtype=float)
    lik = np.ones((T, n_bins), dtype=float)
    causal, _ = _causal_decode(init, A, lik)

    # _acausal_decode expects shape (T, n_bins, 1)
    causal_3d = causal[..., None]
    acausal = _acausal_decode(causal_3d, A)

    assert acausal.shape == causal_3d.shape
    np.testing.assert_allclose(acausal[..., 0], causal, atol=1e-12)


def test_acausal_decode_rows_normalized():
    n_bins = 2
    T = 5
    init = np.array([0.6, 0.4], dtype=float)
    A = stochastic(n_bins)
    lik = np.ones((T, n_bins), dtype=float)
    causal, _ = _causal_decode(init, A, lik)
    acausal = _acausal_decode(causal[..., None], A)
    sums = acausal[..., 0].sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-12)


# ---------- mask ----------


def test_mask_1d():
    v = np.arange(5.0)
    interior = np.array([True, False, True, False, True])
    out = mask(v.copy(), interior)
    assert np.isnan(out[1]) and np.isnan(out[3])
    np.testing.assert_allclose(out[[0, 2, 4]], [0, 2, 4])


def test_mask_last_axis_for_3d():
    v = np.zeros((2, 3, 4), dtype=float)
    interior = np.array([True, False, True, True])
    out = mask(v.copy(), interior)
    # All slices in the last axis where interior=False become NaN
    assert np.isnan(out[:, :, 1]).all()
    assert np.isfinite(out[:, :, 0]).all()


# ---------- check_converged ----------


def test_check_converged_true_and_increasing_true():

    prev = 10.0
    curr = 10.00005
    converged, increasing = check_converged(curr, prev, tolerance=1e-3)
    assert converged
    assert increasing


def test_check_converged_false_and_increasing_false():
    prev = 10.0
    curr = 9.5
    converged, increasing = check_converged(curr, prev, tolerance=1e-6)
    assert not converged
    assert not increasing


# ---------- numerical sanity on random inputs ----------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scaled_likelihood_random_is_positive(dtype):
    rng = np.random.default_rng(0)
    ll = rng.normal(loc=-50.0, scale=20.0, size=(7, 11)).astype(dtype)
    out = scaled_likelihood(ll, axis=1)
    assert out.dtype == dtype
    assert np.all(out > 0.0)
    # Max in each row should be close to 1
    row_max = np.nanmax(out, axis=1)
    np.testing.assert_allclose(row_max, 1.0, rtol=0, atol=1e-6)


# ---------- _expected_transition_counts ----------


def random_hierarchical_model(n_states=2, n_bins=3, n_time=5, seed=0):
    """A small random classifier state space with every quantity positive."""
    rng = np.random.default_rng(seed)
    continuous = rng.random((n_states, n_states, n_bins, n_bins)) + 0.05
    continuous /= continuous.sum(axis=-1, keepdims=True)
    discrete = rng.random((n_states, n_states)) + 0.05
    discrete /= discrete.sum(axis=-1, keepdims=True)
    initial = rng.random((n_states, n_bins, 1))
    initial /= initial.sum()
    likelihood = rng.random((n_time, n_states, n_bins, 1)) + 0.05
    return initial, continuous, discrete, likelihood


def brute_force_pairwise_counts(initial, continuous, discrete, likelihood):
    """Sum over time of the pairwise posteriors of the flat product chain."""
    n_time, n_states, n_bins, _ = likelihood.shape
    n = n_states * n_bins
    # A[(s, i), (s', j)] = D[s, s'] * C[s, s'][i, j]
    A = (discrete[:, :, None, None] * continuous).transpose(0, 2, 1, 3).reshape(n, n)
    L = likelihood.reshape(n_time, n)

    alpha = np.zeros((n_time, n))
    alpha[0] = initial.ravel() * L[0]
    alpha[0] /= alpha[0].sum()
    scale = np.ones(n_time)
    for t in range(1, n_time):
        alpha[t] = (alpha[t - 1] @ A) * L[t]
        scale[t] = alpha[t].sum()
        alpha[t] /= scale[t]
    beta = np.ones((n_time, n))
    for t in range(n_time - 2, -1, -1):
        beta[t] = A @ (L[t + 1] * beta[t + 1]) / scale[t + 1]

    xi = np.zeros((n, n))
    for t in range(n_time - 1):
        xi += (
            alpha[t][:, None] * A * (L[t + 1] * beta[t + 1])[None, :] / scale[t + 1]
        )
    marginal = (alpha * beta).reshape(n_time, n_states, n_bins)
    xi = xi.reshape(n_states, n_bins, n_states, n_bins).transpose(0, 2, 1, 3)
    return xi, marginal


def classify(initial, continuous, discrete, likelihood):
    from replay_trajectory_classification.core import (
        _acausal_classify,
        _causal_classify,
    )

    causal, _ = _causal_classify(initial, continuous, discrete, likelihood)
    acausal = _acausal_classify(causal, continuous, discrete)
    return causal, acausal


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_expected_transition_counts_matches_brute_force_forward_backward(seed):
    from replay_trajectory_classification.core import _expected_transition_counts

    model = random_hierarchical_model(seed=seed)
    causal, acausal = classify(*model)
    initial, continuous, discrete, likelihood = model
    counts = _expected_transition_counts(
        causal, acausal, continuous, discrete, np.ones((2, 2), dtype=bool)
    )
    expected, marginal = brute_force_pairwise_counts(*model)

    np.testing.assert_allclose(acausal[..., 0], marginal, atol=1e-10)
    np.testing.assert_allclose(counts, expected, atol=1e-10)


def test_expected_transition_counts_marginalize_to_the_smoother():
    from replay_trajectory_classification.core import _expected_transition_counts

    model = random_hierarchical_model(n_states=3, n_bins=4, n_time=7, seed=3)
    causal, acausal = classify(*model)
    _, continuous, discrete, _ = model
    counts = _expected_transition_counts(
        causal, acausal, continuous, discrete, np.ones((3, 3), dtype=bool)
    )

    assert counts.sum() == pytest.approx(7 - 1, abs=1e-10)
    # Summing out the destination gives the earlier time, the source the later.
    np.testing.assert_allclose(
        counts.sum(axis=(1, 3)), acausal[:-1, ..., 0].sum(axis=0), atol=1e-10
    )
    np.testing.assert_allclose(
        counts.sum(axis=(0, 2)), acausal[1:, ..., 0].sum(axis=0), atol=1e-10
    )


def test_expected_transition_counts_block_mask_zeros_unselected_blocks():
    from replay_trajectory_classification.core import _expected_transition_counts

    model = random_hierarchical_model(seed=4)
    causal, acausal = classify(*model)
    _, continuous, discrete, _ = model
    full = _expected_transition_counts(
        causal, acausal, continuous, discrete, np.ones((2, 2), dtype=bool)
    )
    block_mask = np.array([[False, True], [False, True]])
    masked = _expected_transition_counts(
        causal, acausal, continuous, discrete, block_mask
    )

    np.testing.assert_array_equal(masked[~block_mask], 0.0)
    np.testing.assert_allclose(masked[block_mask], full[block_mask], atol=1e-14)
