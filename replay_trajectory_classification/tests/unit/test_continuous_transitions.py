# replay_trajectory_classification/tests/unit/test_continuous_transitions.py
import numpy as np
import pytest

from replay_trajectory_classification.continuous_state_transitions import (
    TCM_SR,
    EmpiricalMovement,
    Identity,
    RandomWalk,
    RandomWalkDirection1,
    RandomWalkDirection2,
    Uniform,
)
from replay_trajectory_classification.environments import Environment
from replay_trajectory_classification.observation_model import ObservationModel

# estimate_movement_var may or may not exist; import if present.
try:
    from replay_trajectory_classification.continuous_state_transitions import (
        estimate_movement_var,
    )
except Exception:  # pragma: no cover
    estimate_movement_var = None


# ---------------------- helpers ----------------------


def make_1d_env(name="E", n=11, bin_size=1.0):
    """
    Build a simple 1D environment by fitting a grid on synthetic positions.
    In practice, the resulting number of *interior* bins may differ from `n`.
    """
    x = np.linspace(0, (n - 1) * bin_size, n).reshape(-1, 1)
    env = Environment()
    env.environment_name = name
    env.fit_place_grid(position=x, infer_track_interior=True)
    return env


def flat_mask(env: Environment):
    """Return a flat boolean interior mask; if None, assume all interior."""
    mask = getattr(env, "is_track_interior_", None)
    if mask is None:
        # Infer interior length from centers; assume all interior
        n = getattr(env, "place_bin_centers_", None)
        n_states = len(n) if n is not None else None
        if n_states is None:
            raise RuntimeError("Cannot infer number of states for mask.")
        return np.ones(n_states, dtype=bool)
    return np.asarray(mask).ravel(order="F")


def assert_interior_rows_row_stochastic(
    A: np.ndarray, interior: np.ndarray, atol=1e-12
):
    """Check that interior rows sum to 1, masked rows/cols are all-zero."""
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "Matrix must be square"
    assert np.isfinite(A).all(), "NaN/inf in transition matrix"
    assert (A >= -1e-14).all(), "Negative probabilities"

    # Masked rows & cols are zero
    masked_idx = np.where(~interior)[0]
    if masked_idx.size:
        assert np.all(A[masked_idx] == 0.0)
        assert np.all(A[:, masked_idx] == 0.0)

    # Interior rows are row-stochastic
    np.testing.assert_allclose(A[interior].sum(axis=1), 1.0, atol=atol)


# ---------------------- estimate_movement_var ----------------------


@pytest.mark.skipif(
    estimate_movement_var is None,
    reason="estimate_movement_var not available"
)
def test_estimate_movement_var_returns_cov_and_is_psd():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(1000, 2))
    cov = estimate_movement_var(v, sampling_frequency=50.0)
    assert cov.shape in {(2, 2), (2,)}  # accept either contract
    if cov.ndim == 2:
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)
        eigs = np.linalg.eigvalsh(cov)
        assert np.all(eigs >= -1e-9)


@pytest.mark.skipif(
    estimate_movement_var is None,
    reason="estimate_movement_var not available"
)
def test_estimate_movement_var_ignores_nans_and_scales():
    x = np.arange(100, dtype=float).reshape(-1, 1)
    x[10:15] = np.nan
    cov = estimate_movement_var(x, sampling_frequency=100.0)
    assert np.isfinite(cov).all()


# ---------------------- RandomWalk (non-diffusion) ----------------------


@pytest.mark.parametrize("movement_var", [0.5, 2.0, 6.0])
def test_random_walk_row_stochastic_and_peaked_near_self(movement_var):
    env = make_1d_env(n=9, bin_size=1.0)
    rw = RandomWalk(
        environment_name=env.environment_name,
        movement_var=movement_var,
        movement_mean=0.0,
        use_diffusion=False,
    )
    A = rw.make_state_transition((env,))
    interior = flat_mask(env)
    assert_interior_rows_row_stochastic(A, interior)
    # Argmax per interior row should be on the diagonal (peak near self)
    peak_cols = np.argmax(A[interior], axis=1)
    np.testing.assert_allclose(peak_cols, np.where(interior)[0], atol=0)


def test_random_walk_masks_off_track_bins_if_any():
    env = make_1d_env(n=7, bin_size=1.0)
    # If environment has at least 3 interior bins, mask the middle one
    interior = flat_mask(env).copy()
    interior_idx = np.where(interior)[0]
    if interior_idx.size < 3:
        pytest.skip("Not enough interior bins to test masking.")
    mid = interior_idx[len(interior_idx) // 2]
    interior[mid] = False
    # Apply the modified mask back in env with original shape
    env.is_track_interior_ = interior.reshape(env.is_track_interior_.shape, order="F")

    rw = RandomWalk(
        environment_name=env.environment_name, movement_var=1.0, use_diffusion=False
    )
    A = rw.make_state_transition((env,))
    assert_interior_rows_row_stochastic(A, interior)


# ---------------------- Uniform ----------------------


def test_uniform_is_uniform_over_interior_states():
    env = make_1d_env(n=6)
    uni = Uniform(environment_name=env.environment_name)
    A = uni.make_state_transition((env,))
    interior = flat_mask(env)
    assert_interior_rows_row_stochastic(A, interior)

    # For uniform, each interior row should be identical and uniform over interior columns
    if interior.any():
        row0 = A[interior][0]
        for r in A[interior]:
            np.testing.assert_allclose(r, row0, atol=1e-12)
        n_int = interior.sum()
        expected = np.zeros_like(row0)
        expected[interior] = 1.0 / n_int
        np.testing.assert_allclose(row0, expected, atol=1e-12)


def test_uniform_respects_masked_bins():
    env = make_1d_env(n=5)
    interior = flat_mask(env).copy()
    idx = np.where(interior)[0]
    if idx.size < 2:
        pytest.skip("Not enough interior bins to mask one and keep others.")
    # mask one interior bin
    interior[idx[0]] = False
    env.is_track_interior_ = interior.reshape(env.is_track_interior_.shape, order="F")
    A = Uniform(environment_name=env.environment_name).make_state_transition((env,))
    assert_interior_rows_row_stochastic(A, interior)


# ---------------------- Identity ----------------------


def test_identity_transition_masks_and_normalizes():
    env = make_1d_env(n=4)
    interior = flat_mask(env).copy()
    idx = np.where(interior)[0]
    if idx.size < 2:
        pytest.skip("Not enough interior bins to mask one and test identity.")
    interior[idx[0]] = False
    env.is_track_interior_ = interior.reshape(env.is_track_interior_.shape, order="F")
    ident = Identity(environment_name=env.environment_name)
    A = ident.make_state_transition((env,))
    assert_interior_rows_row_stochastic(A, interior)

    # On the interior subspace, Identity should be identity
    sub = np.ix_(interior, interior)
    np.testing.assert_allclose(A[sub], np.eye(interior.sum()), atol=1e-12)


# ---------------------- Directional random walks ----------------------
# These models bias probability mass to one side; they are not strictly triangular-zeroed.


def _mass_left_right(row: np.ndarray, i: int):
    left = row[: i + 1].sum()
    right = row[i:].sum()  # includes self on both; we'll compare ratios
    return float(left), float(right)


def test_random_walk_direction1_biases_to_higher_indices():
    env = make_1d_env(n=8)
    rwd1 = RandomWalkDirection1(environment_name=env.environment_name, movement_var=2.0)
    A = rwd1.make_state_transition((env,))
    interior = flat_mask(env)
    assert_interior_rows_row_stochastic(A, interior)

    # For each interior row i, mass on >= i should be >= mass on <= i (bias to the right)
    rows = np.where(interior)[0]
    for i in rows:
        left, right = _mass_left_right(A[i], i)
        assert right >= left - 1e-12


def test_random_walk_direction2_biases_to_lower_indices():
    env = make_1d_env(n=8)
    rwd2 = RandomWalkDirection2(environment_name=env.environment_name, movement_var=2.0)
    A = rwd2.make_state_transition((env,))
    interior = flat_mask(env)
    assert_interior_rows_row_stochastic(A, interior)

    # For each interior row i, mass on <= i should be >= mass on >= i (bias to the left)
    rows = np.where(interior)[0]
    for i in rows:
        left, right = _mass_left_right(A[i], i)
        assert left >= right - 1e-12


# ---------------------- Diffusion path sanity (2D) ----------------------


def test_random_walk_with_diffusion_row_stochastic_2d():
    # Build a simple 2D environment by fitting grid on a small 2D point cloud
    xx, yy = np.meshgrid(np.linspace(0, 3, 4), np.linspace(0, 3, 4), indexing="xy")
    pts = np.c_[xx.ravel(), yy.ravel()]  # 16 points on a grid
    env = Environment()
    env.environment_name = "E2"
    env.fit_place_grid(position=pts, infer_track_interior=True)

    rw = RandomWalk(
        environment_name=env.environment_name,
        movement_var=1.0,
        movement_mean=0.0,
        use_diffusion=True,
    )
    A = rw.make_state_transition((env,))
    interior = flat_mask(env)
    assert_interior_rows_row_stochastic(A, interior)


# ---------------------- EmpiricalMovement ----------------------


def make_ramp_env(name="ramp", stop=30.0, step=0.25):
    """A 1D environment plus a strictly increasing trajectory over it.

    The ramp only ever moves forward, so the forward transition matrix must be
    upper triangular under this package's `A[i, j] = p(x_t = j | x_{t-1} = i)`
    convention.
    """
    ramp = np.arange(0.0, stop, step).reshape(-1, 1)
    env = Environment(place_bin_size=1.0)
    env.environment_name = name
    env.fit_place_grid(position=ramp, infer_track_interior=True)
    return env, ramp


def synthesized_labels(n_time):
    """The `encoding_group_labels` the classifiers and decoders substitute.

    See `_ClassifierBase.fit_continuous_state_transition` and
    `_DecoderBase.fit_state_transition`, which both pass
    `np.zeros(n_time, np.int32)` when the caller supplies nothing.
    """
    return np.zeros((n_time,), dtype=np.int32)


def test_empirical_movement_runs_on_numpy_2():
    """EmpiricalMovement builds a row-stochastic matrix under NumPy 2."""
    env, ramp = make_ramp_env()
    A = EmpiricalMovement(environment_name=env.environment_name).make_state_transition(
        (env,), ramp, None, synthesized_labels(len(ramp)), None
    )
    interior = flat_mask(env)
    assert_interior_rows_row_stochastic(A, interior)


def test_empirical_movement_default_encoding_group_matches_observation_model():
    """A `None` default matches no synthesized label and selects nothing."""
    assert EmpiricalMovement().encoding_group == ObservationModel().encoding_group


def test_empirical_movement_with_synthesized_labels_is_not_all_zero():
    """The path the classifiers and decoders actually take.

    They synthesize `np.zeros(n_time)` for `encoding_group_labels`, so an
    `encoding_group` that does not equal 0 selects no time points and the
    histogram comes back empty -- an all-zero matrix, behind nothing louder
    than a RuntimeWarning.
    """
    env, ramp = make_ramp_env()
    A = EmpiricalMovement(environment_name=env.environment_name).make_state_transition(
        (env,), ramp, None, synthesized_labels(len(ramp)), None
    )
    assert A.sum() > 0.0
    assert_interior_rows_row_stochastic(A, flat_mask(env))


def test_empirical_movement_is_forward_in_time():
    """Rows must be the source bin, `A[i, j] = p(x_t = j | x_{t-1} = i)`.

    The ramp only ever moves forward, so every transition goes to a bin index
    at or above the source and the matrix must be upper triangular. The
    transpose, `p(x_{t-1} | x_t)`, would be lower triangular.
    """
    env, ramp = make_ramp_env()
    A = EmpiricalMovement(environment_name=env.environment_name).make_state_transition(
        (env,), ramp, None, synthesized_labels(len(ramp)), None
    )
    assert np.allclose(np.tril(A, -1), 0.0), "matrix is not forward in time"
    assert not np.allclose(np.triu(A, 1), 0.0), "ramp should produce forward mass"


def test_empirical_movement_agrees_with_random_walk_direction1_convention():
    """`RandomWalkDirection1` is upper triangular and means forward movement.

    Pinning both directional kernels to one convention is what stops either
    from being transposed independently later.
    """
    env, ramp = make_ramp_env()
    empirical = EmpiricalMovement(
        environment_name=env.environment_name
    ).make_state_transition((env,), ramp, None, synthesized_labels(len(ramp)), None)
    directional = RandomWalkDirection1(
        environment_name=env.environment_name, movement_var=1.0
    ).make_state_transition((env,))

    assert np.allclose(np.tril(empirical, -1), 0.0)
    assert np.allclose(np.tril(directional, -1), 0.0)


def test_empirical_movement_speedup_moves_mass_forward():
    """`speedup` powers the matrix, so it inherits the orientation.

    A sped-up kernel must still move mass forward on the ramp.
    """
    env, ramp = make_ramp_env()
    A = EmpiricalMovement(
        environment_name=env.environment_name, speedup=5
    ).make_state_transition((env,), ramp, None, synthesized_labels(len(ramp)), None)

    interior = np.where(flat_mask(env))[0]
    source = interior[len(interior) // 3]
    mean_destination = np.average(np.arange(A.shape[1]), weights=A[source])
    assert mean_destination > source, "speedup should move mass forward"


def test_empirical_movement_does_not_pair_across_excluded_intervals():
    """Pairs must not straddle a gap in `is_training`.

    The ramp advances one step at a time, so every real transition moves at
    most a bin or so. Masking the position before pairing joins the last
    selected sample before the gap to the first one after it, fabricating a
    single long jump per excluded interval -- which a successor representation
    then amplifies into a persistent shortcut.
    """
    env, ramp = make_ramp_env()
    is_training = np.ones((len(ramp),), dtype=bool)
    is_training[30:70] = False

    A = EmpiricalMovement(environment_name=env.environment_name).make_state_transition(
        (env,), ramp, is_training, synthesized_labels(len(ramp)), None
    )
    reference = EmpiricalMovement(
        environment_name=env.environment_name
    ).make_state_transition((env,), ramp, None, synthesized_labels(len(ramp)), None)

    source, destination = np.nonzero(A)
    largest_step = np.abs(destination - source).max()
    ref_source, ref_destination = np.nonzero(reference)
    largest_reference_step = np.abs(ref_destination - ref_source).max()
    assert largest_step <= largest_reference_step, (
        f"excluding samples fabricated a {largest_step}-bin transition where "
        f"the trajectory never moves more than {largest_reference_step} bins"
    )


def test_empirical_movement_with_position_range_set():
    """`range` must not be forwarded to a doubled-`bins` histogram.

    `bins=edges_ * 2` has `2 * n_position_dims` entries; `position_range` has
    `n_position_dims`, so passing it raised ValueError for any caller that set
    it on the Environment.
    """
    ramp = np.arange(0.0, 30.0, 0.25).reshape(-1, 1)
    env = Environment(place_bin_size=1.0, position_range=[(0.0, 30.0)])
    env.environment_name = "ranged"
    env.fit_place_grid(position=ramp, infer_track_interior=True)

    A = EmpiricalMovement(environment_name=env.environment_name).make_state_transition(
        (env,), ramp, None, synthesized_labels(len(ramp)), None
    )
    assert_interior_rows_row_stochastic(A, flat_mask(env))


# ---------------------- TCM_SR ----------------------


def make_lap_env(name="laps", length=30.0, n_laps=6, step=0.25):
    """A 1D environment plus a back-and-forth trajectory over it.

    Unlike the one-way ramp, every bin is both entered and left, so no bin is
    absorbing and `zero_diagonal=True` is well defined everywhere.
    """
    leg = np.arange(0.0, length, step)
    laps = np.concatenate([leg if i % 2 == 0 else leg[::-1] for i in range(n_laps)])
    position = laps.reshape(-1, 1)
    env = Environment(place_bin_size=1.0)
    env.environment_name = name
    env.fit_place_grid(position=position, infer_track_interior=True)
    return env, position


def empirical_P(env, position):
    """The row-stochastic interior movement matrix TCM_SR builds internally."""
    from replay_trajectory_classification.continuous_state_transitions import (
        _empirical_movement_counts,
        _normalize_row_probability,
    )

    interior = flat_mask(env)
    counts = _empirical_movement_counts(
        env, position, np.ones((len(position),), dtype=bool)
    )
    counts[~interior] = 0.0
    counts[:, ~interior] = 0.0
    return _normalize_row_probability(counts[np.ix_(interior, interior)])


def rownorm(x):
    s = x.sum(axis=1, keepdims=True)
    return np.divide(x, s, out=np.zeros_like(x), where=s > 0)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.95])
@pytest.mark.parametrize("omega", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("zero_diagonal", [False, True])
def test_tcm_sr_row_stochastic(gamma, omega, zero_diagonal):
    env, laps = make_lap_env()
    A = TCM_SR(
        environment_name=env.environment_name,
        gamma=gamma,
        omega=omega,
        zero_diagonal=zero_diagonal,
    ).make_state_transition((env,), laps, None, synthesized_labels(len(laps)), None)
    assert_interior_rows_row_stochastic(A, flat_mask(env))


@pytest.mark.parametrize("omega", [0.0, 0.25, 0.5, 1.0])
def test_tcm_sr_gamma_zero_has_closed_form(omega):
    """At gamma=0 the SR is P itself, so A = rownorm((1-w) P + w P.T @ P).

    This pins the orientation, the solve, the omega term, the interior masking
    and the normalization against one two-line expression.
    """
    env, ramp = make_ramp_env()
    interior = flat_mask(env)
    P = empirical_P(env, ramp)
    expected = rownorm((1.0 - omega) * P + omega * (P.T @ P))

    A = TCM_SR(
        environment_name=env.environment_name, gamma=0.0, omega=omega
    ).make_state_transition((env,), ramp, None, synthesized_labels(len(ramp)), None)

    np.testing.assert_allclose(A[np.ix_(interior, interior)], expected, atol=1e-12)


@pytest.mark.parametrize("gamma", [0.5, 0.9, 0.95])
def test_tcm_sr_matches_truncated_power_series(gamma):
    """M = sum_k gamma^k P^(k+1), computed without np.linalg.solve."""
    env, ramp = make_ramp_env()
    interior = flat_mask(env)
    P = empirical_P(env, ramp)

    series = np.zeros_like(P)
    term = P.copy()
    for _ in range(4000):
        series += term
        term = gamma * (term @ P)
    expected = rownorm(series)

    A = TCM_SR(
        environment_name=env.environment_name, gamma=gamma, omega=0.0
    ).make_state_transition((env,), ramp, None, synthesized_labels(len(ramp)), None)

    np.testing.assert_allclose(A[np.ix_(interior, interior)], expected, atol=1e-9)


def test_tcm_sr_solve_satisfies_its_defining_equation():
    """(I - gamma P) M == P. Holds whether or not P is fully row stochastic."""
    env, ramp = make_ramp_env()
    interior = flat_mask(env)
    P = empirical_P(env, ramp)
    gamma = 0.95
    M = np.linalg.solve(np.identity(P.shape[0]) - gamma * P, P)
    residual = np.abs((np.identity(P.shape[0]) - gamma * P) @ M - P).max()
    assert residual < 1e-10

    A = TCM_SR(
        environment_name=env.environment_name, gamma=gamma, omega=0.0
    ).make_state_transition((env,), ramp, None, synthesized_labels(len(ramp)), None)
    np.testing.assert_allclose(A[np.ix_(interior, interior)], rownorm(M), atol=1e-10)


def test_tcm_sr_is_forward_in_time():
    """A forward-only trajectory gives a strictly upper triangular kernel."""
    env, ramp = make_ramp_env()
    A = TCM_SR(
        environment_name=env.environment_name, gamma=0.9, omega=0.0
    ).make_state_transition((env,), ramp, None, synthesized_labels(len(ramp)), None)
    assert np.allclose(np.tril(A, -1), 0.0)


def test_tcm_sr_omega_symmetrizes_the_kernel():
    env, ramp = make_ramp_env()
    asymmetry = []
    for omega in (0.0, 0.25, 0.5, 0.75, 1.0):
        A = TCM_SR(
            environment_name=env.environment_name, gamma=0.9, omega=omega
        ).make_state_transition((env,), ramp, None, synthesized_labels(len(ramp)), None)
        asymmetry.append(np.abs(A - A.T).sum())
    assert np.all(np.diff(asymmetry) < 0), "omega should monotonically symmetrize"


def test_tcm_sr_zero_diagonal_removes_self_transitions():
    env, laps = make_lap_env()
    interior = flat_mask(env)
    A = TCM_SR(
        environment_name=env.environment_name, zero_diagonal=True
    ).make_state_transition((env,), laps, None, synthesized_labels(len(laps)), None)
    np.testing.assert_allclose(np.diag(A)[interior], 0.0, atol=1e-15)
    assert_interior_rows_row_stochastic(A, interior)


def test_tcm_sr_zero_diagonal_raises_on_an_absorbing_bin():
    """A one-way ramp ends on a bin the animal never leaves.

    With `omega=0` the successor representation of an absorbing bin has mass
    only on its own diagonal, so zeroing the diagonal leaves the row empty.
    That has to be a loud error, not a silently destroyed bin.
    """
    env, ramp = make_ramp_env()
    with pytest.raises(ValueError, match="zero_diagonal"):
        TCM_SR(
            environment_name=env.environment_name, omega=0.0, zero_diagonal=True
        ).make_state_transition(
            (env,), ramp, None, synthesized_labels(len(ramp)), None
        )


def stranded_setup():
    """An interior bin the `is_training` mask leaves with no transitions.

    The environment's interior is inferred from the *whole* trajectory while the
    kernel is estimated from the training subset, so this mismatch is what a
    routine speed threshold produces at the track ends.
    """
    env, laps = make_lap_env()
    interior_idx = np.where(flat_mask(env))[0]
    target = interior_idx[len(interior_idx) // 2]
    lo, hi = env.place_bin_edges_[target, 0], env.place_bin_edges_[target + 1, 0]
    is_training = ~((laps[:, 0] >= lo) & (laps[:, 0] < hi))
    return env, laps, is_training, target


def test_tcm_sr_raises_on_bins_with_no_observed_transitions():
    env, laps, is_training, target = stranded_setup()
    with pytest.raises(ValueError, match="no observed") as excinfo:
        TCM_SR(environment_name=env.environment_name).make_state_transition(
            (env,), laps, is_training, synthesized_labels(len(laps)), None
        )
    assert str(target) in str(excinfo.value)


def test_tcm_sr_beta_restores_reachability_of_stranded_bins():
    """`beta` must fix in-degree, not just out-degree.

    A uniform floor added after the solve gives a stranded bin a row but leaves
    its column essentially empty, so the decoder still never places the animal
    there. Blending before the solve is what makes the column nonzero.
    """
    env, laps, is_training, _ = stranded_setup()
    interior = flat_mask(env)
    A = TCM_SR(
        environment_name=env.environment_name, beta=0.01
    ).make_state_transition(
        (env,), laps, is_training, synthesized_labels(len(laps)), None
    )
    assert_interior_rows_row_stochastic(A, interior)
    assert np.all(A[:, interior].sum(axis=0) > 0), "an interior bin is unreachable"


def test_tcm_sr_does_not_tunnel_through_masked_bins():
    """Off-track bins must not become a shortcut for the successor representation.

    The animal walks through the barrier bin, so the raw counts connect the two
    sides. Masking zeroes that row and column, which makes the empirical matrix
    block diagonal -- and the successor representation must keep it that way.
    Solving on the full matrix and masking afterwards does not: the SR routes
    probability through the masked bin and the row comes out nearly uniform.

    `beta` is deliberately left at 0. It blends in a `RandomWalk`, which is
    Gaussian in Euclidean space and can legitimately hop a thin barrier.
    """
    env, laps = make_lap_env()
    interior = flat_mask(env).copy()
    interior_idx = np.where(interior)[0]
    barrier = interior_idx[len(interior_idx) // 2]
    interior[barrier] = False
    env.is_track_interior_ = interior.reshape(env.is_track_interior_.shape, order="F")

    A = TCM_SR(environment_name=env.environment_name).make_state_transition(
        (env,), laps, None, synthesized_labels(len(laps)), None
    )

    left = interior_idx[interior_idx < barrier]
    right = interior_idx[interior_idx > barrier]
    assert np.allclose(A[np.ix_(left, right)], 0.0), "mass crossed the barrier"
    assert np.allclose(A[np.ix_(right, left)], 0.0), "mass crossed the barrier"


def realized_symmetric_weight(env, position, gamma, omega, normalize_before_mix):
    """Recover the weight the kernel actually puts on the symmetric term."""
    P = empirical_P(env, position)
    M = np.linalg.solve(np.identity(P.shape[0]) - gamma * P, P)
    G = M.T @ M
    if normalize_before_mix:
        M, G = rownorm(M), rownorm(G)
    A = rownorm((1.0 - omega) * M + omega * G)
    basis = np.stack([rownorm(M).ravel(), rownorm(G).ravel()], axis=1)
    return np.linalg.lstsq(basis, A.ravel(), rcond=None)[0][1]


def test_tcm_sr_omega_realized_weight_depends_on_gamma():
    """Without normalize_before_mix, the realized weight differs from omega.

    M sums to 1/(1-gamma) per row while M.T @ M sums to about its square, so
    normalizing once after mixing gives
    omega_eff = omega / ((1 - omega) * (1 - gamma) + omega).
    """
    env, laps = make_lap_env()
    gamma, omega = 0.95, 0.5
    expected = omega / ((1.0 - omega) * (1.0 - gamma) + omega)
    assert expected == pytest.approx(0.952, abs=1e-3)
    measured = realized_symmetric_weight(env, laps, gamma, omega, False)
    assert measured == pytest.approx(expected, abs=0.02)


def test_tcm_sr_normalize_before_mix_makes_omega_literal():
    env, laps = make_lap_env()
    measured = realized_symmetric_weight(env, laps, 0.95, 0.5, True)
    assert measured == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": 1.0},
        {"gamma": 1.5},
        {"gamma": -0.1},
        {"omega": 1.5},
        {"omega": -0.1},
        {"beta": 2.0},
        {"beta": -0.1},
        {"asymmetry": 1.5},
        {"asymmetry": -0.1},
    ],
)
def test_tcm_sr_validates_its_parameters(kwargs):
    env, laps = make_lap_env()
    with pytest.raises(ValueError):
        TCM_SR(environment_name=env.environment_name, **kwargs).make_state_transition(
            (env,), laps, None, synthesized_labels(len(laps)), None
        )


def test_tcm_sr_raises_when_the_encoding_group_selects_nothing():
    env, laps = make_lap_env()
    with pytest.raises(ValueError, match="no observed transitions"):
        TCM_SR(
            environment_name=env.environment_name, encoding_group=99
        ).make_state_transition(
            (env,), laps, None, synthesized_labels(len(laps)), None
        )


def make_2d_env(name="E2d", n=8, step=0.5):
    """A filled 2D square, densely enough sampled that every bin is visited."""
    grid = np.arange(0.0, n, step)
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    pts = np.c_[xx.ravel(), yy.ravel()]
    walk = np.repeat(pts, 3, axis=0)
    env = Environment(place_bin_size=2.0)
    env.environment_name = name
    env.fit_place_grid(position=walk, infer_track_interior=True)
    return env, walk


def test_tcm_sr_2d_bin_ordering_matches_place_bin_centers():
    """2D bins are flattened in the same order as `place_bin_centers_`.

    A step of one bin in +x must land one flat index away, and a step of one
    bin in +y must land `n_x` away, because `order="F"` flattening indexes as
    `i_x + n_x * i_y` -- the same ordering as `place_bin_centers_` and
    `is_track_interior_.ravel(order="F")`.
    """
    from replay_trajectory_classification.continuous_state_transitions import (
        _empirical_movement_counts,
    )

    env, _ = make_2d_env()
    n_x, _ = env.centers_shape_
    centers = env.place_bin_centers_

    for axis, expected in ((0, {0, 1}), (1, {0, n_x})):
        sweep = centers[np.argsort(centers[:, 1 - axis] * 1e6 + centers[:, axis])]
        counts = _empirical_movement_counts(
            env, sweep, np.ones((len(sweep),), dtype=bool)
        )
        source, destination = np.nonzero(counts)
        offsets = set(np.unique(destination - source)) & expected
        assert expected <= set(np.unique(destination - source)) or offsets, (
            f"axis {axis}: offsets {sorted(set(np.unique(destination - source)))} "
            f"do not contain {sorted(expected)}"
        )


def test_tcm_sr_2d_row_stochastic():
    env, walk = make_2d_env()
    A = TCM_SR(environment_name=env.environment_name, beta=0.01).make_state_transition(
        (env,), walk, None, synthesized_labels(len(walk)), None
    )
    interior = flat_mask(env)
    assert_interior_rows_row_stochastic(A, interior)
    assert np.allclose(A.astype(np.float32)[interior].sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("asymmetry", [0.0, 0.3, 1.0])
def test_tcm_sr_asymmetry_is_literal_forward_weight(asymmetry):
    """asymmetry = 1 - omega with normalize_before_mix."""
    env, laps = make_lap_env()
    labels = synthesized_labels(len(laps))
    A = TCM_SR(
        environment_name=env.environment_name, asymmetry=asymmetry
    ).make_state_transition((env,), laps, None, labels, None)
    expected = TCM_SR(
        environment_name=env.environment_name,
        omega=1.0 - asymmetry,
        normalize_before_mix=True,
    ).make_state_transition((env,), laps, None, labels, None)
    np.testing.assert_allclose(A, expected, atol=1e-12)


def test_tcm_sr_omega_replaces_asymmetry():
    """Setting omega switches to the reference mixture, whatever asymmetry is."""
    env, laps = make_lap_env()
    labels = synthesized_labels(len(laps))
    A = TCM_SR(
        environment_name=env.environment_name, omega=0.3, asymmetry=0.9
    ).make_state_transition((env,), laps, None, labels, None)
    expected = TCM_SR(
        environment_name=env.environment_name, omega=0.3, asymmetry=0.1
    ).make_state_transition((env,), laps, None, labels, None)
    np.testing.assert_allclose(A, expected, atol=1e-12)


def test_tcm_sr_defaults_to_asymmetry_one_half():
    env, laps = make_lap_env()
    labels = synthesized_labels(len(laps))
    A = TCM_SR(environment_name=env.environment_name).make_state_transition(
        (env,), laps, None, labels, None
    )
    expected = TCM_SR(
        environment_name=env.environment_name, omega=0.5, normalize_before_mix=True
    ).make_state_transition((env,), laps, None, labels, None)
    np.testing.assert_allclose(A, expected, atol=1e-12)


def test_tcm_sr_asymmetry_controls_forward_bias():
    """On one-way running, forward mass grows with asymmetry."""
    env, ramp = make_ramp_env()
    labels = synthesized_labels(len(ramp))
    centers = env.place_bin_centers_.ravel()
    i = int(np.flatnonzero(flat_mask(env))[len(centers) // 3])

    def forward_and_backward(asymmetry):
        row = TCM_SR(
            environment_name=env.environment_name, gamma=0.9, asymmetry=asymmetry
        ).make_state_transition((env,), ramp, None, labels, None)[i]
        return row[centers > centers[i]].sum(), row[centers < centers[i]].sum()

    (f0, b0), (f5, b5), (f1, b1) = (forward_and_backward(a) for a in (0.0, 0.5, 1.0))
    assert f0 == pytest.approx(b0, abs=0.05)
    assert f0 - b0 < f5 - b5 < f1 - b1
    # Pure forward SR of one-way running never steps back.
    assert b1 == 0.0


@pytest.mark.parametrize("movement_var", [1.0, 6.0, 40.0])
def test_tcm_sr_reduces_exactly_to_random_walk(movement_var):
    """gamma=0, beta=1, asymmetry=1 is the continuous dynamic itself."""
    env, laps = make_lap_env()
    A = TCM_SR(
        environment_name=env.environment_name,
        gamma=0.0,
        beta=1.0,
        asymmetry=1.0,
        local_movement_var=movement_var,
    ).make_state_transition((env,), laps, None, synthesized_labels(len(laps)), None)
    expected = RandomWalk(
        env.environment_name, movement_var=movement_var
    ).make_state_transition((env,))
    np.testing.assert_allclose(A, expected, atol=1e-12)


# ---------------------- movement time step ----------------------


def lagged_counts_by_brute_force(env, position, is_selected, lag):
    """Histogram of (x_t, x_{t+lag}) over every t whose window is all selected."""
    starts = [
        t
        for t in range(len(position) - lag)
        if np.all(is_selected[t : t + lag + 1])
    ]
    counts, _ = np.histogramdd(
        np.concatenate((position[starts], position[np.add(starts, lag)]), axis=1),
        bins=env.edges_ * 2,
    )
    n_bins = int(np.prod(counts.shape[: position.shape[1]]))
    return counts.reshape((n_bins, n_bins), order="F")


def lagged_P(env, position, lag):
    """Interior row-stochastic movement matrix counted over `lag` samples."""
    from replay_trajectory_classification.continuous_state_transitions import (
        _empirical_movement_counts,
        _normalize_row_probability,
    )

    interior = flat_mask(env)
    counts = _empirical_movement_counts(
        env, position, np.ones((len(position),), dtype=bool), lag=lag
    )
    counts[~interior] = 0.0
    counts[:, ~interior] = 0.0
    return _normalize_row_probability(counts[np.ix_(interior, interior)])


def test_empirical_movement_counts_lag_one_pairs_consecutive_samples():
    from replay_trajectory_classification.continuous_state_transitions import (
        _empirical_movement_counts,
    )

    env, laps = make_lap_env()
    is_selected = np.ones((len(laps),), dtype=bool)
    is_selected[100:130] = False
    np.testing.assert_array_equal(
        _empirical_movement_counts(env, laps, is_selected, lag=1),
        _empirical_movement_counts(env, laps, is_selected),
    )
    np.testing.assert_array_equal(
        _empirical_movement_counts(env, laps, is_selected),
        lagged_counts_by_brute_force(env, laps, is_selected, lag=1),
    )


@pytest.mark.parametrize("lag", [2, 8, 13])
def test_empirical_movement_counts_never_span_an_excluded_interval(lag):
    """Only pairs whose whole window [t, t + lag] is selected are counted."""
    from replay_trajectory_classification.continuous_state_transitions import (
        _empirical_movement_counts,
    )

    env, ramp = make_ramp_env()
    is_selected = np.ones((len(ramp),), dtype=bool)
    is_selected[30:70] = False
    is_selected[85] = False

    counts = _empirical_movement_counts(env, ramp, is_selected, lag=lag)
    np.testing.assert_array_equal(
        counts, lagged_counts_by_brute_force(env, ramp, is_selected, lag)
    )
    assert counts.sum() > 0


def test_empirical_movement_counts_lag_moves_lag_samples_of_distance():
    """The ramp moves 0.25 per sample over 1.0 bins, so 8 samples is 2 bins."""
    from replay_trajectory_classification.continuous_state_transitions import (
        _empirical_movement_counts,
    )

    env, ramp = make_ramp_env()
    counts = _empirical_movement_counts(
        env, ramp, np.ones((len(ramp),), dtype=bool), lag=8
    )
    source, destination = np.nonzero(counts)
    offsets = destination - source
    assert np.all(np.abs(offsets - 2) <= 1)
    assert np.average(offsets, weights=counts[source, destination]) == pytest.approx(
        2.0, abs=0.05
    )


@pytest.mark.parametrize("lag", [0, -1, 1.5])
def test_empirical_movement_counts_rejects_a_bad_lag(lag):
    from replay_trajectory_classification.continuous_state_transitions import (
        _empirical_movement_counts,
    )

    env, ramp = make_ramp_env()
    with pytest.raises(ValueError, match="lag"):
        _empirical_movement_counts(
            env, ramp, np.ones((len(ramp),), dtype=bool), lag=lag
        )


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.9])
@pytest.mark.parametrize("omega", [0.0, 0.5, 1.0])
def test_tcm_sr_movement_time_step_counts_P_over_that_many_samples(gamma, omega):
    """100 ms at 100 Hz is a 10-sample P, then the usual SR and mix."""
    env, laps = make_lap_env()
    A = TCM_SR(
        environment_name=env.environment_name,
        gamma=gamma,
        omega=omega,
        movement_time_step=0.1,
        sampling_frequency=100,
    ).make_state_transition((env,), laps, None, synthesized_labels(len(laps)), None)

    interior = flat_mask(env)
    P = lagged_P(env, laps, lag=10)
    M = np.linalg.solve(np.identity(P.shape[0]) - gamma * P, P)
    expected = np.zeros_like(A)
    expected[np.ix_(interior, interior)] = rownorm(
        np.maximum((1 - omega) * M + omega * (M.T @ M), 0.0)
    )
    np.testing.assert_allclose(A, expected, atol=1e-12)
    assert_interior_rows_row_stochastic(A, interior)


def test_tcm_sr_movement_time_step_of_one_sample_is_the_default():
    env, laps = make_lap_env()
    labels = synthesized_labels(len(laps))
    default = TCM_SR(environment_name=env.environment_name).make_state_transition(
        (env,), laps, None, labels, None
    )
    one_sample = TCM_SR(
        environment_name=env.environment_name,
        movement_time_step=0.01,
        sampling_frequency=100,
    ).make_state_transition((env,), laps, None, labels, None)
    np.testing.assert_array_equal(default, one_sample)
    assert TCM_SR().lag == 1


def test_tcm_sr_longer_movement_time_step_reaches_further():
    """A longer step of P moves more mass away from the source bin."""
    env, laps = make_lap_env()
    labels = synthesized_labels(len(laps))
    centers = env.place_bin_centers_.ravel()
    i = int(np.flatnonzero(flat_mask(env))[len(centers) // 2])

    def mean_distance(movement_time_step):
        row = TCM_SR(
            environment_name=env.environment_name,
            gamma=0.0,
            movement_time_step=movement_time_step,
            sampling_frequency=100,
        ).make_state_transition((env,), laps, None, labels, None)[i]
        return np.sum(row * np.abs(centers - centers[i]))

    assert mean_distance(0.01) < mean_distance(0.05) < mean_distance(0.2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"movement_time_step": 0.1},
        {"sampling_frequency": 100},
        {"movement_time_step": 0.001, "sampling_frequency": 100},
    ],
)
def test_tcm_sr_validates_the_movement_time_step(kwargs):
    env, laps = make_lap_env()
    with pytest.raises(ValueError, match="movement_time_step"):
        TCM_SR(environment_name=env.environment_name, **kwargs).make_state_transition(
            (env,), laps, None, synthesized_labels(len(laps)), None
        )


# ---------------------- TCM_SR fitting: rebuild, learn, m_step ----------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": 0.9, "omega": 0.5},
        {"gamma": 0.9, "omega": 0.5, "normalize_before_mix": True},
        {"gamma": 0.9, "asymmetry": 0.3},
        {
            "gamma": 0.5,
            "asymmetry": 0.7,
            "movement_time_step": 0.04,
            "sampling_frequency": 100,
        },
        {"gamma": 0.5, "omega": 0.2, "zero_diagonal": True},
    ],
)
def test_tcm_sr_rebuild_matches_make_state_transition(kwargs):
    """Rebuilding from the cached P reproduces the kernel, also after a change."""
    env, laps = make_lap_env()
    labels = synthesized_labels(len(laps))
    transition = TCM_SR(environment_name=env.environment_name, **kwargs)
    A = transition.make_state_transition((env,), laps, None, labels, None)
    np.testing.assert_array_equal(transition.rebuild_state_transition(), A)

    transition.gamma = 0.3
    expected = TCM_SR(
        environment_name=env.environment_name, **{**kwargs, "gamma": 0.3}
    ).make_state_transition((env,), laps, None, labels, None)
    np.testing.assert_array_equal(transition.rebuild_state_transition(), expected)


def test_tcm_sr_rebuild_before_fitting_raises():
    with pytest.raises(ValueError, match="make_state_transition"):
        TCM_SR().rebuild_state_transition()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"learn": ("omega",), "asymmetry": 0.5}, "omega is not learnable"),
        ({"learn": ("omega",)}, "omega is not learnable"),
        ({"learn": ("beta",)}, "Cannot learn 'beta'"),
        ({"learn": ("asymmetry",), "omega": 0.5}, "`omega` to be unset"),
    ],
)
def test_tcm_sr_learn_validation(kwargs, match):
    env, laps = make_lap_env()
    with pytest.raises(ValueError, match=match):
        TCM_SR(environment_name=env.environment_name, **kwargs).make_state_transition(
            (env,), laps, None, synthesized_labels(len(laps)), None
        )


def test_tcm_sr_learn_gamma_is_allowed_in_omega_mode():
    env, laps = make_lap_env()
    transition = TCM_SR(
        environment_name=env.environment_name, omega=0.5, learn=("gamma",)
    )
    transition.make_state_transition(
        (env,), laps, None, synthesized_labels(len(laps)), None
    )


def test_tcm_sr_learns_both_parameters_by_default():
    env, laps = make_lap_env()
    transition = TCM_SR(
        environment_name=env.environment_name, learn=("gamma", "asymmetry")
    )
    transition.make_state_transition(
        (env,), laps, None, synthesized_labels(len(laps)), None
    )


def fitted_ramp_tcm_sr(**kwargs):
    """A TCM_SR fit to one-way running, where `asymmetry` is identifiable."""
    env, ramp = make_ramp_env()
    transition = TCM_SR(environment_name=env.environment_name, **kwargs)
    A = transition.make_state_transition(
        (env,), ramp, None, synthesized_labels(len(ramp)), None
    )
    interior = flat_mask(env)
    return transition, A[np.ix_(interior, interior)]


def counts_from_kernel(kernel, seed=0):
    """Expected counts whose rows are proportional to `kernel`'s rows."""
    weights = np.random.default_rng(seed).uniform(1.0, 10.0, kernel.shape[0])
    return weights[:, np.newaxis] * kernel


@pytest.mark.parametrize(
    "learn, start",
    [
        (("gamma", "asymmetry"), {"gamma": 0.3, "asymmetry": 0.2}),
        (("gamma",), {"gamma": 0.3, "asymmetry": 0.7}),
        (("asymmetry",), {"gamma": 0.9, "asymmetry": 0.2}),
    ],
)
def test_tcm_sr_m_step_recovers_the_kernel_that_generated_the_counts(learn, start):
    """Q is maximized by the generating kernel (Gibbs' inequality)."""
    _, true_kernel = fitted_ramp_tcm_sr(gamma=0.9, asymmetry=0.7)
    transition, _ = fitted_ramp_tcm_sr(**start, learn=learn)

    info = transition.m_step(counts_from_kernel(true_kernel))

    assert np.log10(1 / (1 - transition.gamma)) == pytest.approx(1.0, abs=0.05)
    assert transition.asymmetry == pytest.approx(0.7, abs=0.02)
    assert info["q_new"] >= info["q_old"]
    assert info["gamma"] == transition.gamma
    assert info["asymmetry"] == transition.asymmetry


def test_tcm_sr_m_step_fits_gamma_in_omega_mode_without_touching_the_mix():
    _, true_kernel = fitted_ramp_tcm_sr(gamma=0.9, omega=0.5)
    transition, _ = fitted_ramp_tcm_sr(gamma=0.3, omega=0.5, learn=("gamma",))

    transition.m_step(counts_from_kernel(true_kernel))

    assert np.log10(1 / (1 - transition.gamma)) == pytest.approx(1.0, abs=0.05)
    assert transition.omega == 0.5
    assert transition.normalize_before_mix is False
    assert transition.asymmetry == 0.5


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tcm_sr_m_step_never_decreases_q(seed):
    transition, kernel = fitted_ramp_tcm_sr(
        gamma=0.6, asymmetry=0.5, learn=("gamma", "asymmetry")
    )
    rng = np.random.default_rng(seed)
    counts = rng.uniform(0.0, 5.0, kernel.shape) * (kernel > 0)

    first = transition.m_step(counts)
    second = transition.m_step(counts)

    assert first["q_new"] >= first["q_old"]
    assert second["q_new"] >= second["q_old"]
    assert second["q_old"] == pytest.approx(first["q_new"])


def test_tcm_sr_m_step_leaves_unlearned_fields_untouched():
    _, true_kernel = fitted_ramp_tcm_sr(gamma=0.9, asymmetry=0.7)
    transition, _ = fitted_ramp_tcm_sr(
        gamma=0.3, asymmetry=0.2, beta=0.0, learn=("gamma",)
    )
    before = {
        name: getattr(transition, name)
        for name in (
            "asymmetry",
            "omega",
            "beta",
            "normalize_before_mix",
            "zero_diagonal",
        )
    }
    transition.m_step(counts_from_kernel(true_kernel))
    after = {name: getattr(transition, name) for name in before}
    assert after == before


def test_tcm_sr_m_step_without_learn_is_a_no_op():
    transition, kernel = fitted_ramp_tcm_sr(gamma=0.3, asymmetry=0.2)
    info = transition.m_step(counts_from_kernel(kernel))
    assert (transition.gamma, transition.asymmetry) == (0.3, 0.2)
    assert info["n_evaluations"] == 0


@pytest.mark.parametrize(
    "counts_shape, kwargs, match",
    [
        ((3, 3), {}, "expected_counts has shape"),
        (None, {"gamma_bounds": (0.5, 1.0)}, "gamma_bounds"),
        (None, {"gamma_bounds": (0.9, 0.5)}, "gamma_bounds"),
    ],
)
def test_tcm_sr_m_step_validates_its_inputs(counts_shape, kwargs, match):
    transition, kernel = fitted_ramp_tcm_sr(learn=("gamma",))
    counts = np.ones(counts_shape) if counts_shape else kernel
    with pytest.raises(ValueError, match=match):
        transition.m_step(counts, **kwargs)


def test_tcm_sr_m_step_before_fitting_raises():
    with pytest.raises(ValueError, match="make_state_transition"):
        TCM_SR(learn=("gamma",)).m_step(np.ones((3, 3)))
