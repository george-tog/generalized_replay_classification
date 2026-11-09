import numpy as np

from replay_trajectory_classification.environments import Environment

from replay_trajectory_classification.continuous_state_transitions import TCM_SR
from replay_trajectory_classification.sorted_spikes_simulation import (
    PLACE_FIELD_MEANS,
    TRACK_HEIGHT,
    from_directional_position,
    to_directional_position,
    make_hover_replay,
    make_hover_tcm_sr_hover_replay,
    make_tcm_sr_replay,
    simulate_tcm_sr_replay_position,
)
from replay_trajectory_classification.simulate import simulate_position, simulate_time


def test_tcm_sr_replay_position_is_reproducible_by_seed():
    np.testing.assert_array_equal(
        simulate_tcm_sr_replay_position(seed=3),
        simulate_tcm_sr_replay_position(seed=3),
    )
    assert not np.array_equal(
        simulate_tcm_sr_replay_position(seed=3),
        simulate_tcm_sr_replay_position(seed=4),
    )


def test_tcm_sr_replay_position_starts_at_start_position_and_stays_on_track():
    position = simulate_tcm_sr_replay_position(n_time=300, start_position=40.0)

    assert position.shape == (300,)
    assert abs(position[0] - 40.0) < 1.0
    assert position.min() >= 0.0
    assert position.max() <= TRACK_HEIGHT


def test_tcm_sr_replay_position_uses_the_given_transition():
    """A shorter look-ahead takes shorter steps."""
    default_steps = np.diff(simulate_tcm_sr_replay_position(n_time=500))
    short_steps = np.diff(
        simulate_tcm_sr_replay_position(n_time=500, transition=TCM_SR(gamma=0.99))
    )

    assert np.abs(short_steps).mean() < np.abs(default_steps).mean()


def test_tcm_sr_replay_spikes_the_nearest_place_field_every_interval():
    replay_time, test_spikes = make_tcm_sr_replay(n_time=60, spike_interval=3)
    position = simulate_tcm_sr_replay_position(n_time=60)

    assert replay_time.shape == (60,)
    assert test_spikes.shape == (60, PLACE_FIELD_MEANS.shape[0])

    spike_time_ind, neuron_ind = np.nonzero(test_spikes)
    np.testing.assert_array_equal(spike_time_ind, np.arange(0, 60, 3))
    np.testing.assert_array_equal(
        neuron_ind,
        np.argmin(
            np.abs(position[spike_time_ind, np.newaxis] - PLACE_FIELD_MEANS), axis=1
        ),
    )


def test_hover_tcm_sr_hover_replay_hovers_where_the_trajectory_starts_and_ends():
    position = simulate_tcm_sr_replay_position(n_time=80, seed=2)
    start_neuron_ind, end_neuron_ind = np.argmin(
        np.abs(position[[0, -1], np.newaxis] - PLACE_FIELD_MEANS), axis=1
    )
    _, hover_spikes = make_hover_replay()
    n_hover = hover_spikes.shape[0]

    replay_time, test_spikes = make_hover_tcm_sr_hover_replay(n_time=80, seed=2)

    assert test_spikes.shape[0] == replay_time.shape[0] == 2 * n_hover + 80
    assert set(np.nonzero(test_spikes[:n_hover])[1]) == {start_neuron_ind}
    assert set(np.nonzero(test_spikes[-n_hover:])[1]) == {end_neuron_ind}
    np.testing.assert_array_equal(
        test_spikes[n_hover:-n_hover], make_tcm_sr_replay(n_time=80, seed=2)[1]
    )


def test_tcm_sr_replay_position_fits_the_given_run_position():
    """A replay sampled from a shorter track's behavior stays on that track."""
    run_position = simulate_position(simulate_time(100_000, 1000), 60.0, 15.0)
    position = simulate_tcm_sr_replay_position(
        run_position=run_position, n_time=300
    )

    assert abs(position[0] - 30.0) < 1.0
    assert position.max() <= 60.0


def test_tcm_sr_replay_position_default_counts_movement_over_100_ms():
    """The default kernel is a one second look-ahead over 100 ms steps of P."""
    explicit = TCM_SR(gamma=0.9, movement_time_step=0.1, sampling_frequency=1000)
    np.testing.assert_array_equal(
        simulate_tcm_sr_replay_position(n_time=200, seed=5),
        simulate_tcm_sr_replay_position(n_time=200, seed=5, transition=explicit),
    )


def test_tcm_sr_replay_position_does_not_mutate_the_transition():
    transition = TCM_SR(gamma=0.99)
    simulate_tcm_sr_replay_position(n_time=10, transition=transition)

    assert not hasattr(transition, "environment")


def test_tcm_sr_replay_inbound_outbound_neurons_follow_direction():
    n_fields = PLACE_FIELD_MEANS.shape[0]
    position = simulate_tcm_sr_replay_position(n_time=200)
    _, test_spikes = make_tcm_sr_replay(
        n_time=200, spike_interval=1, make_inbound_outbound_neurons=True
    )

    assert test_spikes.shape == (200, 2 * n_fields)
    spike_time_ind, neuron_ind = np.nonzero(test_spikes)
    step = np.diff(position)[spike_time_ind[1:] - 1]
    is_outbound_neuron = neuron_ind[1:] >= n_fields
    np.testing.assert_array_equal(is_outbound_neuron[step > 0], True)
    np.testing.assert_array_equal(is_outbound_neuron[step < 0], False)


def test_hover_tcm_sr_hover_replay_inbound_outbound_neurons_shape():
    _, test_spikes = make_hover_tcm_sr_hover_replay(
        n_time=80, make_inbound_outbound_neurons=True
    )
    assert test_spikes.shape[1] == 2 * PLACE_FIELD_MEANS.shape[0]


def _laps(track_length=60.0, n_samples=40_000):
    return simulate_position(simulate_time(n_samples, 1000), track_length, 15.0)


def test_directional_position_round_trips():
    position = _laps()
    u = to_directional_position(position, 60.0)
    folded, is_outbound = from_directional_position(u, 60.0)

    np.testing.assert_allclose(folded, position)
    # At the far end both directions map to u = L, so direction is ambiguous.
    is_moving = (np.diff(position) != 0) & (position[1:] < 60.0)
    np.testing.assert_array_equal(
        is_outbound[1:][is_moving], np.diff(position)[is_moving] > 0
    )
    assert u.min() >= 0.0 and u.max() <= 120.0


def test_directional_position_only_increases_except_at_the_wrap():
    """40 s of 15 cm/s laps on 60 cm is 5 laps, so 4 wraps from 2L back to 0."""
    u = to_directional_position(_laps(), 60.0)
    step = np.diff(u)
    is_wrap = step < -60.0
    assert is_wrap.sum() == 4
    assert np.all(step[~is_wrap] >= -1e-9)


def test_tcm_sr_on_directional_position_is_forward_including_the_wrap():
    """The one-step movement matrix (gamma=0) on the loop only moves forward.

    With gamma > 0 the successor representation also reaches bins "behind" by
    going forward around the whole loop, so direction is checked on P itself.
    """
    u = to_directional_position(_laps(), 60.0)[:, np.newaxis]
    env = Environment(place_bin_size=1.0)
    env.fit_place_grid(u, infer_track_interior=True)
    A = TCM_SR(
        gamma=0.0, asymmetry=1.0, movement_time_step=0.1, sampling_frequency=1000
    ).make_state_transition((env,), u)
    centers = env.place_bin_centers_.ravel()

    # Forward is increasing u, modulo the loop length.
    loop = 120.0
    signed = (centers[np.newaxis, :] - centers[:, np.newaxis] + loop / 2) % loop - loop / 2
    is_interior = env.is_track_interior_.ravel()
    rows = A[is_interior]
    forward = (rows * (signed[is_interior] > 0)).sum(axis=1)
    backward = (rows * (signed[is_interior] < 0)).sum(axis=1)
    assert np.all(backward < 1e-9)
    # The last bins before the wrap move on to the first bins of the loop.
    last = np.flatnonzero(is_interior)[-1]
    # Running slows to 0 at the turnaround, so most of this row is dwell.
    assert A[last, centers < 5.0].sum() > 0.1
    assert np.median(forward) > 0.5

    # The successor representation keeps almost all of its mass within half a
    # loop ahead. The rest (about 2% here) is reached by continuing forward
    # around the whole loop, not by stepping back.
    M = TCM_SR(
        gamma=0.9, asymmetry=1.0, movement_time_step=0.1, sampling_frequency=1000
    ).make_state_transition((env,), u)[is_interior]
    assert np.median((M * (signed[is_interior] < 0)).sum(axis=1)) < 0.05


def test_tcm_sr_replay_on_directional_position_spikes_by_half():
    u = to_directional_position(_laps(TRACK_HEIGHT, 200_000), TRACK_HEIGHT)
    position = simulate_tcm_sr_replay_position(
        run_position=u, directional_track_length=TRACK_HEIGHT, n_time=200
    )
    _, test_spikes = make_tcm_sr_replay(
        run_position=u, directional_track_length=TRACK_HEIGHT, n_time=200,
        spike_interval=1,
    )
    n_fields = PLACE_FIELD_MEANS.shape[0]
    assert test_spikes.shape == (200, 2 * n_fields)
    _, neuron_ind = np.nonzero(test_spikes)
    np.testing.assert_array_equal(neuron_ind >= n_fields, position < TRACK_HEIGHT)
