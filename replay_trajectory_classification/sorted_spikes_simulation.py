"""Functions for generating sorted spike data.

This module provides simulation functions specifically for sorted (clustered)
spike data used in neural decoding validation and testing.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from replay_trajectory_classification.continuous_state_transitions import (
    TCM_SR,
    estimate_movement_var,
)
from replay_trajectory_classification.environments import Environment
from replay_trajectory_classification.simulate import (
    get_trajectory_direction,
    simulate_neuron_with_place_field,
    simulate_place_field_firing_rate,
    simulate_position,
    simulate_time,
)

SAMPLING_FREQUENCY = 1000
TRACK_HEIGHT = 180
RUNNING_SPEED = 15
PLACE_FIELD_VARIANCE = 6.0**2
PLACE_FIELD_MEANS = np.arange(0, TRACK_HEIGHT + 10, 10, dtype=np.float64)
N_RUNS = 15
REPLAY_SPEEDUP = 120

# Figure Parameters
MM_TO_INCHES = 1.0 / 25.4
ONE_COLUMN = 89.0 * MM_TO_INCHES
ONE_AND_HALF_COLUMN = 140.0 * MM_TO_INCHES
TWO_COLUMN = 178.0 * MM_TO_INCHES
PAGE_HEIGHT = 247.0 * MM_TO_INCHES
GOLDEN_RATIO = (np.sqrt(5) - 1.0) / 2.0


def make_simulated_run_data(
    sampling_frequency: int = SAMPLING_FREQUENCY,
    track_height: float = TRACK_HEIGHT,
    running_speed: float = RUNNING_SPEED,
    n_runs: int = N_RUNS,
    place_field_variance: float = PLACE_FIELD_VARIANCE,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    make_inbound_outbound_neurons: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64]]:
    """Make simulated data of a rat running back and forth
    on a linear maze with sorted spikes.

    Parameters
    ----------
    sampling_frequency : float, optional
        Samples per second
    track_height : float, optional
        Height of the simulated track
    running_speed : float, optional
        Speed of the simulated animal
    n_runs : int, optional
        Number of runs across the track the simulated animal will perform
    place_field_variance : float, optional
        Spatial variance of the place field
    place_field_means : NDArray[np.float64], shape (n_neurons,), optional
        Location of the center of the Gaussian place fields.
    make_inbound_outbound_neurons : bool
        Makes neurons direction selective.

    Returns
    -------
    time : NDArray[np.float64], shape (n_time,)
    position : NDArray[np.float64], shape (n_time,)
        Position of the simualted animal
    sampling_frequency : float
        Samples per second
    spikes : NDArray[np.float64], shape (n_time, n_neurons)
        Binned spike indicator. 1 means spike occured. 0 means no spike occured.
    place_fields : NDArray[np.float64], shape (n_time, n_neurons)

    """
    n_samples = int(n_runs * sampling_frequency * 2 * track_height / running_speed)

    time = simulate_time(n_samples, sampling_frequency)
    position = simulate_position(time, track_height, running_speed)

    if not make_inbound_outbound_neurons:
        place_fields = np.stack(
            [
                simulate_place_field_firing_rate(
                    place_field_mean, position, variance=place_field_variance
                )
                for place_field_mean in place_field_means
            ],
            axis=1,
        )
        spikes = np.stack(
            [
                simulate_neuron_with_place_field(
                    place_field_mean,
                    position,
                    max_rate=15,
                    variance=place_field_variance,
                    sampling_frequency=sampling_frequency,
                )
                for place_field_mean in place_field_means.T
            ],
            axis=1,
        )
    else:
        trajectory_direction = get_trajectory_direction(position)
        place_fields_list: list[NDArray[np.float64]] = []
        spikes_list: list[NDArray[np.float64]] = []
        for direction in np.unique(trajectory_direction):
            is_condition = trajectory_direction == direction
            for place_field_mean in place_field_means:
                place_fields_list.append(
                    simulate_place_field_firing_rate(
                        place_field_mean,
                        position,
                        variance=place_field_variance,
                        is_condition=is_condition,
                    )
                )
                spikes_list.append(
                    simulate_neuron_with_place_field(
                        place_field_mean,
                        position,
                        max_rate=15,
                        variance=place_field_variance,
                        sampling_frequency=sampling_frequency,
                        is_condition=is_condition,
                    )
                )

        place_fields = np.stack(place_fields_list, axis=1)
        spikes = np.stack(spikes_list, axis=1)

    return time, position, sampling_frequency, spikes, place_fields


def make_continuous_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
    track_height: float = TRACK_HEIGHT,
    running_speed: float = RUNNING_SPEED,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    replay_speedup: int = REPLAY_SPEEDUP,
    is_outbound: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Make a simulated continuous replay.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second
    track_height : float, optional
        Height of the simulated track
    running_speed : float, optional
        Speed of the simulated animal
    place_field_means : NDArray[np.float64], optional
        Location of the center of the Gaussian place fields.
    replay_speedup : int, optional
        _description_, by default REPLAY_SPEEDUP
    is_outbound : bool, optional
        _description_, by default True

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_spikes : NDArray[np.float64], shape (n_time, n_neurons)
        Binned spike indicator. 1 means spike occured. 0 means no spike occured.

    """
    replay_speed = running_speed * replay_speedup
    n_samples = int(np.ceil(2 * sampling_frequency * track_height / replay_speed))
    replay_time = simulate_time(n_samples, sampling_frequency)
    true_replay_position = simulate_position(replay_time, track_height, replay_speed)

    # Make inbound or outbound
    replay_time = replay_time[: n_samples // 2]
    if is_outbound:
        true_replay_position = true_replay_position[: n_samples // 2]
    else:
        true_replay_position = true_replay_position[n_samples // 2 :]

    min_times_ind = np.argmin(
        np.abs(true_replay_position[:, np.newaxis] - place_field_means), axis=0
    )

    n_neurons = place_field_means.shape[0]
    test_spikes = np.zeros((replay_time.size, n_neurons))
    test_spikes[
        (
            min_times_ind,
            np.arange(n_neurons),
        )
    ] = 1.0

    return replay_time, test_spikes


def make_hover_replay(
    hover_neuron_ind: Optional[int] = None,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    sampling_frequency: int = SAMPLING_FREQUENCY,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Make a simulated stationary replay.

    Parameters
    ----------
    hover_neuron_ind : int, optional
        _description_, by default None
    place_field_means : NDArray[np.float64], optional
        _description_, by default PLACE_FIELD_MEANS
    sampling_frequency : int, optional
        Samples per second

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_spikes : NDArray[np.float64], shape (n_time, n_neurons)
        Binned spike indicator. 1 means spike occured. 0 means no spike occured.

    """
    n_neurons = place_field_means.shape[0]
    if hover_neuron_ind is None:
        hover_neuron_ind = n_neurons // 2

    N_TIME = 50
    replay_time = np.arange(N_TIME) / sampling_frequency

    spike_time_ind = np.arange(0, N_TIME, 2)

    test_spikes = np.zeros((N_TIME, n_neurons))
    neuron_ind = np.ones((N_TIME // 2,), dtype=int) * hover_neuron_ind

    test_spikes[(spike_time_ind, neuron_ind)] = 1.0

    return replay_time, test_spikes


def make_fragmented_replay(
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    sampling_frequency: int = SAMPLING_FREQUENCY,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Make a simulated fragmented replay.

    Parameters
    ----------
    place_field_means : NDArray[np.float64], optional
        _description_, by default PLACE_FIELD_MEANS
    sampling_frequency : int, optional
        Samples per second

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_spikes : NDArray[np.float64], shape (n_time, n_neurons)
        Binned spike indicator. 1 means spike occured. 0 means no spike occured.

    """
    N_TIME = 10
    replay_time = np.arange(N_TIME) / sampling_frequency
    ind = ([1, 3, 5, 7, 9], [1, -1, 10, -5, 8])
    n_neurons = place_field_means.shape[0]
    test_spikes = np.zeros((N_TIME, n_neurons))
    test_spikes[ind] = 1.0

    return replay_time, test_spikes


def make_hover_continuous_hover_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Make a replay that starts stationary, then is continuous, then is stationary again.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_spikes : NDArray[np.float64], shape (n_time, n_neurons)
        Binned spike indicator. 1 means spike occured. 0 means no spike occured.

    """
    _, test_spikes1 = make_hover_replay(hover_neuron_ind=0)
    _, test_spikes2 = make_continuous_replay()
    _, test_spikes3 = make_hover_replay(hover_neuron_ind=-1)

    test_spikes = np.concatenate((test_spikes1, test_spikes2, test_spikes3))
    replay_time = np.arange(test_spikes.shape[0]) / sampling_frequency

    return replay_time, test_spikes


def make_fragmented_hover_fragmented_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Make a replay that starts fragmented, then is stationary, and then is fragmented again.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_spikes : NDArray[np.float64], shape (n_time, n_neurons)
        Binned spike indicator. 1 means spike occured. 0 means no spike occured.

    """
    _, test_spikes1 = make_fragmented_replay()
    _, test_spikes2 = make_hover_replay(hover_neuron_ind=6)
    _, test_spikes3 = make_fragmented_replay()

    test_spikes = np.concatenate((test_spikes1, test_spikes2, test_spikes3))
    replay_time = np.arange(test_spikes.shape[0]) / sampling_frequency

    return replay_time, test_spikes


def make_fragmented_continuous_fragmented_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Make a replay that is fragmented, then is continuous, then is fragmented again.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_spikes : NDArray[np.float64], shape (n_time, n_neurons)
        Binned spike indicator. 1 means spike occured. 0 means no spike occured.

    """
    _, test_spikes1 = make_fragmented_replay()
    _, test_spikes2 = make_continuous_replay()
    _, test_spikes3 = make_fragmented_replay()

    test_spikes = np.concatenate((test_spikes1, test_spikes2, test_spikes3))
    replay_time = np.arange(test_spikes.shape[0]) / sampling_frequency

    return replay_time, test_spikes


def to_directional_position(
    position: NDArray[np.float64], track_length: float
) -> NDArray[np.float64]:
    """Unfold back-and-forth running on a linear track into one directed loop.

    `u = x` while running outbound (increasing `x`) and `u = 2 * track_length -
    x` while running inbound, so `u` runs from 0 to `2 * track_length` and
    wraps to 0 at the start end. The two branches meet at the far end, so the
    turnaround there is continuous.

    Parameters
    ----------
    position : NDArray[np.float64], shape (n_time,)
        Linear position, between 0 and `track_length`.
    track_length : float

    Returns
    -------
    directional_position : NDArray[np.float64], shape (n_time,)

    """
    position = np.asarray(position, dtype=float).ravel()
    return np.where(_is_outbound(position), position, 2.0 * track_length - position)


def from_directional_position(
    directional_position: NDArray[np.float64], track_length: float
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Fold a directed loop position back to linear position and direction.

    The inverse of `to_directional_position`.

    Returns
    -------
    position : NDArray[np.float64], shape (n_time,)
    is_outbound : NDArray[np.bool_], shape (n_time,)

    """
    directional_position = np.asarray(directional_position, dtype=float).ravel()
    is_outbound = directional_position < track_length
    position = np.where(
        is_outbound, directional_position, 2.0 * track_length - directional_position
    )
    return position, is_outbound


def simulate_tcm_sr_replay_position(
    sampling_frequency: int = SAMPLING_FREQUENCY,
    track_height: float = TRACK_HEIGHT,
    running_speed: float = RUNNING_SPEED,
    n_runs: int = N_RUNS,
    run_position: Optional[NDArray[np.float64]] = None,
    n_time: int = 100,
    start_position: Optional[float] = None,
    transition: Optional[TCM_SR] = None,
    seed: int = 0,
    directional_track_length: Optional[float] = None,
) -> NDArray[np.float64]:
    """Sample a replay trajectory from a TCM-SR transition of running behavior.

    The run behavior is binned with `place_bin_size =
    sqrt(estimate_movement_var(...))`, and the replay is a Markov chain sampled
    from the resulting `TCM_SR` transition, one step per sample.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second
    track_height : float, optional
        Height of the simulated track
    running_speed : float, optional
        Speed of the simulated animal
    n_runs : int, optional
        Number of runs in the behavior the transition is fit to
    run_position : NDArray[np.float64], shape (n_run_time,), optional
        Run behavior to fit the transition to, usually the position the
        classifier was fit on. By default the position of
        `make_simulated_run_data` is regenerated from `track_height`,
        `running_speed` and `n_runs`, which are otherwise ignored.
    n_time : int, optional
        Number of samples in the replay
    start_position : float, optional
        Starting position. By default the middle of the run behavior's range.
    transition : TCM_SR, optional
        Transition to sample from. Its `environment_name` must be "". It is
        copied, so the classifier's own instance can be passed. By default
        `TCM_SR(gamma=0.9, movement_time_step=0.1,
        sampling_frequency=sampling_frequency)`.
    seed : int, optional
        Seed for the random number generator.
    directional_track_length : float, optional
        Set when `run_position` is a directed loop position from
        `to_directional_position` with this track length. The bin size is then
        estimated from the folded linear position, since the loop's wrap would
        inflate it.

    Returns
    -------
    replay_position : NDArray[np.float64], shape (n_time,)
        Position bin center at each sample, in the same coordinate as
        `run_position`.

    """
    if run_position is None:
        n_samples = int(
            n_runs * sampling_frequency * 2 * track_height / running_speed
        )
        run_position = simulate_position(
            simulate_time(n_samples, sampling_frequency), track_height, running_speed
        )
    run_position = np.asarray(run_position, dtype=float).reshape(-1, 1)

    if directional_track_length is None:
        linear_position = run_position
    else:
        linear_position, _ = from_directional_position(
            run_position, directional_track_length
        )
    movement_var = estimate_movement_var(linear_position, sampling_frequency)
    environment = Environment(place_bin_size=np.sqrt(movement_var))
    environment.fit_place_grid(run_position, infer_track_interior=True)

    if transition is None:
        transition = TCM_SR(
            gamma=0.9, movement_time_step=0.1, sampling_frequency=sampling_frequency
        )
    # `make_state_transition` stores the environment on the instance.
    transition = dataclasses.replace(transition)
    state_transition = transition.make_state_transition((environment,), run_position)

    bin_centers = environment.place_bin_centers_.squeeze()
    interior_ind = np.flatnonzero(environment.is_track_interior_.ravel(order="F"))
    if start_position is None:
        start_position = (np.nanmin(run_position) + np.nanmax(run_position)) / 2

    rng = np.random.default_rng(seed)
    bin_ind = np.empty((n_time,), dtype=int)
    bin_ind[0] = interior_ind[
        np.argmin(np.abs(bin_centers[interior_ind] - start_position))
    ]
    for time_ind in range(1, n_time):
        bin_ind[time_ind] = rng.choice(
            bin_centers.size, p=state_transition[bin_ind[time_ind - 1]]
        )

    return bin_centers[bin_ind]


def _is_outbound(position: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Whether each sample is moving outbound (increasing position).

    A sample that does not move keeps the previous sample's direction, and the
    first sample takes the direction of the first move.
    """
    step = np.diff(position)
    moving_ind = np.flatnonzero(step != 0)
    if moving_ind.size == 0:
        return np.ones(position.shape, dtype=bool)
    # Index of the latest move at or before each step, or the first move.
    latest_move = np.maximum.accumulate(
        np.where(step != 0, np.arange(step.size), -1)
    )
    latest_move[latest_move < 0] = moving_ind[0]
    is_outbound = step[latest_move] > 0
    return np.insert(is_outbound, 0, is_outbound[0])


def _nearest_neuron(
    position: NDArray[np.float64],
    place_field_means: NDArray[np.float64],
    is_outbound: Optional[NDArray[np.bool_]] = None,
) -> NDArray[np.int64]:
    """Index of the neuron whose place field is closest to each position.

    With `is_outbound`, neurons are laid out as in
    `make_simulated_run_data(make_inbound_outbound_neurons=True)`: one block of
    inbound neurons followed by one block of outbound neurons, each covering
    `place_field_means`.
    """
    neuron_ind = np.argmin(
        np.abs(np.asarray(position)[:, np.newaxis] - place_field_means), axis=1
    )
    if is_outbound is not None:
        neuron_ind = neuron_ind + is_outbound * place_field_means.shape[0]
    return neuron_ind


def _spike_nearest_place_field(
    position: NDArray[np.float64],
    place_field_means: NDArray[np.float64],
    spike_interval: int,
    make_inbound_outbound_neurons: bool = False,
    directional_track_length: Optional[float] = None,
) -> NDArray[np.float64]:
    """Every `spike_interval` samples, spike the neuron whose place field is
    closest to the current position (and, optionally, prefers the current
    direction of movement).

    With `directional_track_length`, `position` is a directed loop position and
    the direction is read from it rather than from the steps; this implies the
    inbound/outbound neuron layout."""
    if directional_track_length is not None:
        make_inbound_outbound_neurons = True
    n_neurons = place_field_means.shape[0] * (2 if make_inbound_outbound_neurons else 1)
    test_spikes = np.zeros((position.shape[0], n_neurons))
    spike_time_ind = np.arange(0, position.shape[0], spike_interval)
    if directional_track_length is not None:
        linear_position, is_outbound = from_directional_position(
            position[spike_time_ind], directional_track_length
        )
    else:
        linear_position = position[spike_time_ind]
        is_outbound = (
            _is_outbound(position)[spike_time_ind]
            if make_inbound_outbound_neurons
            else None
        )
    neuron_ind = _nearest_neuron(linear_position, place_field_means, is_outbound)
    test_spikes[(spike_time_ind, neuron_ind)] = 1.0

    return test_spikes


def make_tcm_sr_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    spike_interval: int = 2,
    make_inbound_outbound_neurons: bool = False,
    **kwargs,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Make a simulated replay whose trajectory is sampled from a TCM-SR transition.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second
    place_field_means : NDArray[np.float64], optional
        Location of the center of the Gaussian place fields.
    spike_interval : int, optional
        Samples between spikes. Each spike comes from the neuron whose place
        field is closest to the trajectory.
    make_inbound_outbound_neurons : bool, optional
        Lay neurons out as `make_simulated_run_data` does with this option, and
        spike from the block matching the direction of movement.
    **kwargs
        Passed to `simulate_tcm_sr_replay_position` (e.g. `n_time`,
        `start_position`, `transition`, `seed`, `directional_track_length`).
        With `directional_track_length`, the replay is sampled on the directed
        loop and spikes come from the block of the half it is on.

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_spikes : NDArray[np.float64], shape (n_time, n_neurons)
        Binned spike indicator. 1 means spike occured. 0 means no spike occured.

    """
    replay_position = simulate_tcm_sr_replay_position(
        sampling_frequency=sampling_frequency, **kwargs
    )
    test_spikes = _spike_nearest_place_field(
        replay_position, place_field_means, spike_interval,
        make_inbound_outbound_neurons, kwargs.get("directional_track_length"),
    )
    replay_time = simulate_time(test_spikes.shape[0], sampling_frequency)

    return replay_time, test_spikes


def make_hover_tcm_sr_hover_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    spike_interval: int = 2,
    make_inbound_outbound_neurons: bool = False,
    **kwargs,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Make a replay that starts stationary, then follows a TCM-SR trajectory,
    then is stationary again.

    Each stationary period hovers on the neuron closest to where the TCM-SR
    trajectory starts or ends, so the position is continuous across the
    switches.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second
    place_field_means : NDArray[np.float64], optional
        Location of the center of the Gaussian place fields.
    spike_interval : int, optional
        Samples between spikes in the TCM-SR period.
    make_inbound_outbound_neurons : bool, optional
        Lay neurons out as `make_simulated_run_data` does with this option. The
        hovers use the block matching the direction of the TCM-SR step next to
        them.
    **kwargs
        Passed to `simulate_tcm_sr_replay_position` (e.g. `n_time`,
        `start_position`, `transition`, `seed`).

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_spikes : NDArray[np.float64], shape (n_time, n_neurons)
        Binned spike indicator. 1 means spike occured. 0 means no spike occured.

    """
    replay_position = simulate_tcm_sr_replay_position(
        sampling_frequency=sampling_frequency, **kwargs
    )
    directional_track_length = kwargs.get("directional_track_length")
    endpoints = replay_position[[0, -1]]
    if directional_track_length is not None:
        make_inbound_outbound_neurons = True
        endpoints, is_outbound = from_directional_position(
            endpoints, directional_track_length
        )
    elif make_inbound_outbound_neurons:
        is_outbound = _is_outbound(replay_position)[[0, -1]]
    else:
        is_outbound = None
    hover_place_field_means = (
        np.concatenate((place_field_means,) * 2)
        if make_inbound_outbound_neurons
        else place_field_means
    )
    start_neuron_ind, end_neuron_ind = _nearest_neuron(
        endpoints, place_field_means, is_outbound
    )

    _, test_spikes1 = make_hover_replay(
        hover_neuron_ind=start_neuron_ind,
        place_field_means=hover_place_field_means,
        sampling_frequency=sampling_frequency,
    )
    test_spikes2 = _spike_nearest_place_field(
        replay_position, place_field_means, spike_interval,
        make_inbound_outbound_neurons, directional_track_length,
    )
    _, test_spikes3 = make_hover_replay(
        hover_neuron_ind=end_neuron_ind,
        place_field_means=hover_place_field_means,
        sampling_frequency=sampling_frequency,
    )

    test_spikes = np.concatenate((test_spikes1, test_spikes2, test_spikes3))
    replay_time = np.arange(test_spikes.shape[0]) / sampling_frequency

    return replay_time, test_spikes


def make_theta_sweep(
    sampling_frequency: int = SAMPLING_FREQUENCY, n_sweeps: int = 5
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate theta sweeping

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second
    n_sweeps : int, optional
        Number of sweeps to simulate, by default 5

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_spikes : NDArray[np.float64], shape (n_time, n_neurons)
        Binned spike indicator. 1 means spike occured. 0 means no spike occured.

    """
    _, test_spikes1 = make_continuous_replay(is_outbound=False, replay_speedup=145)
    _, test_spikes2 = make_continuous_replay(is_outbound=True, replay_speedup=145)

    test_spikes = np.concatenate(
        [test_spikes1[test_spikes1.shape[0] // 2 :], test_spikes2] * n_sweeps
    )
    replay_time = np.arange(test_spikes.shape[0]) / sampling_frequency

    return replay_time, test_spikes
