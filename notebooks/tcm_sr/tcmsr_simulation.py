"""Simulation code shared by the TCM-SR notebooks.

Run data: an animal running back and forth on a linear track, each run at its
own speed, with direction-selective place cells. Replays: trajectories on the
directed loop `u` (see `to_directional_position`), turned into spikes from the
same place fields. Settings default to `tcmsr_simulation_config`, read at call
time.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

import tcmsr_simulation_config as config
from replay_trajectory_classification.simulate import (
    get_trajectory_direction,
    simulate_neuron_with_place_field,
    simulate_place_field_firing_rate,
)
from replay_trajectory_classification.sorted_spikes_simulation import (
    PLACE_FIELD_VARIANCE,
    TRACK_HEIGHT,
    from_directional_position,
    to_directional_position,
)

RUN_SAMPLING_FREQUENCY = 1000  # Hz, before binning to the decoding bin
RUN_MAX_RATE_HZ = 15  # in-field peak rate of a place cell while running


def _default(value, name):
    return getattr(config, name) if value is None else value


@dataclass
class RunData:
    """Simulated running, binned to the decoding bin."""

    time: NDArray[np.float64]
    linear_distance: NDArray[np.float64]
    directional_position: NDArray[np.float64]
    spikes: NDArray[np.float64]
    place_fields: NDArray[np.float64]
    place_field_means: NDArray[np.float64]
    track_length: float
    sampling_frequency: float


def simulate_laps(
    rng: np.random.Generator,
    n_laps: Optional[int] = None,
    running_speed: Optional[float] = None,
    running_speed_cv: Optional[float] = None,
    track_length: float = TRACK_HEIGHT,
    sampling_frequency: int = RUN_SAMPLING_FREQUENCY,
) -> NDArray[np.float64]:
    """Position of `n_laps` out-and-back laps.

    Each run from one end to the other is a half cosine at its own speed.
    """
    n_laps = _default(n_laps, "N_LAPS")
    running_speed = _default(running_speed, "RUNNING_SPEED_CM_S")
    running_speed_cv = _default(running_speed_cv, "RUNNING_SPEED_CV")
    runs = []
    for run_ind in range(2 * n_laps):
        speed = running_speed * rng.lognormal(
            -running_speed_cv**2 / 2, running_speed_cv)
        duration = track_length / speed
        t = np.arange(int(round(duration * sampling_frequency))) / sampling_frequency
        x = track_length / 2 * (1 - np.cos(np.pi * t / duration))
        runs.append(x if run_ind % 2 == 0 else track_length - x)
    return np.concatenate(runs)


def simulate_run_data(
    seed: int = 0,
    n_cells: Optional[int] = None,
    decode_bin_s: Optional[float] = None,
    track_length: float = TRACK_HEIGHT,
    **lap_kwargs,
) -> RunData:
    """Simulate running and place-cell spikes, then bin them to the decoding bin.

    Cells are laid out inbound then outbound, as `make_simulated_run_data`
    does. Spikes use NumPy's global random state; the laps use `seed`.
    """
    n_cells = _default(n_cells, "N_CELLS")
    decode_bin_s = _default(decode_bin_s, "DECODE_BIN_S")
    assert n_cells % 2 == 0, "half the cells are inbound, half outbound"
    place_field_means = np.linspace(0.0, track_length, n_cells // 2)

    position = simulate_laps(
        np.random.default_rng(seed), track_length=track_length, **lap_kwargs)
    trajectory_direction = get_trajectory_direction(position)
    place_fields, spikes = [], []
    for direction in np.unique(trajectory_direction):
        is_condition = trajectory_direction == direction
        for mean in place_field_means:
            place_fields.append(simulate_place_field_firing_rate(
                mean, position, variance=PLACE_FIELD_VARIANCE,
                is_condition=is_condition))
            spikes.append(simulate_neuron_with_place_field(
                mean, position, max_rate=RUN_MAX_RATE_HZ, variance=PLACE_FIELD_VARIANCE,
                sampling_frequency=RUN_SAMPLING_FREQUENCY, is_condition=is_condition))
    place_fields = np.stack(place_fields, axis=1)
    spikes = np.stack(spikes, axis=1)

    # Bin to the decoding bin: sum the spikes, average position and rate.
    samples_per_bin = int(round(decode_bin_s * RUN_SAMPLING_FREQUENCY))
    assert np.isclose(samples_per_bin, decode_bin_s * RUN_SAMPLING_FREQUENCY), \
        "the decoding bin must be a whole number of samples"
    n_bins = position.size // samples_per_bin

    def to_decoding_bins(x, reduce):
        x = x[:n_bins * samples_per_bin]
        return reduce(x.reshape(n_bins, samples_per_bin, *x.shape[1:]), axis=1)

    linear_distance = to_decoding_bins(position, np.mean)
    return RunData(
        time=np.arange(n_bins) * samples_per_bin / RUN_SAMPLING_FREQUENCY,
        linear_distance=linear_distance,
        directional_position=to_directional_position(linear_distance, track_length),
        spikes=to_decoding_bins(spikes, np.sum),
        place_fields=to_decoding_bins(place_fields, np.mean),
        place_field_means=place_field_means,
        track_length=track_length,
        sampling_frequency=RUN_SAMPLING_FREQUENCY / samples_per_bin,
    )


def make_replay_spikes(
    u: NDArray[np.float64],
    rng: np.random.Generator,
    place_field_means: NDArray[np.float64],
    track_length: float,
    peak_rate_hz: Optional[float] = None,
    decode_bin_s: Optional[float] = None,
    idealized: Optional[bool] = None,
) -> NDArray[np.float64]:
    """Spikes from the place cells along trajectory `u`, in the block of its direction.

    Poisson counts from Gaussian place fields with peak `peak_rate_hz`, or, if
    `idealized`, one spike per bin from the nearest place cell.
    """
    peak_rate_hz = _default(peak_rate_hz, "REPLAY_PEAK_RATE_HZ")
    decode_bin_s = _default(decode_bin_s, "DECODE_BIN_S")
    idealized = _default(idealized, "IDEALIZED_SPIKES")
    x, is_outbound = from_directional_position(u, track_length)
    rate = peak_rate_hz * np.exp(
        -(x[:, np.newaxis] - place_field_means) ** 2 / (2 * PLACE_FIELD_VARIANCE))
    if idealized:
        rate = (rate == rate.max(axis=1, keepdims=True)).astype(float)
    empty = np.zeros_like(rate)
    rate = np.where(is_outbound[:, np.newaxis],
                    np.hstack([empty, rate]), np.hstack([rate, empty]))
    if idealized:
        return rate
    return rng.poisson(rate * decode_bin_s).astype(float)


class ReplayPaths:
    """Replay trajectories on the decoder's place bins of the directed loop."""

    def __init__(self, environment, loop_length: float,
                 kernel: Optional[NDArray[np.float64]] = None):
        self.loop_length = loop_length
        self.kernel = kernel
        self.bin_centers = environment.place_bin_centers_.squeeze()
        self.is_interior = environment.is_track_interior_.ravel()
        self.interior_centers = self.bin_centers[self.is_interior]

    def nearest_bin(self, u: float) -> int:
        nearest = np.argmin(np.abs(self.interior_centers - u))
        return np.flatnonzero(self.is_interior)[nearest]

    def sweep(self, start_u: float, step_cm: float, n_time: int) -> NDArray[np.float64]:
        """Constant-speed movement of `step_cm` per bin, wrapping around the loop."""
        return (self.bin_centers[self.nearest_bin(start_u)]
                + step_cm * np.arange(n_time)) % self.loop_length

    def stationary(self, u: float, n_time: int) -> NDArray[np.float64]:
        return np.full(n_time, self.bin_centers[self.nearest_bin(u)])

    def fragmented(self, rng: np.random.Generator, n_time: int) -> NDArray[np.float64]:
        """An independent random track position every bin."""
        return rng.choice(self.interior_centers, n_time)

    def tcm_sr(self, start_u: float, rng: np.random.Generator,
               n_time: int) -> NDArray[np.float64]:
        """A Markov chain on `kernel`, one step per bin."""
        bin_ind = [self.nearest_bin(start_u)]
        for _ in range(n_time - 1):
            p = self.kernel[bin_ind[-1]]
            bin_ind.append(rng.choice(self.bin_centers.size, p=p / p.sum()))
        return self.bin_centers[bin_ind]


def loop_distance(a, b, loop_length: float):
    """Distance between positions on the loop, the shorter way round."""
    d = np.abs(a - b) % loop_length
    return np.minimum(d, loop_length - d)
