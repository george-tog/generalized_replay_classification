"""Classes for constructing different types of movement models.

This module provides continuous state transition models that describe how
animals move through space, including random walks, empirical movement,
and uniform transitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Hashable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from replay_trajectory_classification.core import atleast_2d
from replay_trajectory_classification.environments import Environment, diffuse_each_bin

logger = logging.getLogger(__name__)


def _normalize_row_probability(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ensure the state transition matrix rows sum to 1.

    Parameters
    ----------
    x : NDArray[np.float64], shape (n_rows, n_cols)
        Input matrix to normalize.

    Returns
    -------
    normalized_x : NDArray[np.float64], shape (n_rows, n_cols)
        Row-normalized matrix where each row sums to 1.

    """
    x /= x.sum(axis=1, keepdims=True)
    x[np.isnan(x)] = 0
    return x


def estimate_movement_var(
    position: NDArray[np.float64], sampling_frequency: int = 1
) -> NDArray[np.float64]:
    """Estimate the movement variance based on position data.

    Parameters
    ----------
    position : NDArray[np.float64], shape (n_time, n_position_dims)
        Position coordinates of the animal over time.
    sampling_frequency : int, optional
        Number of samples per second, by default 1.

    Returns
    -------
    movement_var : NDArray[np.float64], shape (n_position_dims,)
        Covariance matrix of movement scaled by sampling frequency.

    """
    position = atleast_2d(position)
    is_nan = np.any(np.isnan(position), axis=1)
    position = position[~is_nan]
    movement_var = np.cov(np.diff(position, axis=0), rowvar=False)
    return movement_var * sampling_frequency


def _random_walk_on_track_graph(
    place_bin_centers: NDArray[np.float64],
    movement_mean: float,
    movement_var: float,
    place_bin_center_ind_to_node: NDArray[np.float64],
    distance_between_nodes: dict[dict],
) -> NDArray[np.float64]:
    """Estimates the random walk probabilities based on the spatial
    topology given by the track graph.

    Parameters
    ----------
    place_bin_centers : NDArray[np.float64], shape (n_position_bins,)
    movement_mean : float
    movement_var : float
    place_bin_center_ind_to_node : NDArray[np.float64]
        Mapping of place bin center to track graph node
    distance_between_nodes : dict[dict]
        Distance between each pair of track graph nodes with an edge.

    Returns
    -------
    random_walk : NDArray[np.float64], shape (n_position_bins, n_position_bins)

    """
    state_transition = np.zeros((place_bin_centers.size, place_bin_centers.size))
    gaussian = multivariate_normal(mean=movement_mean, cov=movement_var)

    for bin_ind1, node1 in enumerate(place_bin_center_ind_to_node):
        for bin_ind2, node2 in enumerate(place_bin_center_ind_to_node):
            try:
                state_transition[bin_ind1, bin_ind2] = gaussian.pdf(
                    distance_between_nodes[node1][node2]
                )
            except KeyError:
                # bins not on track interior will be -1 and not in distance
                # between nodes
                continue

    return state_transition


def _empirical_movement_counts(
    environment: Environment,
    position: NDArray[np.float64],
    is_selected: NDArray[np.bool_],
    lag: int = 1,
) -> NDArray[np.float64]:
    """Count the animal's observed movements between place bins over `lag` samples.

    `counts[i, j]` is the number of movements from bin `i` to bin `j`, matching
    the `A[i, j] = p(x_t = j | x_{t-1} = i)` convention of this module. Each
    sample `t` is paired with `t + lag`, and a pair is kept only if every sample
    in `[t, t + lag]` is selected. Counting the `lag`-step movement directly
    keeps the animal's momentum, which a matrix power of the one-step counts
    would not.

    `range` is not passed to `np.histogramdd`: it is a no-op with explicit edge
    arrays, and `Environment.position_range` has half the entries the doubled
    `bins` needs, so passing it raises.

    Parameters
    ----------
    environment : Environment
        Fitted environment.
    position : NDArray[np.float64], shape (n_time, n_position_dims)
        Position of the animal.
    is_selected : NDArray[np.bool_], shape (n_time,)
        Time points to estimate the movement from.
    lag : int, optional
        Number of samples between the two ends of each counted movement, by
        default 1.

    Returns
    -------
    movement_counts : NDArray[np.float64], shape (n_position_bins, n_position_bins)

    """
    if int(lag) != lag or lag < 1:
        raise ValueError(f"lag must be a positive integer; got {lag}")
    lag = int(lag)
    position = atleast_2d(position)
    is_selected = np.asarray(is_selected, dtype=bool)
    # Pair first, then keep pairs with no excluded sample in their window.
    # Masking first would pair samples across a gap and invent a long jump.
    n_excluded = np.concatenate(([0], np.cumsum(~is_selected)))
    is_pair = (n_excluded[lag + 1 :] - n_excluded[: -lag - 1]) == 0
    # `edges_ * 2` repeats the edge list: a joint histogram over (x_{t-lag}, x_t).
    counts, _ = np.histogramdd(
        np.concatenate((position[:-lag][is_pair], position[lag:][is_pair]), axis=1),
        bins=environment.edges_ * 2,
    )
    n_position_dims = position.shape[1]
    n_position_bins = np.prod(counts.shape[:n_position_dims])
    # order="F" matches `place_bin_centers_` and `is_track_interior_.ravel("F")`.
    return counts.reshape((n_position_bins, n_position_bins), order="F")


@dataclass
class RandomWalk:
    """A transition where the movement stays locally close in space

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit
    movement_var : float, optional
        How far the animal is can move in one time bin during normal
        movement.
    movement_mean : float, optional
    use_diffusion : bool, optional
        Use diffusion to respect the geometry of the environment
    """

    environment_name: str = ""
    movement_var: float = 6.0
    movement_mean: float = 0.0
    use_diffusion: bool = False

    def make_state_transition(self, environments: tuple[Environment]) -> NDArray[np.float64]:
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : tuple[Environment]
            The existing environments in the model

        Returns
        -------
        state_transition_matrix : NDArray[np.float64], shape (n_position_bins, n_position_bins)

        """
        self.environment = environments[environments.index(self.environment_name)]

        if self.environment.track_graph is None:
            n_position_dims = self.environment.place_bin_centers_.shape[1]

            if (n_position_dims == 1) or not self.use_diffusion:
                transition_matrix = np.stack(
                    [
                        multivariate_normal(
                            mean=center + self.movement_mean, cov=self.movement_var
                        ).pdf(self.environment.place_bin_centers_)
                        for center in self.environment.place_bin_centers_
                    ],
                    axis=1,
                )
            else:
                dx = self.environment.edges_[0][1] - self.environment.edges_[0][0]
                dy = self.environment.edges_[1][1] - self.environment.edges_[1][0]
                n_total_bins = np.prod(self.environment.is_track_interior_.shape)
                transition_matrix = diffuse_each_bin(
                    self.environment.is_track_interior_,
                    self.environment.is_track_boundary_,
                    dx,
                    dy,
                    std=np.sqrt(self.movement_var),
                ).reshape((n_total_bins, -1), order="F")
        else:
            place_bin_center_ind_to_node = np.asarray(
                self.environment.place_bin_centers_nodes_df_.node_id
            )
            transition_matrix = _random_walk_on_track_graph(
                self.environment.place_bin_centers_,
                self.movement_mean,
                self.movement_var,
                place_bin_center_ind_to_node,
                self.environment.distance_between_nodes_,
            )

        is_track_interior = self.environment.is_track_interior_.ravel(order="F")
        transition_matrix[~is_track_interior] = 0.0
        transition_matrix[:, ~is_track_interior] = 0.0

        return _normalize_row_probability(transition_matrix)


@dataclass
class Uniform:
    """
    Attributes
    ----------
    environment_name : str, optional
        Name of first environment to fit
    environment_name2 : str, optional
        Name of second environment to fit if going from one environment to
        another
    """

    environment_name: str = ""
    environment2_name: str = None

    def make_state_transition(self, environments: tuple[Environment]):
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : tuple[Environment]
            The existing environments in the model

        Returns
        -------
        state_transition_matrix : NDArray[np.float64], shape (n_position_bins, n_position_bins)

        """
        self.environment1 = environments[environments.index(self.environment_name)]
        n_bins1 = self.environment1.place_bin_centers_.shape[0]
        is_track_interior1 = self.environment1.is_track_interior_.ravel(order="F")

        if self.environment2_name is None:
            n_bins2 = n_bins1
            is_track_interior2 = is_track_interior1.copy()
        else:
            self.environment2 = environments[environments.index(self.environment2_name)]
            n_bins2 = self.environment2.place_bin_centers_.shape[0]
            is_track_interior2 = self.environment2.is_track_interior_.ravel(order="F")

        transition_matrix = np.ones((n_bins1, n_bins2))

        transition_matrix[~is_track_interior1] = 0.0
        transition_matrix[:, ~is_track_interior2] = 0.0

        return _normalize_row_probability(transition_matrix)


@dataclass
class Identity:
    """A transition where the movement stays within a place bin

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit
    """

    environment_name: str = ""

    def make_state_transition(self, environments: tuple[Environment]):
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : tuple[Environment]
            The existing environments in the model

        Returns
        -------
        state_transition_matrix : NDArray[np.float64], shape (n_position_bins, n_position_bins)

        """
        self.environment = environments[environments.index(self.environment_name)]
        n_bins = self.environment.place_bin_centers_.shape[0]

        transition_matrix = np.identity(n_bins)

        is_track_interior = self.environment.is_track_interior_.ravel(order="F")
        transition_matrix[~is_track_interior] = 0.0
        transition_matrix[:, ~is_track_interior] = 0.0

        return _normalize_row_probability(transition_matrix)


@dataclass
class EmpiricalMovement:
    """A transition matrix trained on the animal's actual movement

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit
    encoding_group : Hashable, optional
        Name of encoding group to fit. By default 0, the label used when no
        `encoding_group_labels` are given. `None` would select no time points.
    speedup : int, optional
        Used to make the empirical transition matrix "faster", means allowing for
        all the same transitions made by the animal but sped up by
        `speedup` times. So `speedup​=20` means 20x faster than the
        animal's movement.
    """

    environment_name: str = ""
    encoding_group: Hashable = 0
    speedup: int = 1

    def make_state_transition(
        self,
        environments: tuple[Environment],
        position: NDArray[np.float64],
        is_training: NDArray[np.float64] = None,
        encoding_group_labels: NDArray[np.float64] = None,
        environment_labels: NDArray[np.float64] = None,
    ):
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : tuple[Environment]
            The existing environments in the model
        position : NDArray[np.float64]
            Position of the animal
        is_training : NDArray[np.float64], optional
            Boolean array that determines what data to train the place fields on, by default None
        encoding_group_labels : NDArray[np.float64], shape (n_time,), optional
            If place fields should correspond to each state, label each time point with the group name
            For example, Some points could correspond to inbound trajectories and some outbound, by default None
        environment_labels : NDArray[np.float64], shape (n_time,), optional
            If there are multiple environments, label each time point with the environment name, by default None

        Returns
        -------
        state_transition_matrix : NDArray[np.float64], shape (n_position_bins, n_position_bins)
            `state_transition_matrix[i, j]` is
            `p(position_t = j | position_{t-1} = i)`, as in `RandomWalk` and the
            forward recursions in `core`.

        """
        self.environment = environments[environments.index(self.environment_name)]

        n_time = position.shape[0]
        if is_training is None:
            is_training = np.ones((n_time,), dtype=bool)

        if encoding_group_labels is None:
            is_encoding = np.ones((n_time,), dtype=bool)
        else:
            is_encoding = encoding_group_labels == self.encoding_group

        if environment_labels is None:
            is_environment = np.ones((n_time,), dtype=bool)
        else:
            is_environment = environment_labels == self.environment_name

        state_transition = _normalize_row_probability(
            _empirical_movement_counts(
                self.environment,
                position,
                is_training & is_encoding & is_environment,
            )
        )

        return np.linalg.matrix_power(state_transition, self.speedup)


def _tcm_sr_successor(
    movement: NDArray[np.float64], gamma: float, normalize_before_mix: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The successor representation `M` of `movement` and the term `M.T @ M`.

    Never modifies `movement`, so a cached `P` can be reused across `gamma`.

    Parameters
    ----------
    movement : NDArray[np.float64], shape (n_interior_bins, n_interior_bins)
        Row-stochastic empirical movement matrix `P` on the track interior.
    gamma : float
        Successor representation discount, in [0, 1).
    normalize_before_mix : bool
        Row normalize both terms.

    Returns
    -------
    successor : NDArray[np.float64], shape (n_interior_bins, n_interior_bins)
        `M = (I - gamma * P)^-1 @ P`.
    retrieved : NDArray[np.float64], shape (n_interior_bins, n_interior_bins)
        `M.T @ M`.

    """
    # cond(I - gamma * P) is about 1 / (1 - gamma), so a direct solve is fine.
    successor = np.linalg.solve(
        np.identity(movement.shape[0]) - gamma * movement, movement
    )
    retrieved = successor.T @ successor
    if normalize_before_mix:
        successor = _normalize_row_probability(successor)
        retrieved = _normalize_row_probability(retrieved)
    return successor, retrieved


def _tcm_sr_mix(
    successor: NDArray[np.float64],
    retrieved: NDArray[np.float64],
    omega: float,
    zero_diagonal: bool,
) -> NDArray[np.float64]:
    """Mix the two TCM-SR terms into a row-stochastic kernel.

    Builds a new array, so `successor` and `retrieved` can be reused.

    Parameters
    ----------
    successor, retrieved : NDArray[np.float64], shape (n_bins, n_bins)
        The outputs of `_tcm_sr_successor`.
    omega : float
        Weight of `retrieved`, in [0, 1].
    zero_diagonal : bool
        Remove self-transitions before normalizing.

    Returns
    -------
    kernel : NDArray[np.float64], shape (n_bins, n_bins)

    """
    local_kernel = (1.0 - omega) * successor + omega * retrieved

    if zero_diagonal:
        np.fill_diagonal(local_kernel, 0.0)

    # Clip before normalizing: the solve can emit tiny negative entries.
    local_kernel = _normalize_row_probability(np.maximum(local_kernel, 0.0))

    if np.any(local_kernel.sum(axis=1) == 0):
        raise ValueError(
            "`zero_diagonal=True` emptied at least one row: those bins only "
            "ever transition to themselves, so removing the diagonal leaves "
            "nowhere to go. Set `zero_diagonal=False` or `beta` > 0."
        )
    return local_kernel


def _tcm_sr_kernel(
    movement: NDArray[np.float64],
    gamma: float,
    omega: float,
    normalize_before_mix: bool,
    zero_diagonal: bool,
) -> NDArray[np.float64]:
    """The TCM-SR kernel on the track interior, given the movement matrix `P`."""
    successor, retrieved = _tcm_sr_successor(movement, gamma, normalize_before_mix)
    return _tcm_sr_mix(successor, retrieved, omega, zero_diagonal)


def _tcm_sr_expected_log_likelihood(
    expected_counts: NDArray[np.float64], kernel: NDArray[np.float64]
) -> float:
    """Expected complete-data log likelihood of a kernel, the EM M-step objective.

    `Q = sum_ij E[i, j] * log(K[i, j])` over entries with `E > 0`. Zeros in `K`
    are floored at the smallest positive float, so such a kernel scores very
    low but finite.

    Parameters
    ----------
    expected_counts : NDArray[np.float64], shape (n_bins, n_bins)
        Expected number of transitions from bin `i` to bin `j`.
    kernel : NDArray[np.float64], shape (n_bins, n_bins)

    Returns
    -------
    expected_log_likelihood : float

    """
    is_count = expected_counts > 0
    return float(
        np.sum(
            expected_counts[is_count]
            * np.log(np.maximum(kernel[is_count], np.finfo(np.float64).tiny))
        )
    )


#: Names of the `TCM_SR` fields that `TCM_SR.m_step` can fit.
_TCM_SR_LEARNABLE = ("gamma", "asymmetry")


@dataclass
class TCM_SR:
    """A transition given by a Temporal Context Model successor representation.

    The animal's empirical movement matrix `P` is turned into a multi-step
    successor representation `M` and mixed with a symmetric retrieved-context
    term `M.T @ M`:

        M = (I - gamma * P)^-1 @ P
        A = asymmetry * rownorm(M) + (1 - asymmetry) * rownorm(M.T @ M)

    `gamma` sets how far ahead the kernel looks, and `asymmetry` how strongly it
    favors the direction the animal moved in. Both can be fit to replay by EM
    (see `learn`). Set `movement_time_step` so that `gamma` means the same
    look-ahead in seconds whatever the position sampling rate.

    Both terms are computed on the track interior only. Unlike `RandomWalk`, which
    scores transitions by Euclidean distance between bin centers, this kernel
    scores them by how reachable one bin is from another given the transitions
    the animal actually made.

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit
    encoding_group : Hashable, optional
        Name of encoding group to fit. By default 0, the label used when no
        `encoding_group_labels` are given.
    gamma : float, optional
        Successor representation discount, in [0, 1). The look-ahead horizon is
        `1 / (1 - gamma)` steps of `P`, or `movement_time_step / (1 - gamma)`
        seconds when `movement_time_step` is set. By default 0.95.
    asymmetry : float, optional
        Weight on the directional term `M`, in [0, 1], with `1 - asymmetry` on
        the symmetric term `M.T @ M`. `asymmetry = 1` is the plain forward
        successor representation and `asymmetry = 0` the symmetric term alone.
        By default 0.5. Ignored when `omega` is set.

        The kernel can only be as asymmetric as `P`. Behavior that runs a
        linear track equally in both directions gives a symmetric `P`, and
        then no setting of `asymmetry` produces a forward bias. Give each
        direction its own segment of position (a directed coordinate), or fit
        each direction separately (e.g. `encoding_group` with direction
        labels), to get a forward kernel.
    omega : float, optional
        Alternative to `asymmetry`: weight `omega` on `M.T @ M` and
        `1 - omega` on `M`, normalized once after mixing. By default None,
        which uses `asymmetry`.

        Without `normalize_before_mix`, the realized weight on `M.T @ M`
        differs from `omega`, because `M` sums to `1 / (1 - gamma)` per row
        while `M.T @ M` sums to `colsum(M) / (1 - gamma)`.
    beta : float, optional
        Weight of a local random walk blended into `P` before the successor
        representation is solved. Use a small value when interior bins have no
        observed outgoing transitions. By default 0.0.
    local_movement_var : float, optional
        Variance of the random walk that `beta` blends in. By default 6.0. With
        `gamma=0`, `beta=1` and `asymmetry=1` the kernel is
        `RandomWalk(movement_var=local_movement_var)`.
    zero_diagonal : bool, optional
        Remove the kernel's diagonal. Set this only when another state supplies
        the dwell. By default False.
    normalize_before_mix : bool, optional
        Used only with `omega`. Row normalize `M` and `M.T @ M` before mixing,
        so that `omega` is the literal weight on `M.T @ M`. By default False.
    movement_time_step : float, optional
        Time step of `P`, in seconds: each position sample is paired with the
        sample `movement_time_step` later. Requires `sampling_frequency`. By
        default None, which pairs consecutive samples.

        It is also the kernel's minimum step, so `movement_time_step` divided
        by the decoding bin is the slowest replay compression it can represent
        (100x for a 100 ms step and 1 ms bins).
    sampling_frequency : float, optional
        Position samples per second, used with `movement_time_step`. By
        default None.
    learn : tuple of str, optional
        Parameters that
        `estimate_parameters(estimate_continuous_transition=True)` fits to
        replay: any of `"gamma"` and `"asymmetry"`. By default empty. `omega`
        is not learnable.

    """

    environment_name: str = ""
    encoding_group: Hashable = 0
    gamma: float = 0.95
    omega: Optional[float] = None
    beta: float = 0.0
    local_movement_var: float = 6.0
    zero_diagonal: bool = False
    normalize_before_mix: bool = False
    asymmetry: float = 0.5
    movement_time_step: Optional[float] = None
    sampling_frequency: Optional[float] = None
    learn: tuple[str, ...] = ()

    @property
    def lag(self) -> int:
        """Samples between the two ends of each movement counted into `P`."""
        if (self.movement_time_step is None) != (self.sampling_frequency is None):
            raise ValueError(
                "movement_time_step and sampling_frequency must be set together; "
                f"got movement_time_step={self.movement_time_step}, "
                f"sampling_frequency={self.sampling_frequency}"
            )
        if self.movement_time_step is None:
            return 1
        lag = int(round(self.movement_time_step * self.sampling_frequency))
        if lag < 1:
            raise ValueError(
                f"movement_time_step={self.movement_time_step} s is shorter than "
                f"one sample at sampling_frequency={self.sampling_frequency}"
            )
        return lag

    def _validate(self) -> int:
        """Check the field ranges and `learn`, and return the lag of `P`.

        Run when the kernel is built, not in `__post_init__`, as
        `DiagonalDiscrete` does.
        """
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError(f"gamma must be in [0, 1); got {self.gamma}")
        if self.omega is not None and not 0.0 <= self.omega <= 1.0:
            raise ValueError(f"omega must be between 0 and 1; got {self.omega}")
        if self.omega is None and not (
            self.asymmetry is not None and 0.0 <= self.asymmetry <= 1.0
        ):
            raise ValueError(
                f"asymmetry must be between 0 and 1; got {self.asymmetry}"
            )
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"beta must be between 0 and 1; got {self.beta}")
        for name in self.learn:
            if name == "omega":
                raise ValueError(
                    "omega is not learnable; learn `asymmetry` instead."
                )
            if name not in _TCM_SR_LEARNABLE:
                raise ValueError(
                    f"Cannot learn {name!r}; learnable parameters are "
                    f"{_TCM_SR_LEARNABLE}."
                )
        if "asymmetry" in self.learn and self.omega is not None:
            raise ValueError("Learning `asymmetry` requires `omega` to be unset.")
        return self.lag

    def _mixing(self) -> tuple[float, bool]:
        """The weight on `M.T @ M` and whether both terms are row normalized."""
        if self.omega is not None:
            return self.omega, self.normalize_before_mix
        return 1.0 - self.asymmetry, True

    def make_state_transition(
        self,
        environments: tuple[Environment],
        position: NDArray[np.float64],
        is_training: NDArray[np.bool_] = None,
        encoding_group_labels: NDArray[np.int64] = None,
        environment_labels: NDArray[np.int64] = None,
    ) -> NDArray[np.float64]:
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : tuple[Environment]
            The existing environments in the model
        position : NDArray[np.float64]
            Position of the animal
        is_training : NDArray[np.bool_], optional
            Boolean array that determines what data to train on, by default None
        encoding_group_labels : NDArray[np.int64], shape (n_time,), optional
            Label each time point with the encoding group name, by default None
        environment_labels : NDArray[np.int64], shape (n_time,), optional
            Label each time point with the environment name, by default None

        Returns
        -------
        state_transition_matrix : NDArray[np.float64], shape (n_bins, n_bins)
            Rows are `position_{t-1}` and columns are `position_t`. Off-track
            rows and columns are zero and interior rows sum to one.

        """
        lag = self._validate()

        self.environment = environments[environments.index(self.environment_name)]

        n_time = position.shape[0]
        is_selected = np.ones((n_time,), dtype=bool)
        if is_training is not None:
            is_selected &= np.asarray(is_training).squeeze()
        if encoding_group_labels is not None and self.encoding_group is not None:
            is_selected &= encoding_group_labels == self.encoding_group
        if environment_labels is not None:
            is_selected &= environment_labels == self.environment_name

        is_track_interior = self.environment.is_track_interior_.ravel(order="F")
        interior_ind = np.ix_(is_track_interior, is_track_interior)
        n_interior_bins = int(is_track_interior.sum())
        if n_interior_bins < 2:
            raise ValueError(
                "TCM_SR needs at least 2 track interior bins; got "
                f"{n_interior_bins}."
            )

        counts = _empirical_movement_counts(
            self.environment, position, is_selected, lag=lag
        )
        if counts.sum() == 0:
            raise ValueError(
                "TCM_SR found no observed transitions. Selected "
                f"{int(is_selected.sum())} of {n_time} time points; check "
                "`is_training` and that "
                f"`encoding_group={self.encoding_group!r}` matches the labels."
            )

        # Position samples can land in bins `get_track_interior` excluded. Drop
        # them before solving, so no mass routes through them across a barrier.
        counts[~is_track_interior] = 0.0
        counts[:, ~is_track_interior] = 0.0
        movement = _normalize_row_probability(counts[interior_ind])

        if self.beta > 0.0:
            local = RandomWalk(
                self.environment_name, movement_var=self.local_movement_var
            ).make_state_transition(environments)
            movement = _normalize_row_probability(
                (1.0 - self.beta) * movement + self.beta * local[interior_ind]
            )

        is_unreachable = movement.sum(axis=1) == 0
        if np.any(is_unreachable):
            unreachable = np.flatnonzero(is_track_interior)[is_unreachable]
            raise ValueError(
                f"{is_unreachable.sum()} track interior bins have no observed "
                f"outgoing transitions (bins {unreachable.tolist()} at "
                f"{self.environment.place_bin_centers_[unreachable].tolist()}). "
                "Their probability mass would be silently destroyed by the "
                "forward pass. Align `is_training` with the track interior, "
                "raise `Environment.bin_count_threshold`, or set `beta` "
                "(e.g. 0.01) to blend in a local random walk."
            )

        # Cached for `rebuild_state_transition` and `m_step`. Plain attributes,
        # not fields, so `dataclasses.replace` copies an unfitted instance.
        self.movement_ = movement
        self.is_track_interior_ = is_track_interior

        return self.rebuild_state_transition()

    def rebuild_state_transition(self) -> NDArray[np.float64]:
        """Rebuild the kernel for the current parameters from the cached `P`.

        Returns
        -------
        state_transition_matrix : NDArray[np.float64], shape (n_bins, n_bins)
            As returned by `make_state_transition`.

        """
        if getattr(self, "movement_", None) is None:
            raise ValueError(
                "TCM_SR has no movement matrix yet; call `make_state_transition` "
                "(or fit the classifier) first."
            )
        self._validate()
        omega, normalize_before_mix = self._mixing()
        local_kernel = _tcm_sr_kernel(
            self.movement_, self.gamma, omega, normalize_before_mix, self.zero_diagonal
        )
        n_position_bins = self.is_track_interior_.shape[0]
        state_transition = np.zeros((n_position_bins, n_position_bins))
        state_transition[np.ix_(self.is_track_interior_, self.is_track_interior_)] = (
            local_kernel
        )
        return state_transition

    def m_step(
        self,
        expected_counts: NDArray[np.float64],
        *,
        gamma_bounds: tuple[float, float] = (0.0, 0.99),
        n_horizon_grid: int = 12,
        n_asymmetry_grid: int = 11,
        maxfev: int = 200,
    ) -> dict:
        """Update the parameters in `learn` to maximize the expected log likelihood.

        Maximizes `Q = sum_ij E[i, j] * log(K[i, j])` with a coarse grid
        followed by a bounded Powell search. The best of the current point, the
        grid best and the refinement is kept, preferring the current point on a
        tie, so `Q` never decreases and the EM loop stays monotone. `gamma` is
        searched in log horizon.

        Parameters
        ----------
        expected_counts : NDArray[np.float64], shape (n_interior_bins, n_interior_bins)
            Expected transition counts on the track interior from the E-step,
            summed over every cell of the classifier that uses this instance.
        gamma_bounds : tuple of float, optional
            Search range for `gamma`, by default (0.0, 0.99).
        n_horizon_grid : int, optional
            Number of log-spaced horizons in the coarse grid, by default 12.
        n_asymmetry_grid : int, optional
            Number of evenly spaced `asymmetry` values in the coarse grid, by
            default 11.
        maxfev : int, optional
            Maximum objective evaluations in the Powell refinement, by default
            200.

        Returns
        -------
        info : dict
            The old and new `gamma` and `asymmetry`, `q_old`, `q_new`,
            `n_evaluations`, and `accepted`, one of "current", "grid" or
            "refined".

        """
        self._validate()
        # With `omega` set the mixture is fixed and `asymmetry` plays no part.
        gamma_old = self.gamma
        asymmetry_old = self.asymmetry if self.omega is None else None
        learn_gamma = "gamma" in self.learn
        learn_asymmetry = "asymmetry" in self.learn
        if not (learn_gamma or learn_asymmetry):
            return dict(
                gamma_old=gamma_old,
                gamma=gamma_old,
                asymmetry_old=asymmetry_old,
                asymmetry=asymmetry_old,
                q_old=np.nan,
                q_new=np.nan,
                n_evaluations=0,
                accepted="current",
            )
        if getattr(self, "movement_", None) is None:
            raise ValueError(
                "TCM_SR has no movement matrix yet; call `make_state_transition` "
                "(or fit the classifier) first."
            )
        expected_counts = np.asarray(expected_counts, dtype=np.float64)
        if expected_counts.shape != self.movement_.shape:
            raise ValueError(
                f"expected_counts has shape {expected_counts.shape}; the track "
                f"interior needs {self.movement_.shape}."
            )
        gamma_low, gamma_high = gamma_bounds
        if not 0.0 <= gamma_low < gamma_high < 1.0:
            raise ValueError(
                f"gamma_bounds must satisfy 0 <= low < high < 1; got {gamma_bounds}"
            )

        # Log horizon: log(1 / (1 - gamma)) = -log1p(-gamma).
        log_horizon_low = -np.log1p(-gamma_low)
        log_horizon_span = -np.log1p(-gamma_high) - log_horizon_low

        def to_gamma(u: float) -> float:
            return float(-np.expm1(-(log_horizon_low + u * log_horizon_span)))

        def to_unit(gamma: float) -> float:
            return float(
                np.clip((-np.log1p(-gamma) - log_horizon_low) / log_horizon_span, 0, 1)
            )

        omega_fixed, normalize_before_mix = self._mixing()
        successor_cache: dict[float, tuple] = {}
        n_evaluations = 0

        def expected_log_likelihood(gamma: float, asymmetry: Optional[float]) -> float:
            nonlocal n_evaluations
            n_evaluations += 1
            if gamma not in successor_cache:
                if len(successor_cache) > 64:
                    successor_cache.clear()
                successor_cache[gamma] = _tcm_sr_successor(
                    self.movement_, gamma, normalize_before_mix
                )
            omega = omega_fixed if asymmetry is None else 1.0 - asymmetry
            try:
                kernel = _tcm_sr_mix(
                    *successor_cache[gamma], omega, self.zero_diagonal
                )
            except ValueError:
                return -np.inf
            return _tcm_sr_expected_log_likelihood(expected_counts, kernel)

        q_old = expected_log_likelihood(gamma_old, asymmetry_old)
        tolerance = 1e-12 * max(1.0, abs(q_old))
        best = (q_old, gamma_old, asymmetry_old, "current")

        gammas = [gamma_old]
        if learn_gamma:
            gammas += [to_gamma(u) for u in np.linspace(0.0, 1.0, n_horizon_grid)]
        asymmetries = [asymmetry_old]
        if learn_asymmetry:
            asymmetries += [float(a) for a in np.linspace(0.0, 1.0, n_asymmetry_grid)]
        grid_best = best
        for gamma in gammas:
            for asymmetry in asymmetries:
                q = expected_log_likelihood(gamma, asymmetry)
                if q > grid_best[0]:
                    grid_best = (q, gamma, asymmetry, "grid")

        def unpack(x: NDArray[np.float64]) -> tuple[float, Optional[float]]:
            x = np.clip(x, 0.0, 1.0)
            gamma = to_gamma(x[0]) if learn_gamma else gamma_old
            asymmetry = float(x[-1]) if learn_asymmetry else asymmetry_old
            return gamma, asymmetry

        x0 = []
        if learn_gamma:
            x0.append(to_unit(grid_best[1]))
        if learn_asymmetry:
            x0.append(grid_best[2])
        result = minimize(
            lambda x: -expected_log_likelihood(*unpack(x)),
            np.asarray(x0),
            method="Powell",
            bounds=[(0.0, 1.0)] * len(x0),
            options=dict(maxfev=maxfev, xtol=1e-4, ftol=1e-10),
        )
        refined_gamma, refined_asymmetry = unpack(result.x)
        refined = (
            expected_log_likelihood(refined_gamma, refined_asymmetry),
            refined_gamma,
            refined_asymmetry,
            "refined",
        )

        for candidate in (grid_best, refined):
            if candidate[0] > best[0] + tolerance:
                best = candidate
        q_new, self.gamma, asymmetry_new, accepted = best
        if learn_asymmetry:
            self.asymmetry = asymmetry_new

        horizon = 1.0 / (1.0 - self.gamma)
        if self.movement_time_step is None:
            horizon_text = f"{horizon:.3g} steps"
        else:
            horizon_text = f"{horizon * self.movement_time_step:.3g} s"
        logger.info(
            f"TCM_SR m_step: gamma {gamma_old:.4g} -> {self.gamma:.4g} "
            f"(horizon {horizon_text}), asymmetry {asymmetry_old} -> "
            f"{self.asymmetry}, Q gain {q_new - q_old:.4g}, "
            f"{n_evaluations} evaluations, accepted {accepted}"
        )
        if learn_gamma and to_unit(self.gamma) > 1.0 - 1e-3:
            logger.warning(
                f"TCM_SR gamma ended on its upper bound {gamma_high}; the data "
                "favor a longer horizon than `gamma_bounds` allows."
            )

        return dict(
            gamma_old=gamma_old,
            gamma=self.gamma,
            asymmetry_old=asymmetry_old,
            asymmetry=self.asymmetry,
            q_old=q_old,
            q_new=q_new,
            n_evaluations=n_evaluations,
            accepted=accepted,
        )


#: Transitions whose `make_state_transition` needs the animal's position.
#: `classifier.py` and `decoder.py` dispatch on this; register new ones here.
POSITION_DEPENDENT_TRANSITIONS = (EmpiricalMovement, TCM_SR)


@dataclass
class RandomWalkDirection1:
    """A Gaussian random walk in that can only go one direction

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit
    movement_var : float, optional
        How far the animal is can move in one time bin during normal
        movement.
    """

    environment_name: str = ""
    movement_var: float = 6.0

    def make_state_transition(self, environments: tuple[Environment]):
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : tuple[Environment]
            The existing environments in the model

        Returns
        -------
        state_transition_matrix : NDArray[np.float64], shape (n_position_bins, n_position_bins)

        """
        self.environment = environments[environments.index(self.environment_name)]
        random = RandomWalk(
            self.environment_name, self.movement_var
        ).make_state_transition(environments)

        return _normalize_row_probability(np.triu(random))


@dataclass
class RandomWalkDirection2:
    """A Gaussian random walk in that can only go one direction

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit
    movement_var : float, optional
        How far the animal is can move in one time bin during normal
        movement.
    """

    environment_name: str = ""
    movement_var: float = 6.0

    def make_state_transition(self, environments: tuple[Environment]):
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : tuple[Environment]
            The existing environments in the model

        Returns
        -------
        state_transition_matrix : NDArray[np.float64], shape (n_position_bins, n_position_bins)

        """
        self.environment = environments[environments.index(self.environment_name)]
        random = RandomWalk(
            self.environment_name, self.movement_var
        ).make_state_transition(environments)

        return _normalize_row_probability(np.tril(random))
