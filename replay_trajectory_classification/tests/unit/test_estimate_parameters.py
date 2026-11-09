"""Fitting TCM_SR's gamma and asymmetry with the classifier's generalized EM."""

import numpy as np
import pytest

from replay_trajectory_classification import (
    DiagonalDiscrete,
    Environment,
    Identity,
    SortedSpikesClassifier,
    SortedSpikesDecoder,
    TCM_SR,
    Uniform,
)
from replay_trajectory_classification.simulate import simulate_neuron_with_place_field
from replay_trajectory_classification.sorted_spikes_simulation import (
    _spike_nearest_place_field,
)

SAMPLING_FREQUENCY = 100
RUNNING_SPEED = 20.0
TRACK_LENGTH = 100.0
PLACE_FIELD_MEANS = np.arange(0.0, TRACK_LENGTH + 5.0, 5.0)
TRUE_GAMMA, TRUE_ASYMMETRY = 0.95, 0.8


def simulate_one_way_running(n_laps=20, seed=0):
    """Laps that run 0 -> 100 cm, pause at each end, and jump back to the start.

    One-way running makes `P` directional, so `asymmetry` is identifiable. The
    first sample of each lap is excluded from training, so no movement pair
    spans the jump back to the start.
    """
    run = np.arange(0.0, TRACK_LENGTH, RUNNING_SPEED / SAMPLING_FREQUENCY)
    lap = np.concatenate([np.zeros(50), run, np.full(50, TRACK_LENGTH)])
    position = np.tile(lap, n_laps)
    is_training = np.ones(position.shape, dtype=bool)
    is_training[np.arange(n_laps) * lap.size] = False

    np.random.seed(seed)
    spikes = np.stack(
        [
            simulate_neuron_with_place_field(
                mean,
                position[:, np.newaxis],
                max_rate=15,
                variance=36.0,
                sampling_frequency=SAMPLING_FREQUENCY,
            )
            for mean in PLACE_FIELD_MEANS
        ],
        axis=1,
    )
    return position, is_training, spikes


def make_classifier(tcm_sr):
    """Stationary, TCM-SR and fragmented dynamics with a fixed discrete matrix."""
    return SortedSpikesClassifier(
        environments=Environment(place_bin_size=3.0),
        continuous_transition_types=[
            [Identity(), tcm_sr, Uniform()],
            [Identity(), tcm_sr, Uniform()],
            [Uniform(), Uniform(), Uniform()],
        ],
        discrete_transition_type=DiagonalDiscrete(0.98),
        sorted_spikes_algorithm="spiking_likelihood_kde",
        sorted_spikes_algorithm_params={"position_std": 3.0},
    )


def simulate_replay(environment, kernel, n_events=25, seed=1):
    """Alternate hover, TCM-SR and fragmented segments on the classifier's grid.

    Each TCM-SR segment is a Markov chain sampled from `kernel` and starts in
    the first 60 cm, so it has room to run forward before the end of the
    track. Every bin, the neuron with the nearest place field spikes.
    """
    rng = np.random.default_rng(seed)
    centers = environment.place_bin_centers_.squeeze()
    interior = np.flatnonzero(environment.is_track_interior_.ravel(order="F"))
    path = []
    for _ in range(n_events):
        path += [rng.choice(interior)] * 30
        segment = [rng.choice(interior[centers[interior] < 60.0])]
        for _ in range(59):
            segment.append(rng.choice(centers.size, p=kernel[segment[-1]]))
        path += segment
        path += list(rng.choice(interior, 15))
    return _spike_nearest_place_field(centers[np.asarray(path)], PLACE_FIELD_MEANS, 1)


def tcm_sr(**kwargs):
    return TCM_SR(
        movement_time_step=0.1, sampling_frequency=SAMPLING_FREQUENCY, **kwargs
    )


@pytest.fixture(scope="module")
def data():
    position, is_training, spikes = simulate_one_way_running()
    probe = make_classifier(tcm_sr(gamma=TRUE_GAMMA, asymmetry=TRUE_ASYMMETRY))
    probe.fit(position, spikes, is_training=is_training)
    kernel = probe.continuous_state_transition_[1, 1]
    replay_spikes = simulate_replay(probe.environments[0], kernel)
    fit_args = dict(position=position, spikes=spikes, is_training=is_training)
    return fit_args, dict(spikes=replay_spikes)


@pytest.fixture(scope="module")
def fitted(data):
    fit_args, predict_args = data
    transition = tcm_sr(gamma=0.5, asymmetry=0.3, learn=("gamma", "asymmetry"))
    classifier = make_classifier(transition)
    results, data_log_likelihoods = classifier.estimate_parameters(
        fit_args,
        dict(predict_args),
        estimate_continuous_transition=True,
        estimate_discrete_transition=False,
        max_iter=20,
    )
    return classifier, transition, results, data_log_likelihoods


def test_estimate_parameters_recovers_tcm_sr_parameters(fitted):
    classifier, transition, _, data_log_likelihoods = fitted

    assert np.all(np.diff(data_log_likelihoods) >= -1e-6 * abs(data_log_likelihoods[0]))
    assert np.log10(1 / (1 - transition.gamma)) == pytest.approx(
        np.log10(1 / (1 - TRUE_GAMMA)), abs=0.3
    )
    assert transition.asymmetry == pytest.approx(TRUE_ASYMMETRY, abs=0.15)
    np.testing.assert_array_equal(
        classifier.discrete_state_transition_,
        DiagonalDiscrete(0.98).make_state_transition(3),
    )
    history = classifier.continuous_transition_parameter_history_
    assert len(history) == len(data_log_likelihoods) - 1
    assert all(list(step) == [((0, 1), (1, 1))] for step in history)


def test_em_estimate_agrees_with_the_forward_pass_grid(fitted):
    """The marginal likelihood cross-check: no grid point beats the EM optimum."""
    classifier, transition, _, _ = fitted
    em_score = classifier.score_continuous_transition(transition)
    grid = {
        (horizon, asymmetry): classifier.score_continuous_transition(
            transition, gamma=1 - 1 / horizon, asymmetry=asymmetry
        )
        for horizon in (5, 10, 20, 40, 80)
        for asymmetry in (0.5, 0.8, 1.0)
    }
    best_horizon, best_asymmetry = max(grid, key=grid.get)

    assert em_score >= max(grid.values()) - 1.0
    em_horizon = 1 / (1 - transition.gamma)
    nearest_horizon = min(
        (5, 10, 20, 40, 80), key=lambda h: abs(np.log(h) - np.log(em_horizon))
    )
    nearest_asymmetry = min(
        (0.5, 0.8, 1.0), key=lambda a: abs(a - transition.asymmetry)
    )
    assert (best_horizon, best_asymmetry) == (nearest_horizon, nearest_asymmetry)


def test_shared_instance_cells_stay_identical(fitted):
    classifier, transition, _, _ = fitted
    rebuilt = transition.rebuild_state_transition()
    continuous = classifier.continuous_state_transition_
    np.testing.assert_array_equal(continuous[0, 1], rebuilt)
    np.testing.assert_array_equal(continuous[1, 1], rebuilt)


def test_score_continuous_transition_restores_the_model(fitted):
    classifier, transition, results, _ = fitted
    gamma, asymmetry = transition.gamma, transition.asymmetry
    continuous = classifier.continuous_state_transition_.copy()

    assert classifier.score_continuous_transition(transition) == pytest.approx(
        results.data_log_likelihood
    )
    classifier.score_continuous_transition(transition, gamma=0.2, asymmetry=0.1)

    assert (transition.gamma, transition.asymmetry) == (gamma, asymmetry)
    np.testing.assert_array_equal(classifier.continuous_state_transition_, continuous)


def test_score_continuous_transition_rejects_unknown_inputs(fitted):
    classifier, transition, _, _ = fitted
    with pytest.raises(ValueError, match="not in continuous_transition_types"):
        classifier.score_continuous_transition(tcm_sr(), gamma=0.5)
    with pytest.raises(ValueError, match="no fields"):
        classifier.score_continuous_transition(transition, horizon=5)


def test_estimate_parameters_defaults_leave_the_continuous_transition_unchanged(data):
    fit_args, predict_args = data
    classifier = make_classifier(tcm_sr(gamma=0.5, asymmetry=0.3))
    classifier.fit(**fit_args)
    before = classifier.continuous_state_transition_.copy()

    classifier.estimate_parameters(fit_args, dict(predict_args), max_iter=2)

    np.testing.assert_array_equal(classifier.continuous_state_transition_, before)
    assert classifier.continuous_transition_parameter_history_ == []


def test_estimate_continuous_transition_without_learn_raises(data):
    fit_args, predict_args = data
    classifier = make_classifier(tcm_sr(gamma=0.5, asymmetry=0.3))
    with pytest.raises(ValueError, match="learn"):
        classifier.estimate_parameters(
            fit_args, dict(predict_args), estimate_continuous_transition=True
        )


def test_estimate_continuous_transition_needs_the_acausal_posterior(data):
    fit_args, predict_args = data
    classifier = make_classifier(
        tcm_sr(gamma=0.5, asymmetry=0.3, learn=("gamma", "asymmetry"))
    )
    with pytest.raises(ValueError, match="acausal"):
        classifier.estimate_parameters(
            fit_args,
            dict(predict_args, is_compute_acausal=False),
            estimate_continuous_transition=True,
        )


def running_only(fit_args):
    """The training mask without the pauses at either end of the track."""
    position = fit_args["position"]
    return fit_args["is_training"] & (position > 0.0) & (position < TRACK_LENGTH)


def test_continuous_transition_is_training_separates_the_two_masks(data):
    fit_args, _ = data
    is_running = running_only(fit_args)

    classifier = make_classifier(tcm_sr(gamma=0.5, asymmetry=0.3))
    classifier.fit(
        **dict(fit_args, is_training=is_running),
        continuous_transition_is_training=fit_args["is_training"],
    )
    movement_reference = make_classifier(tcm_sr(gamma=0.5, asymmetry=0.3))
    movement_reference.fit(**fit_args)
    # The default RandomWalk does not learn from behavior, so this fits the
    # place fields on running alone without needing a movement mask.
    place_field_reference = SortedSpikesClassifier(
        environments=Environment(place_bin_size=3.0),
        sorted_spikes_algorithm="spiking_likelihood_kde",
        sorted_spikes_algorithm_params={"position_std": 3.0},
    )
    place_field_reference.fit(**dict(fit_args, is_training=is_running))

    np.testing.assert_array_equal(
        classifier.continuous_state_transition_,
        movement_reference.continuous_state_transition_,
    )
    # The running-only mask differs from the default, so the place fields do too.
    place_fields = classifier.place_fields_["", 0].values
    assert not np.allclose(place_fields, movement_reference.place_fields_["", 0].values)
    np.testing.assert_array_equal(
        place_fields, place_field_reference.place_fields_["", 0].values
    )


def test_continuous_transition_is_training_defaults_to_is_training(data):
    fit_args, _ = data
    default = make_classifier(tcm_sr(gamma=0.5, asymmetry=0.3)).fit(**fit_args)
    explicit = make_classifier(tcm_sr(gamma=0.5, asymmetry=0.3)).fit(
        **fit_args, continuous_transition_is_training=fit_args["is_training"]
    )
    np.testing.assert_array_equal(
        default.continuous_state_transition_, explicit.continuous_state_transition_
    )


def test_continuous_transition_is_training_reaches_estimate_parameters(data):
    fit_args, predict_args = data
    transition = tcm_sr(gamma=0.5, asymmetry=0.3, learn=("gamma", "asymmetry"))
    classifier = make_classifier(transition)
    classifier.estimate_parameters(
        dict(
            fit_args,
            is_training=running_only(fit_args),
            continuous_transition_is_training=fit_args["is_training"],
        ),
        dict(predict_args),
        estimate_continuous_transition=True,
        estimate_discrete_transition=False,
        max_iter=1,
    )
    reference = tcm_sr(gamma=0.5, asymmetry=0.3)
    reference.make_state_transition(
        classifier.environments,
        fit_args["position"][:, np.newaxis],
        is_training=fit_args["is_training"],
    )
    np.testing.assert_array_equal(transition.movement_, reference.movement_)


def test_decoder_continuous_transition_is_training(data):
    fit_args, _ = data
    position, spikes = fit_args["position"], fit_args["spikes"]

    def fit_decoder(**kwargs):
        return SortedSpikesDecoder(
            environment=Environment(place_bin_size=3.0),
            transition_type=tcm_sr(gamma=0.5, asymmetry=0.3),
            sorted_spikes_algorithm="spiking_likelihood_kde",
            sorted_spikes_algorithm_params={"position_std": 3.0},
        ).fit(position, spikes, **kwargs)

    split = fit_decoder(
        is_training=running_only(fit_args),
        continuous_transition_is_training=fit_args["is_training"],
    )
    reference = fit_decoder(is_training=fit_args["is_training"])
    np.testing.assert_array_equal(split.state_transition_, reference.state_transition_)
