"""Simulation settings shared by the TCM-SR notebooks. Edit and rerun them."""

N_CELLS = 100               # place cells, half inbound and half outbound
RUNNING_SPEED_CM_S = 80.0   # mean running speed of the simulated laps
RUNNING_SPEED_CV = 0.25     # lap-to-lap coefficient of variation of running speed
N_LAPS = 50                 # out-and-back laps of run data the classifier is fit on
REPLAY_SPEEDUP = 10         # replay speed relative to running speed
DECODE_BIN_S = 0.003        # decoding time bin (s), a whole number of 1 ms samples
PLACE_BIN_SIZE_CM = 2.0     # spatial bin of the decoder
REPLAY_PEAK_RATE_HZ = 20.0  # in-field peak rate of a place cell during a replay
EVENT_S = 0.2               # duration of a simulated replay (s)
IDEALIZED_SPIKES = False    # True: nearest place cell spikes every bin; False: Poisson

SETTINGS = (
    "N_CELLS", "RUNNING_SPEED_CM_S", "RUNNING_SPEED_CV", "N_LAPS", "REPLAY_SPEEDUP",
    "DECODE_BIN_S", "PLACE_BIN_SIZE_CM", "REPLAY_PEAK_RATE_HZ", "EVENT_S",
    "IDEALIZED_SPIKES",
)


def print_settings():
    for name in SETTINGS:
        print(f"{name} = {globals()[name]}")
