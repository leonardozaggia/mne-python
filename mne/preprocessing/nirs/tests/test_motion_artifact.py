# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mne.datasets import testing
from mne.datasets.testing import data_path
from mne.io import read_raw_nirx
from mne.preprocessing.nirs import (
    detect_motion_artifacts,
    detect_motion_artifacts_by_channel,
    motion_artifact,
    motion_artifact_by_channel,
    optical_density,
)
from mne.preprocessing.nirs._motion_artifact import (
    _tinc_ch_from_annotations,
    _tinc_from_annotations,
)

fname_nirx_15_2 = (
    data_path(download=False) / "NIRx" / "nirscout" / "nirx_15_2_recording"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inject_spike(raw_od, picks, start, stop, amplitude=100.0):
    """Inject a large spike into all picks between start and stop samples."""
    raw_spike = raw_od.copy()
    raw_spike._data[picks, start:stop] += amplitude
    return raw_spike


# ---------------------------------------------------------------------------
# Tests: detect_motion_artifacts (global mask)
# ---------------------------------------------------------------------------


@testing.requires_testing_data
@pytest.mark.parametrize("fname", [fname_nirx_15_2])
def test_detect_motion_artifacts_returns_tinc(fname):
    """Output is a boolean array of length n_times."""
    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    n_times = raw_od._data.shape[1]

    tInc = detect_motion_artifacts(raw_od)

    assert isinstance(tInc, np.ndarray)
    assert tInc.dtype == bool
    assert tInc.shape == (n_times,)


@testing.requires_testing_data
@pytest.mark.parametrize("fname", [fname_nirx_15_2])
def test_detect_motion_artifacts_clean_data_all_true(fname):
    """Clean data (low amplitude) should produce an all-True mask."""
    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)

    # Use very high thresholds so nothing gets flagged
    tInc = detect_motion_artifacts(raw_od, stdev_thresh=1e9, amp_thresh=1e9)

    assert tInc.all(), "Expected all samples clean with impossibly high thresholds"


@testing.requires_testing_data
@pytest.mark.parametrize("fname", [fname_nirx_15_2])
def test_detect_motion_artifacts_spike_is_flagged(fname):
    """A large injected spike should be flagged as bad (False)."""
    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    picks = list(range(raw_od._data.shape[0]))
    n_times = raw_od._data.shape[1]

    spike_start = n_times // 3
    spike_stop = spike_start + 5
    raw_spike = _inject_spike(raw_od, picks, spike_start, spike_stop, amplitude=200.0)

    tInc = detect_motion_artifacts(raw_spike, stdev_thresh=10.0, amp_thresh=10.0)

    # At least the spike region should be flagged
    assert not tInc[spike_start:spike_stop].all(), (
        "Expected spike region to be flagged as bad"
    )


@testing.requires_testing_data
@pytest.mark.parametrize("fname", [fname_nirx_15_2])
def test_detect_motion_artifacts_dilation(fname):
    """Bad region should extend beyond the spike due to t_mask dilation."""
    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    picks = list(range(raw_od._data.shape[0]))
    n_times = raw_od._data.shape[1]
    fs = raw_od.info["sfreq"]

    # Single-sample spike right in the middle, away from edges
    spike = n_times // 2
    raw_spike = _inject_spike(raw_od, picks, spike, spike + 1, amplitude=500.0)

    t_mask = 0.5  # seconds
    tInc = detect_motion_artifacts(
        raw_spike, t_motion=0.1, t_mask=t_mask, stdev_thresh=1.0, amp_thresh=1.0
    )

    n_mask = int(np.round(t_mask * fs))
    # Region *before* the spike should also be bad due to dilation
    assert not tInc[max(0, spike - n_mask) : spike].all(), (
        "Expected dilation to extend bad region before spike"
    )


def test_detect_motion_artifacts_wrong_type():
    """Passing a non-Raw object should raise TypeError."""
    with pytest.raises(TypeError):
        detect_motion_artifacts(np.zeros((10, 100)))


def test_detect_motion_artifacts_requires_nirs():
    """Passing non-fNIRS Raw should raise an error."""
    import mne

    # Use a minimal raw with eeg channels (no fNIRS chromophores)
    info = mne.create_info(ch_names=["EEG1"], sfreq=10.0, ch_types=["eeg"])
    raw_eeg = mne.io.RawArray(np.zeros((1, 100)), info)
    with pytest.raises((RuntimeError, ValueError)):
        detect_motion_artifacts(raw_eeg)


# ---------------------------------------------------------------------------
# Tests: detect_motion_artifacts_by_channel (per-channel mask)
# ---------------------------------------------------------------------------


@testing.requires_testing_data
@pytest.mark.parametrize("fname", [fname_nirx_15_2])
def test_detect_motion_artifacts_by_channel_shape(fname):
    """Output is a 2-D boolean array (n_picks, n_times)."""
    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    from mne.preprocessing.nirs import _validate_nirs_info

    picks = _validate_nirs_info(raw_od.info)
    n_times = raw_od._data.shape[1]

    tIncCh = detect_motion_artifacts_by_channel(raw_od)

    assert isinstance(tIncCh, np.ndarray)
    assert tIncCh.dtype == bool
    assert tIncCh.shape == (len(picks), n_times)


@testing.requires_testing_data
@pytest.mark.parametrize("fname", [fname_nirx_15_2])
def test_detect_motion_artifacts_by_channel_selective(fname):
    """Only the spiked channel should have bad samples; others remain clean."""
    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    from mne.preprocessing.nirs import _validate_nirs_info

    picks = _validate_nirs_info(raw_od.info)
    n_times = raw_od._data.shape[1]
    spike_start = n_times // 3
    spike_stop = spike_start + 3

    # Spike only the first pick
    raw_spike = raw_od.copy()
    raw_spike._data[picks[0], spike_start:spike_stop] += 500.0

    tIncCh = detect_motion_artifacts_by_channel(
        raw_spike, stdev_thresh=10.0, amp_thresh=10.0
    )

    # First channel should have some bad samples
    assert not tIncCh[0, spike_start:spike_stop].all(), (
        "Expected spiked channel to have bad samples"
    )
    # Other channels should be entirely clean (no spike injected)
    assert tIncCh[1:].all(), "Expected non-spiked channels to remain clean"


@testing.requires_testing_data
@pytest.mark.parametrize("fname", [fname_nirx_15_2])
def test_detect_motion_artifacts_by_channel_vs_global(fname):
    """Global mask is the AND of all per-channel masks."""
    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)

    tInc_global = detect_motion_artifacts(
        raw_od, stdev_thresh=20.0, amp_thresh=2.0
    )
    tIncCh = detect_motion_artifacts_by_channel(
        raw_od, stdev_thresh=20.0, amp_thresh=2.0
    )

    expected_global = tIncCh.all(axis=0)
    assert_array_equal(tInc_global, expected_global)


# ---------------------------------------------------------------------------
# Tests: aliases
# ---------------------------------------------------------------------------


def test_aliases():
    """Short aliases should point to the same functions."""
    assert motion_artifact is detect_motion_artifacts
    assert motion_artifact_by_channel is detect_motion_artifacts_by_channel


# ---------------------------------------------------------------------------
# Tests: annotation helpers
# ---------------------------------------------------------------------------


@testing.requires_testing_data
@pytest.mark.parametrize("fname", [fname_nirx_15_2])
def test_tinc_from_annotations(fname):
    """BAD annotations should produce False in the returned mask."""
    import mne

    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    n_times = raw_od._data.shape[1]
    fs = raw_od.info["sfreq"]

    # Set a 1-second BAD annotation at t=1.0 s
    onset = 1.0
    duration = 1.0
    raw_od.set_annotations(mne.Annotations([onset], [duration], ["BAD_motion"]))

    tInc = _tinc_from_annotations(raw_od, n_times)

    bad_start = int(np.round((onset - raw_od.first_time) * fs))
    bad_stop = min(n_times, bad_start + int(np.round(duration * fs)))

    assert not tInc[bad_start:bad_stop].any(), (
        "Annotated region should be False (artifact)"
    )
    assert tInc[:bad_start].all(), "Region before annotation should be True (clean)"


@testing.requires_testing_data
@pytest.mark.parametrize("fname", [fname_nirx_15_2])
def test_tinc_ch_from_annotations_shape(fname):
    """_tinc_ch_from_annotations returns correct shape and matches global."""
    import mne

    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    from mne.preprocessing.nirs import _validate_nirs_info

    picks = _validate_nirs_info(raw_od.info)
    n_times = raw_od._data.shape[1]

    raw_od.set_annotations(mne.Annotations([1.0], [1.0], ["BAD_motion"]))

    tIncCh = _tinc_ch_from_annotations(raw_od, len(picks), n_times)
    tInc = _tinc_from_annotations(raw_od, n_times)

    assert tIncCh.shape == (len(picks), n_times)
    # Every row should equal the global mask (annotations are channel-agnostic)
    for row in tIncCh:
        assert_array_equal(row, tInc)
