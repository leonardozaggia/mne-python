# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# The core logic for this implementation was adapted from the Cedalion project
# (https://github.com/ibs-lab/cedalion), which is originally based on Homer3
# (https://github.com/BUNPC/Homer3).

import numpy as np
from scipy.ndimage import binary_dilation

from ...io import BaseRaw
from ...utils import _validate_type, logger, verbose
from ..nirs import _validate_nirs_info


def _tinc_from_annotations(raw, n_times):
    """Build a global tInc mask from ``BAD`` annotations on *raw*.

    Parameters
    ----------
    raw : instance of Raw
        Raw object whose annotations are inspected.
    n_times : int
        Number of time samples.

    Returns
    -------
    tInc : ndarray of bool, shape (n_times,)
        ``True`` = clean sample, ``False`` = motion artifact.
    """
    tInc = np.ones(n_times, dtype=bool)
    fs = raw.info["sfreq"]
    for ann in raw.annotations:
        if ann["description"].upper().startswith("BAD"):
            onset_sec = ann["onset"] - raw.first_time
            start = max(0, int(np.round(onset_sec * fs)))
            stop = min(n_times, start + int(np.round(ann["duration"] * fs)))
            tInc[start:stop] = False
    return tInc


def _tinc_ch_from_annotations(raw, n_picks, n_times):
    """Build a per-channel tIncCh mask tiled from ``BAD`` annotations.

    Since MNE annotations are global (not per-channel), the same mask is
    applied to every channel.

    Parameters
    ----------
    raw : instance of Raw
        Raw object whose annotations are inspected.
    n_picks : int
        Number of fNIRS channels.
    n_times : int
        Number of time samples.

    Returns
    -------
    tIncCh : ndarray of bool, shape (n_picks, n_times)
        ``True`` = clean sample, ``False`` = motion artifact.
    """
    global_mask = _tinc_from_annotations(raw, n_times)
    return np.tile(global_mask, (n_picks, 1))


def _detect_channel(signal, fs, t_motion, t_mask, stdev_thresh, amp_thresh):
    """Detect motion artifacts in a single fNIRS channel.

    Uses a sliding window to compute the signal standard deviation and
    peak-to-peak amplitude.  Windows that exceed either threshold are flagged,
    then each flagged region is dilated by ``t_mask`` seconds.

    Parameters
    ----------
    signal : ndarray, shape (n_times,)
        Single-channel signal (optical density or hemoglobin).
    fs : float
        Sampling frequency in Hz.
    t_motion : float
        Sliding window duration in seconds.
    t_mask : float
        Dilation (padding) duration in seconds applied around each flagged
        region.
    stdev_thresh : float
        Standard-deviation threshold.
    amp_thresh : float
        Peak-to-peak amplitude threshold.

    Returns
    -------
    mask : ndarray of bool, shape (n_times,)
        ``True`` = clean sample, ``False`` = motion artifact.
    """
    n = len(signal)
    n_motion = max(2, int(np.round(t_motion * fs)))
    n_mask = max(1, int(np.round(t_mask * fs)))

    # Sliding window std and peak-to-peak using stride tricks (fast, no copies)
    windows = np.lib.stride_tricks.sliding_window_view(signal, n_motion)
    std_arr = windows.std(axis=-1)
    ptp_arr = windows.max(axis=-1) - windows.min(axis=-1)

    # Boolean array of flagged window *start* positions
    bad_starts = (std_arr > stdev_thresh) | (ptp_arr > amp_thresh)

    # Expand: mark the full window extent as bad
    bad_samples = np.zeros(n, dtype=bool)
    bad_samples[: len(bad_starts)] = bad_starts
    if bad_samples.any():
        # Grow each bad start to cover the full n_motion window
        bad_samples = binary_dilation(
            bad_samples, structure=np.ones(n_motion, dtype=bool)
        )
        # Pad by t_mask on each side
        bad_samples = binary_dilation(bad_samples, iterations=n_mask)

    return ~bad_samples  # True = clean


@verbose
def detect_motion_artifacts(
    raw,
    t_motion=0.5,
    t_mask=1.0,
    stdev_thresh=50.0,
    amp_thresh=5.0,
    *,
    verbose=None,
):
    """Detect motion artifacts across all fNIRS channels (global mask).

    Implements Homer3's ``hmrR_MotionArtifact`` algorithm
    :footcite:`HuppertEtAl2009`.

    For each channel a sliding window of ``t_motion`` seconds is moved along
    the signal.  If the window's standard deviation exceeds ``stdev_thresh``
    **or** its peak-to-peak amplitude exceeds ``amp_thresh``, the window is
    flagged as a motion artifact.  The flagged region is then dilated by
    ``t_mask`` seconds on each side.  The returned global mask is the logical
    AND across all channels: a sample is clean only if **every** channel is
    clean at that time point.

    Parameters
    ----------
    raw : instance of Raw
        The raw fNIRS data (optical density or hemoglobin).
    t_motion : float
        Duration (s) of the sliding window.  Default is ``0.5``.
    t_mask : float
        Duration (s) to pad around each detected artifact.  Default is
        ``1.0``.
    stdev_thresh : float
        Standard-deviation threshold.  Default is ``50.0``.
    amp_thresh : float
        Peak-to-peak amplitude threshold.  Default is ``5.0``.
    %(verbose)s

    Returns
    -------
    tInc : ndarray of bool, shape (n_times,)
        Global motion-artifact mask.  ``True`` = clean sample,
        ``False`` = motion artifact.

    See Also
    --------
    detect_motion_artifacts_by_channel : Per-channel version.
    motion_correct_pca : PCA-based correction that accepts this mask.
    motion_correct_spline : Spline-based correction that accepts this mask.

    Notes
    -----
    There is a shorter alias ``mne.preprocessing.nirs.motion_artifact``
    that can be used instead of this function.

    References
    ----------
    .. footbibliography::
    """
    _validate_type(raw, BaseRaw, "raw")
    raw = raw.copy().load_data()
    picks = _validate_nirs_info(raw.info)

    if not len(picks):
        raise RuntimeError(
            "Motion artifact detection should be run on optical density "
            "or hemoglobin data."
        )

    n_times = raw._data.shape[1]
    fs = raw.info["sfreq"]

    tInc = np.ones(n_times, dtype=bool)
    for pick in picks:
        ch_mask = _detect_channel(
            raw._data[pick], fs, t_motion, t_mask, stdev_thresh, amp_thresh
        )
        tInc &= ch_mask  # global: bad in any channel → bad globally

    n_bad = int((~tInc).sum())
    logger.info(
        "Detected %d bad samples (%.1f s) out of %d (%.1f%%).",
        n_bad,
        n_bad / fs,
        n_times,
        n_bad / n_times * 100,
    )
    return tInc


@verbose
def detect_motion_artifacts_by_channel(
    raw,
    t_motion=0.5,
    t_mask=1.0,
    stdev_thresh=50.0,
    amp_thresh=5.0,
    *,
    verbose=None,
):
    """Detect motion artifacts per fNIRS channel (channel-wise mask).

    Implements Homer3's ``hmrR_MotionArtifactByChannel`` algorithm
    :footcite:`HuppertEtAl2009`.

    Same detection procedure as :func:`detect_motion_artifacts` but returns
    a separate boolean mask for each channel instead of a single global mask.

    Parameters
    ----------
    raw : instance of Raw
        The raw fNIRS data (optical density or hemoglobin).
    t_motion : float
        Duration (s) of the sliding window.  Default is ``0.5``.
    t_mask : float
        Duration (s) to pad around each detected artifact.  Default is
        ``1.0``.
    stdev_thresh : float
        Standard-deviation threshold.  Default is ``50.0``.
    amp_thresh : float
        Peak-to-peak amplitude threshold.  Default is ``5.0``.
    %(verbose)s

    Returns
    -------
    tIncCh : ndarray of bool, shape (n_picks, n_times)
        Per-channel motion-artifact mask.  ``True`` = clean sample,
        ``False`` = motion artifact.  ``n_picks`` is the number of fNIRS
        channels returned by ``_validate_nirs_info``.

    See Also
    --------
    detect_motion_artifacts : Global (single-mask) version.
    motion_correct_spline : Spline-based correction that accepts this mask.

    Notes
    -----
    There is a shorter alias
    ``mne.preprocessing.nirs.motion_artifact_by_channel``
    that can be used instead of this function.

    References
    ----------
    .. footbibliography::
    """
    _validate_type(raw, BaseRaw, "raw")
    raw = raw.copy().load_data()
    picks = _validate_nirs_info(raw.info)

    if not len(picks):
        raise RuntimeError(
            "Motion artifact detection should be run on optical density "
            "or hemoglobin data."
        )

    n_times = raw._data.shape[1]
    fs = raw.info["sfreq"]

    tIncCh = np.ones((len(picks), n_times), dtype=bool)
    for ch_idx, pick in enumerate(picks):
        tIncCh[ch_idx] = _detect_channel(
            raw._data[pick], fs, t_motion, t_mask, stdev_thresh, amp_thresh
        )
    return tIncCh


# short aliases
motion_artifact = detect_motion_artifacts
motion_artifact_by_channel = detect_motion_artifacts_by_channel
