"""
Dynamic Direction of Arrival Tracking GUI
MUSIC DoA + IMM Kalman Filter + MVDR/LCMV Beamforming
Adaptive jammer version v5 - Tx1 internal jammer + automatic MVDR/LCMV switching
1 ADALM-Pluto / AD9361
2 RX channels + Tx0 QPSK SOI + Tx1 CW jammer

Required packages:
    pip install numpy pyqt5 pyqtgraph pyadi-iio

Run:
    python pluto_music_imm_lcmv_mvdr_auto_jammer_v5.py

System chain:
    1) Receive 2-channel IQ from PlutoSDR
    2) Estimate covariance matrix
    3) Estimate raw DoA with MUSIC
    4) If Tx1 jammer is OFF, IMM Kalman tracks the dynamic SOI DoA
       and MVDR is used for beamforming.
    5) If Tx1 jammer is ON, SOI is assumed known at broadside
       and IMM Kalman tracks the dominant MUSIC peak as the jammer DoA.
    6) In jammer-present mode, LCMV keeps unity gain at SOI broadside
       and places a null at the IMM-tracked jammer DoA.
    7) When Tx1 jammer is cut, the worker returns to MVDR + SOI tracking.

Notes:
    - RSSI In and RSSI Out are both calculated from IQ power with the same
      calibrated offset:
          RSSI[dBm] ≈ 10log10(mean(|IQ|^2)) + DBM_OFFSET
    - Hardware RSSI register is intentionally not used because it can be
      misleading as an absolute dBm value in this setup.
"""

import sys
import time
from dataclasses import dataclass

import numpy as np

from PyQt5.QtCore import QObject, QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg
import adi


# ============================================================
# Configuration
# ============================================================

@dataclass
class SystemConfig:
    sdr_uri: str = "ip:192.168.2.1"

    sample_rate: float = 5e6
    center_freq: float = 2.1e9
    qpsk_if_freq: float = 100e3
    symbol_rate: float = 250e3

    rx_gain: float = 3.0
    tx_gain: float = -20.0
    phase_offset: float = -0.3

    rx_buffer_size: int = 2**14
    tx_buffer_len: int = 2**18

    use_transmitter: bool = True
    use_imm_for_mvdr: bool = True

    # Adaptive mode switch.
    # True: processing mode follows Tx1 jammer state automatically:
    #       Tx1 OFF -> MVDR + IMM tracks SOI
    #       Tx1 ON  -> LCMV + IMM tracks jammer, SOI fixed at broadside
    jammer_mode: bool = True
    soi_broadside_deg: float = 0.0

    # Internal jammer from Tx1. In this project setup Tx0 is QPSK SOI and
    # Tx1 is the CW jammer. This checkbox is the practical jammer ON/OFF switch.
    use_internal_jammer: bool = True
    jammer_if_freq: float = 100e3
    jammer_gain: float = -3.0

    # Approximate IQ-power-to-dBm calibration offset.
    # You can tune this experimentally.
    dbm_offset: float = -110.0

    c: float = 299_792_458.0
    nr: int = 2
    n_sources: int = 1


# ============================================================
# Signal generation and array processing
# ============================================================

def generate_qpsk_if(num_samples, fs, symbol_rate_hz, if_freq_hz, seed=7):
    """
    Generate a cyclic QPSK SOI at a digital IF.

    Output:
        complex64 IQ samples scaled for Pluto Tx.
    """
    rng = np.random.default_rng(seed)

    sps = max(1, int(round(fs / symbol_rate_hz)))
    num_symbols = int(np.ceil(num_samples / sps))

    symbol_map = np.array(
        [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
        dtype=np.complex64,
    ) / np.sqrt(2)

    symbols = symbol_map[rng.integers(0, 4, num_symbols)]
    bb = np.repeat(symbols, sps)[:num_samples]

    t = np.arange(num_samples) / float(fs)
    iq = bb * np.exp(1j * 2 * np.pi * if_freq_hz * t)

    iq = iq / np.max(np.abs(iq)) * (2**14)
    return iq.astype(np.complex64)




def generate_cw_jammer_if(num_samples, fs, if_freq_hz):
    """
    Generate a strong cyclic CW jammer at a digital IF.

    CW is intentionally simple for stage-1 testing because it creates a clear
    dominant spatial source in the covariance matrix.
    """
    t = np.arange(num_samples) / float(fs)
    iq = np.exp(1j * 2 * np.pi * if_freq_hz * t)
    iq = iq / np.max(np.abs(iq)) * (2**14)
    return iq.astype(np.complex64)


def iq_power_db(raw_data):
    """
    Raw IQ power in dB.

    This is NOT dBFS. It directly uses the power of the IQ samples:
        P[dB] = 10log10(mean(|IQ|^2))
    """
    raw_data = np.asarray(raw_data).squeeze()

    if raw_data.size <= 1:
        return -120.0

    power = np.mean(np.abs(raw_data) ** 2)
    return float(10 * np.log10(np.maximum(power, 1e-12)))


def iq_power_dbm_est(raw_data, dbm_offset=-110.0):
    """
    Approximate dBm estimate from IQ power.

    Model:
        P[dBm] ≈ 10log10(mean(|IQ|^2)) + DBM_OFFSET

    DBM_OFFSET is an experimental calibration constant.
    For this project GUI, the same method is used for both RSSI In and RSSI Out.
    """
    return iq_power_db(raw_data) + dbm_offset


def steering_vector(theta_rad, nr, d, center_freq, c):
    """
    Uniform linear array steering vector.

    theta_rad:
        Direction of arrival in radians.
    """
    n = np.arange(nr)

    a = np.exp(
        -2j
        * np.pi
        * d
        * center_freq
        / c
        * n
        * np.sin(theta_rad)
    )

    return a.reshape(-1, 1)


def apply_rx_phase_calibration(x, phase_offset):
    """
    Apply fixed inter-channel phase calibration to Rx1.

    x shape:
        [2, N]
    """
    x_cal = x.copy()

    # Rx0 reference, Rx1 calibrated.
    x_cal[1, :] *= np.exp(-1j * phase_offset)

    return x_cal


def covariance_matrix(x):
    """
    Estimate spatial covariance matrix.

    R = X X^H / Ns

    x shape:
        [Nr, Ns]
    """
    x = x - np.mean(x, axis=1, keepdims=True)
    R = (x @ x.conj().T) / x.shape[1]

    return R


def music_spectrum(R, theta_grid_rad, cfg):
    """
    MUSIC pseudo-spectrum.

    For 2 RX and 1 SOI:
        signal subspace dimension = 1
        noise subspace dimension = 1
    """
    if cfg.n_sources >= cfg.nr:
        raise ValueError("MUSIC requires n_sources < number of RX channels.")

    eigvals, eigvecs = np.linalg.eigh(R)

    # Largest eigenvalue corresponds to signal subspace.
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    # Noise subspace.
    en = eigvecs[:, cfg.n_sources:]

    d = cfg.c / (2 * cfg.center_freq)

    p_music = []

    for theta in theta_grid_rad:
        a = steering_vector(theta, cfg.nr, d, cfg.center_freq, cfg.c)

        denom = a.conj().T @ en @ en.conj().T @ a
        denom = np.maximum(np.abs(denom.item()), 1e-12)

        p_music.append(1.0 / denom)

    p_music = np.asarray(p_music, dtype=float)

    p_music_db = 10 * np.log10(np.maximum(p_music, 1e-12))
    p_music_db = p_music_db - np.max(p_music_db)

    return p_music_db


def mvdr_spectrum(R, theta_grid_rad, cfg):
    """
    MVDR/Capon spatial spectrum.
    This is only for visualization.

    Beamforming output itself is calculated with mvdr_weights().
    """
    loading = 1e-6 * np.trace(R).real / cfg.nr
    R_loaded = R + loading * np.eye(cfg.nr)

    Rinv = np.linalg.pinv(R_loaded)

    d = cfg.c / (2 * cfg.center_freq)

    p_mvdr = []

    for theta in theta_grid_rad:
        a = steering_vector(theta, cfg.nr, d, cfg.center_freq, cfg.c)

        denom = a.conj().T @ Rinv @ a
        denom = np.maximum(np.real(denom.item()), 1e-12)

        p_mvdr.append(1.0 / denom)

    p_mvdr = np.asarray(p_mvdr, dtype=float)

    p_mvdr_db = 10 * np.log10(np.maximum(p_mvdr, 1e-12))
    p_mvdr_db = p_mvdr_db - np.max(p_mvdr_db)

    return p_mvdr_db


def mvdr_weights(theta_rad, R, cfg):
    """
    Calculate MVDR beamforming weights for the selected SOI direction.
    """
    loading = 1e-6 * np.trace(R).real / cfg.nr
    R_loaded = R + loading * np.eye(cfg.nr)

    Rinv = np.linalg.pinv(R_loaded)

    d = cfg.c / (2 * cfg.center_freq)
    a = steering_vector(theta_rad, cfg.nr, d, cfg.center_freq, cfg.c)

    w = (Rinv @ a) / (a.conj().T @ Rinv @ a)

    return w.squeeze()


def lcmv_weights(soi_theta_rad, jammer_theta_rad, R, cfg):
    """
    Calculate LCMV weights for jammer-present operation.

    Constraints:
        w^H a_soi = 1   -> keep SOI direction
        w^H a_jam = 0   -> place a null at jammer direction

    With 2 RX antennas, these two constraints are the practical limit.
    If the matrix becomes ill-conditioned, the function falls back to MVDR
    steered to the SOI direction instead of crashing.
    """
    loading = 1e-5 * np.trace(R).real / cfg.nr
    R_loaded = R + loading * np.eye(cfg.nr)
    Rinv = np.linalg.pinv(R_loaded)

    d = cfg.c / (2 * cfg.center_freq)
    a_soi = steering_vector(soi_theta_rad, cfg.nr, d, cfg.center_freq, cfg.c)
    a_jam = steering_vector(jammer_theta_rad, cfg.nr, d, cfg.center_freq, cfg.c)

    C = np.hstack([a_soi, a_jam])
    f = np.array([[1.0 + 0j], [0.0 + 0j]])

    G = C.conj().T @ Rinv @ C

    if np.linalg.cond(G) > 1e8:
        return mvdr_weights(soi_theta_rad, R, cfg), False

    w = Rinv @ C @ np.linalg.pinv(G) @ f
    return w.squeeze(), True


# ============================================================
# MUSIC quality metrics for adaptive Kalman measurement noise
# ============================================================

def music_peak_sharpness_db(theta_deg, spectrum_db, peak_idx, guard_deg=6.0):
    """
    Measure MUSIC peak sharpness.

    Since the spectrum is normalized to 0 dB at its maximum:
        sharpness = peak_level - strongest_sidelobe_outside_guard

    Larger sharpness means a more reliable DoA measurement.
    """
    theta_deg = np.asarray(theta_deg)
    spectrum_db = np.asarray(spectrum_db)

    peak_angle = theta_deg[peak_idx]
    peak_level = spectrum_db[peak_idx]

    mask = np.abs(theta_deg - peak_angle) > guard_deg

    if not np.any(mask):
        return 0.0

    side_level = np.max(spectrum_db[mask])
    return float(peak_level - side_level)


def adaptive_measurement_variance(sharpness_db, cond_r, rssi_dbm):
    """
    Adaptive measurement noise for the IMM Kalman filter.

    Output:
        R_var   : measurement variance in deg^2
        R_sigma : measurement standard deviation in deg

    Logic:
        - Sharp MUSIC peak  -> trust the measurement more.
        - Wide/flat peak    -> trust the prediction more.
        - Ill-conditioned R -> increase measurement uncertainty.
        - Very weak signal  -> increase measurement uncertainty.
    """
    sigma = 1.2  # base measurement sigma in degrees

    if sharpness_db < 2.0:
        sigma *= 5.0
    elif sharpness_db < 4.0:
        sigma *= 3.0
    elif sharpness_db < 8.0:
        sigma *= 1.7

    if cond_r > 1e5:
        sigma *= 3.0
    elif cond_r > 1e4:
        sigma *= 2.0
    elif cond_r > 1e3:
        sigma *= 1.3

    if rssi_dbm < -90.0:
        sigma *= 1.5

    sigma = float(np.clip(sigma, 0.7, 25.0))
    return sigma**2, sigma


# ============================================================
# 3-model IMM Kalman tracker: CP/static + CV + CA
# ============================================================

class IMMDoATracker:
    """
    Interacting Multiple Model Kalman tracker for DoA.

    State:
        x = [theta, theta_dot, theta_ddot]^T

    Units:
        theta      : degree
        theta_dot  : degree / second
        theta_ddot : degree / second^2

    Models:
        CP/static:
            best when source is stationary.
        CV:
            best when source moves with approximately constant angular velocity.
        CA:
            best when source accelerates or maneuvers.
    """

    def __init__(self):
        self.model_names = ["CP", "CV", "CA"]
        self.n_models = 3
        self.dim_x = 3

        self.H = np.array([[1.0, 0.0, 0.0]])

        # Row: current model, column: next model
        self.PI = np.array(
            [
                [0.94, 0.05, 0.01],
                [0.04, 0.92, 0.04],
                [0.02, 0.08, 0.90],
            ],
            dtype=float,
        )

        self.mu = np.array([0.60, 0.30, 0.10], dtype=float)

        self.x_models = None
        self.P_models = None

        self.x_combined = np.zeros(self.dim_x)
        self.P_combined = np.eye(self.dim_x)

        self.initialized = False

    def initialize(self, theta_deg):
        x0 = np.array([theta_deg, 0.0, 0.0], dtype=float)
        P0 = np.diag([8.0**2, 40.0**2, 200.0**2])

        self.x_models = np.vstack([x0.copy() for _ in range(self.n_models)])
        self.P_models = np.stack([P0.copy() for _ in range(self.n_models)])

        self.x_combined = x0.copy()
        self.P_combined = P0.copy()

        self.mu = np.array([0.60, 0.30, 0.10], dtype=float)
        self.initialized = True

    def model_matrices(self, model_name, dt):
        dt = float(np.clip(dt, 0.005, 0.500))

        if model_name == "CP":
            # Static model:
            # angle stays almost constant, velocity and acceleration decay.
            F = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.15, 0.0],
                    [0.0, 0.0, 0.10],
                ],
                dtype=float,
            )

            Q = np.diag(
                [
                    (0.06**2) * dt,
                    (0.40**2) * dt,
                    (1.00**2) * dt,
                ]
            )

        elif model_name == "CV":
            # Constant angular velocity model.
            # Acceleration is kept as a weakly decaying state for IMM compatibility.
            F = np.array(
                [
                    [1.0, dt, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.25],
                ],
                dtype=float,
            )

            Q = np.diag(
                [
                    (0.10**2) * dt,
                    (1.50**2) * dt,
                    (4.00**2) * dt,
                ]
            )

        elif model_name == "CA":
            # Constant angular acceleration model.
            F = np.array(
                [
                    [1.0, dt, 0.5 * dt * dt],
                    [0.0, 1.0, dt],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )

            Q = np.diag(
                [
                    (0.18**2) * dt,
                    (3.00**2) * dt,
                    (12.00**2) * dt,
                ]
            )

        else:
            raise ValueError(f"Unknown IMM model: {model_name}")

        return F, Q

    def _clip_state(self, x):
        x = x.copy()
        x[0] = np.clip(x[0], -90.0, 90.0)
        x[1] = np.clip(x[1], -220.0, 220.0)
        x[2] = np.clip(x[2], -1200.0, 1200.0)
        return x

    def kf_predict_update(self, x, P, z, R_var, model_name, dt):
        F, Q = self.model_matrices(model_name, dt)

        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        x_pred = self._clip_state(x_pred)

        z_pred = float((self.H @ x_pred).item())
        innovation = float(z - z_pred)

        S = float((self.H @ P_pred @ self.H.T).item() + R_var)
        S = max(S, 1e-9)

        K = (P_pred @ self.H.T) / S

        x_upd = x_pred + (K[:, 0] * innovation)

        I = np.eye(self.dim_x)
        KH = K @ self.H
        P_upd = (I - KH) @ P_pred @ (I - KH).T + K * R_var * K.T

        x_upd = self._clip_state(x_upd)
        P_upd = 0.5 * (P_upd + P_upd.T)

        likelihood = np.exp(-0.5 * innovation * innovation / S) / np.sqrt(2.0 * np.pi * S)
        likelihood = float(max(likelihood, 1e-300))

        return x_upd, P_upd, likelihood, innovation, S

    def step(self, z_deg, dt, R_var):
        """
        Run one IMM update.

        Returns:
            dictionary with filtered angle, velocity, acceleration, model probabilities.
        """
        z_deg = float(np.clip(z_deg, -90.0, 90.0))

        if not self.initialized:
            self.initialize(z_deg)
            return {
                "theta_deg": float(self.x_combined[0]),
                "theta_dot": float(self.x_combined[1]),
                "theta_ddot": float(self.x_combined[2]),
                "mu": self.mu.copy(),
                "model_names": list(self.model_names),
                "innovation": 0.0,
            }

        # -----------------------------
        # IMM mixing
        # -----------------------------
        c_j = self.PI.T @ self.mu
        c_j = np.maximum(c_j, 1e-12)

        mixed_x = np.zeros_like(self.x_models)
        mixed_P = np.zeros_like(self.P_models)

        for j in range(self.n_models):
            mixing_probs = self.PI[:, j] * self.mu / c_j[j]

            x0_j = np.sum(self.x_models * mixing_probs[:, None], axis=0)

            P0_j = np.zeros((self.dim_x, self.dim_x))
            for i in range(self.n_models):
                dx = (self.x_models[i] - x0_j).reshape(-1, 1)
                P0_j += mixing_probs[i] * (self.P_models[i] + dx @ dx.T)

            mixed_x[j] = x0_j
            mixed_P[j] = P0_j

        # -----------------------------
        # Model-conditioned KF updates
        # -----------------------------
        new_x = np.zeros_like(self.x_models)
        new_P = np.zeros_like(self.P_models)
        likelihoods = np.zeros(self.n_models)
        innovations = np.zeros(self.n_models)

        for j, name in enumerate(self.model_names):
            x_j, P_j, ll_j, innov_j, _ = self.kf_predict_update(
                mixed_x[j],
                mixed_P[j],
                z_deg,
                R_var,
                name,
                dt,
            )

            new_x[j] = x_j
            new_P[j] = P_j
            likelihoods[j] = ll_j
            innovations[j] = innov_j

        # -----------------------------
        # Mode probability update
        # -----------------------------
        mu_new = likelihoods * c_j
        mu_sum = np.sum(mu_new)

        if mu_sum <= 1e-300 or not np.isfinite(mu_sum):
            mu_new = np.ones(self.n_models) / self.n_models
        else:
            mu_new = mu_new / mu_sum

        self.mu = mu_new
        self.x_models = new_x
        self.P_models = new_P

        # -----------------------------
        # Combination
        # -----------------------------
        x_comb = np.sum(self.x_models * self.mu[:, None], axis=0)

        P_comb = np.zeros((self.dim_x, self.dim_x))
        for j in range(self.n_models):
            dx = (self.x_models[j] - x_comb).reshape(-1, 1)
            P_comb += self.mu[j] * (self.P_models[j] + dx @ dx.T)

        self.x_combined = self._clip_state(x_comb)
        self.P_combined = 0.5 * (P_comb + P_comb.T)

        avg_innovation = float(np.sum(self.mu * innovations))

        return {
            "theta_deg": float(self.x_combined[0]),
            "theta_dot": float(self.x_combined[1]),
            "theta_ddot": float(self.x_combined[2]),
            "mu": self.mu.copy(),
            "model_names": list(self.model_names),
            "innovation": avg_innovation,
        }


# ============================================================
# Worker thread
# ============================================================

class DoAWorker(QObject):
    frame_ready = pyqtSignal(object)
    log_message = pyqtSignal(str)
    error_message = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self._running = False
        self.sdr = None

        self.tracker = IMMDoATracker()
        self.tracker_target = None

        self.internal_jammer_started = False

        self.theta_scan_deg = np.linspace(-90, 90, 361)
        self.theta_scan_rad = np.deg2rad(self.theta_scan_deg)

    def stop(self):
        self._running = False

    def _safe_destroy_tx_buffer(self):
        try:
            self.sdr.tx_destroy_buffer()
        except Exception:
            pass

    def start_tx_waveforms(self, include_jammer=False):
        """
        Configure cyclic TX waveforms.

        TX modes:
            - Tx1 jammer OFF: Tx0 QPSK SOI only.
            - Tx1 jammer ON : Tx0 QPSK SOI + Tx1 CW jammer.

        Project setup:
            Tx0 = QPSK SOI
            Tx1 = stronger CW jammer

        If your Pluto/AD9361 image does not expose Tx1, disable Tx1 jammer
        and use an external jammer source instead.
        """
        cfg = self.cfg

        self.sdr.tx_lo = int(cfg.center_freq)
        self.sdr.tx_cyclic_buffer = True
        self._safe_destroy_tx_buffer()

        waveforms = []
        enabled_channels = []

        if cfg.use_transmitter:
            enabled_channels.append(0)
            waveforms.append(
                generate_qpsk_if(
                    num_samples=cfg.tx_buffer_len,
                    fs=cfg.sample_rate,
                    symbol_rate_hz=cfg.symbol_rate,
                    if_freq_hz=cfg.qpsk_if_freq,
                    seed=7,
                )
            )

        if include_jammer and cfg.use_internal_jammer:
            enabled_channels.append(1)
            waveforms.append(
                generate_cw_jammer_if(
                    num_samples=cfg.tx_buffer_len,
                    fs=cfg.sample_rate,
                    if_freq_hz=cfg.jammer_if_freq,
                )
            )

        if not enabled_channels:
            self.internal_jammer_started = False
            self.log_message.emit("Dahili TX kapalı. Harici SOI/jammer kaynağı bekleniyor.")
            return

        try:
            # Önemli sıra:
            # pyadi-iio bazı Pluto/AD9361 contextlerinde TX gain attribute objesini
            # tx_enabled_channels ayarlanmadan oluşturmuyor. Gain'i önce yazmak
            # 'NoneType object has no attribute attrs' hatasına yol açabiliyor.
            self.sdr.tx_enabled_channels = enabled_channels

            if 0 in enabled_channels:
                self.sdr.tx_hardwaregain_chan0 = int(cfg.tx_gain)

            if 1 in enabled_channels:
                self.sdr.tx_hardwaregain_chan1 = int(cfg.jammer_gain)

            if len(waveforms) == 1:
                self.sdr.tx(waveforms[0])
            else:
                self.sdr.tx(waveforms)
        except Exception as exc:
            raise RuntimeError(
                f"TX başlatılamadı. enabled_channels={enabled_channels}. "
                "Pluto/firmware Tx1 desteklemiyorsa internal jammer seçeneğini kapatıp harici jammer kullan. "
                f"Alt hata: {exc}"
            ) from exc

        if include_jammer and cfg.use_internal_jammer:
            self.internal_jammer_started = True
            self.log_message.emit(
                f"Tx0 QPSK SOI + Tx1 CW jammer başlatıldı | "
                f"SOI gain={cfg.tx_gain:.0f} dB, jammer gain={cfg.jammer_gain:.0f} dB, "
                f"jammer IF={cfg.jammer_if_freq:.0f} Hz"
            )
        elif cfg.use_transmitter:
            self.internal_jammer_started = False
            self.log_message.emit(
                f"Tx0 üzerinden QPSK SOI yayını başlatıldı | SOI gain={cfg.tx_gain:.0f} dB"
            )

    def configure_sdr(self):
        cfg = self.cfg

        self.sdr = adi.ad9361(uri=cfg.sdr_uri)

        self.log_message.emit("Pluto bağlantısı kuruldu.")

        # -----------------------------
        # RX configuration
        # -----------------------------
        self.sdr.sample_rate = int(cfg.sample_rate)
        cfg.sample_rate = int(self.sdr.sample_rate)

        self.sdr.rx_lo = int(cfg.center_freq)
        self.sdr.rx_enabled_channels = [0, 1]
        self.sdr.rx_buffer_size = int(cfg.rx_buffer_size)

        self.sdr.gain_control_mode_chan0 = "manual"
        self.sdr.gain_control_mode_chan1 = "manual"

        self.sdr.rx_hardwaregain_chan0 = int(cfg.rx_gain)
        self.sdr.rx_hardwaregain_chan1 = int(cfg.rx_gain)

        # -----------------------------
        # TX configuration
        # -----------------------------
        self.internal_jammer_started = False

        start_jammer_now = bool(cfg.use_internal_jammer)
        self.start_tx_waveforms(include_jammer=start_jammer_now)

        if cfg.jammer_mode:
            self.log_message.emit(
                f"Adaptif mod aktif: Tx1 jammer AÇIK ise LCMV + IMM JAMMER takibi; "
                f"Tx1 jammer KAPALI ise MVDR + IMM SOI takibi. SOI_ref={cfg.soi_broadside_deg:+.2f} deg"
            )
        else:
            self.log_message.emit(
                "Adaptif mod kapalı: sistem MVDR + IMM SOI takibinde kalacak."
            )

    def cleanup(self):
        if self.sdr is not None:
            try:
                self.sdr.tx_destroy_buffer()
            except Exception:
                pass

        self.log_message.emit("SDR tamponları temizlendi.")

    def tracker_step(self, target_label, measurement_deg, dt, R_var):
        """
        Reinitialize IMM when the tracked physical object changes.
        This avoids a long transient when switching from SOI tracking to jammer tracking.
        """
        if self.tracker_target != target_label:
            self.tracker = IMMDoATracker()
            self.tracker.initialize(measurement_deg)
            self.tracker_target = target_label
            self.log_message.emit(
                f"IMM takip hedefi değişti: {target_label} | başlangıç={measurement_deg:+.2f} deg"
            )

        return self.tracker.step(
            z_deg=measurement_deg,
            dt=dt,
            R_var=R_var,
        )

    def update_internal_jammer_tx_if_needed(self):
        """
        Keep Tx1 jammer state synchronized with Jammer Mode.

        This runs inside the SDR worker loop, so TX buffer destroy/restart is
        done in the same thread that owns the Pluto object.
        """
        desired = bool(self.cfg.use_internal_jammer)

        if desired == bool(self.internal_jammer_started):
            return

        self.log_message.emit(
            f"Tx yeniden yapılandırılıyor: Tx1 jammer {'AÇIK' if desired else 'KAPALI'}"
        )
        self.start_tx_waveforms(include_jammer=desired)


    def run(self):
        self._running = True
        frame_idx = 0
        last_frame_time = None

        try:
            self.configure_sdr()

            while self._running:
                t0 = time.time()

                if last_frame_time is None:
                    dt = 0.05
                else:
                    dt = t0 - last_frame_time

                last_frame_time = t0
                dt = float(np.clip(dt, 0.005, 0.500))

                # Synchronize Tx1 jammer with current Jammer Mode before RX.
                # This allows practical live switching between no-jammer and
                # jammer tests from the GUI.
                self.update_internal_jammer_tx_if_needed()

                # Receive data
                data = self.sdr.rx()
                x = np.asarray(data, dtype=np.complex64)

                if x.ndim != 2 or x.shape[0] != self.cfg.nr:
                    raise RuntimeError(
                        f"Beklenen veri şekli [2, N], gelen veri şekli: {x.shape}"
                    )

                # Phase calibration
                x = apply_rx_phase_calibration(x, self.cfg.phase_offset)

                # Covariance matrix
                R = covariance_matrix(x)

                # MUSIC DoA spectrum
                p_music_db = music_spectrum(
                    R,
                    self.theta_scan_rad,
                    self.cfg,
                )

                # MVDR/Capon visualization spectrum
                p_mvdr_db = mvdr_spectrum(
                    R,
                    self.theta_scan_rad,
                    self.cfg,
                )

                # Raw DoA estimate from MUSIC peak
                doa_idx = int(np.argmax(p_music_db))
                doa_music_deg = float(self.theta_scan_deg[doa_idx])

                # RSSI input metric before filtering/beamforming
                pin_dbm = float(iq_power_dbm_est(x, self.cfg.dbm_offset))

                # Measurement quality
                cond_r = float(np.linalg.cond(R))
                sharpness_db = music_peak_sharpness_db(
                    self.theta_scan_deg,
                    p_music_db,
                    doa_idx,
                    guard_deg=6.0,
                )

                R_var, R_sigma = adaptive_measurement_variance(
                    sharpness_db=sharpness_db,
                    cond_r=cond_r,
                    rssi_dbm=pin_dbm,
                )

                # --------------------------------------------------------
                # Adaptive jammer-aware logic
                # --------------------------------------------------------
                # Tx1 jammer OFF:
                #   Dominant MUSIC peak is the SOI. IMM tracks SOI.
                #   MVDR is used because there is no jammer-null constraint.
                # Tx1 jammer ON:
                #   SOI is fixed at the known broadside angle.
                #   Dominant MUSIC peak is the stronger jammer. IMM tracks jammer.
                #   LCMV preserves SOI and places a null at jammer DoA.
                jammer_tx_active = bool(self.internal_jammer_started)
                jammer_mode_active = bool(self.cfg.jammer_mode and jammer_tx_active)

                if jammer_mode_active:
                    tracking_target = "JAMMER"
                else:
                    tracking_target = "SOI"

                imm = self.tracker_step(
                    target_label=tracking_target,
                    measurement_deg=doa_music_deg,
                    dt=dt,
                    R_var=R_var,
                )

                doa_imm_deg = float(imm["theta_deg"])

                if jammer_mode_active:
                    soi_doa_deg = float(self.cfg.soi_broadside_deg)
                    jammer_doa_deg = doa_imm_deg
                    doa_beamformer_deg = soi_doa_deg

                    soi_rad = float(np.deg2rad(soi_doa_deg))
                    jammer_rad = float(np.deg2rad(jammer_doa_deg))
                    w, lcmv_ok = lcmv_weights(soi_rad, jammer_rad, R, self.cfg)
                    beamformer_name = "LCMV" if lcmv_ok else "LCMV->MVDR fallback"
                else:
                    soi_doa_deg = doa_imm_deg
                    jammer_doa_deg = None

                    if self.cfg.use_imm_for_mvdr:
                        doa_beamformer_deg = doa_imm_deg
                    else:
                        doa_beamformer_deg = doa_music_deg

                    beamformer_rad = float(np.deg2rad(doa_beamformer_deg))
                    w = mvdr_weights(beamformer_rad, R, self.cfg)
                    beamformer_name = "MVDR"

                y_beamformed = w.conj().T @ x

                # RSSI output metric.
                # Important: suppression is meaningful only when Tx1 jammer is active
                # and LCMV/nulling mode is being used. In no-jammer MVDR mode,
                # pin-pout is merely an input/output power difference, not jammer
                # suppression, so the GUI/log should not display it as suppression.
                pout_dbm = float(iq_power_dbm_est(y_beamformed, self.cfg.dbm_offset))
                suppression_valid = bool(jammer_mode_active)
                suppression_db = float(pin_dbm - pout_dbm) if suppression_valid else None

                # Diagnostics
                eigvals = np.linalg.eigvalsh(R)
                elapsed_ms = 1e3 * (time.time() - t0)

                frame_idx += 1

                self.frame_ready.emit(
                    {
                        "frame_idx": frame_idx,
                        "theta_scan_deg": self.theta_scan_deg,
                        "music_db": p_music_db,
                        "mvdr_db": p_mvdr_db,
                        "doa_music_deg": doa_music_deg,
                        "doa_imm_deg": doa_imm_deg,
                        "doa_mvdr_deg": doa_beamformer_deg,
                        "beamformer_name": beamformer_name,
                        "soi_doa_deg": float(soi_doa_deg),
                        "jammer_doa_deg": None if jammer_doa_deg is None else float(jammer_doa_deg),
                        "jammer_detected": bool(jammer_mode_active),
                        "tracking_target": tracking_target,
                        "jammer_mode": bool(self.cfg.jammer_mode),
                        "soi_fixed_broadside": bool(jammer_mode_active),
                        "suppression_db": suppression_db,
                        "suppression_valid": suppression_valid,
                        "theta_dot": float(imm["theta_dot"]),
                        "theta_ddot": float(imm["theta_ddot"]),
                        "mu": imm["mu"],
                        "model_names": imm["model_names"],
                        "innovation": float(imm["innovation"]),
                        "R_sigma": float(R_sigma),
                        "sharpness_db": float(sharpness_db),
                        "pin_dbm": pin_dbm,
                        "pout_dbm": pout_dbm,
                        "cond_r": cond_r,
                        "eigvals": eigvals,
                        "dt": dt,
                        "elapsed_ms": elapsed_ms,
                    }
                )

        except Exception as exc:
            self.error_message.emit(str(exc))

        finally:
            self.cleanup()
            self.finished.emit()


# ============================================================
# GUI
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dynamic DoA Tracking - MUSIC + IMM Kalman + MVDR/LCMV + Tx1 Jammer - PlutoSDR")
        self.resize(1380, 850)

        self.thread = None
        self.worker = None

        self.history_max = 500
        self.hist_frame = []
        self.hist_music = []
        self.hist_imm = []

        pg.setConfigOptions(antialias=True)

        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        main_layout = QHBoxLayout(root)

        controls = self._build_control_panel()
        plots = self._build_plot_panel()

        main_layout.addWidget(controls, 0)
        main_layout.addWidget(plots, 1)

        self.setCentralWidget(root)

    def _build_control_panel(self):
        box = QGroupBox("Sistem Ayarları")
        layout = QVBoxLayout(box)

        form = QFormLayout()

        self.uri_edit = QLineEdit("ip:192.168.2.1")

        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(100e3, 20e6)
        self.fs_spin.setDecimals(0)
        self.fs_spin.setSingleStep(1e6)
        self.fs_spin.setValue(5e6)

        self.fc_spin = QDoubleSpinBox()
        self.fc_spin.setRange(70e6, 6e9)
        self.fc_spin.setDecimals(0)
        self.fc_spin.setSingleStep(100e6)
        self.fc_spin.setValue(2.1e9)

        self.if_spin = QDoubleSpinBox()
        self.if_spin.setRange(-2e6, 2e6)
        self.if_spin.setDecimals(0)
        self.if_spin.setSingleStep(25e3)
        self.if_spin.setValue(100e3)

        self.rs_spin = QDoubleSpinBox()
        self.rs_spin.setRange(10e3, 2e6)
        self.rs_spin.setDecimals(0)
        self.rs_spin.setSingleStep(50e3)
        self.rs_spin.setValue(250e3)

        self.rx_gain_spin = QDoubleSpinBox()
        self.rx_gain_spin.setRange(-3, 70)
        self.rx_gain_spin.setDecimals(0)
        self.rx_gain_spin.setSingleStep(1)
        self.rx_gain_spin.setValue(3)

        self.tx_gain_spin = QDoubleSpinBox()
        self.tx_gain_spin.setRange(-88, 0)
        self.tx_gain_spin.setDecimals(0)
        self.tx_gain_spin.setSingleStep(1)
        self.tx_gain_spin.setValue(-20)

        self.phase_spin = QDoubleSpinBox()
        self.phase_spin.setRange(-np.pi, np.pi)
        self.phase_spin.setDecimals(4)
        self.phase_spin.setSingleStep(0.05)
        self.phase_spin.setValue(-0.3)

        self.buffer_spin = QSpinBox()
        self.buffer_spin.setRange(1024, 262144)
        self.buffer_spin.setSingleStep(1024)
        self.buffer_spin.setValue(2**14)

        self.dbm_offset_spin = QDoubleSpinBox()
        self.dbm_offset_spin.setRange(-200.0, 0.0)
        self.dbm_offset_spin.setDecimals(2)
        self.dbm_offset_spin.setSingleStep(1.0)
        self.dbm_offset_spin.setValue(-110.0)

        self.use_tx_check = QCheckBox("Tx0 QPSK SOI gönder")
        self.use_tx_check.setChecked(True)

        self.use_imm_mvdr_check = QCheckBox("MVDR yönü için IMM kullan")
        self.use_imm_mvdr_check.setChecked(True)

        self.jammer_mode_check = QCheckBox("Adaptif mod: Tx1 açıkken LCMV/JAMMER, kesilince MVDR/SOI")
        self.jammer_mode_check.setChecked(True)
        self.jammer_mode_check.stateChanged.connect(self.on_jammer_mode_changed)

        self.soi_broadside_spin = QDoubleSpinBox()
        self.soi_broadside_spin.setRange(-90.0, 90.0)
        self.soi_broadside_spin.setDecimals(1)
        self.soi_broadside_spin.setSingleStep(1.0)
        self.soi_broadside_spin.setValue(0.0)
        self.soi_broadside_spin.valueChanged.connect(self.on_broadside_changed)

        self.use_internal_jammer_check = QCheckBox("Tx1 CW jammer gönder / kes")
        self.use_internal_jammer_check.setChecked(True)
        self.use_internal_jammer_check.stateChanged.connect(self.on_internal_jammer_changed)

        self.jammer_if_spin = QDoubleSpinBox()
        self.jammer_if_spin.setRange(-2e6, 2e6)
        self.jammer_if_spin.setDecimals(0)
        self.jammer_if_spin.setSingleStep(25e3)
        self.jammer_if_spin.setValue(100e3)

        self.jammer_gain_spin = QDoubleSpinBox()
        self.jammer_gain_spin.setRange(-88, 0)
        self.jammer_gain_spin.setDecimals(0)
        self.jammer_gain_spin.setSingleStep(1)
        self.jammer_gain_spin.setValue(-3)

        form.addRow("SDR URI", self.uri_edit)
        form.addRow("Sample Rate [Hz]", self.fs_spin)
        form.addRow("Center Freq [Hz]", self.fc_spin)
        form.addRow("QPSK IF [Hz]", self.if_spin)
        form.addRow("Symbol Rate [sym/s]", self.rs_spin)
        form.addRow("RX Gain [dB]", self.rx_gain_spin)
        form.addRow("SOI TX0 Gain [dB]", self.tx_gain_spin)
        form.addRow("Rx1 Phase Cal. [rad]", self.phase_spin)
        form.addRow("RX Buffer Size", self.buffer_spin)
        form.addRow("DBM Offset", self.dbm_offset_spin)
        form.addRow("", self.use_tx_check)
        form.addRow("", self.use_imm_mvdr_check)
        form.addRow("", self.jammer_mode_check)
        form.addRow("Broadside SOI Angle [deg]", self.soi_broadside_spin)
        form.addRow("", self.use_internal_jammer_check)
        form.addRow("Jammer IF [Hz]", self.jammer_if_spin)
        form.addRow("Jammer TX1 Gain [dB]", self.jammer_gain_spin)

        layout.addLayout(form)

        button_row = QHBoxLayout()

        self.start_btn = QPushButton("Başlat")
        self.stop_btn = QPushButton("Durdur")
        self.stop_btn.setEnabled(False)

        self.start_btn.clicked.connect(self.start_system)
        self.stop_btn.clicked.connect(self.stop_system)

        button_row.addWidget(self.start_btn)
        button_row.addWidget(self.stop_btn)

        layout.addLayout(button_row)

        metrics_box = QGroupBox("Canlı Ölçümler")
        metrics_layout = QGridLayout(metrics_box)

        self.music_doa_label = QLabel("-- deg")
        self.imm_doa_label = QLabel("-- deg")
        self.mvdr_doa_label = QLabel("-- deg")
        self.beamformer_label = QLabel("--")
        self.soi_lock_label = QLabel("-- deg")
        self.jammer_doa_label = QLabel("--")
        self.tracking_target_label = QLabel("--")

        self.pin_label = QLabel("-- dBm")
        self.pout_label = QLabel("-- dBm")
        self.suppression_label = QLabel("-- dB")

        self.cp_label = QLabel("-- %")
        self.cv_label = QLabel("-- %")
        self.ca_label = QLabel("-- %")

        self.vel_label = QLabel("-- deg/s")
        self.acc_label = QLabel("-- deg/s²")

        self.sharpness_label = QLabel("-- dB")
        self.rsigma_label = QLabel("-- deg")
        self.cond_label = QLabel("--")
        self.time_label = QLabel("-- ms")
        self.frame_label = QLabel("--")

        label_style = "font-size: 16px; font-weight: bold;"

        for lab in [
            self.music_doa_label,
            self.imm_doa_label,
            self.mvdr_doa_label,
            self.beamformer_label,
            self.soi_lock_label,
            self.jammer_doa_label,
            self.tracking_target_label,
            self.pin_label,
            self.pout_label,
            self.suppression_label,
            self.cp_label,
            self.cv_label,
            self.ca_label,
        ]:
            lab.setStyleSheet(label_style)

        r = 0
        metrics_layout.addWidget(QLabel("Dominant MUSIC DoA:"), r, 0)
        metrics_layout.addWidget(self.music_doa_label, r, 1)
        r += 1

        metrics_layout.addWidget(QLabel("IMM Tracked DoA:"), r, 0)
        metrics_layout.addWidget(self.imm_doa_label, r, 1)
        r += 1

        metrics_layout.addWidget(QLabel("Beamformer Steering / SOI DoA:"), r, 0)
        metrics_layout.addWidget(self.mvdr_doa_label, r, 1)
        r += 1

        metrics_layout.addWidget(QLabel("Beamformer:"), r, 0)
        metrics_layout.addWidget(self.beamformer_label, r, 1)
        r += 1

        metrics_layout.addWidget(QLabel("SOI DoA / Reference:"), r, 0)
        metrics_layout.addWidget(self.soi_lock_label, r, 1)
        r += 1

        metrics_layout.addWidget(QLabel("Jammer DoA:"), r, 0)
        metrics_layout.addWidget(self.jammer_doa_label, r, 1)
        r += 1

        metrics_layout.addWidget(QLabel("IMM Target:"), r, 0)
        metrics_layout.addWidget(self.tracking_target_label, r, 1)
        r += 1

        metrics_layout.addWidget(QLabel("RSSI In:"), r, 0)
        metrics_layout.addWidget(self.pin_label, r, 1)
        r += 1

        metrics_layout.addWidget(QLabel("RSSI Out:"), r, 0)
        metrics_layout.addWidget(self.pout_label, r, 1)
        r += 1

        metrics_layout.addWidget(QLabel("Jammer Suppression:"), r, 0)
        metrics_layout.addWidget(self.suppression_label, r, 1)
        r += 1

        metrics_layout.addWidget(QLabel("CP Prob.:"), r, 0)
        metrics_layout.addWidget(self.cp_label, r, 1)
        r += 1

        metrics_layout.addWidget(QLabel("CV Prob.:"), r, 0)
        metrics_layout.addWidget(self.cv_label, r, 1)
        r += 1

        metrics_layout.addWidget(QLabel("CA Prob.:"), r, 0)
        metrics_layout.addWidget(self.ca_label, r, 1)
        r += 1

        metrics_layout.addWidget(QLabel("Angular Velocity:"), r, 0)
        metrics_layout.addWidget(self.vel_label, r, 1)
        r += 1

        metrics_layout.addWidget(QLabel("Angular Accel.:"), r, 0)
        metrics_layout.addWidget(self.acc_label, r, 1)
        r += 1

        metrics_layout.addWidget(QLabel("MUSIC Sharpness:"), r, 0)
        metrics_layout.addWidget(self.sharpness_label, r, 1)
        r += 1

        metrics_layout.addWidget(QLabel("Kalman R σ:"), r, 0)
        metrics_layout.addWidget(self.rsigma_label, r, 1)
        r += 1

        metrics_layout.addWidget(QLabel("cond(R):"), r, 0)
        metrics_layout.addWidget(self.cond_label, r, 1)
        r += 1

        metrics_layout.addWidget(QLabel("Frame Time:"), r, 0)
        metrics_layout.addWidget(self.time_label, r, 1)
        r += 1

        metrics_layout.addWidget(QLabel("Frame:"), r, 0)
        metrics_layout.addWidget(self.frame_label, r, 1)

        layout.addWidget(metrics_box)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(160)

        layout.addWidget(QLabel("Log"))
        layout.addWidget(self.log_box)

        layout.addStretch(1)

        return box

    def _build_plot_panel(self):
        box = QWidget()
        layout = QVBoxLayout(box)

        title = QLabel("MUSIC Spectrum + IMM Kalman Trajectory")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")

        layout.addWidget(title)

        # ------------------------------------------------------------
        # Spectrum plot
        # ------------------------------------------------------------
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setBackground("w")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.25)

        self.spectrum_plot.setLabel("bottom", "Direction of Arrival", units="deg")
        self.spectrum_plot.setLabel("left", "Normalized Spectrum", units="dB")

        self.spectrum_plot.setXRange(-90, 90)
        self.spectrum_plot.setYRange(-40, 1)

        self.spectrum_plot.addLegend(offset=(10, 10))

        # MUSIC: strong blue
        self.music_curve = self.spectrum_plot.plot(
            [],
            [],
            pen=pg.mkPen(color=(0, 90, 255), width=3),
            name="MUSIC",
        )

        # MVDR/Capon: strong orange dashed line
        self.mvdr_curve = self.spectrum_plot.plot(
            [],
            [],
            pen=pg.mkPen(color=(230, 120, 0), style=Qt.DashLine, width=3),
            name="MVDR/Capon",
        )

        # MUSIC peak marker: blue
        self.music_peak_marker = pg.ScatterPlotItem(
            size=13,
            brush=pg.mkBrush(0, 90, 255),
            pen=pg.mkPen(0, 40, 180, width=2),
        )
        self.spectrum_plot.addItem(self.music_peak_marker)

        # Raw MUSIC vertical line
        self.raw_doa_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            movable=False,
            pen=pg.mkPen(color=(0, 90, 255), style=Qt.DotLine, width=2),
        )
        self.spectrum_plot.addItem(self.raw_doa_line)

        # IMM filtered vertical line
        self.imm_doa_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            movable=False,
            pen=pg.mkPen(color=(220, 40, 40), style=Qt.SolidLine, width=2),
        )
        self.spectrum_plot.addItem(self.imm_doa_line)

        # SOI reference / MVDR steering line: green dashed
        self.soi_lock_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            movable=False,
            pen=pg.mkPen(color=(20, 150, 60), style=Qt.DashDotLine, width=2),
        )
        self.spectrum_plot.addItem(self.soi_lock_line)

        layout.addWidget(self.spectrum_plot, 2)

        # ------------------------------------------------------------
        # Trajectory plot
        # ------------------------------------------------------------
        self.traj_plot = pg.PlotWidget()
        self.traj_plot.setBackground("w")
        self.traj_plot.showGrid(x=True, y=True, alpha=0.25)

        self.traj_plot.setLabel("bottom", "Frame Index")
        self.traj_plot.setLabel("left", "DoA Trajectory", units="deg")

        self.traj_plot.setYRange(-90, 90)
        self.traj_plot.addLegend(offset=(10, 10))

        self.traj_music_curve = self.traj_plot.plot(
            [],
            [],
            pen=pg.mkPen(color=(0, 90, 255, 130), width=2),
            symbol="o",
            symbolSize=4,
            symbolBrush=pg.mkBrush(0, 90, 255, 120),
            symbolPen=pg.mkPen(0, 90, 255, 120),
            name="Raw MUSIC DoA",
        )

        self.traj_imm_curve = self.traj_plot.plot(
            [],
            [],
            pen=pg.mkPen(color=(220, 40, 40), width=3),
            name="IMM Filtered DoA",
        )

        layout.addWidget(self.traj_plot, 1)

        

        return box

    def build_config(self):
        return SystemConfig(
            sdr_uri=self.uri_edit.text().strip(),
            sample_rate=float(self.fs_spin.value()),
            center_freq=float(self.fc_spin.value()),
            qpsk_if_freq=float(self.if_spin.value()),
            symbol_rate=float(self.rs_spin.value()),
            rx_gain=float(self.rx_gain_spin.value()),
            tx_gain=float(self.tx_gain_spin.value()),
            phase_offset=float(self.phase_spin.value()),
            rx_buffer_size=int(self.buffer_spin.value()),
            dbm_offset=float(self.dbm_offset_spin.value()),
            use_transmitter=bool(self.use_tx_check.isChecked()),
            use_imm_for_mvdr=bool(self.use_imm_mvdr_check.isChecked()),
            jammer_mode=bool(self.jammer_mode_check.isChecked()),
            soi_broadside_deg=float(self.soi_broadside_spin.value()),
            use_internal_jammer=bool(self.use_internal_jammer_check.isChecked()),
            jammer_if_freq=float(self.jammer_if_spin.value()),
            jammer_gain=float(self.jammer_gain_spin.value()),
        )

    def set_controls_enabled(self, enabled):
        widgets = [
            self.uri_edit,
            self.fs_spin,
            self.fc_spin,
            self.if_spin,
            self.rs_spin,
            self.rx_gain_spin,
            self.tx_gain_spin,
            self.phase_spin,
            self.buffer_spin,
            self.dbm_offset_spin,
            self.use_tx_check,
            self.use_imm_mvdr_check,
            # Adaptif mod, broadside angle and Tx1 ON/OFF intentionally stay enabled
            # during runtime. The worker loop restarts Tx0/Tx1 buffers when
            # internal jammer state changes.
            self.jammer_if_spin,
            self.jammer_gain_spin,
        ]

        for widget in widgets:
            widget.setEnabled(enabled)

        self.start_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(not enabled)

    def on_jammer_mode_changed(self, *args):
        """Allow live switching between dynamic SOI tracking and jammer tracking."""
        if self.worker is not None:
            self.worker.cfg.jammer_mode = bool(self.jammer_mode_check.isChecked())
            self.worker.cfg.soi_broadside_deg = float(self.soi_broadside_spin.value())
            mode_txt = "ADAPTIF" if self.worker.cfg.jammer_mode else "MVDR/SOI ZORLA"
            self.append_log(
                f"Canlı adaptif mod değişti: {mode_txt} | SOI_ref={self.worker.cfg.soi_broadside_deg:+.1f} deg"
            )

    def on_internal_jammer_changed(self, *args):
        """Live Tx1 jammer ON/OFF switch. The worker restarts TX buffers safely."""
        if self.worker is not None:
            self.worker.cfg.use_internal_jammer = bool(self.use_internal_jammer_check.isChecked())
            self.worker.cfg.soi_broadside_deg = float(self.soi_broadside_spin.value())
            state_txt = "AÇIK" if self.worker.cfg.use_internal_jammer else "KESİLDİ/KAPALI"
            self.append_log(
                f"Canlı Tx1 jammer durumu: {state_txt}. "
                "Worker TX buffer'ı güncelleyip uygun MVDR/LCMV moduna geçecek."
            )

    def on_broadside_changed(self, *args):
        """Update broadside SOI reference during live external-jammer tests."""
        if self.worker is not None:
            self.worker.cfg.soi_broadside_deg = float(self.soi_broadside_spin.value())

    def clear_history(self):
        self.hist_frame.clear()
        self.hist_music.clear()
        self.hist_imm.clear()

        self.traj_music_curve.setData([], [])
        self.traj_imm_curve.setData([], [])

    def start_system(self):
        """
        Start or restart the SDR processing thread.
        """
        if self.thread is not None:
            self.append_log("Sistem zaten çalışıyor.")
            return

        cfg = self.build_config()

        self.log_box.clear()
        self.clear_history()

        self.append_log("Sistem başlatılıyor...")
        self.append_log(
            f"IMM modeli: CP/static + CV + CA | MVDR IMM kullanımı: {cfg.use_imm_for_mvdr}"
        )
        self.append_log(
            f"RSSI hesabı: 10log10(mean(|IQ|^2)) + {cfg.dbm_offset:.2f} dB"
        )
        self.append_log(
            f"Adaptive Tx1-jammer MVDR/LCMV | adaptive_mode={cfg.jammer_mode} | "
            f"SOI_ref={cfg.soi_broadside_deg:+.1f} deg | "
            f"Tx0_SOI_gain={cfg.tx_gain:.0f} dB | Tx1_JAM_gain={cfg.jammer_gain:.0f} dB | "
            f"Tx1_jammer={cfg.use_internal_jammer}"
        )

        self.thread = QThread(self)
        self.worker = DoAWorker(cfg)

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.worker.frame_ready.connect(self.update_frame)
        self.worker.log_message.connect(self.append_log)
        self.worker.error_message.connect(self.show_error)

        # Correct shutdown order:
        # 1) Worker emits finished
        # 2) Thread event loop quits
        # 3) Worker is deleted safely
        # 4) Thread emits finished
        # 5) GUI references are cleared
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)

        self.thread.finished.connect(self.on_thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)

        self.set_controls_enabled(False)
        self.thread.start()

    def stop_system(self):
        """
        Stop only the running SDR loop.
        The GUI remains open and Start can be pressed again afterwards.
        """
        if self.worker is not None:
            self.append_log("Durdurma isteği gönderildi...")
            self.stop_btn.setEnabled(False)
            self.worker.stop()

    def on_thread_finished(self):
        """
        Called only after QThread has fully stopped.
        Safe place to clear worker/thread references and re-enable Start.
        """
        self.append_log("Sistem durdu. Tekrar başlatılabilir.")

        self.worker = None
        self.thread = None

        self.set_controls_enabled(True)

    def update_frame(self, result):
        theta = result["theta_scan_deg"]
        music_db = result["music_db"]
        mvdr_db = result["mvdr_db"]

        doa_music = result["doa_music_deg"]
        doa_imm = result["doa_imm_deg"]

        self.music_curve.setData(theta, music_db)
        self.mvdr_curve.setData(theta, mvdr_db)

        self.raw_doa_line.setValue(doa_music)
        self.imm_doa_line.setValue(doa_imm)
        self.soi_lock_line.setValue(result["soi_doa_deg"])

        # MUSIC peak marker
        peak_idx = int(np.argmax(music_db))
        self.music_peak_marker.setData(
            [theta[peak_idx]],
            [music_db[peak_idx]],
        )

        # Trajectory history
        self.hist_frame.append(result["frame_idx"])
        self.hist_music.append(doa_music)
        self.hist_imm.append(doa_imm)

        if len(self.hist_frame) > self.history_max:
            self.hist_frame = self.hist_frame[-self.history_max:]
            self.hist_music = self.hist_music[-self.history_max:]
            self.hist_imm = self.hist_imm[-self.history_max:]

        self.traj_music_curve.setData(self.hist_frame, self.hist_music)
        self.traj_imm_curve.setData(self.hist_frame, self.hist_imm)

        if len(self.hist_frame) > 2:
            right = self.hist_frame[-1]
            left = max(0, right - self.history_max)
            self.traj_plot.setXRange(left, right, padding=0.02)

        mu = result["mu"]

        self.music_doa_label.setText(f"{result['doa_music_deg']:+.2f} deg")
        self.imm_doa_label.setText(f"{result['doa_imm_deg']:+.2f} deg")
        self.mvdr_doa_label.setText(f"{result['doa_mvdr_deg']:+.2f} deg")
        self.beamformer_label.setText(result.get("beamformer_name", "--"))

        if result.get("jammer_detected", False):
            self.soi_lock_label.setText(f"{result['soi_doa_deg']:+.2f} deg (fixed)")
        else:
            self.soi_lock_label.setText(f"{result['soi_doa_deg']:+.2f} deg (dynamic)")

        if result["jammer_doa_deg"] is None:
            self.jammer_doa_label.setText("Yok / SOI")
        else:
            self.jammer_doa_label.setText(f"{result['jammer_doa_deg']:+.2f} deg")

        self.tracking_target_label.setText(result["tracking_target"])

        self.pin_label.setText(f"{result['pin_dbm']:+.2f} dBm")
        self.pout_label.setText(f"{result['pout_dbm']:+.2f} dBm")

        if result.get("suppression_valid", False) and result.get("suppression_db") is not None:
            self.suppression_label.setText(f"{result['suppression_db']:+.2f} dB")
        else:
            self.suppression_label.setText("-- dB")

        self.cp_label.setText(f"{100.0 * mu[0]:.1f} %")
        self.cv_label.setText(f"{100.0 * mu[1]:.1f} %")
        self.ca_label.setText(f"{100.0 * mu[2]:.1f} %")

        self.vel_label.setText(f"{result['theta_dot']:+.2f} deg/s")
        self.acc_label.setText(f"{result['theta_ddot']:+.2f} deg/s²")

        self.sharpness_label.setText(f"{result['sharpness_db']:.2f} dB")
        self.rsigma_label.setText(f"{result['R_sigma']:.2f} deg")
        self.cond_label.setText(f"{result['cond_r']:.2e}")
        self.time_label.setText(f"{result['elapsed_ms']:.1f} ms")
        self.frame_label.setText(str(result["frame_idx"]))

        if result["frame_idx"] % 5 == 0:
            supp_txt = (
                f"{result['suppression_db']:+6.2f} dB"
                if result.get("suppression_valid", False) and result.get("suppression_db") is not None
                else "---"
            )
            self.append_log(
                f"Frame {result['frame_idx']:05d} | "
                f"MUSIC={result['doa_music_deg']:+7.2f} deg | "
                f"IMM={result['doa_imm_deg']:+7.2f} deg({result['tracking_target']}) | "
                f"SOI={result['soi_doa_deg']:+7.2f} deg | "
                f"JAM={result['jammer_doa_deg'] if result['jammer_doa_deg'] is not None else '---'} | "
                f"BF={result.get('beamformer_name', '--')}@{result['doa_mvdr_deg']:+7.2f} deg | "
                f"CP/CV/CA={100*mu[0]:4.1f}/{100*mu[1]:4.1f}/{100*mu[2]:4.1f}% | "
                f"RSSI In={result['pin_dbm']:+7.2f} dBm | "
                f"RSSI Out={result['pout_dbm']:+7.2f} dBm | "
                f"JamSupp={supp_txt} | "
                f"Rσ={result['R_sigma']:.2f} deg | "
                f"Sharp={result['sharpness_db']:.2f} dB | "
                f"cond(R)={result['cond_r']:.2e} | "
                f"t={result['elapsed_ms']:.1f} ms"
            )

    def append_log(self, text):
        self.log_box.append(text)

    def show_error(self, text):
        self.append_log(f"HATA: {text}")

    def closeEvent(self, event):
        """
        Close the GUI safely.

        Unlike the Stop button, closing the window should wait briefly for the
        worker thread to terminate cleanly.
        """
        if self.worker is not None:
            self.worker.stop()

        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(2000)

        event.accept()


# ============================================================
# Main
# ============================================================

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MUSIC IMM MVDR PlutoSDR GUI")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
