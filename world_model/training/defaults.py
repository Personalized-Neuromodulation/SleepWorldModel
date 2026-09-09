"""Shared defaults for HSP SSL training and debugging."""

HSP_SSL_TARGET_SAMPLE_RATES = {
    "eeg": 100.0,
    "eog": 100.0,
    "emg": 100.0,
    "ecg": 100.0,
    "resp": 25.0,
    "spo2": 25.0,
}

__all__ = ["HSP_SSL_TARGET_SAMPLE_RATES"]
