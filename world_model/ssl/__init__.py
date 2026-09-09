"""Minimal two-view SSL with SIGReg."""

from .config import SSLConfig
from .losses import SSLLoss, SSLLossOutput
from .model import SSLModel, SSLOutput
from .sigreg import SIGReg

__all__ = ["SSLConfig", "SSLModel", "SSLOutput", "SSLLoss", "SSLLossOutput", "SIGReg"]
