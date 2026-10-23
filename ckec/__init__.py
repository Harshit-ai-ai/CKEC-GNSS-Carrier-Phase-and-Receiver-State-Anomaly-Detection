"""
ckec — Cross-Modal Kinematic-Electromagnetic Coupling for GNSS Spoofing Detection.

Package layout
--------------
ckec.ingestion      Dataset loaders and metadata validators
ckec.preprocessing  Signal cleaning, alignment, phase unwrapping
ckec.features       Feature extraction (AGC, phase, geometry)
ckec.detectors      Baseline and CKEC detectors
ckec.evaluation     Metrics, calibration, ROC/PR curves
ckec.simulation     Controlled synthetic signal generation
ckec.utils          Shared helpers (config, logging, I/O)
"""

__version__ = "0.1.0"
