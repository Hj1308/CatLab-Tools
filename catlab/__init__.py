# CatLab-Tools — Catalyst Reaction Analysis Suite
# BET/BJH/T-Plot → https://github.com/Hj1308/BET_analyser
from .catalyst_analytics import (
    convert_to_mmol_L,
    SampleInfo,
    calc_conversion,
    calc_tof,
    calc_toc_removal,
    KineticsAnalyser,
)

__version__ = "1.1.0"
__author__  = "Hoda Jafari"
