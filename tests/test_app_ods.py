# tests/test_app_ods.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
# app_ods.py:100 calls st.set_page_config() at module import time;
# stub it so importing app_ods is side-effect free outside `streamlit run`.
st.set_page_config = lambda *args, **kwargs: None

import numpy as np
import pandas as pd
from unittest.mock import MagicMock

import app_ods


class TestNonConvergedSentinelHandling:
    """Regression: failed fits carry converged=False (not R2=-999), and every
    consumer must skip them via the converged flag -- never via numeric R2
    comparisons, because np.nan comparisons are always False."""

    @staticmethod
    def _failed(error="Optimal parameters not found"):
        return {"R2": np.nan, "aicc": np.nan, "converged": False, "error": error}

    def test_get_valid_models_skips_nonconverged(self):
        ok = {"R2": 0.99, "aicc": 5.0}
        res = {"Zero-order": self._failed(), "Pseudo-first": ok}
        assert app_ods._get_valid_models(res, ["Zero-order", "Pseudo-first"]) == {"Pseudo-first": ok}

    def test_best_model_returns_none_when_all_nonconverged(self):
        res = {"Zero-order": self._failed()}
        assert app_ods._best_model(res, ["Zero-order"]) is None

    def test_best_model_excludes_nonconverged_from_competition(self):
        ok_pso = {"R2": 0.95, "aicc": 5.0}
        ok_fo  = {"R2": 0.90, "aicc": 8.0}
        res = {"Zero-order": self._failed(),
               "Pseudo-first": ok_fo,
               "Pseudo-second-order": ok_pso}
        assert app_ods._best_model(
            res, ["Zero-order", "Pseudo-first", "Pseudo-second-order"]
        ) == "Pseudo-second-order"

    def test_residuals_skips_nonconverged_model_without_keyerror(self, monkeypatch):
        st = MagicMock()
        st.columns.return_value = (MagicMock(), MagicMock())
        st.expander.return_value = MagicMock()
        st.selectbox.side_effect = ["Cat-A Removal (%)", "Zero-order"]  # catalyst, then model
        monkeypatch.setattr(app_ods, "st", st)

        df = pd.DataFrame({
            "Time (min)": [0, 5, 10, 15, 20],
            "Cat-A Removal (%)": [0, 20, 40, 60, 80],
        })
        monkeypatch.setattr(app_ods, "_load_kinetic_data",
                            lambda uploaded: (df, "Time (min)", ["Cat-A Removal (%)"]))
        all_res = {m: self._failed() for m in app_ods.MODEL_NAMES}
        monkeypatch.setattr(app_ods, "_fit_nonlinear", lambda t, c, C0: all_res)

        app_ods._tab_residuals({"C0": 0.015}, MagicMock())  # must return, not KeyError

        assert st.error.called
