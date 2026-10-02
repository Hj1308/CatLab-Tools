# tests/test_app_ui.py
# End-to-end tests of the Streamlit app (app_ods.py) driven with
# streamlit.testing.v1.AppTest.
import io
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from catlab.kinetics_engine import MODEL_NAMES

pytestmark = pytest.mark.ui

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app_ods.py")

TIME = [0, 15, 30, 45, 60, 90, 120]
CAT_A = [0, 25, 44, 58, 68, 81, 88]
CAT_B = [0, 22, 41, 59, 74, 87, 94]


def _kinetic_frame():
    return pd.DataFrame({"Time (min)": TIME, "CatA Removal (%)": CAT_A, "CatB Removal (%)": CAT_B})


@pytest.fixture(scope="module")
def csv_bytes():
    return _kinetic_frame().to_csv(index=False).encode("utf-8")


@pytest.fixture(scope="module")
def xlsx_bytes():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        _kinetic_frame().to_excel(writer, sheet_name="Raw_Data", index=False)
    return buf.getvalue()


@pytest.fixture
def app():
    return AppTest.from_file(APP_PATH, default_timeout=180).run()


def _upload_csv(app, data):
    app.file_uploader(key="shared_file").upload("data.csv", data, "text/csv").run()
    return app


def _upload_xlsx(app, data):
    app.file_uploader(key="shared_file").upload(
        "data.xlsx", data, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ).run()
    return app


# ================================================================
# STEP 1 — shared fixtures and smoke tests
# ================================================================


class TestSmoke:
    def test_app_loads_with_nine_tabs(self, app):
        assert len(app.tabs) == 9
        assert not app.exception

    def test_csv_upload_has_no_errors_in_any_tab(self, app, csv_bytes):
        _upload_csv(app, csv_bytes)
        assert not app.exception
        assert not [e.value for e in app.error if "Cannot read file" in e.value]

    def test_xlsx_upload_has_no_errors_in_any_tab(self, app, xlsx_bytes):
        _upload_xlsx(app, xlsx_bytes)
        assert not app.exception
        assert not app.error


# ================================================================
# STEP 2 — Tab 1 kinetic fitting
# ================================================================


def _run_analysis(app):
    [b for b in app.tabs[0].button if "Run" in str(b.label)][0].click().run()
    return app


def _summary(app):
    for d in app.tabs[0].dataframe:
        cols = list(d.value.columns)
        if "Best Model" in cols and "R²" in cols:
            return d.value
    return None


class TestTabKinetics:
    def test_summary_has_one_row_per_catalyst(self, app, csv_bytes):
        _upload_csv(app, csv_bytes)
        _run_analysis(app)
        summary = _summary(app)
        assert summary is not None
        assert list(summary["Catalyst"]) == ["CatA", "CatB"]

    def test_best_model_is_one_of_model_names(self, app, csv_bytes):
        _upload_csv(app, csv_bytes)
        _run_analysis(app)
        summary = _summary(app)
        assert summary is not None
        for model in summary["Best Model"]:
            assert model in MODEL_NAMES

    def test_excluding_a_point_drops_points_used_by_one(self, app, csv_bytes):
        _upload_csv(app, csv_bytes)
        multiselect = app.tabs[0].multiselect(key="excl_CatA Removal (%)")
        assert "t = 15 min" in multiselect.options
        multiselect.select("t = 15 min").run()
        _run_analysis(app)
        captions = [c.value for c in app.tabs[0].caption]
        match = re.search(r"CatA: (\d+) points used \(1 excluded\)", "\n".join(captions))
        assert match is not None
        # 7 points, minus the t=0 anchor (not counted), minus the excluded point.
        assert int(match.group(1)) == len(TIME) - 1 - 1

    def test_time_h_column_warns_not_in_minutes(self, app):
        data = b"Time (h),CatA Removal (%)\n0,0\n1,25\n2,44\n3,58\n4,68\n6,81\n8,88\n"
        _upload_csv(app, data)
        assert any("not in minutes" in w.value for w in app.tabs[0].warning)

    def test_t0_row_does_not_change_fit(self, app, csv_bytes):
        frame = _kinetic_frame()
        with_t0 = frame.to_csv(index=False).encode("utf-8")
        without_t0 = frame.iloc[1:].to_csv(index=False).encode("utf-8")

        app_with = AppTest.from_file(APP_PATH, default_timeout=180).run()
        _upload_csv(app_with, with_t0)
        _run_analysis(app_with)
        r2_with = _summary(app_with).set_index("Catalyst")["R²"]

        app_without = AppTest.from_file(APP_PATH, default_timeout=180).run()
        _upload_csv(app_without, without_t0)
        _run_analysis(app_without)
        r2_without = _summary(app_without).set_index("Catalyst")["R²"]

        assert list(r2_with.index) == list(r2_without.index)
        for cat in r2_with.index:
            assert np.isclose(r2_with[cat], r2_without[cat], rtol=1e-9)
