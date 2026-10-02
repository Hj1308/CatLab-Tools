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
from tests.synthetic_data import SYNTHETIC_REMOVAL, T_SPARSE

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

    def test_five_point_dataset_warns_too_few_points(self, app):
        rem = SYNTHETIC_REMOVAL["B_initial_drop"]
        rows = ["Time (min),CatA Removal (%)"]
        rows += [f"{t},{r}" for t, r in zip(T_SPARSE, rem)]
        data = ("\n".join(rows) + "\n").encode("utf-8")
        _upload_csv(app, data)
        _run_analysis(app)
        assert any("informative data points" in w.value for w in app.tabs[0].warning)

    def test_seven_point_dataset_has_no_too_few_points_warning(self, app, csv_bytes):
        _upload_csv(app, csv_bytes)
        _run_analysis(app)
        assert not any("informative data points" in w.value for w in app.tabs[0].warning)

    def test_summary_has_n_fitted_column(self, app, csv_bytes):
        _upload_csv(app, csv_bytes)
        _run_analysis(app)
        summary = _summary(app)
        assert summary is not None
        assert "n (fitted)" in summary.columns


# ================================================================
# STEP 3 — Tabs 2-9
# ================================================================


class TestTabLinearization:
    def test_linearization_reports_r2_for_each_catalyst(self, app, csv_bytes):
        _upload_csv(app, csv_bytes)
        for d in app.tabs[1].dataframe:
            cols = list(d.value.columns)
            if set(cols) >= {"Catalyst", "Zero-order", "Pseudo-first"}:
                pivot = d.value
                assert list(pivot["Catalyst"]) == ["CatA", "CatB"]
                for model in ["Zero-order", "Pseudo-first", "Pseudo-second-order", "Elovich"]:
                    assert model in pivot.columns
                    assert pivot[model].between(0.0, 1.0).all()
                return
        pytest.fail("linearization R² pivot table not found")

    def test_initial_drop_dataset_warns_pfo_intercept(self, app):
        rem = SYNTHETIC_REMOVAL["B_initial_drop"]
        rows = ["Time (min),CatA Removal (%)"]
        rows += [f"{t},{r}" for t, r in zip(T_SPARSE, rem)]
        data = ("\n".join(rows) + "\n").encode("utf-8")
        _upload_csv(app, data)
        warnings = [w.value for w in app.tabs[1].warning]
        assert any("does not pass through the origin" in w for w in warnings)


class TestTabRemoval:
    def test_removal_renders_efficiency_plots(self, app, csv_bytes):
        _upload_csv(app, csv_bytes)
        # Streamlit renamed the element type ("imgs" -> "image"); accept both.
        imgs = [
            c
            for c in app.tabs[2].children.values()
            if getattr(c, "type", None) in ("imgs", "image")
        ]
        assert len(imgs) >= 2  # efficiency vs time + final-efficiency bar chart
        assert not app.exception


class TestTabTonTof:
    def test_option_b_shows_mass_normalized_tof(self, app, csv_bytes):
        _upload_csv(app, csv_bytes)
        app.tabs[3].radio[0].set_value(
            "Option B — Carbon-based / Metal-free  (mass-normalized TOF)"
        ).run()
        tab = app.tabs[3]
        cols = {c for d in tab.dataframe for c in d.value.columns}
        assert "TOF_mass_avg (µmol·g⁻¹·min⁻¹)" in cols
        assert any("Catalyst_Properties" in i.value for i in tab.info)


class TestTabParameterEffect:
    def test_temperature_sweep_reveals_arrhenius_inputs(self, app):
        app.tabs[4].selectbox[1].set_value("Temperature (Arrhenius)").run()
        labels = [n.label for n in app.tabs[4].number_input]
        assert "k at ref T (min⁻¹)" in labels
        assert "Eₐ (kJ/mol)" in labels


class TestTabOxidant:
    def test_measured_h2o2_reveals_per_catalyst_inputs(self, app, csv_bytes):
        _upload_csv(app, csv_bytes)
        app.tabs[5].radio[0].set_value("Yes — I will enter measured consumption").run()
        tab = app.tabs[5]
        labels = [n.label for n in tab.number_input]
        assert any("CatA Removal (%)" in label for label in labels)
        assert any("CatB Removal (%)" in label for label in labels)
        cols = {c for d in tab.dataframe for c in d.value.columns}
        assert "η (%)" in cols


class TestTabComparison:
    def test_comparison_uploads_table_and_offers_axes(self, app):
        cmp = (
            b"Experiment,T (C),O/S,kapp (1/min),R2\n"
            b"Run-1,25,2,0.012,0.989\nRun-2,40,4,0.028,0.994\nRun-3,60,6,0.055,0.997\n"
        )
        app.file_uploader(key="cmp_upload").upload("cmp.csv", cmp, "text/csv").run()
        tab = app.tabs[6]
        assert tab.dataframe and list(tab.dataframe[0].value["Experiment"]) == [
            "Run-1",
            "Run-2",
            "Run-3",
        ]
        labels = [s.label for s in tab.selectbox]
        assert "X axis" in labels and "Y axis" in labels


class TestTabArrhenius:
    def test_arrhenius_extracts_ea_from_two_temperatures(self, app, csv_bytes):
        # The 60 °C run must be faster than the 25 °C run; identical files would
        # give the same k at both temperatures and Ea = 0, testing nothing.
        fast = pd.DataFrame(
            {
                "Time (min)": TIME,
                "CatA Removal (%)": [0, 45, 68, 81, 89, 96, 98],
                "CatB Removal (%)": [0, 41, 66, 82, 91, 97, 99],
            }
        )
        fast_bytes = fast.to_csv(index=False).encode("utf-8")
        app.file_uploader(key="arrhenius_files").set_value(
            [("t25.csv", csv_bytes, "text/csv"), ("t60.csv", fast_bytes, "text/csv")]
        ).run()
        tab = app.tabs[7]
        tab.number_input(key="arr_T_t25.csv").set_value(25.0)
        tab.number_input(key="arr_T_t60.csv").set_value(60.0)
        tab.selectbox(key="arr_model_choice").set_value("Pseudo-first")
        [b for b in tab.button if "Arrhenius" in str(b.label)][0].click().run()
        tab = app.tabs[7]
        assert not tab.error
        result = tab.dataframe[0].value
        assert "Eₐ (kJ/mol)" in result.columns
        assert list(result["n_T"]) == [2, 2]
        ea = pd.to_numeric(result["Eₐ (kJ/mol)"], errors="coerce")
        assert (ea > 0).all(), f"faster run at 60 °C must give Ea > 0, got {list(ea)}"


class TestTabResiduals:
    def test_switching_model_changes_r2_metric(self, app, csv_bytes):
        _upload_csv(app, csv_bytes)
        tab = app.tabs[8]
        r2_before = float(tab.metric[0].value)
        tab.selectbox(key="resid_model").set_value("Pseudo-first").run()
        r2_after = float(app.tabs[8].metric[0].value)
        assert r2_before != r2_after
        assert 0.0 <= r2_before <= 1.0
        assert 0.0 <= r2_after <= 1.0
