# tests/test_app_ui.py
# End-to-end tests of the Streamlit app (app_ods.py) driven with
# streamlit.testing.v1.AppTest.
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

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
