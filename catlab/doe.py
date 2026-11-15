# catlab/doe.py
# Design of experiments and response-surface analysis for ODS optimisation
# (pure numpy/scipy, no streamlit). Designs follow Montgomery, Design and Analysis of
# Experiments, ch. 6 and 11; Box & Behnken (1960) Technometrics 2, 455-475.
import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats as scipy_stats

DESIGNS = ("Full factorial (2-level)", "Box-Behnken", "CCD (face-centred)", "CCD (rotatable)")


# -- coded design matrices -----------------------------------------------------
def full_factorial_2level(k):
    """All 2^k corners at coded -1/+1, standard (Yates) order."""
    return np.array(list(itertools.product([-1.0, 1.0], repeat=k)))[:, ::-1]


def box_behnken(k):
    """Box-Behnken runs without centre points: for every pair of factors a 2^2 factorial
    at +/-1 with all other factors at 0 (k = 3: 12 runs, 4: 24, 5: 40). This pairwise
    construction equals Box & Behnken's design only for 3 <= k <= 5; for k >= 6 they use
    other incomplete-block bases, so larger k is refused."""
    if not 3 <= k <= 5:
        raise ValueError("Box-Behnken is implemented for 3 to 5 factors")
    rows = []
    for i, j in itertools.combinations(range(k), 2):
        for a, b in itertools.product([-1.0, 1.0], repeat=2):
            r = np.zeros(k)
            r[i], r[j] = a, b
            rows.append(r)
    return np.array(rows)


def ccd_alpha(k, kind):
    """Axial distance: 1 for face-centred, (2^k)^(1/4) for rotatable."""
    return 1.0 if kind == "face" else float((2**k) ** 0.25)


def central_composite(k, kind="face"):
    """CCD without centre points: 2^k factorial plus 2k axial points at +/-alpha."""
    a = ccd_alpha(k, kind)
    axial = np.vstack([s * a * np.eye(k)[i] for i in range(k) for s in (-1.0, 1.0)])
    return np.vstack([full_factorial_2level(k), axial])


def coded_design(k, design, n_center):
    if design == "Full factorial (2-level)":
        X = full_factorial_2level(k)
    elif design == "Box-Behnken":
        X = box_behnken(k)
    elif design == "CCD (face-centred)":
        X = central_composite(k, "face")
    elif design == "CCD (rotatable)":
        X = central_composite(k, "rotatable")
    else:
        raise ValueError(f"unknown design {design!r}")
    return np.vstack([X, np.zeros((int(n_center), k))])


def make_design(factors, design, n_center=3, seed=0, response="Removal (%)"):
    """Run sheet in real units.

    factors: list of (name, low, high); low/high map to coded -1/+1 (a rotatable CCD
    places its axial runs beyond them, at +/-alpha). Rows are in randomised run order
    (reproducible with `seed`); 'Std order' keeps the design order. The response column
    is left empty for the experimenter."""
    names = [f[0] for f in factors]
    lo = np.array([float(f[1]) for f in factors])
    hi = np.array([float(f[2]) for f in factors])
    if np.any(hi <= lo):
        raise ValueError("each factor needs high > low")
    Xc = coded_design(len(factors), design, n_center)
    mid, half = (hi + lo) / 2, (hi - lo) / 2
    real = mid + Xc * half
    df = pd.DataFrame(real, columns=names)
    for i, n in enumerate(names):
        df[f"{n} [coded]"] = np.round(Xc[:, i], 4)
    df.insert(0, "Std order", np.arange(1, len(df) + 1))
    order = np.random.default_rng(seed).permutation(len(df))
    df = df.iloc[order].reset_index(drop=True)
    df.insert(0, "Run", np.arange(1, len(df) + 1))
    df[response] = np.nan
    return df


# -- response-surface fit -------------------------------------------------------
@dataclass
class RSMFit:
    terms: list
    coef: np.ndarray
    se: np.ndarray
    p: np.ndarray
    cov: np.ndarray
    n: int
    df_resid: int
    r2: float
    adj_r2: float
    pred_r2: float
    f_model: float
    p_model: float
    lack_of_fit: dict
    curvature: dict
    dropped: list
    coding: dict
    warnings: list = field(default_factory=list)
    resid: np.ndarray = None
    fitted: np.ndarray = None

    def table(self):
        return pd.DataFrame(
            {"Term": self.terms, "Coefficient (coded)": self.coef, "SE": self.se, "p": self.p}
        )


def _term_columns(Xc, names, order):
    cols, labels = [np.ones(len(Xc))], ["Intercept"]
    k = Xc.shape[1]
    for i in range(k):
        cols.append(Xc[:, i])
        labels.append(names[i])
    if order >= 2:
        for i, j in itertools.combinations(range(k), 2):
            cols.append(Xc[:, i] * Xc[:, j])
            labels.append(f"{names[i]}·{names[j]}")
        for i in range(k):
            cols.append(Xc[:, i] ** 2)
            labels.append(f"{names[i]}²")
    return np.column_stack(cols), labels


def _code(df, names, coding=None):
    """Coded factor matrix. Uses '<name> [coded]' columns when present (sheets made by
    make_design); otherwise maps each factor's min/max to -1/+1, or uses `coding`
    {name: (low, high)} when given."""
    if coding is None and all(f"{n} [coded]" in df for n in names):
        Xc = df[[f"{n} [coded]" for n in names]].to_numpy(float)
        real = df[names].to_numpy(float)
        coding = {}
        for i, n in enumerate(names):
            b, a = np.polyfit(Xc[:, i], real[:, i], 1)  # real = a + b * coded
            coding[n] = (float(a - b), float(a + b))
        return Xc, coding
    real = df[names].to_numpy(float)
    if coding is None:
        coding = {n: (real[:, i].min(), real[:, i].max()) for i, n in enumerate(names)}
    lo = np.array([coding[n][0] for n in names], float)
    hi = np.array([coding[n][1] for n in names], float)
    return (real - (hi + lo) / 2) / ((hi - lo) / 2), coding


def fit_rsm(df, factors, response, model="quadratic", coding=None):
    """Least-squares response surface in coded units.

    model: 'linear', 'interaction' (linear + two-factor interactions) or 'quadratic'.
    Terms that the design cannot estimate (aliased, e.g. squares in a 2-level
    factorial) are dropped and listed in `dropped`. Reports coefficient SE and
    p-values, R², adjusted R², predicted R² (PRESS), the model F-test, the
    lack-of-fit F-test against pure error from replicated settings, and a
    centre-point curvature test for 2-level factorial designs."""
    d = df.dropna(subset=list(factors) + [response]).reset_index(drop=True)
    y = d[response].to_numpy(float)
    Xc, coding = _code(d, list(factors), coding)
    order = 1 if model == "linear" else 2
    X, labels = _term_columns(Xc, list(factors), order)
    if model == "interaction":
        keep = [i for i, t in enumerate(labels) if not t.endswith("²")]
        X, labels = X[:, keep], [labels[i] for i in keep]
    # drop terms that do not raise the rank (aliased with earlier ones)
    kept, dropped, rank = [], [], 0
    for j in range(X.shape[1]):
        r = np.linalg.matrix_rank(X[:, kept + [j]])
        if r > rank:
            kept.append(j)
            rank = r
        else:
            dropped.append(labels[j])
    X, labels = X[:, kept], [labels[i] for i in kept]
    n, p = X.shape
    warns = []
    if n <= p:
        raise ValueError(f"{n} runs cannot fit {p} terms; add runs or use a smaller model")
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    fitted = X @ beta
    e = y - fitted
    dfr = n - p
    sse = float(e @ e)
    sst = float(((y - y.mean()) ** 2).sum())
    s2 = sse / dfr
    cov = s2 * XtX_inv
    se = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        tval = beta / se
    pval = 2 * scipy_stats.t.sf(np.abs(tval), dfr)
    r2 = 1 - sse / sst if sst > 0 else np.nan
    adj = 1 - (1 - r2) * (n - 1) / dfr if sst > 0 else np.nan
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    if np.any(h > 1 - 1e-10):
        pred_r2 = np.nan
        warns.append(
            "Predicted R² not available: a run has leverage 1 (it alone determines a term)."
        )
    else:
        press = float(((e / (1 - h)) ** 2).sum())
        pred_r2 = 1 - press / sst if sst > 0 else np.nan
    if p > 1 and sse > 0:
        f_model = ((sst - sse) / (p - 1)) / s2
        p_model = float(scipy_stats.f.sf(f_model, p - 1, dfr))
    else:
        f_model, p_model = np.nan, np.nan
    # pure error from replicated settings
    key = [tuple(np.round(r, 6)) for r in Xc]
    groups = {}
    for i, kk in enumerate(key):
        groups.setdefault(kk, []).append(i)
    ss_pe = sum(float(((y[g] - y[g].mean()) ** 2).sum()) for g in groups.values())
    df_pe = n - len(groups)
    df_lof = len(groups) - p
    lof = {"df_lof": df_lof, "df_pe": df_pe, "F": np.nan, "p": np.nan}
    if df_pe > 0 and df_lof > 0 and ss_pe > 0:
        F = ((sse - ss_pe) / df_lof) / (ss_pe / df_pe)
        lof.update(F=float(F), p=float(scipy_stats.f.sf(F, df_lof, df_pe)))
    elif df_pe == 0:
        warns.append("No replicated runs: lack of fit cannot be tested. Add centre points.")
    # curvature test (2-level factorial + centre points)
    curv = {}
    centre = np.all(np.abs(Xc) < 1e-9, axis=1)
    corner = np.all(np.abs(np.abs(Xc) - 1) < 1e-9, axis=1)
    if centre.sum() >= 2 and corner.any() and corner.sum() + centre.sum() == n and ss_pe > 0:
        diff = y[corner].mean() - y[centre].mean()
        se_d = np.sqrt((ss_pe / df_pe) * (1 / corner.sum() + 1 / centre.sum()))
        curv = {
            "difference": float(diff),
            "p": float(2 * scipy_stats.t.sf(abs(diff / se_d), df_pe)),
        }
    if dropped:
        warns.append("Not estimable with this design and dropped: " + ", ".join(dropped) + ".")
    if not np.isnan(pred_r2) and not np.isnan(adj) and adj - pred_r2 > 0.2:
        warns.append(
            f"Predicted R² ({pred_r2:.2f}) is far below adjusted R² ({adj:.2f}): the model "
            "is overfitted or a few runs dominate it. Do not use it to predict."
        )
    if "removal" in response.lower() and np.mean(y >= 95) > 0.3:
        warns.append(
            "More than 30 % of runs reach ≥ 95 % removal. A capped response hides the "
            "factor effects; use removal at an earlier time or the rate constant k as the response."
        )
    return RSMFit(
        labels,
        beta,
        se,
        pval,
        cov,
        n,
        dfr,
        r2,
        adj,
        pred_r2,
        f_model,
        p_model,
        lof,
        curv,
        dropped,
        coding,
        warns,
        e,
        fitted,
    )


def _x_row(xc, fit, factors):
    X, labels = _term_columns(np.atleast_2d(xc), list(factors), 2)
    idx = [labels.index(t) for t in fit.terms]
    return X[:, idx]


def predict(fit, factors, xc):
    """Prediction and its standard error at coded point(s) xc."""
    X = _x_row(np.atleast_2d(xc), fit, factors)
    yhat = X @ fit.coef
    se = np.sqrt(np.einsum("ij,jk,ik->i", X, fit.cov, X))
    return yhat, se


def optimum(fit, factors, maximize=True, bounds=None, n_start=20, seed=0):
    """Best predicted response inside the coded box (default: -1..+1 per factor).
    Returns coded and real settings, the prediction with a 95 % confidence interval
    of the mean, and which factors sit on the boundary (extrapolation risk)."""
    k = len(factors)
    bounds = bounds or [(-1.0, 1.0)] * k
    sign = -1.0 if maximize else 1.0
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_start):
        x0 = np.array([rng.uniform(a, b) for a, b in bounds])
        r = optimize.minimize(
            lambda x: sign * predict(fit, factors, x)[0][0], x0, bounds=bounds, method="L-BFGS-B"
        )
        if best is None or r.fun < best.fun:
            best = r
    xc = best.x
    yhat, se = predict(fit, factors, xc)
    t = scipy_stats.t.ppf(0.975, fit.df_resid)
    real = {}
    for i, n in enumerate(factors):
        lo, hi = fit.coding[n]
        real[n] = float((hi + lo) / 2 + xc[i] * (hi - lo) / 2)
    edge = [
        factors[i] for i, (a, b) in enumerate(bounds) if min(abs(xc[i] - a), abs(xc[i] - b)) < 1e-3
    ]
    return {
        "coded": xc,
        "real": real,
        "prediction": float(yhat[0]),
        "ci95": (float(yhat[0] - t * se[0]), float(yhat[0] + t * se[0])),
        "at_boundary": edge,
    }
