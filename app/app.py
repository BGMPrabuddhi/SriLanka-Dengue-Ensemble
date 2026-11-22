# AI-assistance disclosure: This file was generated with the help of Claude Code (Anthropic) and reviewed, run and tested by B. G. M. Prabuddhi.
"""Sri Lanka Dengue Early-Warning Dashboard (Streamlit).

Run from the repository root:   streamlit run app/app.py
Everything is loaded from models/ and data/processed/ (nothing is retrained at start-up); all model results
and written findings come from models/results.csv and models/summary.json, produced by the notebooks.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))           # make `src` importable (needed to un-pickle the models)

from src import config, evaluate as E, features as F, models as M  # noqa: E402
from src.viz import AXIS, FAMILY_COLORS, GRID, INK, INK_2, MUTED, RISK_COLORS, SERIES, SURFACE  # noqa: E402

st.set_page_config(page_title="Sri Lanka Dengue Early Warning", page_icon="🦟", layout="wide")

RISK_TEXT = {"Low": "#ffffff", "Medium": "#0b0b0b", "High": "#ffffff"}
RISK_ICON = {"Low": "▼", "Medium": "■", "High": "▲"}
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
CORE_SHORT = {"Bagging": "Random Forest", "Boosting": "XGBoost", "Stacking": "RF+XGB+LGBM+Cat → Ridge"}

st.markdown(f"""
<style>
  .block-container {{padding-top: 2rem; max-width: 1200px;}}
  .kpi {{background:#ffffff; border:1px solid rgba(11,11,11,0.10); border-radius:10px; padding:14px 16px; height:100%;}}
  .kpi .label {{color:{INK_2}; font-size:0.85rem;}}
  .kpi .value {{color:{INK}; font-size:1.7rem; font-weight:650; line-height:1.25;}}
  .kpi .sub {{color:{MUTED}; font-size:0.82rem;}}
  .badge {{display:inline-block; padding:3px 12px; border-radius:999px; font-weight:650;}}
  .story {{font-size:1.08rem; color:{INK_2}; line-height:1.6; max-width:900px;}}
  .panel {{background:#ffffff; border:1px solid rgba(11,11,11,0.10); border-left:5px solid {SERIES[0]};
           border-radius:8px; padding:14px 20px; margin:6px 0 10px 0;}}
</style>
""", unsafe_allow_html=True)


# ====================================================================== data & models (cached)
@st.cache_data(show_spinner=False)
def load_tables():
    proc, mdir = config.DATA_PROCESSED, config.MODELS_DIR
    t = {
        "dengue": pd.read_csv(proc / "dengue_clean.csv", parse_dates=["start_date", "end_date"]),
        "forecast": pd.read_csv(F.FORECAST_CSV, parse_dates=["origin_date", "target_date"]),
        "results": E.results_table(pd.read_csv(E.RESULTS_CSV)),
        "preds": pd.read_csv(E.PREDICTIONS_CSV, parse_dates=["target_date"]),
        "arima_res": pd.read_csv(mdir / "hybrid_arima_results.csv"),
        "arima_preds": pd.read_csv(mdir / "hybrid_arima_predictions.csv", parse_dates=["target_date"]),
        "summary": json.loads((mdir / "summary.json").read_text(encoding="utf-8")),
        "registry": json.loads((mdir / "registry.json").read_text()),
        "cfg": json.loads(F.FEATURE_CONFIG_JSON.read_text()),
    }
    d = t["dengue"]
    nat = d.groupby("start_date").agg(cases=("cases", lambda s: s.sum(min_count=1)),
                                      missing=("missing_report", "sum"))
    nat.loc[nat["missing"] >= 15, "cases"] = np.nan      # weeks the source reported unreliably
    t["national"] = nat["cases"]
    return t


@st.cache_resource(show_spinner="Loading the trained model…")
def load_model(name: str):
    return M.load_predictor(name)


@st.cache_data(show_spinner=False)
def recent_thresholds(years: int = 3) -> dict:
    """Risk thresholds: each region's 50th (Medium) / 75th (High) percentile of weekly cases, last `years` years."""
    d = load_tables()["dengue"]
    cutoff = d["start_date"].max() - pd.DateOffset(years=years)
    r = d[(d["start_date"] > cutoff) & d["cases"].notna()]
    return {k: {"medium": float(np.percentile(s, 50)), "high": float(np.percentile(s, 75))}
            for k, s in r.groupby("district")["cases"]}


def risk_of(value: float, thr: dict) -> str:
    if value > thr["high"]:
        return "High"
    return "Medium" if value > thr["medium"] else "Low"


@st.cache_data(show_spinner=False)
def trend_table() -> pd.DataFrame:
    """Early-warning trend from reported data: last 4 reported weeks vs the 4 before."""
    rows = []
    for dist, g in load_tables()["dengue"].sort_values("start_date").groupby("district"):
        s = g["cases"].dropna()
        last4, prev4 = s.iloc[-4:].mean(), s.iloc[-8:-4].mean()
        pct = 100 * (last4 - prev4) / prev4 if prev4 > 0 else np.nan
        label = "■ Stable" if pd.isna(pct) or abs(pct) <= 15 else ("▲ Rising" if pct > 0 else "▼ Falling")
        rows.append({"District": dist, "4-week trend": label})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def forecast_table(model_name: str) -> pd.DataFrame:
    t = load_tables()
    feats = t["cfg"]["numeric_features"] + t["cfg"]["categorical_features"]
    fc = t["forecast"].sort_values("district").reset_index(drop=True)
    pred = np.clip(load_model(model_name).predict(fc[feats]), 0, None)
    thr = recent_thresholds()
    out = pd.DataFrame({"District": fc["district"], "Last reported week": fc["cases_lag1"].round(0),
                        "Forecast next week": pred.round(0)})
    last = out["Last reported week"]
    out["Change %"] = np.where(last > 0, 100 * (out["Forecast next week"] - last) / last, np.nan)
    out["Medium above"] = [thr[d]["medium"] for d in out["District"]]
    out["High above"] = [thr[d]["high"] for d in out["District"]]
    out["Risk"] = [risk_of(p, thr[d]) for p, d in zip(out["Forecast next week"], out["District"])]
    out["vs High"] = out["Forecast next week"] / out["High above"].replace(0, np.nan)
    return out.merge(trend_table(), on="District", how="left")


@st.cache_data(show_spinner=False)
def whatif_ranges() -> tuple[dict, float]:
    f = pd.read_csv(F.FEATURES_CSV, usecols=["district", "cases_lag1", "rain_mm_lag8"])
    return f.groupby("district")["cases_lag1"].max().to_dict(), float(f["rain_mm_lag8"].quantile(0.995))


# ====================================================================== UI helpers
def style(fig, height=380, title=None):
    fig.update_layout(
        height=height, template="plotly_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_2, size=13), margin=dict(l=10, r=10, t=56 if title else 24, b=10),
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16, color=INK)) if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)), hovermode="x unified")
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, showgrid=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, rangemode="tozero")
    return fig


def badge(level: str) -> str:
    return (f'<span class="badge" style="background:{RISK_COLORS[level]};color:{RISK_TEXT[level]}">'
            f'{RISK_ICON[level]} {level} risk</span>')


def kpi(col, label, value, sub="", border=None):
    top = f"border-top:5px solid {border};" if border else ""
    col.markdown(f'<div class="kpi" style="{top}"><div class="label">{label}</div><div class="value">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def fmt_date(d) -> str:
    return pd.Timestamp(d).strftime("%d %b %Y")


def fmt_table(df: pd.DataFrame, highlight: set | None = None):
    """Two decimals for numbers; optional highlighted rows (by 'Model' column)."""
    sty = df.style.format({c: "{:.2f}" for c in df.columns if df[c].dtype != object})
    if highlight:
        sty = sty.apply(lambda r: ["background-color:#e6f0fc;font-weight:600" if r.get("Model") in highlight
                                   else "" for _ in r], axis=1)
    return sty


T = load_tables()
S = T["summary"]
RES = T["results"]
NAIVE = RES.set_index("model").loc[E.NAIVE_NAME]
DEPLOYED = S["deployed_model"]
BEST_TEST = S["best_on_test"]["model"]
FEATS = T["cfg"]["numeric_features"] + T["cfg"]["categorical_features"]
LAST_WEEK = T["dengue"]["start_date"].max()
TARGET_WEEK = pd.Timestamp(T["cfg"]["forecast_target_week"])
LATE_NOTE = "The latest 1–2 weeks may still rise because some reports arrive late."
ESTIMATE_NOTE = (f"Forecasts are model estimates, not reported cases (typical error about "
                 f"±{S['overall_winner_validation']['MAE']:.0f} cases per region-week).")


# ====================================================================== pages
def page_home():
    st.title("Dengue in Sri Lanka: where will it strike next week?")
    st.markdown(
        '<p class="story">Dengue is one of Sri Lanka\'s biggest public-health problems: in 2017 alone more than '
        '170,000 cases were reported. Cases are reported every week for 26 health regions, but the health ministry has '
        'to send fogging teams, hospital capacity and awareness campaigns <b>before</b> the next week\'s cases appear. '
        'This dashboard forecasts <b>next week\'s cases in every region</b> and compares bagging, boosting, stacking and '
        'hybrid models to find the most reliable forecaster.</p>', unsafe_allow_html=True)

    nat = T["national"].dropna()
    fc = forecast_table(DEPLOYED)
    latest, prev = nat.iloc[-1], nat.iloc[-2]
    top = fc.sort_values("vs High", ascending=False).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Cases reported, latest week", f"{latest:,.0f}", f"week of {fmt_date(nat.index[-1])}")
    kpi(c2, "Change vs previous week", f"{100 * (latest - prev) / prev:+.1f}%", f"previous week: {prev:,.0f}")
    kpi(c3, "Highest-risk region next week", top["District"], f"forecast {top['Forecast next week']:,.0f} cases")
    kpi(c4, "Regions at High risk next week", f"{int((fc['Risk'] == 'High').sum())} of 26",
        f"week of {fmt_date(TARGET_WEEK)}")
    st.caption(f"{ESTIMATE_NOTE} {LATE_NOTE}")

    s = T["national"]
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(color=SERIES[0], width=2),
                               name="national cases", hovertemplate="%{y:,.0f} cases<extra></extra>"))
    fig.add_annotation(x=s.idxmax(), y=s.max(), text=f"<b>2017 record outbreak</b><br>{s.max():,.0f} cases in one week",
                       showarrow=True, arrowhead=0, ax=70, ay=10, font=dict(color=INK_2), align="left")
    st.plotly_chart(style(fig, 380, "Weekly dengue cases in Sri Lanka, 2007–2026"), use_container_width=True)
    st.caption("Gaps in the line are weeks the source reported incorrectly; they are left blank rather than invented.")


def page_models():
    st.title("Which ensemble approach worked best, and why?")
    st.markdown(f"Every model forecasts **next week's cases in each of the 26 regions**, scored on the same "
                f"**{S['n_test_rows']:,} region-weeks** ({S['test_period'][0]} to {S['test_period'][1]}) that were never "
                f"used for training. **MAE** = average error in cases per region-week (lower is better). "
                f"The naive rule *“next week = this week”* scores **MAE {NAIVE.MAE:.2f}**.")

    # --- the three strategies required by the brief
    core = S["core_comparison"]
    best_core = min(core, key=lambda f: core[f]["MAE"])
    cols = st.columns(3)
    for col, fam in zip(cols, ["Bagging", "Boosting", "Stacking"]):
        r = core[fam]
        kpi(col, f"<b>{fam}</b> · {CORE_SHORT[fam]}{' 🏆' if fam == best_core else ''}", f"MAE {r['MAE']:.2f}",
            f"{r['MAE_vs_naive_pct']:+.1f}% vs naive", border=FAMILY_COLORS[fam])

    # --- results table grouped by family
    st.subheader("Results by family")
    tbl = RES[["family", "model", "MAE", "RMSE", "R2", "MAE_vs_naive_pct"]].rename(
        columns={"family": "Family", "model": "Model", "R2": "R²", "MAE_vs_naive_pct": "% vs naive"})
    st.dataframe(fmt_table(tbl, {BEST_TEST, DEPLOYED}), hide_index=True, use_container_width=True,
                 height=36 * len(tbl) + 38)
    st.caption(f"Highlighted: **{BEST_TEST}** (lowest test error) and **{DEPLOYED}** (chosen in advance by the 2022 "
               "validation error; used for the forecasts on this dashboard).")

    # --- % vs naive, coloured by family
    r = RES[RES["model"] != E.NAIVE_NAME].sort_values("MAE_vs_naive_pct")
    fig = go.Figure(go.Bar(y=r["model"], x=r["MAE_vs_naive_pct"], orientation="h",
                           marker_color=[FAMILY_COLORS[f] for f in r["family"]], customdata=r["family"],
                           text=r["MAE_vs_naive_pct"].map("{:+.1f}%".format), textposition="outside", cliponaxis=False,
                           hovertemplate="%{y} (%{customdata}): %{x:+.1f}% vs naive<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=INK_2, width=1))
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(range=[max(r["MAE_vs_naive_pct"].min(), -60) - 8, r["MAE_vs_naive_pct"].max() + 8])
    st.plotly_chart(style(fig, 520, "Improvement over the naive rule (% lower MAE; right of 0 = better)"),
                    use_container_width=True)
    st.markdown(" &nbsp; ".join(f'<span style="color:{c}">■</span> {f}' for f, c in FAMILY_COLORS.items()
                                if f in set(RES["family"])), unsafe_allow_html=True)

    # --- normal vs outbreak weeks
    st.subheader("Where the models struggle: normal weeks vs outbreak weeks")
    names = [E.NAIVE_NAME] + [core[f]["model"] for f in ["Bagging", "Boosting", "Stacking"]] + [BEST_TEST]
    names = list(dict.fromkeys(names))
    sub = RES.set_index("model").loc[names]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=sub["MAE_normal"], name="normal weeks", marker_color=SERIES[0]))
    fig.add_trace(go.Bar(x=names, y=sub["MAE_outbreak"], name="outbreak weeks", marker_color=SERIES[1]))
    fig.update_layout(barmode="group", bargap=0.25, yaxis_title="MAE (cases)")
    st.plotly_chart(style(fig, 380), use_container_width=True)
    st.caption(f"Outbreak weeks = cases above the region's usual 75th percentile ({S['outbreak_week_share']:.0%} of the "
               "test period). All models beat naive in normal weeks; outbreak weeks decide the ranking.")

    # --- actual vs predicted
    st.subheader("Actual vs predicted")
    p = T["preds"]
    models = [c for c in p.columns if c not in ("district", "target_date", "target")]
    regions = sorted(p["district"].unique())
    c1, c2 = st.columns(2)
    model = c1.selectbox("Model", models, index=models.index(BEST_TEST), key="avp_model")
    region = c2.selectbox("District", regions, index=regions.index("Colombo"), key="avp_region")
    q = p[p["district"] == region].sort_values("target_date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q["target_date"], y=q["target"], name="actual", line=dict(color=INK, width=2.5)))
    fig.add_trace(go.Scatter(x=q["target_date"], y=q[model], name=model, line=dict(color=SERIES[0], width=1.8)))
    if model != E.NAIVE_NAME:
        fig.add_trace(go.Scatter(x=q["target_date"], y=q[E.NAIVE_NAME], name="naive (last week)",
                                 line=dict(color=MUTED, width=1, dash="dot")))
    st.plotly_chart(style(fig, 400, f"{region}: one-week-ahead forecasts on the test period"), use_container_width=True)

    # --- why
    st.subheader("Which approach performed best, and why")
    st.markdown('<div class="panel"><ol style="margin:0;padding-left:1.2rem">' +
                "".join(f"<li style='margin:6px 0'>{md_bold(t)}</li>" for t in S["dashboard_findings"]) +
                "</ol></div>", unsafe_allow_html=True)

    st.subheader("The model families in one line")
    for fam, text in S["family_explanations"].items():
        st.markdown(f'<span style="color:{FAMILY_COLORS[fam]}">■</span> **{fam}**: {text}', unsafe_allow_html=True)


def md_bold(text: str) -> str:
    """Convert **bold** markdown to <b> for use inside an HTML block."""
    parts = text.split("**")
    return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))


def page_district():
    st.title("District forecast & risk")
    fc = forecast_table(DEPLOYED)
    districts = sorted(fc["District"])
    default = fc.sort_values("vs High", ascending=False).iloc[0]["District"]
    district = st.selectbox("Choose a district", districts, index=districts.index(default), key="dist_region")
    row = fc.set_index("District").loc[district]

    c1, c2, c3 = st.columns(3)
    kpi(c1, f"Last reported week ({fmt_date(LAST_WEEK)})", f"{row['Last reported week']:,.0f}", "cases")
    kpi(c2, f"Forecast, week of {fmt_date(TARGET_WEEK)}", f"{row['Forecast next week']:,.0f}",
        f"{row['Change %']:+.0f}% vs last week" if pd.notna(row["Change %"]) else "")
    c3.markdown(f'<div class="kpi"><div class="label">Risk next week</div><div style="margin:8px 0">'
                f'{badge(row["Risk"])}</div><div class="sub">4-week trend: {row["4-week trend"]}</div></div>',
                unsafe_allow_html=True)

    hist = T["dengue"][T["dengue"]["district"] == district].set_index("start_date")["cases"]
    hist = hist[hist.index > LAST_WEEK - pd.Timedelta(weeks=104)]
    last_obs = hist.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="reported cases",
                             line=dict(color=SERIES[0], width=2), hovertemplate="%{y:,.0f}<extra>reported</extra>"))
    fig.add_trace(go.Scatter(x=[last_obs.index[-1], TARGET_WEEK], y=[last_obs.iloc[-1], row["Forecast next week"]],
                             mode="lines+markers", name="next-week forecast",
                             line=dict(color=RISK_COLORS[row["Risk"]], width=2, dash="dot"),
                             marker=dict(size=[0, 13], color=RISK_COLORS[row["Risk"]], line=dict(color="white", width=2)),
                             hovertemplate="%{y:,.0f}<extra>forecast</extra>"))
    for lvl, dash in [("Medium", "dot"), ("High", "dash")]:
        fig.add_hline(y=row[f"{lvl} above"], line=dict(color=MUTED, width=1, dash=dash),
                      annotation_text=f"{lvl} threshold", annotation_position="top left", annotation_font_color=INK_2)
    st.plotly_chart(style(fig, 380, f"{district}: last two years and next week's forecast"), use_container_width=True)
    st.info("**Notes.** Risk thresholds are each district's own 50th (Medium) and 75th (High) percentile of weekly "
            f"cases over the **last 3 years**. {LATE_NOTE} {ESTIMATE_NOTE}")

    st.subheader("Risk for all 26 regions next week")
    view = fc.sort_values("vs High", ascending=False)
    fig = go.Figure(go.Bar(
        x=view["District"], y=view["Forecast next week"], marker_color=[RISK_COLORS[r] for r in view["Risk"]],
        customdata=np.stack([view["Risk"], view["High above"]], axis=1),
        hovertemplate="%{x}: %{y:,.0f} forecast cases · %{customdata[0]} risk "
                      "(High above %{customdata[1]:.0f})<extra></extra>"))
    fig.update_layout(hovermode="closest", yaxis_title="forecast cases")
    st.plotly_chart(style(fig, 340), use_container_width=True)
    st.markdown(" &nbsp; ".join(f'<span style="color:{RISK_COLORS[k]}">■</span> {RISK_ICON[k]} {k}'
                                for k in ["High", "Medium", "Low"]), unsafe_allow_html=True)
    st.dataframe(view[["District", "Risk", "4-week trend", "Last reported week", "Forecast next week", "Change %",
                       "High above"]],
                 hide_index=True, use_container_width=True,
                 column_config={"Change %": st.column_config.NumberColumn(format="%+.0f%%"),
                                "Last reported week": st.column_config.NumberColumn(format="%d"),
                                "Forecast next week": st.column_config.NumberColumn(format="%d"),
                                "High above": st.column_config.NumberColumn(format="%.0f")})


def page_hybrids():
    st.title("Hybrid models")
    notes = S["hybrid_notes"]

    st.subheader("RF + ARIMA: a time-series model corrected by a Random Forest")
    which = st.radio("Series", ["National", "Colombo"], horizontal=True, key="hyb_series")
    ar = T["arima_res"][T["arima_res"]["series"] == which][["model", "MAE", "RMSE", "MAE_vs_naive_pct"]]
    ar = ar.rename(columns={"model": "Model", "MAE_vs_naive_pct": "% vs naive"})
    st.dataframe(fmt_table(ar, {"Hybrid RF+SARIMA"}), hide_index=True, use_container_width=True)
    ap = T["arima_preds"][T["arima_preds"]["series"] == which]
    fig = go.Figure(go.Scatter(x=ap["target_date"], y=ap["actual"], name="actual", line=dict(color=INK, width=2.5)))
    for m, c in [("SARIMA alone", SERIES[1]), ("RF alone", SERIES[0]), ("Hybrid RF+SARIMA", FAMILY_COLORS["Hybrid"])]:
        fig.add_trace(go.Scatter(x=ap["target_date"], y=ap[m], name=m, line=dict(color=c, width=1.6)))
    st.plotly_chart(style(fig, 380, f"{which}: one-week-ahead forecasts, 2023–2026"), use_container_width=True)
    st.markdown(f'<div class="panel">{notes[0] if which == "National" else notes[1]}</div>', unsafe_allow_html=True)

    st.subheader("RF + ANN: a Random Forest combined with a neural network")
    names = ["Random Forest", "ANN (MLP)", "Hybrid RF+ANN (weighted avg)", "Hybrid RF->ANN (RF as input)",
             E.NAIVE_NAME]
    tbl = RES.set_index("model").loc[names, ["MAE", "RMSE", "R2", "MAE_vs_naive_pct"]].reset_index()
    tbl = tbl.rename(columns={"model": "Model", "R2": "R²", "MAE_vs_naive_pct": "% vs naive"})
    st.dataframe(fmt_table(tbl, {"Hybrid RF+ANN (weighted avg)"}), hide_index=True, use_container_width=True)
    st.markdown(f'<div class="panel">{notes[2]}<br>{notes[3]}</div>', unsafe_allow_html=True)


def page_explain():
    st.title("What drives the forecast?")
    figs = config.FIGURES_DIR
    st.image(str(figs / "shap_summary.png"), use_container_width=True)
    st.caption("SHAP: each dot is one region-week; dots to the right push the forecast up, red = high value. Last "
               "week's cases and the 4-week average dominate. Weather adds a smaller push from rain and humidity 5–9 "
               "weeks earlier (breeding, mosquito development and incubation take about two months).")
    with st.expander("Feature importance: Random Forest and XGBoost"):
        c1, c2 = st.columns(2)
        c1.image(str(figs / "fi_random_forest.png"), use_container_width=True)
        c2.image(str(figs / "fi_xgboost.png"), use_container_width=True)

    st.subheader("What-if: change the inputs, watch the forecast")
    fc = T["forecast"].set_index("district")
    c1, c2 = st.columns([1, 2])
    district = c1.selectbox("District", sorted(fc.index), index=sorted(fc.index).index("Colombo"), key="wi_region")
    base = fc.loc[[district]].reset_index()
    case_max, rain_max = whatif_ranges()
    lag1 = c2.slider("Cases reported last week", 0, int(max(case_max[district], base["cases_lag1"].iloc[0]) * 1.2),
                     int(base["cases_lag1"].iloc[0]), key=f"wi_cases_{district}")
    rain8 = c2.slider("Rainfall 8 weeks ago (mm in that week)", 0, int(rain_max),
                      int(round(base["rain_mm_lag8"].iloc[0])), key=f"wi_rain_{district}")

    x = base.copy()
    x["cases_lag1"] = lag1                                   # keep the rolling features consistent with lag 1
    lags = x[[f"cases_lag{k}" for k in range(1, 5)]].to_numpy(dtype=float)[0]
    x["rolling_mean_4"], x["rolling_std_4"] = lags.mean(), lags.std(ddof=1)
    x["rolling_mean_8"] = base["rolling_mean_8"] + (lag1 - base["cases_lag1"]) / 8
    x["rain_mm_lag8"] = rain8

    model = load_model(DEPLOYED)
    p_base = float(np.clip(model.predict(base[FEATS]), 0, None)[0])
    p_new = float(np.clip(model.predict(x[FEATS]), 0, None)[0])
    k1, k2, k3 = st.columns(3)
    kpi(k1, "Forecast with the real latest data", f"{p_base:,.0f}", f"week of {fmt_date(TARGET_WEEK)}")
    kpi(k2, "Forecast with your inputs", f"{p_new:,.0f}", f"{p_new - p_base:+,.0f} cases")
    k3.markdown(f'<div class="kpi"><div class="label">Risk with your inputs</div><div style="margin:8px 0">'
                f'{badge(risk_of(p_new, recent_thresholds()[district]))}</div></div>', unsafe_allow_html=True)
    st.caption(f"Live prediction from the saved **{DEPLOYED}** model (model estimates, not reported cases). Tree "
               "models learn step-like rules, so small slider moves can leave the forecast unchanged and then jump.")


def page_explorer():
    st.title("Data explorer")
    d = T["dengue"]
    last_year = int(d["year"].max())
    regions_all = sorted(d["district"].unique())
    top4 = list(d.groupby("district")["cases"].sum().sort_values(ascending=False).index[:4])
    c1, c2 = st.columns([1, 2])
    y0, y1 = c1.slider("Years", 2007, last_year, (2017, last_year), key="ex_years")
    sel = c2.multiselect("Districts (empty = all 26)", regions_all, default=top4, key="ex_regions")
    regions = sel or regions_all
    sub = d[d["year"].between(y0, y1) & d["district"].isin(regions)]
    reported = sub.dropna(subset=["cases"])
    if reported.empty:
        st.warning("No reported data for this selection. Widen the years or choose other districts.")
        return

    g = sub.groupby("start_date")
    total = g["cases"].sum(min_count=1).mask(g["missing_report"].mean() >= 0.5)
    if total.dropna().empty:
        total = g["cases"].sum(min_count=1)
    k1, k2, k3 = st.columns(3)
    kpi(k1, "Reported cases", f"{reported['cases'].sum():,.0f}", f"{y0}–{y1}")
    kpi(k2, "Average per week", f"{total.mean():,.0f}", "selected districts combined")
    kpi(k3, "Busiest week", f"{total.max():,.0f}", f"week of {fmt_date(total.idxmax())}")

    fig = go.Figure()
    if sel and len(sel) <= 6:
        for i, r in enumerate(sel):
            s = sub[sub["district"] == r].set_index("start_date")["cases"]
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=r, mode="lines", line=dict(color=SERIES[i], width=1.8)))
    else:
        fig.add_trace(go.Scatter(x=total.index, y=total.values, name="total", mode="lines",
                                 line=dict(color=SERIES[0], width=2)))
    st.plotly_chart(style(fig, 340, "Weekly reported cases" + ("" if sel and len(sel) <= 6 else " (combined)")),
                    use_container_width=True)

    tot = d[d["year"].between(y0, y1)].groupby("district")["cases"].sum().sort_values()
    fig = go.Figure(go.Bar(y=tot.index, x=tot.values, orientation="h",
                           marker_color=[SERIES[0] if (not sel or r in sel) else "#c9c8c1" for r in tot.index],
                           text=[f"{v:,.0f}" for v in tot.values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:,.0f} cases<extra></extra>"))
    fig.update_layout(hovermode="closest")
    st.plotly_chart(style(fig, 620, f"Reported cases by district, {y0}–{y1} (selected in blue)"),
                    use_container_width=True)

    stats = reported.groupby("district")["cases"].agg(["sum", "mean", "max"]).reset_index()
    stats.columns = ["District", "Total", "Weekly mean", "Busiest week"]
    st.dataframe(stats.sort_values("Total", ascending=False), hide_index=True, use_container_width=True,
                 column_config={"Total": st.column_config.NumberColumn(format="%d"),
                                "Weekly mean": st.column_config.NumberColumn(format="%.1f"),
                                "Busiest week": st.column_config.NumberColumn(format="%d")})
    if y1 == last_year:
        st.caption(f"{last_year} has data up to {fmt_date(LAST_WEEK)}. {LATE_NOTE}")


def page_about():
    st.title("About")
    st.markdown(f"""
**Data sources**
* **Dengue cases:** weekly cases for 26 health regions (RDHS areas), 2006–2026, from the R package **denguedatahub**
  (Talagala, T. S., GitHub `thiyangt/denguedatahub`), compiled from the Weekly Epidemiological Reports of the
  Epidemiology Unit, Ministry of Health, Sri Lanka.
* **Weather:** daily rainfall, temperature and humidity for each region from the **Open-Meteo Historical Weather API**
  (Zippenfenig, P., 2023; ERA5 reanalysis), aggregated to the reporting weeks.

**Methods**
* Each forecast stands at the end of week *t* and predicts cases in week *t+1* for one region, using the last 4 weeks of
  cases, 4- and 8-week averages, weather 1–8 weeks earlier, week of year and region.
* Time-based split: train 2007–2021, validation 2022, test 2023–2026; tuning with time-series cross-validation.
* Models: **bagging** (bagged trees, Random Forest, Extra Trees), **boosting** (AdaBoost, Gradient Boosting, XGBoost,
  LightGBM, CatBoost), **stacking** and voting, **hybrids** (RF + SARIMA, RF + neural network), all compared with the
  naive "same as last week" rule.
* Reporting errors in the source (missing years, copied weeks, counts in the wrong region) were left blank, never invented.

**Limitations:** one-week horizon only; the latest weeks may be revised; no mosquito, serotype, population or travel
data; one weather point per region; the 2023–2026 test years were dominated by outbreaks.

**Code:** [{config.GITHUB_REPO_URL}]({config.GITHUB_REPO_URL})

---
**AI-use disclosure.** This project was built with the help of generative AI (Claude Code by Anthropic) for the code,
notebooks, this dashboard and documentation. The student, B. G. M. Prabuddhi, chose the problem and data, made or
approved the data and modelling decisions, and reviewed, ran and interpreted all results. Details in `AI_DISCLOSURE.md`.
""")


# ====================================================================== navigation
PAGES = {"Home": page_home, "Model comparison": page_models, "District forecast & risk": page_district,
         "Hybrid models": page_hybrids, "Explainability & what-if": page_explain, "Data explorer": page_explorer,
         "About": page_about}
st.sidebar.markdown("## 🦟 Dengue Early Warning")
st.sidebar.caption("Sri Lanka · next-week forecasts for 26 health regions")
page = st.sidebar.radio("Go to", list(PAGES), label_visibility="collapsed", key="page")
st.sidebar.divider()
st.sidebar.caption(f"Data up to week of **{fmt_date(LAST_WEEK)}**")
st.sidebar.caption(f"Forecast for week of **{fmt_date(TARGET_WEEK)}**")
st.sidebar.caption(f"Forecasting model: **{DEPLOYED}**")
PAGES[page]()
