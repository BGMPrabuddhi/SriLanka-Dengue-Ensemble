# AI-assistance disclosure: This file was generated with the help of Claude Code (Anthropic) and reviewed, run and tested by B. G. M. Prabuddhi.
"""Sri Lanka Dengue Early-Warning Dashboard (Streamlit).

Run from the repository root:   streamlit run app/app.py
Everything is loaded from models/ and data/processed/ - nothing is retrained at start-up.
Numbers shown on the Model Comparison page come from models/results.csv and models/summary.json.
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

st.markdown(f"""
<style>
  .block-container {{padding-top: 2rem; max-width: 1250px;}}
  h1, h2, h3 {{letter-spacing: -0.01em;}}
  .kpi {{background: #ffffff; border: 1px solid rgba(11,11,11,0.10); border-radius: 10px; padding: 14px 16px;}}
  .kpi .label {{color: {INK_2}; font-size: 0.85rem;}}
  .kpi .value {{color: {INK}; font-size: 1.75rem; font-weight: 650; line-height: 1.25;}}
  .kpi .sub {{color: {MUTED}; font-size: 0.82rem;}}
  .badge {{display: inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 650; font-size: 0.95rem;}}
  .story {{font-size: 1.05rem; color: {INK_2}; line-height: 1.6;}}
  .panel {{background: #ffffff; border: 1px solid rgba(11,11,11,0.10); border-left: 5px solid {SERIES[0]};
           border-radius: 8px; padding: 16px 20px; margin: 6px 0 14px 0;}}
</style>
""", unsafe_allow_html=True)


# ====================================================================== loading (cached)
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
        "summary": json.loads((mdir / "summary.json").read_text()),
        "registry": json.loads((mdir / "registry.json").read_text()),
        "cfg": json.loads(F.FEATURE_CONFIG_JSON.read_text()),
    }
    d = t["dengue"]
    nat = d.groupby("start_date").agg(cases=("cases", lambda s: s.sum(min_count=1)),
                                      missing=("missing_report", "sum"))
    nat.loc[nat["missing"] >= 15, "cases"] = np.nan      # national reporting gaps / unreliable weeks
    t["national"] = nat["cases"]
    return t


@st.cache_resource(show_spinner="Loading the trained model…")
def load_model(name: str):
    return M.load_predictor(name)


@st.cache_data(show_spinner=False)
def recent_thresholds(years: int = 3) -> dict:
    """Operational risk thresholds: each district's 50th / 75th percentile over the last `years` years."""
    d = load_tables()["dengue"]
    cutoff = d["start_date"].max() - pd.DateOffset(years=years)
    r = d[(d["start_date"] > cutoff) & d["cases"].notna()]
    return {k: {"medium": float(np.percentile(s, 50)), "high": float(np.percentile(s, 75))}
            for k, s in r.groupby("district")["cases"]}


def risk_of(value: float, thr: dict) -> str:
    return "High" if value > thr["high"] else "Medium" if value > thr["medium"] else "Low"


@st.cache_data(show_spinner=False)
def forecast_table(model_name: str) -> pd.DataFrame:
    t = load_tables()
    feats = t["cfg"]["numeric_features"] + t["cfg"]["categorical_features"]
    fc = t["forecast"].sort_values("district").reset_index(drop=True)
    pred = np.clip(load_model(model_name).predict(fc[feats]), 0, None)
    thr = recent_thresholds()
    out = pd.DataFrame({
        "District": fc["district"], "Last reported week": fc["cases_lag1"].round(0),
        "Forecast next week": pred.round(0),
    })
    out["Change %"] = np.where(out["Last reported week"] > 0,
                               100 * (out["Forecast next week"] - out["Last reported week"]) / out["Last reported week"], np.nan)
    out["Medium above"] = [thr[d]["medium"] for d in out["District"]]
    out["High above"] = [thr[d]["high"] for d in out["District"]]
    out["Risk"] = [risk_of(p, thr[d]) for p, d in zip(out["Forecast next week"], out["District"])]
    out["vs High threshold"] = out["Forecast next week"] / out["High above"].replace(0, np.nan)
    return out


# ====================================================================== small UI helpers
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


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def fmt_date(d) -> str:
    return pd.Timestamp(d).strftime("%d %b %Y")


T = load_tables()
S = T["summary"]
DEPLOYED = S["deployed_model"]
FEATS = T["cfg"]["numeric_features"] + T["cfg"]["categorical_features"]
LAST_WEEK = T["dengue"]["start_date"].max()
TARGET_WEEK = pd.Timestamp(T["cfg"]["forecast_target_week"])
LATE_NOTE = ("The most recent 1–2 weeks may still rise as late notifications arrive, so the latest reported "
             "numbers can be an under-count.")

# ====================================================================== sidebar
st.sidebar.markdown("## 🦟 Dengue Early Warning")
st.sidebar.caption("Sri Lanka · weekly forecasts for 26 health regions")
PAGES = ["Home", "District forecast & risk", "Model comparison", "Hybrid models", "Explainability & what-if", "About"]
page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed", key="page")
st.sidebar.divider()
st.sidebar.caption(f"Data up to week of **{fmt_date(LAST_WEEK)}**")
st.sidebar.caption(f"Forecast for week of **{fmt_date(TARGET_WEEK)}**")
st.sidebar.caption(f"Forecasting model: **{DEPLOYED}**")


# ====================================================================== pages
def page_home():
    st.title("Where will dengue strike next week?")
    st.markdown(
        '<p class="story">Dengue is one of Sri Lanka\'s biggest public-health burdens: in 2017 alone more than '
        '170,000 cases were reported. Every week, the Epidemiology Unit publishes case counts for 26 health regions. '
        'This dashboard turns those counts, plus the last two months of rainfall, humidity and temperature, into a '
        '<b>forecast for next week in every region</b> and a simple <b>Low / Medium / High risk level</b>, so resources '
        '(fogging teams, hospital beds, awareness campaigns) can be sent where they will be needed. Behind it, we compared '
        'bagging, boosting, stacking and hybrid machine-learning models to find the most reliable forecaster.</p>',
        unsafe_allow_html=True)

    nat = T["national"].dropna()
    fc = forecast_table(DEPLOYED)
    latest, prev = nat.iloc[-1], nat.iloc[-2]
    top = fc.sort_values("vs High threshold", ascending=False).iloc[0]
    n_high = int((fc["Risk"] == "High").sum())
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Cases reported, latest week", f"{latest:,.0f}", f"week of {fmt_date(nat.index[-1])}")
    kpi(c2, "Change vs previous week", f"{100 * (latest - prev) / prev:+.1f}%", f"previous week: {prev:,.0f}")
    kpi(c3, "Highest-risk region next week", top["District"],
        f"forecast {top['Forecast next week']:,.0f} cases ({top['Risk']} risk)")
    kpi(c4, "Regions at High risk next week", f"{n_high} of 26", f"week of {fmt_date(TARGET_WEEK)}")
    st.caption("⚠️ " + LATE_NOTE)

    s = T["national"]
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(color=SERIES[0], width=2),
                               name="national cases", hovertemplate="%{y:,.0f} cases<extra></extra>"))
    peak_date = s.idxmax()
    fig.add_annotation(x=peak_date, y=s.max(), text=f"2017 record outbreak<br>{s.max():,.0f} cases in one week",
                       showarrow=True, arrowhead=0, ax=60, ay=10, font=dict(color=INK_2), align="left")
    fig.add_vrect(x0=s.index[-2], x1=s.index[-1] + pd.Timedelta(days=6), fillcolor=MUTED, opacity=0.15, line_width=0,
                  annotation_text="may be incomplete", annotation_position="top left")
    st.plotly_chart(style(fig, 360, "Weekly dengue cases in Sri Lanka, 2007–2026"), use_container_width=True)
    st.caption("Gaps in the line are weeks where the source data were incomplete or unreliable (see About).")

    top10 = fc.sort_values("Forecast next week", ascending=True).tail(10)
    fig = go.Figure(go.Bar(
        y=top10["District"], x=top10["Forecast next week"], orientation="h",
        marker=dict(color=[RISK_COLORS[r] for r in top10["Risk"]]),
        text=[f"{v:,.0f} · {RISK_ICON[r]} {r}" for v, r in zip(top10["Forecast next week"], top10["Risk"])],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:,.0f} forecast cases<extra></extra>"))
    fig.update_layout(hovermode="closest")
    st.plotly_chart(style(fig, 400, f"Top 10 regions by forecast cases, week of {fmt_date(TARGET_WEEK)}"),
                    use_container_width=True)


def page_district():
    st.title("District forecast & risk")
    fc = forecast_table(DEPLOYED)
    default = fc.sort_values("vs High threshold", ascending=False).iloc[0]["District"]
    districts = sorted(fc["District"])
    district = st.selectbox("Choose a health region", districts, index=districts.index(default), key="dist_region")
    row = fc.set_index("District").loc[district]

    c1, c2, c3 = st.columns([1, 1, 1.3])
    kpi(c1, f"Last reported week ({fmt_date(LAST_WEEK)})", f"{row['Last reported week']:,.0f}", "cases")
    kpi(c2, f"Forecast, week of {fmt_date(TARGET_WEEK)}", f"{row['Forecast next week']:,.0f}",
        f"{row['Change %']:+.0f}% vs last week" if pd.notna(row["Change %"]) else "")
    c3.markdown(f'<div class="kpi"><div class="label">Risk level next week</div>'
                f'<div style="margin:8px 0">{badge(row["Risk"])}</div>'
                f'<div class="sub">Medium above {row["Medium above"]:.0f} · High above {row["High above"]:.0f} cases</div></div>',
                unsafe_allow_html=True)

    hist = T["dengue"][(T["dengue"]["district"] == district)].set_index("start_date")["cases"]
    hist = hist[hist.index > LAST_WEEK - pd.Timedelta(weeks=104)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="reported cases",
                             line=dict(color=SERIES[0], width=2), hovertemplate="%{y:,.0f}<extra>reported</extra>"))
    last_obs = hist.dropna()
    fig.add_trace(go.Scatter(x=[last_obs.index[-1], TARGET_WEEK], y=[last_obs.iloc[-1], row["Forecast next week"]],
                             mode="lines+markers", name="next-week forecast",
                             line=dict(color=RISK_COLORS[row["Risk"]], width=2, dash="dot"),
                             marker=dict(size=[0, 13], color=RISK_COLORS[row["Risk"]], line=dict(color="white", width=2)),
                             hovertemplate="%{y:,.0f}<extra>forecast</extra>"))
    for lvl, dash in [("medium", "dot"), ("high", "dash")]:
        fig.add_hline(y=row[f"{lvl.capitalize()} above"], line=dict(color=MUTED, width=1, dash=dash),
                      annotation_text=f"{lvl.capitalize()} threshold", annotation_position="top left",
                      annotation_font_color=INK_2)
    st.plotly_chart(style(fig, 380, f"{district}: last two years and next week's forecast"), use_container_width=True)
    st.info("**How risk is defined here.** Thresholds are **recent baselines**: the 50th (Medium) and 75th (High) percentile "
            "of each region's weekly cases over the **last 3 years**, so 'High' means busier than usual *for that region "
            "recently*. (Model evaluation in the notebooks used thresholds from the 2007–2021 training years only, to avoid "
            "using future information.)")
    st.caption("⚠️ " + LATE_NOTE)

    st.subheader("All 26 regions, next week")
    view = fc.sort_values("vs High threshold", ascending=False)
    fig = go.Figure(go.Bar(
        x=view["District"], y=view["vs High threshold"],
        marker=dict(color=[RISK_COLORS[r] for r in view["Risk"]]),
        customdata=np.stack([view["Forecast next week"], view["High above"], view["Risk"]], axis=1),
        hovertemplate="%{x}: forecast %{customdata[0]:,.0f} vs High threshold %{customdata[1]:.0f} "
                      "(%{customdata[2]})<extra></extra>"))
    fig.add_hline(y=1, line=dict(color=INK_2, width=1, dash="dash"), annotation_text="High-risk threshold",
                  annotation_position="top right")
    fig.update_layout(hovermode="closest", yaxis_title="forecast ÷ High threshold")
    st.plotly_chart(style(fig, 360, "Forecast relative to each region's High-risk threshold"), use_container_width=True)
    st.dataframe(
        view[["District", "Risk", "Last reported week", "Forecast next week", "Change %", "Medium above", "High above"]],
        hide_index=True, use_container_width=True,
        column_config={"Change %": st.column_config.NumberColumn(format="%+.0f%%"),
                       "Last reported week": st.column_config.NumberColumn(format="%d"),
                       "Forecast next week": st.column_config.NumberColumn(format="%d"),
                       "Medium above": st.column_config.NumberColumn(format="%.0f"),
                       "High above": st.column_config.NumberColumn(format="%.0f")})


def page_models():
    st.title("Which ensemble approach forecasts dengue best?")
    res = T["results"]
    st.markdown(f'<div class="panel"><b>Which approach performed best, and why</b><br>{S["why_best"]}</div>',
                unsafe_allow_html=True)
    with st.expander("Key findings in plain English", expanded=True):
        for k in S["key_findings"]:
            st.markdown(f"- {k}")

    naive = res.set_index("model").loc[E.NAIVE_NAME]
    st.markdown(f"All models forecast **next week's cases for each of 26 regions** and are scored on the same "
                f"**{S['n_test_rows']:,} region-weeks** from {S['test_period'][0]} to {S['test_period'][1]}, which were "
                f"never used for training. MAE = average error in cases per region-week. The simple rule *“next week = this "
                f"week”* scores **MAE {naive.MAE:.2f}**; positive *% vs naive* means better than that rule.")

    tabs = st.tabs(["Results table", "Accuracy", "Normal vs outbreak weeks", "Actual vs predicted", "The ensemble families"])
    with tabs[0]:
        show = res[["family", "model", "MAE", "RMSE", "R2", "sMAPE", "MAE_vs_naive_pct", "MAE_normal", "MAE_outbreak",
                    "val_MAE", "train_time_s"]].copy()
        winners = {DEPLOYED, S["best_on_test"]["model"]}

        def hl(r):
            return ["background-color: #e6f0fc; font-weight: 600" if r["model"] in winners else "" for _ in r]
        st.dataframe(show.style.apply(hl, axis=1).format({c: "{:.2f}" for c in show.columns if show[c].dtype != object}),
                     hide_index=True, use_container_width=True, height=560)
        st.caption(f"Highlighted: **{DEPLOYED}** (chosen by the pre-set rule: lowest 2022 validation MAE, used for the "
                   f"forecasts) and **{S['best_on_test']['model']}** (lowest test MAE). MAE_normal / MAE_outbreak split the "
                   "test error into normal weeks and outbreak weeks (cases above the region's training 75th percentile).")
    with tabs[1]:
        r = res.sort_values("MAE", ascending=False)
        c1, c2 = st.columns(2)
        fig = go.Figure(go.Bar(y=r["model"], x=r["MAE"], orientation="h",
                               marker_color=[FAMILY_COLORS[f] for f in r["family"]],
                               text=r["MAE"].map("{:.2f}".format), textposition="outside", cliponaxis=False,
                               customdata=r["family"], hovertemplate="%{y} (%{customdata}): MAE %{x:.2f}<extra></extra>"))
        fig.add_vline(x=naive.MAE, line=dict(color=MUTED, dash="dash"), annotation_text="naive")
        fig.update_layout(hovermode="closest")
        c1.plotly_chart(style(fig, 560, "Test MAE (lower is better)"), use_container_width=True)
        rr = res[res["model"] != E.NAIVE_NAME].sort_values("MAE_vs_naive_pct")
        fig = go.Figure(go.Bar(y=rr["model"], x=rr["MAE_vs_naive_pct"], orientation="h",
                               marker_color=[FAMILY_COLORS[f] for f in rr["family"]],
                               text=rr["MAE_vs_naive_pct"].map("{:+.1f}%".format), textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:+.1f}% vs naive<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=INK_2, width=1))
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(range=[max(rr["MAE_vs_naive_pct"].min(), -60) - 8, max(rr["MAE_vs_naive_pct"].max(), 0) + 12])
        c2.plotly_chart(style(fig, 560, "MAE improvement vs naive (%)"), use_container_width=True)
        st.markdown(" ".join(f'<span style="color:{c}">■</span> {f}' for f, c in FAMILY_COLORS.items()),
                    unsafe_allow_html=True)
    with tabs[2]:
        fam = S["winner_per_family"]
        names = [E.NAIVE_NAME] + [fam[f]["model"] for f in fam]
        sub = res.set_index("model").loc[names]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=names, y=sub["MAE_normal"], name="normal weeks", marker_color=SERIES[0]))
        fig.add_trace(go.Bar(x=names, y=sub["MAE_outbreak"], name="outbreak weeks", marker_color=SERIES[1]))
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05, hovermode="x unified", yaxis_title="MAE (cases)")
        st.plotly_chart(style(fig, 420, "Error in normal weeks vs outbreak weeks (best model of each family)"),
                        use_container_width=True)
        st.markdown(f"All models beat the naive rule in **normal weeks**. In **outbreak weeks** "
                    f"({S['outbreak_week_share']:.0%} of the test period) the tree models forecast too low, because cases "
                    "rise faster than the patterns learned from 2007–2021, so outbreak weeks decide the overall ranking. "
                    "Combinations that lift outbreak forecasts (stacking, the RF + neural-network hybrid) did best there.")
    with tabs[3]:
        p = T["preds"]
        c1, c2 = st.columns([1, 2])
        district = c1.selectbox("Region", sorted(p["district"].unique()), index=sorted(p["district"].unique()).index("Colombo"), key="avp_region")
        model_cols = [c for c in p.columns if c not in ("district", "target_date", "target")]
        default = [m for m in [E.NAIVE_NAME, DEPLOYED, S["best_on_test"]["model"]] if m in model_cols]
        chosen = c2.multiselect("Models (up to 4)", model_cols, default=list(dict.fromkeys(default)), max_selections=4, key="avp_models")
        q = p[p["district"] == district].sort_values("target_date")
        fig = go.Figure(go.Scatter(x=q["target_date"], y=q["target"], name="actual", line=dict(color=INK, width=2.5)))
        for i, m in enumerate(chosen):
            fig.add_trace(go.Scatter(x=q["target_date"], y=q[m], name=m, line=dict(color=SERIES[i], width=1.6)))
        st.plotly_chart(style(fig, 420, f"{district}: actual vs one-week-ahead forecasts (test period)"), use_container_width=True)
    with tabs[4]:
        cols = st.columns(3)
        for i, (fam, text) in enumerate(S["family_explanations"].items()):
            cols[i % 3].markdown(f'<div class="kpi" style="min-height:150px;border-top:4px solid {FAMILY_COLORS[fam]}">'
                                 f'<b>{fam}</b><div class="sub" style="font-size:0.92rem;color:{INK_2}">{text}</div></div>',
                                 unsafe_allow_html=True)
            cols[i % 3].write("")


def page_hybrids():
    st.title("Hybrid models")
    st.markdown("A hybrid combines two *different kinds* of model so that each covers the other's blind spot.")
    for note in S.get("hybrid_notes", []):
        st.markdown(f"- {note}")

    st.subheader("RF + SARIMA (time series)")
    st.caption("SARIMA (a statistical seasonal time-series model) forecasts the weekly series; a Random Forest then "
               "predicts SARIMA's error from recent cases and weather. Walk-forward, one week ahead, 2023–2026.")
    which = st.radio("Series", ["National", "Colombo"], horizontal=True, key="hyb_series")
    ar = T["arima_res"][T["arima_res"]["series"] == which].drop(columns="series")
    st.dataframe(ar.style.format({c: "{:.2f}" for c in ar.columns if ar[c].dtype != object}), hide_index=True,
                 use_container_width=True)
    ap = T["arima_preds"][T["arima_preds"]["series"] == which]
    fig = go.Figure(go.Scatter(x=ap["target_date"], y=ap["actual"], name="actual", line=dict(color=INK, width=2.5)))
    for m, c in [("SARIMA alone", SERIES[1]), ("RF alone", SERIES[0]), ("Hybrid RF+SARIMA", FAMILY_COLORS["Hybrid"])]:
        fig.add_trace(go.Scatter(x=ap["target_date"], y=ap[m], name=m, line=dict(color=c, width=1.6)))
    st.plotly_chart(style(fig, 400, f"{which}: one-week-ahead forecasts"), use_container_width=True)

    st.subheader("RF + ANN (all 26 regions)")
    st.caption("(a) weighted average of a Random Forest and a neural network (weight tuned on 2022); "
               "(b) the forest's forecast fed into the neural network as an extra input.")
    names = [E.NAIVE_NAME, "Random Forest", "ANN (MLP)", "Hybrid RF+ANN (weighted avg)", "Hybrid RF->ANN (RF as input)"]
    sub = T["results"].set_index("model").loc[[n for n in names if n in set(T["results"]["model"])]]
    tbl = sub[["family", "MAE", "RMSE", "MAE_vs_naive_pct", "MAE_normal", "MAE_outbreak", "val_MAE"]].reset_index()
    st.dataframe(tbl.style.format({c: "{:.2f}" for c in tbl.columns if tbl[c].dtype != object}), hide_index=True,
                 use_container_width=True)
    w = T["registry"].get("rf_ann_weighted", {}).get("weight_rf")
    if w is not None:
        st.markdown(f"Tuned weight: **{w:.2f} × Random Forest + {1 - w:.2f} × neural network**.")


def page_explain():
    st.title("What drives the forecast?")
    figs = config.FIGURES_DIR
    t1, t2, t3 = st.tabs(["SHAP (best tree model)", "Random Forest importance", "XGBoost importance"])
    t1.image(str(figs / "shap_summary.png"), use_container_width=True)
    t1.caption("Each dot is one region-week. Red = high feature value; dots to the right push the forecast up. "
               "Recent case counts dominate; rainfall and humidity about 5 and 7 weeks earlier add a smaller push.")
    t2.image(str(figs / "fi_random_forest.png"), use_container_width=True)
    t3.image(str(figs / "fi_xgboost.png"), use_container_width=True)

    st.subheader("What-if: change the inputs and see the forecast")
    fc = T["forecast"].set_index("district")
    c1, c2 = st.columns([1, 2])
    district = c1.selectbox("Region", sorted(fc.index), index=sorted(fc.index).index("Colombo"), key="wi_region")
    base = fc.loc[[district]].reset_index()
    train = pd.read_csv(F.FEATURES_CSV, usecols=["district", "cases_lag1", "rain_mm_lag8"])
    dmax = float(train.loc[train["district"] == district, "cases_lag1"].max())
    lag1 = c2.slider("Cases reported last week", 0, int(max(dmax, base["cases_lag1"].iloc[0]) * 1.2),
                     int(base["cases_lag1"].iloc[0]), key=f"wi_cases_{district}")
    rain8 = c2.slider("Rainfall 8 weeks ago (mm in that week)", 0, int(train["rain_mm_lag8"].quantile(0.995)),
                      int(round(base["rain_mm_lag8"].iloc[0])), key=f"wi_rain_{district}")
    week = c2.slider("Week of the year being forecast", 1, 52, int(base["week"].iloc[0]), key=f"wi_week_{district}")

    x = base.copy()
    x["cases_lag1"] = lag1
    lags = x[[f"cases_lag{k}" for k in range(1, 5)]].to_numpy(dtype=float)[0]
    x["rolling_mean_4"] = lags.mean()
    x["rolling_std_4"] = lags.std(ddof=1)
    x["rolling_mean_8"] = base["rolling_mean_8"] + (lag1 - base["cases_lag1"]) / 8
    x["rain_mm_lag8"] = rain8
    angle = 2 * np.pi * week / 52.18
    x["week_sin"], x["week_cos"] = np.sin(angle), np.cos(angle)

    model = load_model(DEPLOYED)
    p_base = float(np.clip(model.predict(base[FEATS]), 0, None)[0])
    p_new = float(np.clip(model.predict(x[FEATS]), 0, None)[0])
    level = risk_of(p_new, recent_thresholds()[district])
    k1, k2, k3 = st.columns(3)
    kpi(k1, "Forecast with real latest data", f"{p_base:,.0f}", f"week of {fmt_date(TARGET_WEEK)}")
    kpi(k2, "Forecast with your inputs", f"{p_new:,.0f}", f"{p_new - p_base:+,.0f} cases")
    k3.markdown(f'<div class="kpi"><div class="label">Risk with your inputs</div><div style="margin:8px 0">'
                f'{badge(level)}</div><div class="sub">recent-baseline thresholds</div></div>', unsafe_allow_html=True)
    st.caption(f"Live prediction from the saved **{DEPLOYED}** model. Changing last week's cases also updates the 4- and "
               "8-week averages consistently. Because tree models learn step-like rules, small slider moves can leave "
               "the forecast unchanged and then jump.")


def page_about():
    st.title("About this project")
    st.markdown(f"""
**Data sources**
* **Dengue cases:** weekly cases for 26 Regional Director of Health Services areas, 2006–2026, from the R package
  **denguedatahub** (Talagala, T. S., *denguedatahub: A Tidy Format Datasets of Dengue by Country*, CRAN / GitHub
  `thiyangt/denguedatahub`), compiled from the Weekly Epidemiological Reports of the Epidemiology Unit, Ministry of Health, Sri Lanka.
* **Weather:** daily rainfall, mean temperature and relative humidity at each region's main town from the
  **Open-Meteo Historical Weather API** (Zippenfenig, P., 2023, Open-Meteo.com; ERA5 reanalysis, Copernicus/ECMWF),
  aggregated to the reporting weeks.

**Method in brief**
* Each row is one forecast: standing at the end of week *t*, predict cases in week *t+1* for one region.
  Features: cases in the last 4 weeks, 4- and 8-week averages, weather 1–8 weeks earlier, week of year, region.
* Chronological split: train 2007–2021, validation 2022, test 2023–2026 (the latest 2 weeks excluded).
* Models: **bagging** (bagged trees, Random Forest, Extra Trees), **boosting** (AdaBoost, Gradient Boosting, XGBoost,
  LightGBM, CatBoost), **stacking/voting**, and **hybrids** (RF + SARIMA, RF + neural network). Tuned with time-series
  cross-validation; selected on the 2022 validation year; every model compared with the naive "same as last week" rule.
* Data problems found in the source (missing years, copied weeks, counts assigned to the wrong region) were marked
  as missing, never invented.

**Limitations:** one-week horizon only; recent weeks may be revised; no data on mosquito surveys, virus serotypes,
population or travel; one weather point per region; 2023–2026 was dominated by outbreaks.

**Code:** [{config.GITHUB_REPO_URL}]({config.GITHUB_REPO_URL})

---
**AI-use disclosure.** This project was built with the help of generative AI (Claude Code by Anthropic) for code,
notebooks, this dashboard and documentation. The student, B. G. M. Prabuddhi, chose the problem and data, made or approved
the data and modelling decisions, and reviewed, ran and interpreted all results. See `AI_DISCLOSURE.md` in the repository.
""")


{"Home": page_home, "District forecast & risk": page_district, "Model comparison": page_models,
 "Hybrid models": page_hybrids, "Explainability & what-if": page_explain, "About": page_about}[page]()
