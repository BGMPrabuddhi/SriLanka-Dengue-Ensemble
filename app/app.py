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
from src.viz import AXIS, FAMILY_COLORS, GRID, INK, INK_2, MUTED, RISK_COLORS, SERIES  # noqa: E402

st.set_page_config(page_title="Sri Lanka Dengue Early Warning", page_icon="🦟", layout="wide",
                   initial_sidebar_state="expanded")

RISK_TEXT = {"Low": "#ffffff", "Medium": "#0b0b0b", "High": "#ffffff"}
RISK_ICON = {"Low": "▼", "Medium": "■", "High": "▲"}
FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
CORE_SHORT = {"Bagging": "Random Forest", "Boosting": "XGBoost", "Stacking": "RF+XGB+LGBM+Cat → Ridge"}
FAMILY_ICON = {"Baseline": "📏", "Bagging": "🌳", "Boosting": "🚀", "Stacking": "🧱", "Neural net": "🧠", "Hybrid": "🔗"}
GOOD, BAD = "#0ca30c", "#d03b3b"

# ====================================================================== look & feel
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="st-"], .stMarkdown, button, input {{ font-family: {FONT}; }}
.block-container {{ padding-top: 3.4rem; padding-bottom: 3rem; max-width: 1220px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}

/* page header banner */
.hero {{ background: linear-gradient(120deg, #0d366b 0%, #1c5cab 55%, #2a78d6 100%); color: #fff;
        border-radius: 18px; padding: 26px 32px 24px; margin: 0 0 20px; position: relative; overflow: hidden;
        box-shadow: 0 12px 30px rgba(13, 54, 107, 0.18); }}
.hero::after {{ content: "🦟"; position: absolute; right: 28px; top: 10px; font-size: 76px; opacity: 0.12; }}
.hero .eyebrow {{ text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.72rem; font-weight: 600;
                 color: rgba(255,255,255,0.72); margin-bottom: 6px; }}
.hero h1 {{ color: #fff; font-size: 2rem; font-weight: 700; margin: 0 0 8px; padding: 0; line-height: 1.2; }}
.hero p {{ color: rgba(255,255,255,0.9); font-size: 1.02rem; line-height: 1.6; margin: 0; max-width: 900px; }}

/* cards */
.kpi {{ background: #fff; border: 1px solid #e6e9ef; border-radius: 14px; padding: 16px 18px; height: 100%;
       box-shadow: 0 2px 10px rgba(16, 24, 40, 0.05); transition: transform .15s ease, box-shadow .15s ease; }}
.kpi:hover {{ transform: translateY(-2px); box-shadow: 0 10px 24px rgba(16, 24, 40, 0.10); }}
.kpi .head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
.kpi .icon {{ width: 34px; height: 34px; border-radius: 10px; display: inline-flex; align-items: center;
             justify-content: center; font-size: 17px; background: #eaf2fc; flex: 0 0 34px; }}
.kpi .label {{ color: {INK_2}; font-size: 0.84rem; font-weight: 500; line-height: 1.3; }}
.kpi .value {{ color: {INK}; font-size: 1.75rem; font-weight: 700; line-height: 1.2; }}
.kpi .sub {{ color: {MUTED}; font-size: 0.82rem; margin-top: 4px; }}
.chip {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }}
.badge {{ display: inline-block; padding: 5px 14px; border-radius: 999px; font-weight: 700; font-size: 1rem; }}
/* st.container(border=True) -> white card. Plain wrappers (page, sidebar, columns) carry the empty style class
   st-emotion-cache-0, bordered ones get their own generated class, so exclude the empty one. */
div[data-testid="stVerticalBlockBorderWrapper"]:not(.st-emotion-cache-0) {{
       background: #fff; border-radius: 16px; border-color: #e6e9ef !important;
       box-shadow: 0 2px 12px rgba(16, 24, 40, 0.05); }}

/* section titles */
.section {{ display: flex; align-items: center; gap: 10px; margin: 26px 0 4px; }}
.section .bar {{ width: 6px; height: 24px; border-radius: 3px; background: linear-gradient(#2a78d6, #0d366b); }}
.section h3 {{ margin: 0; padding: 0; font-size: 1.28rem; font-weight: 650; color: {INK}; }}
.section-sub {{ color: {INK_2}; font-size: 0.92rem; margin: 0 0 10px 16px; }}

/* callouts */
.callout {{ display: flex; gap: 16px; align-items: center; border-radius: 14px; padding: 14px 20px; margin: 4px 0 12px;
           background: linear-gradient(90deg, #eaf6ee, #f5fbf7); border: 1px solid #cbe9d6; color: {INK}; }}
.callout .big {{ font-size: 32px; line-height: 1; }}
.note {{ border-radius: 12px; padding: 12px 16px; background: #f1f5fb; border: 1px solid #dbe5f3; color: {INK_2};
        font-size: 0.92rem; margin: 6px 0; }}
.finding {{ display: flex; gap: 14px; padding: 12px 2px; border-bottom: 1px solid #eef1f5; color: {INK}; line-height: 1.55; }}
.finding:last-child {{ border-bottom: 0; }}
.finding .num {{ flex: 0 0 30px; height: 30px; border-radius: 50%; background: linear-gradient(135deg, #2a78d6, #0d366b);
                color: #fff; font-weight: 700; display: flex; align-items: center; justify-content: center; }}
.fam {{ background: #fff; border: 1px solid #e6e9ef; border-radius: 14px; padding: 14px 16px; height: 100%;
       box-shadow: 0 2px 10px rgba(16,24,40,0.04); }}
.fam .t {{ font-weight: 650; color: {INK}; margin-bottom: 4px; }}
.fam .d {{ color: {INK_2}; font-size: 0.9rem; line-height: 1.45; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0d366b 0%, #14467f 100%); }}
section[data-testid="stSidebar"] * {{ color: #e9eff8 !important; }}
/* navigation items styled as buttons */
section[data-testid="stSidebar"] div[role="radiogroup"] {{ display: flex; flex-direction: column; gap: 8px; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
       padding: 10px 14px; border-radius: 10px; width: 100%; margin: 0; cursor: pointer;
       background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.14);
       box-shadow: 0 2px 6px rgba(0,0,0,0.18);
       transition: background .15s ease, transform .15s ease, box-shadow .15s ease; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
       background: rgba(255,255,255,0.14); transform: translateY(-1px); box-shadow: 0 6px 14px rgba(0,0,0,0.24); }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
       background: #ffffff; border-color: #ffffff; box-shadow: 0 6px 16px rgba(0,0,0,0.28); }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{ color: #0d366b !important;
       font-weight: 700; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{ display: none; }}
.brand {{ padding: 6px 4px 14px; }}
.brand .logo {{ font-size: 1.35rem; font-weight: 700; }}
.brand .tag {{ font-size: 0.82rem; opacity: 0.75; }}
.sidefoot {{ background: rgba(255,255,255,0.08); border-radius: 12px; padding: 12px 14px; font-size: 0.82rem;
            line-height: 1.6; }}
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
        height=height, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK_2, size=13), margin=dict(l=8, r=8, t=52 if title else 20, b=8),
        title=dict(text=title or "", x=0, xanchor="left", font=dict(size=15, color=INK)),   # "" = no title
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=dict(text="")),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)), hovermode="x unified")
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, showgrid=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, rangemode="tozero")
    return fig


def chart(fig, height=380, title=None):
    st.plotly_chart(style(fig, height, title), use_container_width=True, config={"displayModeBar": False})


def hero(eyebrow: str, title: str, text: str = ""):
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1>'
                f'{f"<p>{text}</p>" if text else ""}</div>', unsafe_allow_html=True)


def section(title: str, sub: str = ""):
    st.markdown(f'<div class="section"><div class="bar"></div><h3>{title}</h3></div>'
                f'{f"<div class=section-sub>{sub}</div>" if sub else ""}', unsafe_allow_html=True)


def kpi(col, icon, label, value, sub="", accent=None, tint="#eaf2fc"):
    top = f"border-top:4px solid {accent};" if accent else ""
    col.markdown(f'<div class="kpi" style="{top}"><div class="head"><span class="icon" style="background:{tint}">'
                 f'{icon}</span><span class="label">{label}</span></div><div class="value">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def badge(level: str) -> str:
    return (f'<span class="badge" style="background:{RISK_COLORS[level]};color:{RISK_TEXT[level]}">'
            f'{RISK_ICON[level]} {level} risk</span>')


def chip(text: str, good: bool) -> str:
    c = GOOD if good else BAD
    return f'<span class="chip" style="background:{c}1a;color:{c}">{text}</span>'


def note(text: str):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def md_bold(text: str) -> str:
    """Convert **bold** markdown to <b> for use inside an HTML block."""
    parts = text.split("**")
    return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))


def fmt_date(d) -> str:
    return pd.Timestamp(d).strftime("%d %b %Y")


def fmt_table(df: pd.DataFrame, highlight: set | None = None):
    """Two decimals for numbers; optional highlighted rows (by 'Model' column)."""
    sty = df.style.format({c: "{:.2f}" for c in df.columns if df[c].dtype != object})
    if highlight:
        sty = sty.apply(lambda r: ["background-color:#e6f0fc;font-weight:600" if r.get("Model") in highlight
                                   else "" for _ in r], axis=1)
    return sty


def legend(items: dict):
    st.markdown(" &nbsp;&nbsp; ".join(f'<span style="color:{c};font-size:1.1rem">■</span> {k}' for k, c in items.items()),
                unsafe_allow_html=True)


T = load_tables()
S = T["summary"]
RES = T["results"]
NAIVE = RES.set_index("model").loc[E.NAIVE_NAME]
DEPLOYED = S["deployed_model"]
BEST_TEST = S["best_on_test"]["model"]
FEATS = T["cfg"]["numeric_features"] + T["cfg"]["categorical_features"]
LAST_WEEK = T["national"].dropna().index.max()      # latest week with reliable reports
TARGET_WEEK = pd.Timestamp(T["cfg"]["forecast_target_week"])
LATE_NOTE = "The latest 1–2 weeks may still rise because some reports arrive late."
ESTIMATE_NOTE = (f"Forecasts are model estimates, not reported cases (typical error about "
                 f"±{S['overall_winner_validation']['MAE']:.0f} cases per district-week).")


# ====================================================================== pages
def page_home():
    hero("Sri Lanka · weekly dengue early warning", "Where will dengue strike next week?",
         "Dengue is one of Sri Lanka's biggest public-health problems: in 2017 alone more than 170,000 cases were "
         "reported. Cases are reported every week for 26 health regions, but the health ministry has to send fogging "
         "teams, hospital capacity and awareness campaigns <b>before</b> the next week's cases appear. This dashboard "
         "forecasts <b>next week's cases in every region</b> and compares bagging, boosting, stacking and hybrid models "
         "to find the most reliable forecaster.")

    nat = T["national"].dropna()
    fc = forecast_table(DEPLOYED)
    latest, prev = nat.iloc[-1], nat.iloc[-2]
    change = 100 * (latest - prev) / prev
    top = fc.sort_values("vs High", ascending=False).iloc[0]
    counts = fc["Risk"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "🦟", "Cases reported, latest week", f"{latest:,.0f}", f"week of {fmt_date(nat.index[-1])}")
    kpi(c2, "📉" if change < 0 else "📈", "Change vs previous week",
        f'<span style="color:{GOOD if change < 0 else BAD}">{"▼" if change < 0 else "▲"} {abs(change):.1f}%</span>',
        f"previous week: {prev:,.0f}", tint="#e9f7ee" if change < 0 else "#fbeaea")
    kpi(c3, "📍", "Highest-risk district next week", top["District"],
        f"forecast {top['Forecast next week']:,.0f} cases · {top['Risk']} risk", tint="#fff4dc")
    kpi(c4, "🚨", "Districts at High risk next week", f"{counts.get('High', 0)} of 26",
        " · ".join(f'<span style="color:{RISK_COLORS[k]}">{RISK_ICON[k]}</span> {counts.get(k, 0)} {k}'
                   for k in ["High", "Medium", "Low"]), tint="#fbeaea")
    st.caption(f"{ESTIMATE_NOTE} {LATE_NOTE}")

    with st.container(border=True):
        s = T["national"]
        fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(color=SERIES[0], width=2),
                                   fill="tozeroy", fillcolor="rgba(42,120,214,0.10)", name="national cases",
                                   hovertemplate="%{y:,.0f} cases<extra></extra>"))
        fig.add_annotation(x=s.idxmax(), y=s.max(), text=f"<b>2017 record outbreak</b><br>{s.max():,.0f} cases in one week",
                           showarrow=True, arrowhead=0, ax=80, ay=10, font=dict(color=INK), align="left",
                           bgcolor="rgba(255,255,255,0.85)")
        chart(fig, 400, "Weekly dengue cases in Sri Lanka, 2007–2026")
        st.caption("Gaps are weeks the source reported incorrectly; they are left blank rather than invented.")


def page_models():
    hero("Model comparison", "Which ensemble approach worked best, and why?",
         f"Every model forecasts next week's cases in each of the 26 regions and is scored on the same "
         f"{S['n_test_rows']:,} district-weeks ({S['test_period'][0]} to {S['test_period'][1]}) that were never used for "
         f"training. <b>MAE</b> = average error in cases per district-week (lower is better). The naive rule "
         f"<i>“next week = this week”</i> scores <b>MAE {NAIVE.MAE:.2f}</b>.")

    core = S["core_comparison"]
    best_core = min(core, key=lambda f: core[f]["MAE"])
    bt = S["best_on_test"]
    st.markdown(f'<div class="callout"><div class="big">🏆</div><div><b>{best_core}</b> was the best of the three ensemble '
                f'strategies (MAE {core[best_core]["MAE"]:.2f}, {core[best_core]["MAE_vs_naive_pct"]:+.1f}% vs naive). '
                f'Best model overall: <b>{bt["model"]}</b> ({bt["MAE_vs_naive_pct"]:+.1f}% vs naive).</div></div>',
                unsafe_allow_html=True)
    cols = st.columns(3)
    for col, fam in zip(cols, ["Bagging", "Boosting", "Stacking"]):
        r = core[fam]
        pct = r["MAE_vs_naive_pct"]
        trophy = " 🏆" if fam == best_core else ""
        detail = (chip(f"{pct:+.1f}% vs naive", pct > 0) +
                  f" &nbsp; normal weeks {r['MAE_normal']:.2f} · outbreaks {r['MAE_outbreak']:.2f}")
        kpi(col, FAMILY_ICON[fam], f"<b>{fam}</b> · {CORE_SHORT[fam]}{trophy}", f"MAE {r['MAE']:.2f}", detail,
            accent=FAMILY_COLORS[fam], tint=FAMILY_COLORS[fam] + "1f")

    section("Results by family", "Test-set errors for every model; highlighted rows are the winners.")
    with st.container(border=True):
        tbl = RES[["family", "model", "MAE", "RMSE", "R2", "MAE_vs_naive_pct"]].rename(
            columns={"family": "Family", "model": "Model", "R2": "R²", "MAE_vs_naive_pct": "% vs naive"})
        st.dataframe(fmt_table(tbl, {BEST_TEST, DEPLOYED}), hide_index=True, use_container_width=True,
                     height=36 * len(tbl) + 38)
        st.caption(f"Highlighted: **{BEST_TEST}** (lowest test error) and **{DEPLOYED}** (chosen in advance by the "
                   "2022 validation error and used for the forecasts on this dashboard).")

    section("Improvement over the naive rule", "Bars to the right of 0 beat “next week = this week”.")
    with st.container(border=True):
        r = RES[RES["model"] != E.NAIVE_NAME].sort_values("MAE_vs_naive_pct")
        fig = go.Figure(go.Bar(y=r["model"], x=r["MAE_vs_naive_pct"], orientation="h",
                               marker=dict(color=[FAMILY_COLORS[f] for f in r["family"]], cornerradius=4),
                               customdata=r["family"], text=r["MAE_vs_naive_pct"].map("{:+.1f}%".format),
                               textposition="outside", cliponaxis=False,
                               hovertemplate="%{y} (%{customdata}): %{x:+.1f}% vs naive<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=INK_2, width=1))
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(range=[max(r["MAE_vs_naive_pct"].min(), -60) - 8, r["MAE_vs_naive_pct"].max() + 8])
        chart(fig, 500)
        legend({f: c for f, c in FAMILY_COLORS.items() if f in set(RES["family"])})

    section("Where the models struggle", "Error in normal weeks vs outbreak weeks (cases above the district's usual "
            f"75th percentile, {S['outbreak_week_share']:.0%} of the test period).")
    with st.container(border=True):
        names = list(dict.fromkeys([E.NAIVE_NAME] + [core[f]["model"] for f in ["Bagging", "Boosting", "Stacking"]]
                                   + [BEST_TEST]))
        sub = RES.set_index("model").loc[names]
        labels = [n.replace(" (RF+XGB+LGBM+Cat -> Ridge)", "").replace(" (weighted avg)", "") for n in names]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=sub["MAE_normal"], name="normal weeks", marker=dict(color=SERIES[0], cornerradius=4)))
        fig.add_trace(go.Bar(x=labels, y=sub["MAE_outbreak"], name="outbreak weeks", marker=dict(color=SERIES[1], cornerradius=4)))
        fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08, yaxis_title="MAE (cases)")
        chart(fig, 360)
        st.caption("All models beat naive in normal weeks; outbreak weeks decide the ranking.")

    section("Actual vs predicted", "Pick a model and a district.")
    with st.container(border=True):
        p = T["preds"]
        models = [c for c in p.columns if c not in ("district", "target_date", "target")]
        regions = sorted(p["district"].unique())
        c1, c2 = st.columns(2)
        model = c1.selectbox("Model", models, index=models.index(BEST_TEST), key="avp_model")
        region = c2.selectbox("District", regions, index=regions.index("Colombo"), key="avp_region")
        q = p[p["district"] == region].sort_values("target_date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=q["target_date"], y=q["target"], name="actual", line=dict(color=INK, width=2.5)))
        fig.add_trace(go.Scatter(x=q["target_date"], y=q[model], name=model, line=dict(color=SERIES[0], width=2)))
        if model != E.NAIVE_NAME:
            fig.add_trace(go.Scatter(x=q["target_date"], y=q[E.NAIVE_NAME], name="naive (last week)",
                                     line=dict(color=MUTED, width=1, dash="dot")))
        chart(fig, 400, f"{region}: one-week-ahead forecasts, 2023–2026")

    section("Which approach performed best, and why")
    with st.container(border=True):
        st.markdown("".join(f'<div class="finding"><div class="num">{i}</div><div>{md_bold(t)}</div></div>'
                            for i, t in enumerate(S["dashboard_findings"], 1)), unsafe_allow_html=True)

    section("The model families in one line")
    fams = list(S["family_explanations"].items())
    for row in (fams[:3], fams[3:]):
        cols = st.columns(3)
        for col, (fam, text) in zip(cols, row):
            col.markdown(f'<div class="fam" style="border-top:4px solid {FAMILY_COLORS[fam]}"><div class="t">'
                         f'{FAMILY_ICON[fam]} {fam}</div><div class="d">{text}</div></div>', unsafe_allow_html=True)
        st.write("")


def page_district():
    hero("District forecast & risk", "Next week's outlook, district by district",
         "Pick a district to see its recent history, next week's forecast and its risk level. Risk compares the "
         "forecast with that district's own recent weeks.")
    fc = forecast_table(DEPLOYED)
    districts = sorted(fc["District"])
    default = fc.sort_values("vs High", ascending=False).iloc[0]["District"]
    district = st.selectbox("Choose a district", districts, index=districts.index(default), key="dist_region")
    row = fc.set_index("District").loc[district]

    c1, c2, c3 = st.columns(3)
    kpi(c1, "🗓️", f"Last reported week ({fmt_date(LAST_WEEK)})", f"{row['Last reported week']:,.0f}", "cases")
    ch = row["Change %"]
    kpi(c2, "🔮", f"Forecast, week of {fmt_date(TARGET_WEEK)}", f"{row['Forecast next week']:,.0f}",
        chip(f"{ch:+.0f}% vs last week", ch < 0) if pd.notna(ch) else "", tint="#efeafd")
    c3.markdown(f'<div class="kpi"><div class="head"><span class="icon" style="background:#fbeaea">🚦</span>'
                f'<span class="label">Risk next week</span></div><div style="margin:6px 0 4px">{badge(row["Risk"])}</div>'
                f'<div class="sub">4-week trend: {row["4-week trend"]}</div></div>', unsafe_allow_html=True)

    with st.container(border=True):
        hist = T["dengue"][T["dengue"]["district"] == district].set_index("start_date")["cases"]
        hist = hist[hist.index > LAST_WEEK - pd.Timedelta(weeks=104)]
        last_obs = hist.dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="reported cases",
                                 line=dict(color=SERIES[0], width=2), fill="tozeroy", fillcolor="rgba(42,120,214,0.08)",
                                 hovertemplate="%{y:,.0f}<extra>reported</extra>"))
        fig.add_trace(go.Scatter(x=[last_obs.index[-1], TARGET_WEEK], y=[last_obs.iloc[-1], row["Forecast next week"]],
                                 mode="lines+markers", name="next-week forecast",
                                 line=dict(color=RISK_COLORS[row["Risk"]], width=2.5, dash="dot"),
                                 marker=dict(size=[0, 14], color=RISK_COLORS[row["Risk"]], line=dict(color="white", width=2)),
                                 hovertemplate="%{y:,.0f}<extra>forecast</extra>"))
        for lvl, dash, pos in [("Medium", "dot", "bottom left"), ("High", "dash", "top left")]:
            fig.add_hline(y=row[f"{lvl} above"], line=dict(color=MUTED, width=1, dash=dash),
                          annotation_text=f"{lvl} threshold", annotation_position=pos, annotation_font_color=INK_2)
        chart(fig, 380, f"{district}: last two years and next week's forecast")
    note(f"ℹ️ Risk thresholds are each district's own 50th (Medium) and 75th (High) percentile of weekly cases over the "
         f"<b>last 3 years</b>. {LATE_NOTE} {ESTIMATE_NOTE}")

    section("Risk for all 26 districts next week")
    with st.container(border=True):
        view = fc.sort_values("vs High", ascending=False)
        fig = go.Figure(go.Bar(
            x=view["District"], y=view["Forecast next week"],
            marker=dict(color=[RISK_COLORS[r] for r in view["Risk"]], cornerradius=4),
            customdata=np.stack([view["Risk"], view["High above"]], axis=1),
            hovertemplate="%{x}: %{y:,.0f} forecast cases · %{customdata[0]} risk "
                          "(High above %{customdata[1]:.0f})<extra></extra>"))
        fig.update_layout(hovermode="closest", yaxis_title="forecast cases")
        chart(fig, 340)
        legend({f"{RISK_ICON[k]} {k}": RISK_COLORS[k] for k in ["High", "Medium", "Low"]})
        st.dataframe(view[["District", "Risk", "4-week trend", "Last reported week", "Forecast next week", "Change %",
                           "High above"]],
                     hide_index=True, use_container_width=True,
                     column_config={"Change %": st.column_config.NumberColumn(format="%+.0f%%"),
                                    "Last reported week": st.column_config.NumberColumn(format="%d"),
                                    "Forecast next week": st.column_config.NumberColumn(format="%d"),
                                    "High above": st.column_config.NumberColumn(format="%.0f")})


def page_hybrids():
    hero("Hybrid models", "Two different kinds of model, combined",
         "A hybrid pairs models that make different kinds of mistakes, so each can cover the other's blind spot.")
    notes = S["hybrid_notes"]

    section("RF + ARIMA", "A seasonal time-series model (SARIMA) whose errors are corrected by a Random Forest.")
    with st.container(border=True):
        which = st.radio("Series", ["National", "Colombo"], horizontal=True, key="hyb_series")
        ar = T["arima_res"][T["arima_res"]["series"] == which][["model", "MAE", "RMSE", "MAE_vs_naive_pct"]]
        ar = ar.rename(columns={"model": "Model", "MAE_vs_naive_pct": "% vs naive"})
        st.dataframe(fmt_table(ar, {"Hybrid RF+SARIMA"}), hide_index=True, use_container_width=True)
        ap = T["arima_preds"][T["arima_preds"]["series"] == which]
        fig = go.Figure(go.Scatter(x=ap["target_date"], y=ap["actual"], name="actual", line=dict(color=INK, width=2.5)))
        for m, c in [("SARIMA alone", SERIES[1]), ("RF alone", SERIES[0]), ("Hybrid RF+SARIMA", FAMILY_COLORS["Hybrid"])]:
            fig.add_trace(go.Scatter(x=ap["target_date"], y=ap[m], name=m, line=dict(color=c, width=1.7)))
        chart(fig, 380, f"{which}: one-week-ahead forecasts, 2023–2026")
    note(f"💡 {notes[0] if which == 'National' else notes[1]}")

    section("RF + ANN", "A Random Forest combined with a neural network, compared with each on its own.")
    with st.container(border=True):
        names = ["Random Forest", "ANN (MLP)", "Hybrid RF+ANN (weighted avg)", "Hybrid RF->ANN (RF as input)",
                 E.NAIVE_NAME]
        tbl = RES.set_index("model").loc[names, ["MAE", "RMSE", "R2", "MAE_vs_naive_pct"]].reset_index()
        tbl = tbl.rename(columns={"model": "Model", "R2": "R²", "MAE_vs_naive_pct": "% vs naive"})
        st.dataframe(fmt_table(tbl, {"Hybrid RF+ANN (weighted avg)"}), hide_index=True, use_container_width=True)
        sub = tbl[tbl["Model"] != E.NAIVE_NAME]
        fig = go.Figure(go.Bar(x=sub["Model"], y=sub["% vs naive"], text=sub["% vs naive"].map("{:+.1f}%".format),
                               textposition="outside", cliponaxis=False,
                               marker=dict(color=[FAMILY_COLORS["Bagging"], FAMILY_COLORS["Neural net"],
                                                  FAMILY_COLORS["Hybrid"], FAMILY_COLORS["Hybrid"]], cornerradius=4),
                               hovertemplate="%{x}: %{y:+.1f}% vs naive<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=INK_2, width=1))
        fig.update_layout(hovermode="closest")
        fig.update_yaxes(rangemode="normal", title="% vs naive")
        chart(fig, 320)
    note(f"💡 {notes[2]}<br>💡 {notes[3]}")


def page_explain():
    hero("Explainability & what-if", "What drives the forecast?",
         "Which inputs the model relies on, and a live forecast you can change with the sliders.")
    figs = config.FIGURES_DIR
    with st.container(border=True):
        st.image(str(figs / "shap_summary.png"), use_container_width=True)
        st.caption("SHAP: each dot is one district-week; dots to the right push the forecast up, red = high value. Last "
                   "week's cases and the 4-week average dominate. Weather adds a smaller push from rain and humidity 5–9 "
                   "weeks earlier (breeding, mosquito development and incubation take about two months).")
    with st.expander("Feature importance: Random Forest and XGBoost"):
        c1, c2 = st.columns(2)
        c1.image(str(figs / "fi_random_forest.png"), use_container_width=True)
        c2.image(str(figs / "fi_xgboost.png"), use_container_width=True)

    section("What-if: change the inputs, watch the forecast")
    fc = T["forecast"].set_index("district")
    with st.container(border=True):
        left, right = st.columns([1, 1.3], gap="large")
        district = left.selectbox("District", sorted(fc.index), index=sorted(fc.index).index("Colombo"), key="wi_region")
        base = fc.loc[[district]].reset_index()
        case_max, rain_max = whatif_ranges()
        lag0, rain0 = int(base["cases_lag1"].iloc[0]), int(round(base["rain_mm_lag8"].iloc[0]))
        lag1 = left.slider("Cases reported last week", 0, int(max(case_max[district], lag0) * 1.2), lag0,
                           key=f"wi_cases_{district}")
        rain8 = left.slider("Rainfall 8 weeks ago (mm in that week)", 0, int(rain_max), rain0, key=f"wi_rain_{district}")

        x = base.copy()                                      # only change what the user moved
        if lag1 != lag0:                                     # keep the 4- and 8-week averages consistent with lag 1
            delta = lag1 - float(base["cases_lag1"].iloc[0])
            x["cases_lag1"] = lag1
            x["rolling_mean_4"] = base["rolling_mean_4"] + delta / 4
            x["rolling_mean_8"] = base["rolling_mean_8"] + delta / 8
            lags = x[[f"cases_lag{k}" for k in range(1, 5)]].to_numpy(dtype=float)[0]
            x["rolling_std_4"] = np.nanstd(lags, ddof=1)
        if rain8 != rain0:
            x["rain_mm_lag8"] = rain8
        model = load_model(DEPLOYED)
        p_base = float(np.clip(model.predict(base[FEATS]), 0, None)[0])
        p_new = float(np.clip(model.predict(x[FEATS]), 0, None)[0])
        level = risk_of(p_new, recent_thresholds()[district])
        k1, k2 = right.columns(2)
        kpi(k1, "📊", "Forecast with the real data", f"{p_base:,.0f}", f"week of {fmt_date(TARGET_WEEK)}")
        kpi(k2, "🎛️", "Forecast with your inputs", f"{p_new:,.0f}",
            chip(f"{p_new - p_base:+,.0f} cases", p_new <= p_base), tint="#efeafd")
        right.markdown(f'<div style="margin-top:14px">{badge(level)}</div>', unsafe_allow_html=True)
        right.caption(f"Live prediction from the saved **{DEPLOYED}** model (an estimate, not reported cases). Tree "
                      "models learn step-like rules, so small slider moves can leave the forecast unchanged.")


def page_explorer():
    hero("Data explorer", "Reported dengue cases by year and district",
         "Filter the reported data (no model involved) to compare districts and periods.")
    d = T["dengue"]
    last_year = int(d["year"].max())
    regions_all = sorted(d["district"].unique())
    top4 = list(d.groupby("district")["cases"].sum().sort_values(ascending=False).index[:4])
    with st.container(border=True):
        c1, c2 = st.columns([1, 2], gap="large")
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
    kpi(k1, "🧮", "Reported cases", f"{reported['cases'].sum():,.0f}", f"{y0}–{y1}")
    kpi(k2, "📅", "Average per week", f"{total.mean():,.0f}", "selected districts combined")
    kpi(k3, "⛰️", "Busiest week", f"{total.max():,.0f}", f"week of {fmt_date(total.idxmax())}", tint="#fbeaea")

    with st.container(border=True):
        fig = go.Figure()
        if sel and len(sel) <= 6:
            for i, r in enumerate(sel):
                s = sub[sub["district"] == r].set_index("start_date")["cases"]
                fig.add_trace(go.Scatter(x=s.index, y=s.values, name=r, mode="lines", line=dict(color=SERIES[i], width=1.8)))
        else:
            fig.add_trace(go.Scatter(x=total.index, y=total.values, name="total", mode="lines", fill="tozeroy",
                                     fillcolor="rgba(42,120,214,0.10)", line=dict(color=SERIES[0], width=2)))
        chart(fig, 340, "Weekly reported cases" + ("" if sel and len(sel) <= 6 else " (combined)"))

    left, right = st.columns([1.15, 1], gap="medium")
    with left.container(border=True):
        tot = d[d["year"].between(y0, y1)].groupby("district")["cases"].sum().sort_values()
        fig = go.Figure(go.Bar(y=tot.index, x=tot.values, orientation="h",
                               marker=dict(color=[SERIES[0] if (not sel or r in sel) else "#c9c8c1" for r in tot.index],
                                           cornerradius=3),
                               text=[f"{v:,.0f}" for v in tot.values], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:,.0f} cases<extra></extra>"))
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(range=[0, tot.max() * 1.22])        # room for the value labels
        chart(fig, 640, f"Cases by district, {y0}–{y1}")
    with right.container(border=True):
        stats = reported.groupby("district")["cases"].agg(["sum", "mean", "max"]).reset_index()
        stats.columns = ["District", "Total", "Weekly mean", "Busiest week"]
        st.markdown("**Summary statistics**")
        st.dataframe(stats.sort_values("Total", ascending=False), hide_index=True, use_container_width=True,
                     height=min(36 * len(stats) + 38, 600),
                     column_config={"Total": st.column_config.NumberColumn(format="%d"),
                                    "Weekly mean": st.column_config.NumberColumn(format="%.1f"),
                                    "Busiest week": st.column_config.NumberColumn(format="%d")})
    if y1 == last_year:
        st.caption(f"{last_year} has data up to {fmt_date(LAST_WEEK)}. {LATE_NOTE}")


def page_about():
    hero("About", "Data, methods and limitations")
    c1, c2 = st.columns(2, gap="medium")
    with c1.container(border=True):
        st.markdown("""
#### 📦 Data sources
* **Dengue cases:** weekly cases for 26 health regions (RDHS areas), 2006–2026, from the R package **denguedatahub**
  (Talagala, T. S., GitHub `thiyangt/denguedatahub`), compiled from the Weekly Epidemiological Reports of the
  Epidemiology Unit, Ministry of Health, Sri Lanka.
* **Weather:** daily rainfall, temperature and humidity for each region from the **Open-Meteo Historical Weather API**
  (Zippenfenig, P., 2023; ERA5 reanalysis), aggregated to the reporting weeks.

#### ⚠️ Limitations
One-week horizon only; the latest weeks may be revised; no mosquito, serotype, population or travel data; one weather
point per region; the 2023–2026 test years were dominated by outbreaks.
""")
    with c2.container(border=True):
        st.markdown("""
#### 🛠️ Methods
* Each forecast stands at the end of week *t* and predicts cases in week *t+1* for one region, from the last 4 weeks of
  cases, 4- and 8-week averages, weather 1–8 weeks earlier, week of year and region.
* Time-based split: train 2007–2021, validation 2022, test 2023–2026; tuning with time-series cross-validation.
* **Bagging** (bagged trees, Random Forest, Extra Trees), **boosting** (AdaBoost, Gradient Boosting, XGBoost,
  LightGBM, CatBoost), **stacking** and voting, **hybrids** (RF + SARIMA, RF + neural network), all compared with the
  naive "same as last week" rule.
* Reporting errors in the source (missing years, copied weeks, counts in the wrong region) were left blank, never invented.
""")
    st.markdown(f'<div class="note">💻 <b>Code:</b> <a href="{config.GITHUB_REPO_URL}">{config.GITHUB_REPO_URL}</a></div>',
                unsafe_allow_html=True)
    note("🤖 <b>AI-use disclosure.</b> This project was built with the help of generative AI (Claude Code by Anthropic) "
         "for the code, notebooks, this dashboard and documentation. The student, B. G. M. Prabuddhi, chose the problem "
         "and data, made or approved the data and modelling decisions, and reviewed, ran and interpreted all results. "
         "Details in <code>AI_DISCLOSURE.md</code>.")


# ====================================================================== navigation
PAGES = {"Home": page_home, "Model comparison": page_models, "District forecast & risk": page_district,
         "Hybrid models": page_hybrids, "Explainability & what-if": page_explain, "Data explorer": page_explorer,
         "About": page_about}
st.sidebar.markdown('<div class="brand"><div class="logo">Dengue Early Warning</div>'
                    '<div class="tag">Sri Lanka · next-week forecasts for 26 regions</div></div>', unsafe_allow_html=True)
page = st.sidebar.radio("Go to", list(PAGES), label_visibility="collapsed", key="page")
st.sidebar.markdown(f'<div class="sidefoot">Data up to <b>{fmt_date(LAST_WEEK)}</b><br>'
                    f'Forecast for <b>{fmt_date(TARGET_WEEK)}</b><br>Model: <b>{DEPLOYED}</b></div>',
                    unsafe_allow_html=True)
PAGES[page]()
