import os
import sys
import time
import pandas as pd
import streamlit as st
import altair as alt
import yaml
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
try:
    import google.generativeai as genai  # optional, for Gemini sentiment
except Exception:
    genai = None

BASE_DIR = os.path.dirname(__file__)
PARQUET_DIR = os.path.join(BASE_DIR, "data", "parquet")
DAILY_DIR = os.path.join(PARQUET_DIR, "aggregates_daily")
TOPS_DIR = os.path.join(PARQUET_DIR, "top_posts_monthly")
POSTS_DIR = os.path.join(PARQUET_DIR, "posts")
INGEST_DIR = os.path.join(BASE_DIR, "data", "ingest")

st.set_page_config(page_title="Brand Sentiment (Reddit)", layout="wide")

@st.cache_data(show_spinner=False)
def load_daily():
    if not os.path.exists(DAILY_DIR):
        return pd.DataFrame()
    return pd.read_parquet(DAILY_DIR)

@st.cache_data(show_spinner=False)
def load_tops():
    if not os.path.exists(TOPS_DIR):
        return pd.DataFrame()
    return pd.read_parquet(TOPS_DIR)

@st.cache_data(show_spinner=False)
def load_posts_for(day: pd.Timestamp, brand: str):
    # Pull posts for that day and brand for event keyword analysis
    if not os.path.exists(POSTS_DIR):
        return pd.DataFrame()
    y = day.strftime("%Y"); m = day.strftime("%m"); d = day.strftime("%d")
    part_path = os.path.join(POSTS_DIR, f"year={y}", f"month={m}", f"day={d}", f"brand={brand}")
    if not os.path.exists(part_path):
        return pd.DataFrame()
    return pd.read_parquet(part_path)

st.title("Brand Sentiment on Reddit: Trends and Top Posts")

with st.sidebar:
    st.markdown("### Data control")
    if st.button("Refresh data cache"):
        load_daily.clear()
        load_tops.clear()
        st.success("Refreshed. Scroll to charts.")

# Data
daily = load_daily()
tops = load_tops()

if daily.empty:
    st.info("No aggregates found. Run the batch processor first: python batch_process.py")
    st.stop()

brands = sorted(daily["brand"].dropna().unique().tolist())
col1, col2, col3 = st.columns([2,2,1])
with col1:
    brand = st.selectbox("Brand", brands, index=0)
with col2:
    years = sorted(daily["year"].dropna().unique().tolist()) if "year" in daily.columns else []
    year = st.selectbox("Year", years, index=len(years)-1 if years else 0)
with col3:
    roll = st.selectbox("7-day rolling?", ["Yes", "No"], index=0)

# Filter
df = daily.copy()
if "year" in df.columns and year:
    df = df[df["year"] == year]

df = df[df["brand"] == brand]

# Prepare timeseries
df = df.rename(columns={"event_date": "date"})
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Rolling
if roll == "Yes":
    for c in ["mentions", "avg_compound", "positive", "negative", "neutral", "event_hits"]:
        if c in df.columns:
            df[c + "_roll7"] = df[c].rolling(7, min_periods=1).mean()

# For each date, compute a representative/top post (permalink + excerpt) so we can
# show the post excerpt on hover and make the point clickable (open top post in new tab).
# This is done in the dashboard (fast for modest datasets). If it becomes slow,
# we should precompute and store `top_permalink` in the batch output.
def _get_top_for_row(r):
    try:
        posts = load_posts_for(r["date"], brand)
        if posts.empty:
            return pd.Series({"top_permalink": "", "top_text": ""})
        top = posts.sort_values("score", ascending=False).iloc[0]
        permalink = top.get("permalink") if "permalink" in top else top.get("url", "")
        text = top.get("text", "")
        # simple cleanup, truncate for tooltip readability
        text = (text.replace("\n", " ").strip())[:400]
        return pd.Series({"top_permalink": permalink or "", "top_text": text})
    except Exception:
        return pd.Series({"top_permalink": "", "top_text": ""})

if not df.empty:
    # apply row-wise to add columns
    df = pd.concat([df.reset_index(drop=True), df.apply(_get_top_for_row, axis=1).reset_index(drop=True)], axis=1)

# Charts
left, right = st.columns(2)
with left:
    st.subheader("Mentions per day")
    st.caption("'Mentions per day' = number of posts/comments in the dataset that matched the brand keywords/events for that date.")
    y_col = "mentions_roll7" if roll == "Yes" and "mentions_roll7" in df else "mentions"
    # Build a line + interactive points layer. Points carry a top_permalink and top_text
    # so hovering shows the top post excerpt and clicking opens the permalink (href).
    base = alt.Chart(df).encode(x="date:T")
    line = base.mark_line().encode(
        y=alt.Y(f"{y_col}:Q", title="Mentions")
    )
    points = base.mark_point(filled=True, size=100).encode(
        y=alt.Y(f"{y_col}:Q"),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("mentions:Q", title="Mentions"),
            alt.Tooltip("event_hits:Q", title="Event hits"),
            alt.Tooltip("top_text:N", title="Top post (excerpt)")
        ],
        href=alt.Href("top_permalink:N")
    )
    layered = (line + points).properties(height=300)
    st.altair_chart(layered, use_container_width=True)

with right:
    st.subheader("Avg sentiment (compound)")
    y_col2 = "avg_compound_roll7" if roll == "Yes" and "avg_compound_roll7" in df else "avg_compound"
    # Layered chart with clickable points similar to the Mentions chart: points show top post excerpt
    base2 = alt.Chart(df).encode(x="date:T")
    line2 = base2.mark_line().encode(
        y=alt.Y(f"{y_col2}:Q", title="Compound")
    )
    points2 = base2.mark_point(filled=True, size=100).encode(
        y=alt.Y(f"{y_col2}:Q"),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("avg_compound:Q", title="Compound"),
            alt.Tooltip("top_text:N", title="Top post (excerpt)")
        ],
        href=alt.Href("top_permalink:N")
    )
    layered2 = (line2 + points2).properties(height=300)
    st.altair_chart(layered2, use_container_width=True)

st.markdown("---")

# Top posts (monthly)
if tops.empty:
    st.info("No monthly top posts found.")
else:
    st.subheader("Top posts per month")
    t = tops[tops["brand"] == brand]
    months = sorted(set(zip(t["year"], t["month"])) )
    if months:
        y0, m0 = months[-1]
        colm1, colm2 = st.columns(2)
        with colm1:
            y_sel = st.selectbox("Top posts year", sorted({y for y,_ in months}), index=sorted({y for y,_ in months}).index(y0))
        with colm2:
            m_list = sorted({m for y,m in months if y == y_sel})
            m_sel = st.selectbox("Month", m_list, index=len(m_list)-1 if m_list else 0)
        tt = t[(t["year"] == y_sel) & (t["month"] == m_sel)]
        if not tt.empty:
            cols = [
                "score", "event_ts", "subreddit", "author", "compound", "sentiment", "event_tags", "permalink", "text"
            ]
            tt = tt[cols].sort_values("score", ascending=False)
            # Clickable links
            tt["link"] = tt["permalink"].apply(lambda u: f"[Open]({u})" if isinstance(u, str) else "")
            st.dataframe(tt[["score", "event_ts", "subreddit", "compound", "sentiment", "event_tags", "link", "text"]], use_container_width=True, height=420)
        else:
            st.info("No top posts for the selected month.")

st.caption("Data source: Reddit. Processing: PySpark VADER. Dashboard: Streamlit. Parquet-backed for reliability.")

# -----------------------------
# Live tracking section
# -----------------------------
st.markdown("---")
st.header("Live tracking (every 10s)")

@st.cache_data(show_spinner=False)
def load_cfg(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_brand_keywords(cfg: dict, brand_name: str):
    brands = (cfg or {}).get("brands", [])
    target = None
    for b in brands:
        if b.get("name") == brand_name or b.get("canonical", "").lower() == str(brand_name).lower():
            target = b
            break
    if not target:
        return []
    return [k for k in target.get("keywords", []) if k]

def match_brand_text(text: str, keywords: list[str]) -> bool:
    if not text or not keywords:
        return False
    tl = str(text).lower()
    for k in keywords:
        try:
            if k and k.lower() in tl:
                return True
        except Exception:
            continue
    return False

_live_analyzer = SentimentIntensityAnalyzer()

def compute_sentiment_vader(text: str) -> float:
    if not text:
        return 0.0
    try:
        return float(_live_analyzer.polarity_scores(str(text)).get("compound", 0.0))
    except Exception:
        return 0.0

def compute_sentiment_gemini(text: str, api_key: str) -> float:
    # Very simple JSON-style prompt asking for a normalized compound score in [-1,1]
    if not text or not api_key or genai is None:
        return compute_sentiment_vader(text)
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Return only a JSON object with keys: compound (float -1..1). "
            "Task: sentiment of the following text. Text:"\
        ) + "\n" + str(text)[:4000]
        resp = model.generate_content(prompt)
        out = (resp.text or "").strip()
        # naive parse for compound
        import json, re
        comp = 0.0
        try:
            data = json.loads(out)
            comp = float(data.get("compound", 0.0))
        except Exception:
            m = re.search(r"compound\s*[:=]\s*(-?\d+\.\d+|-?\d+)", out, re.I)
            if m:
                comp = float(m.group(1))
        # clip to [-1,1]
        return max(-1.0, min(1.0, comp))
    except Exception:
        return compute_sentiment_vader(text)

def compute_sentiment(text: str) -> float:
    provider = st.session_state.get("sentiment_provider", "vader")
    if provider == "gemini":
        api_key = st.session_state.get("gemini_api_key") or os.environ.get("GOOGLE_API_KEY")
        return compute_sentiment_gemini(text, api_key)
    return compute_sentiment_vader(text)

def _to_epoch_seconds(series: pd.Series) -> pd.Series:
    """Coerce timestamps to epoch seconds.
    Tries: datetime64 -> seconds; numeric (ms or s) -> seconds; string -> parse -> seconds.
    Returns float series with NaNs where conversion failed.
    """
    # datetime64 directly
    if getattr(series.dtype, 'kind', None) == 'M':
        return (series.view('int64') // 10**9)
    # numeric path
    s_num = pd.to_numeric(series, errors='coerce')
    s_sec_from_num = s_num.where(s_num < 1e12, (s_num // 1000))
    # if some are NaN, try parse as datetime strings
    needs_dt = s_sec_from_num.isna()
    if needs_dt.any():
        parsed = pd.to_datetime(series[needs_dt], errors='coerce', utc=True)
        s_sec_from_num.loc[needs_dt] = (parsed.view('int64') // 10**9)
    return s_sec_from_num

def gather_new_since(start_epoch: int, brand_name: str, keywords: list[str]):
    # Scan NDJSON in data/ingest and pick rows with fetched_at >= start_epoch
    rows = []
    if not os.path.isdir(INGEST_DIR):
        return pd.DataFrame(rows)
    for fn in os.listdir(INGEST_DIR):
        if not fn.endswith(".json"):
            continue
        fpath = os.path.join(INGEST_DIR, fn)
        try:
            dfj = pd.read_json(fpath, lines=True)
        except Exception:
            continue
        if dfj.empty:
            continue
        # choose timestamp column
        ts_col = "fetched_at" if "fetched_at" in dfj.columns else ("created_utc" if "created_utc" in dfj.columns else None)
        if ts_col is None:
            continue
        # normalize to epoch seconds and filter
        fa_sec = _to_epoch_seconds(dfj[ts_col])
        dfj = dfj[fa_sec.fillna(0) >= float(start_epoch)]
        if dfj.empty or "text" not in dfj.columns:
            continue
        # Filter by brand keywords and compute sentiment
        dfj = dfj[dfj["text"].astype(str).apply(lambda t: match_brand_text(t, keywords))]
        if dfj.empty:
            continue
        dfj["compound"] = dfj["text"].astype(str).apply(compute_sentiment)
        rows.append(dfj)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

# Session state init
if "tracking" not in st.session_state:
    st.session_state.tracking = False
if "track_start" not in st.session_state:
    st.session_state.track_start = None
if "live_points" not in st.session_state:
    st.session_state.live_points = []  # list of dicts: {ts, mentions, avg_compound, top_permalink, top_text}

cfg = load_cfg(os.path.join(BASE_DIR, "config", "brands.yml"))
brand_keywords = get_brand_keywords(cfg, brand)

# Sentiment provider controls
if "sentiment_provider" not in st.session_state:
    st.session_state.sentiment_provider = "vader"
prov_col = st.columns([2,2,3])
with prov_col[0]:
    prov = st.selectbox("Sentiment model", ["VADER (offline)", "Gemini (API)"], index=0)
    st.session_state.sentiment_provider = "gemini" if prov.startswith("Gemini") else "vader"
with prov_col[1]:
    if st.session_state.sentiment_provider == "gemini":
        st.session_state.gemini_api_key = st.text_input("Gemini API key", type="password", value=st.session_state.get("gemini_api_key", ""))
        if genai is None:
            st.info("Install google-generativeai to use Gemini: pip install google-generativeai")

controls = st.columns([1,1,3])
with controls[0]:
    if not st.session_state.tracking:
        if st.button("Start tracking"):
            st.session_state.tracking = True
            st.session_state.track_start = int(time.time())
            st.session_state.live_points = []
            st.rerun()
    else:
        if st.button("Stop tracking"):
            st.session_state.tracking = False
            st.rerun()
with controls[1]:
    st.write("")
    st.write("Tracking: " + ("ON" if st.session_state.tracking else "OFF"))

live_container = st.container()

def append_live_point(posts_df: pd.DataFrame):
    now_ts = pd.Timestamp.utcnow()
    if posts_df is None or posts_df.empty:
        st.session_state.live_points.append({
            "ts": now_ts, "mentions": 0, "avg_compound": 0.0,
            "top_permalink": "", "top_text": ""
        })
        return
    mentions = int(len(posts_df))
    avg_compound = float(posts_df["compound"].mean()) if "compound" in posts_df.columns else 0.0
    # choose top post by score if present, else by highest compound
    order_col = "score" if "score" in posts_df.columns else "compound"
    top = posts_df.sort_values(order_col, ascending=False).iloc[0]
    permalink = None
    if isinstance(top, pd.Series):
        permalink = top.get("permalink") or top.get("url") or ""
    text = top.get("text", "") if isinstance(top, pd.Series) else ""
    text = (str(text).replace("\n", " ").strip())[:400]
    st.session_state.live_points.append({
        "ts": now_ts,
        "mentions": mentions,
        "avg_compound": avg_compound,
        "top_permalink": permalink or "",
        "top_text": text,
    })
    # Keep last ~200 points
    if len(st.session_state.live_points) > 200:
        st.session_state.live_points = st.session_state.live_points[-200:]

if st.session_state.tracking:
    # Backfill window to include very recent items even if fetched slightly before start
    bf_mins = st.sidebar.number_input("Live backfill window (mins)", min_value=0, max_value=60, value=2, step=1,
                                      help="Include posts newer than now - this window (helps when producer timing differs)")
    since_epoch = max(int(st.session_state.track_start or time.time()), int(time.time()) - bf_mins * 60)
    new_posts = gather_new_since(since_epoch, brand, brand_keywords)
    append_live_point(new_posts)

    live_df = pd.DataFrame(st.session_state.live_points)
    if not live_df.empty:
        c1, c2 = live_container.columns(2)
        with c1:
            st.subheader("Live mentions")
            base_l = alt.Chart(live_df).encode(x=alt.X("ts:T", title="Time (UTC)"))
            line_l = base_l.mark_line().encode(y=alt.Y("mentions:Q", title="Mentions/interval"))
            pts_l = base_l.mark_point(filled=True, size=90).encode(
                y=alt.Y("mentions:Q"),
                tooltip=[
                    alt.Tooltip("ts:T", title="Time"),
                    alt.Tooltip("mentions:Q", title="Mentions"),
                    alt.Tooltip("top_text:N", title="Top post (excerpt)")
                ],
                href=alt.Href("top_permalink:N")
            )
            st.altair_chart((line_l + pts_l).properties(height=280), use_container_width=True)
        with c2:
            st.subheader("Live avg sentiment")
            base_s = alt.Chart(live_df).encode(x=alt.X("ts:T", title="Time (UTC)"))
            line_s = base_s.mark_line().encode(y=alt.Y("avg_compound:Q", title="Compound"))
            pts_s = base_s.mark_point(filled=True, size=90).encode(
                y=alt.Y("avg_compound:Q"),
                tooltip=[
                    alt.Tooltip("ts:T", title="Time"),
                    alt.Tooltip("avg_compound:Q", title="Compound"),
                    alt.Tooltip("top_text:N", title="Top post (excerpt)")
                ],
                href=alt.Href("top_permalink:N")
            )
            st.altair_chart((line_s + pts_s).properties(height=280), use_container_width=True)
        # Show last top post as a regular clickable link (works even if chart clicks are blocked)
        last_nonempty = live_df[live_df["top_permalink"].astype(str) != ""]
        if not last_nonempty.empty:
            last_row = last_nonempty.iloc[-1]
            st.markdown(f"Latest top post: [Open]({last_row['top_permalink']})")
            st.caption(str(last_row.get("top_text", ""))[:300])
        else:
            st.info("No matching new posts yet. Ensure your producer is writing to data/ingest and brand keywords match.")

    # Auto-refresh every ~10 seconds while tracking
    time.sleep(10)
    st.rerun()
