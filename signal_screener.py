"""
Signal Screener - Detect financial inflection points from quarterly data.
Data: SEC EDGAR XBRL API (free, 17+ years history) with yfinance fallback.
Computation: pure pandas, no AI needed for signal detection.

Modes:
  python signal_screener.py NVDA AAPL            # existing: terminal output
  python signal_screener.py scan --min-revenue 5M # full scan: all SEC companies
  python signal_screener.py update --days 3       # incremental: recent filings
  python signal_screener.py report                # generate HTML from cache
  python signal_screener.py filter --severity HIGH # query cached data with filters
"""

import argparse
import sys
import math
import json
import os
import random
import time
import tempfile
import threading
import urllib.request
import urllib.error
import webbrowser
from datetime import datetime, timedelta

# ── ANSI colors (Windows compatible) ──────────────────────────────────
if sys.platform == "win32":
    import ctypes
    k = ctypes.windll.kernel32
    k.SetConsoleMode(k.GetStdHandle(-11), 7)
    sys.stdout.reconfigure(encoding="utf-8")

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")
FACTS_CACHE_DIR = os.path.join(CACHE_DIR, "facts")
RESULTS_CACHE_DIR = os.path.join(CACHE_DIR, "results")
SCAN_HISTORY_FILE = os.path.join(CACHE_DIR, "scan_history.json")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports")
SEC_UA = "SignalScreener/1.0 (research@example.com)"
FACTS_CACHE_TTL = 30 * 86400  # 30 days

# ── Signal thresholds ─────────────────────────────────────────────────
THRESHOLDS = {
    "incr_margin_yoy": 0.10,       # YoY incremental OPM > 10pp above trailing avg
    "rev_growth_accel": 0.05,      # YoY revenue growth acceleration > 5pp
    "gross_margin_inflect": 0.02,  # Gross margin reversal > 2pp
    "op_leverage": 1.5,            # Rev growth / OpEx growth ratio
    "fcf_margin_improve": 0.05,    # FCF margin improvement > 5pp
}

SUBCOMMANDS = {"scan", "update", "report", "filter", "serve"}


# ══════════════════════════════════════════════════════════════════════
#  Rate limiter
# ══════════════════════════════════════════════════════════════════════

class RateLimiter:
    """Rate limiter with random jitter. Thread-safe, avoids SEC throttling."""

    def __init__(self, base_interval=0.3, jitter=0.2):
        self._base = base_interval   # avg ~3 req/s
        self._jitter = jitter        # +/- random variation
        self._lock = threading.Lock()
        self._last = 0.0
        self._count = 0

    def wait(self):
        with self._lock:
            self._count += 1
            # Every 50 requests, take a longer pause (2-4s)
            if self._count % 50 == 0:
                time.sleep(2.0 + random.random() * 2.0)
                self._last = time.monotonic()
                return

            interval = self._base + random.uniform(-self._jitter, self._jitter)
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed < interval:
                time.sleep(interval - elapsed)
            self._last = time.monotonic()


_rate_limiter = RateLimiter(base_interval=1.5, jitter=0.8)


# ══════════════════════════════════════════════════════════════════════
#  Safe JSON I/O helpers
# ══════════════════════════════════════════════════════════════════════

def _safe_json_load(path):
    """Load JSON from file. Returns None on any error (corrupt, missing)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        # Corrupt cache — delete it
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def _atomic_json_write(path, data):
    """Write JSON atomically: write to .tmp then rename."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        # On Windows, os.replace handles atomic overwrite
        os.replace(tmp_path, path)
    except OSError:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


# ══════════════════════════════════════════════════════════════════════
#  Formatting helpers
# ══════════════════════════════════════════════════════════════════════

def fmt_pct(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:+.1%}" if v >= 0 else f"{v:.1%}"


def fmt_b(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    av = abs(v)
    sign = "-" if v < 0 else ""
    if av >= 1e9:
        return f"{sign}${av/1e9:.1f}B"
    if av >= 1e6:
        return f"{sign}${av/1e6:.0f}M"
    return f"{sign}${av:,.0f}"


def _parse_revenue_str(s):
    """Parse revenue strings like '5M', '1B', '500K' into numeric values."""
    s = s.strip().upper()
    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9}
    if s[-1] in multipliers:
        return float(s[:-1]) * multipliers[s[-1]]
    return float(s)


# ══════════════════════════════════════════════════════════════════════
#  SEC EDGAR data layer
# ══════════════════════════════════════════════════════════════════════

def _sec_get(url, retries=3):
    """HTTP GET with SEC-required User-Agent, rate limiting, and retry."""
    _rate_limiter.wait()
    req = urllib.request.Request(url, headers={"User-Agent": SEC_UA})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 404:
                return None
            if e.code == 429:
                wait = 2 ** attempt + 1
                time.sleep(wait)
                continue
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, OSError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            raise
    raise last_err


def _sec_get_text(url, retries=3):
    """HTTP GET returning raw text (for index files)."""
    _rate_limiter.wait()
    req = urllib.request.Request(url, headers={"User-Agent": SEC_UA})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 404:
                return None
            if e.code == 429:
                wait = 2 ** attempt + 1
                time.sleep(wait)
                continue
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, OSError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            raise
    raise last_err


def _load_ticker_cik_map():
    """Load SEC ticker->CIK mapping. Cached locally for 7 days."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, "company_tickers.json")

    # Use cache if fresh
    if os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < 7 * 86400:
            data = _safe_json_load(cache_file)
            if data:
                return data

    raw = _sec_get("https://www.sec.gov/files/company_tickers.json")
    # Reshape: {ticker: cik_str(zero-padded)}
    mapping = {}
    for k, v in raw.items():
        mapping[v["ticker"].upper()] = str(v["cik_str"]).zfill(10)
    _atomic_json_write(cache_file, mapping)
    return mapping


def _build_cik_to_ticker_map():
    """Build reverse CIK->ticker mapping."""
    ticker_cik = _load_ticker_cik_map()
    return {cik: ticker for ticker, cik in ticker_cik.items()}


# ── Facts caching ────────────────────────────────────────────────────

def _get_cached_facts(cik):
    """Load cached SEC facts for a CIK. Returns None if missing or stale."""
    os.makedirs(FACTS_CACHE_DIR, exist_ok=True)
    path = os.path.join(FACTS_CACHE_DIR, f"CIK{cik}.json")
    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < FACTS_CACHE_TTL:
            return _safe_json_load(path)
    return None


def _save_cached_facts(cik, data):
    """Save SEC facts to cache."""
    os.makedirs(FACTS_CACHE_DIR, exist_ok=True)
    path = os.path.join(FACTS_CACHE_DIR, f"CIK{cik}.json")
    _atomic_json_write(path, data)


# ── Results caching ──────────────────────────────────────────────────

def _make_serializable(obj):
    """Convert an object to JSON-serializable form (handle datetime, etc)."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(x) for x in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def _save_result(ticker, data):
    """Save computed result (company data + signals) to cache."""
    os.makedirs(RESULTS_CACHE_DIR, exist_ok=True)
    path = os.path.join(RESULTS_CACHE_DIR, f"{ticker.upper()}.json")
    _atomic_json_write(path, _make_serializable(data))


def _load_result(ticker):
    """Load cached result for a ticker."""
    path = os.path.join(RESULTS_CACHE_DIR, f"{ticker.upper()}.json")
    return _safe_json_load(path)


# ── Scan history ─────────────────────────────────────────────────────

def _load_scan_history():
    """Load scan history. Returns list of scan records."""
    data = _safe_json_load(SCAN_HISTORY_FILE)
    return data if isinstance(data, list) else []


def _append_scan_history(record):
    """Append a scan record to history."""
    history = _load_scan_history()
    history.append(record)
    _atomic_json_write(SCAN_HISTORY_FILE, history)


# ══════════════════════════════════════════════════════════════════════
#  Quarterly data extraction
# ══════════════════════════════════════════════════════════════════════

def _extract_quarterly(entries, n_quarters=20):
    """Extract single-quarter values from SEC XBRL entries.

    SEC data contains both quarterly and cumulative (YTD) values.
    Single quarters have ~80-100 day periods.
    Q4 is not reported directly -- must be computed as FY - Q1 - Q2 - Q3.
    """
    quarterly = {}  # end_date -> value
    annual = {}     # end_date -> value
    ytd_by_end = {} # end_date -> {days: value}

    for e in entries:
        start = e.get("start")
        end = e.get("end")
        val = e.get("val")
        if not start or not end or val is None:
            continue

        days = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days

        # Single quarter: ~80-100 days
        if 75 <= days <= 105:
            if end not in quarterly:
                quarterly[end] = val

        # Full year: ~355-370 days
        elif 350 <= days <= 375:
            if end not in annual:
                annual[end] = val

        # YTD periods (for Q4 computation): store all
        if end not in ytd_by_end:
            ytd_by_end[end] = {}
        ytd_by_end[end][days] = val

    # Compute Q4 = FY - (Q1 + Q2 + Q3) = FY - 9-month YTD
    for fy_end, fy_val in annual.items():
        if fy_end in quarterly:
            continue  # already have Q4
        # Look for 9-month YTD ending ~90 days before FY end
        fy_date = datetime.strptime(fy_end, "%Y-%m-%d")
        # Find 9-month cumulative (260-280 days) in the same fiscal year
        for end_date, periods in ytd_by_end.items():
            for days, val in periods.items():
                if 255 <= days <= 285:
                    # Check if this YTD is from the same fiscal year
                    ytd_end = datetime.strptime(end_date, "%Y-%m-%d")
                    if abs((fy_date - ytd_end).days) < 100 and ytd_end < fy_date:
                        q4_val = fy_val - val
                        quarterly[fy_end] = q4_val
                        break
            if fy_end in quarterly:
                break

    # Sort by date, return most recent n_quarters
    sorted_q = sorted(quarterly.items(), key=lambda x: x[0])
    return sorted_q[-n_quarters:]


def _extract_quarterly_cashflow(facts_us_gaap, concept, n_quarters=20):
    """Extract quarterly cash flow data. Cash flow in 10-Q is always YTD cumulative,
    so we need to subtract previous quarter YTD to get single quarter values."""
    if concept not in facts_us_gaap:
        return {}

    entries = facts_us_gaap[concept].get("units", {}).get("USD", [])
    if not entries:
        return {}

    # Collect all entries by (end_date, period_days)
    by_end = {}
    annual = {}
    for e in entries:
        start = e.get("start")
        end = e.get("end")
        val = e.get("val")
        if not start or not end or val is None:
            continue
        days = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
        key = (end, days)
        if key not in by_end:
            by_end[key] = val
        if 350 <= days <= 375:
            annual[end] = val

    # Group by fiscal year start (look for entries with same start date)
    # Strategy: for each fiscal year, collect Q1(~90d), H1(~180d), 9M(~270d), FY(~365d)
    # Then: Q1=Q1, Q2=H1-Q1, Q3=9M-H1, Q4=FY-9M
    by_start = {}
    for e in entries:
        start = e.get("start")
        end = e.get("end")
        val = e.get("val")
        if not start or not end or val is None:
            continue
        days = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
        if start not in by_start:
            by_start[start] = []
        by_start[start].append({"end": end, "days": days, "val": val})

    quarterly = {}
    for start, periods in by_start.items():
        # Sort by duration
        periods.sort(key=lambda x: x["days"])
        # Deduplicate by days bucket
        buckets = {}
        for p in periods:
            if 75 <= p["days"] <= 105:
                buckets["Q1"] = p
            elif 170 <= p["days"] <= 195:
                buckets["H1"] = p
            elif 255 <= p["days"] <= 285:
                buckets["9M"] = p
            elif 350 <= p["days"] <= 375:
                buckets["FY"] = p

        if "Q1" in buckets:
            quarterly[buckets["Q1"]["end"]] = buckets["Q1"]["val"]
        if "H1" in buckets and "Q1" in buckets:
            quarterly[buckets["H1"]["end"]] = buckets["H1"]["val"] - buckets["Q1"]["val"]
        if "9M" in buckets and "H1" in buckets:
            quarterly[buckets["9M"]["end"]] = buckets["9M"]["val"] - buckets["H1"]["val"]
        if "FY" in buckets and "9M" in buckets:
            quarterly[buckets["FY"]["end"]] = buckets["FY"]["val"] - buckets["9M"]["val"]

    sorted_q = sorted(quarterly.items(), key=lambda x: x[0])
    return dict(sorted_q[-n_quarters:])


def _assess_data_quality(quarters):
    """Assess data completeness and continuity for quality reporting."""
    n = len(quarters)
    if n == 0:
        return {"n_quarters": 0, "coverage": {}, "gaps": [], "warnings": ["No data"]}

    # Coverage per metric
    metric_keys = [
        ("revenue", "revenue"),
        ("gross_profit", "gross_profit"),
        ("op_income", "operating_income"),
        ("fcf", "fcf"),
    ]
    coverage = {}
    for label, key in metric_keys:
        count = sum(1 for q in quarters if q.get(key) is not None)
        coverage[label] = (count, n)

    # Check continuity -- consecutive quarters should be ~90 days apart
    gaps = []
    for i in range(1, n):
        d_curr = quarters[i]["date"]
        d_prev = quarters[i - 1]["date"]
        if isinstance(d_curr, str):
            d_curr = datetime.fromisoformat(d_curr)
        if isinstance(d_prev, str):
            d_prev = datetime.fromisoformat(d_prev)
        diff = (d_curr - d_prev).days
        if diff > 120:  # > 4 months = missing quarter(s)
            gaps.append((quarters[i - 1]["quarter_label"], quarters[i]["quarter_label"], diff))

    # Build human-readable warnings
    warnings = []
    for label, (count, total) in coverage.items():
        if count == 0:
            warnings.append(f"{label}: no data")
        elif count < total * 0.8:  # less than 80% coverage
            warnings.append(f"{label}: {count}/{total}Q")

    if gaps:
        for prev_q, next_q, days in gaps:
            missed = round(days / 90) - 1
            warnings.append(f"gap: {missed}Q missing between {prev_q} and {next_q}")

    return {
        "n_quarters": n,
        "coverage": coverage,
        "gaps": gaps,
        "warnings": warnings,
    }


def _find_entries(us_gaap, concepts):
    """Merge entries from all matching concepts, dedup by (start, end).
    Companies change concept names over time (e.g., SalesRevenueNet -> RevenueFromContract...),
    so a single concept may not cover all quarters."""
    seen = set()
    merged = []
    for c in concepts:
        if c in us_gaap:
            entries = us_gaap[c].get("units", {}).get("USD", [])
            for e in entries:
                key = (e.get("start", ""), e.get("end", ""))
                if key not in seen:
                    seen.add(key)
                    merged.append(e)
    return merged


# ══════════════════════════════════════════════════════════════════════
#  Data fetching (SEC)
# ══════════════════════════════════════════════════════════════════════

def fetch_data_sec(ticker=None, cik=None, n_quarters=20, use_cache=True,
                   force_refresh=False, skip_market_cap=False):
    """Fetch quarterly data from SEC EDGAR XBRL API.

    Args:
        ticker: Ticker symbol (used for CIK lookup if cik not given)
        cik: CIK string (zero-padded). If given, ticker lookup is skipped.
        n_quarters: Number of quarters to extract
        use_cache: Use cached facts if available
        force_refresh: Force re-download even if cache exists
        skip_market_cap: Skip yfinance market cap lookup (for batch scans)
    """
    if cik is None:
        if ticker is None:
            return None
        cik_map = _load_ticker_cik_map()
        cik = cik_map.get(ticker.upper())
        if not cik:
            return None

    if ticker is None:
        # Reverse lookup
        cik_to_ticker = _build_cik_to_ticker_map()
        ticker = cik_to_ticker.get(cik, f"CIK{cik}")

    # Try cache first
    facts = None
    if use_cache and not force_refresh:
        facts = _get_cached_facts(cik)

    if facts is None:
        facts = _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
        if facts is None:
            return None
        if use_cache:
            _save_cached_facts(cik, facts)

    entity_name = facts.get("entityName", ticker)
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    if not us_gaap:
        return None

    # ── Income statement items ────────────────────────────────────
    rev_concepts = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                    "RevenueFromContractWithCustomerIncludingAssessedTax",
                    "SalesRevenueNet", "SalesRevenueGoodsNet"]
    cogs_concepts = ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
                     "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"]
    gp_concepts = ["GrossProfit"]
    oi_concepts = ["OperatingIncomeLoss"]
    ni_concepts = ["NetIncomeLoss"]
    opex_concepts = ["OperatingExpenses", "CostsAndExpenses"]
    rd_concepts = ["ResearchAndDevelopmentExpense"]
    sga_concepts = ["SellingGeneralAndAdministrativeExpense"]

    rev_q = dict(_extract_quarterly(_find_entries(us_gaap, rev_concepts), n_quarters))
    cogs_q = dict(_extract_quarterly(_find_entries(us_gaap, cogs_concepts), n_quarters))
    gp_q = dict(_extract_quarterly(_find_entries(us_gaap, gp_concepts), n_quarters))
    oi_q = dict(_extract_quarterly(_find_entries(us_gaap, oi_concepts), n_quarters))
    ni_q = dict(_extract_quarterly(_find_entries(us_gaap, ni_concepts), n_quarters))
    opex_q = dict(_extract_quarterly(_find_entries(us_gaap, opex_concepts), n_quarters))
    rd_q = dict(_extract_quarterly(_find_entries(us_gaap, rd_concepts), n_quarters))
    sga_q = dict(_extract_quarterly(_find_entries(us_gaap, sga_concepts), n_quarters))

    # ── Cash flow items ───────────────────────────────────────────
    ocf_concepts = [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        "NetCashProvidedByOperatingActivities",
        "NetCashProvidedByOperatingActivitiesContinuingOperations",
    ]
    ocf_q = {}
    for c in ocf_concepts:
        ocf_q = _extract_quarterly_cashflow(us_gaap, c, n_quarters)
        if ocf_q:
            break
    capex_q = _extract_quarterly_cashflow(us_gaap,
                "PaymentsToAcquirePropertyPlantAndEquipment", n_quarters)

    # ── Assemble quarterly data ───────────────────────────────────
    all_dates = sorted(set(rev_q.keys()))
    if not all_dates:
        return None

    data = []
    for d in all_dates:
        dt = datetime.strptime(d, "%Y-%m-%d")
        rev = rev_q.get(d)
        cogs = cogs_q.get(d)
        gp = gp_q.get(d)
        oi = oi_q.get(d)
        ocf = ocf_q.get(d)
        capex = capex_q.get(d)

        # Compute gross profit if not directly available
        if gp is None and rev is not None and cogs is not None:
            gp = rev - cogs

        # Compute OpEx from components if not directly available
        opex = opex_q.get(d)
        if opex is None and gp is not None and oi is not None:
            opex = gp - oi

        # FCF = OCF - CapEx
        fcf = None
        if ocf is not None and capex is not None:
            fcf = ocf - capex  # capex is already positive in SEC data
        elif ocf is not None:
            fcf = ocf  # if no capex data, just use OCF as approximation

        row = {
            "date": dt,
            "quarter_label": dt.strftime("%Y-%m") + f" (Q{(dt.month-1)//3+1})",
            "revenue": rev,
            "gross_profit": gp,
            "operating_income": oi,
            "net_income": ni_q.get(d),
            "opex": opex,
            "cogs": cogs,
            "rd": rd_q.get(d),
            "sga": sga_q.get(d),
            "fcf": fcf,
            "ocf": ocf,
            "capex": -capex if capex else None,  # normalize to negative
        }
        data.append(row)

    # Get market cap from yfinance (quick, just info)
    market_cap = None
    if not skip_market_cap:
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
            market_cap = info.get("marketCap")
        except Exception:
            pass

    # ── Data quality assessment ────────────────────────────────
    data_quality = _assess_data_quality(data)

    return {
        "ticker": ticker,
        "name": entity_name,
        "market_cap": market_cap,
        "quarters": data,
        "source": "SEC EDGAR",
        "data_quality": data_quality,
    }


def fetch_data_yfinance(ticker):
    """Fallback: fetch from yfinance (5 quarters only)."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    info = t.info or {}
    company_name = info.get("shortName", info.get("longName", ticker))
    market_cap = info.get("marketCap", None)

    qf = t.quarterly_financials
    qcf = t.quarterly_cashflow
    if qf is None or qf.empty:
        return None

    qf = qf[sorted(qf.columns)]
    if qcf is not None and not qcf.empty:
        qcf = qcf[sorted(qcf.columns)]

    def safe_get(df, label, col):
        try:
            import pandas as pd
            v = df.loc[label, col]
            return None if pd.isna(v) else float(v)
        except (KeyError, TypeError):
            return None

    data = []
    for q in qf.columns:
        row = {
            "date": q,
            "quarter_label": q.strftime("%Y-%m") + f" (Q{(q.month-1)//3+1})",
            "revenue": safe_get(qf, "Total Revenue", q),
            "gross_profit": safe_get(qf, "Gross Profit", q),
            "operating_income": safe_get(qf, "Operating Income", q),
            "net_income": safe_get(qf, "Net Income", q),
            "opex": safe_get(qf, "Operating Expense", q),
            "cogs": safe_get(qf, "Cost Of Revenue", q),
            "rd": safe_get(qf, "Research And Development", q),
            "sga": safe_get(qf, "Selling General And Administration", q),
            "fcf": safe_get(qcf, "Free Cash Flow", q) if qcf is not None and q in qcf.columns else None,
            "ocf": safe_get(qcf, "Operating Cash Flow", q) if qcf is not None and q in qcf.columns else None,
            "capex": safe_get(qcf, "Capital Expenditure", q) if qcf is not None and q in qcf.columns else None,
        }
        data.append(row)

    return {
        "ticker": ticker,
        "name": company_name,
        "market_cap": market_cap,
        "quarters": data,
        "source": "yfinance",
    }


def fetch_data(ticker, n_quarters=20, source="auto"):
    """Fetch data with auto source selection: SEC EDGAR first, yfinance fallback."""
    if source in ("sec", "auto"):
        try:
            data = fetch_data_sec(ticker=ticker, n_quarters=n_quarters)
            if data and len(data["quarters"]) >= 4:
                return data
        except Exception as e:
            if source == "sec":
                raise
            # Fall through to yfinance

    if source in ("yf", "auto"):
        try:
            data = fetch_data_yfinance(ticker)
            if data:
                data["source"] = "yfinance"
                return data
        except Exception:
            pass

    return None


# ══════════════════════════════════════════════════════════════════════
#  Signal computation
# ══════════════════════════════════════════════════════════════════════

def compute_signals(company_data):
    """Compute all signal indicators from quarterly data."""
    quarters = company_data["quarters"]
    n = len(quarters)
    signals = []

    def _q_date(q):
        """Get date from quarter, handling both datetime and string."""
        d = q["date"]
        if isinstance(d, str):
            return datetime.fromisoformat(d)
        return d

    def _find_yoy_idx(i):
        """Find index of quarter ~1 year before quarters[i], or None.
        Handles data gaps by checking actual dates instead of assuming i-4."""
        target = _q_date(quarters[i])
        best = None
        for j in range(i - 1, -1, -1):
            diff = (target - _q_date(quarters[j])).days
            if 330 <= diff <= 400:
                if best is None or abs(diff - 365) < abs((target - _q_date(quarters[best])).days - 365):
                    best = j
            elif diff > 400:
                break
        return best

    for i in range(n):
        q = quarters[i]
        rev = q["revenue"]
        gp = q["gross_profit"]
        oi = q["operating_income"]
        fcf = q.get("fcf")

        # Basic margins (use `is not None` to handle zero values correctly)
        q["gross_margin"] = gp / rev if rev is not None and rev != 0 and gp is not None else None
        q["op_margin"] = oi / rev if rev is not None and rev != 0 and oi is not None else None
        q["fcf_margin"] = fcf / rev if rev is not None and rev != 0 and fcf is not None else None

        if i == 0:
            continue

        prev = quarters[i - 1]
        prev_rev = prev["revenue"]
        prev_oi = prev["operating_income"]

        # ── Signal 0: Operating Profit Turned Positive ─────────
        curr_opm = q.get("op_margin")
        prev_opm = prev.get("op_margin")
        if curr_opm is not None and prev_opm is not None:
            if prev_opm < 0 and curr_opm > 0:
                signals.append({
                    "quarter": q["quarter_label"],
                    "signal": "OP MARGIN TURNED POSITIVE (QoQ)",
                    "detail": f"Op margin: {fmt_pct(prev_opm)} -> {fmt_pct(curr_opm)}",
                    "severity": "HIGH",
                })
        # YoY turn positive
        yoy_i = _find_yoy_idx(i)
        if yoy_i is not None:
            prev_y_opm_val = quarters[yoy_i].get("op_margin")
            if curr_opm is not None and prev_y_opm_val is not None:
                if prev_y_opm_val < 0 and curr_opm > 0:
                    signals.append({
                        "quarter": q["quarter_label"],
                        "signal": "OP MARGIN TURNED POSITIVE (YoY)",
                        "detail": f"Op margin: {fmt_pct(prev_y_opm_val)} (year-ago) -> {fmt_pct(curr_opm)}",
                        "severity": "HIGH",
                    })

        # ── Signal 1: YoY Incremental Operating Margin ──────────
        #    Only meaningful when BOTH this quarter and year-ago quarter have positive OP margin
        if yoy_i is not None:
            prev_y = quarters[yoy_i]
            prev_y_rev = prev_y.get("revenue")
            prev_y_oi = prev_y.get("operating_income")
            prev_y_opm = prev_y.get("op_margin")
            if rev is not None and prev_y_rev is not None and oi is not None and prev_y_oi is not None:
                delta_rev_yoy = rev - prev_y_rev
                delta_oi_yoy = oi - prev_y_oi
                if abs(delta_rev_yoy) > 1e6:
                    incr_margin_yoy = delta_oi_yoy / delta_rev_yoy
                    q["incr_op_margin_yoy"] = incr_margin_yoy

                    # Only signal when both quarters have positive op margin
                    if (prev_y_opm is not None and prev_y_opm > 0
                            and curr_opm is not None and curr_opm > 0
                            and incr_margin_yoy > prev_y_opm + THRESHOLDS["incr_margin_yoy"]):
                        signals.append({
                            "quarter": q["quarter_label"],
                            "signal": "INCREMENTAL MARGIN (YoY)",
                            "detail": f"YoY Incr OPM: {fmt_pct(incr_margin_yoy)} vs year-ago OPM: {fmt_pct(prev_y_opm)}",
                            "severity": "HIGH" if incr_margin_yoy > prev_y_opm + 0.20 else "MEDIUM",
                        })

        # ── Signal 2: YoY Revenue Growth + Acceleration ────────────
        # Compute YoY growth (uses date-based lookup to handle data gaps)
        if yoy_i is not None:
            r0 = rev
            r4 = quarters[yoy_i].get("revenue")
            if r0 is not None and r4 is not None and r4 > 0:
                q["rev_growth_yoy"] = (r0 - r4) / r4

        # Compute YoY acceleration (this Q's YoY vs prev Q's YoY)
        prev_q_yoy_i = _find_yoy_idx(i - 1) if i >= 1 else None
        if yoy_i is not None and prev_q_yoy_i is not None:
            r0 = rev
            r4 = quarters[yoy_i].get("revenue")
            r1 = quarters[i - 1].get("revenue") if i >= 1 else None
            r5 = quarters[prev_q_yoy_i].get("revenue")
            if r0 is not None and r4 is not None and r1 is not None and r5 is not None and r4 > 0 and r5 > 0:
                yoy_curr = (r0 - r4) / r4
                yoy_prev = (r1 - r5) / r5
                yoy_accel = yoy_curr - yoy_prev
                q["rev_growth_yoy_accel"] = yoy_accel

                if yoy_accel > THRESHOLDS["rev_growth_accel"]:
                    signals.append({
                        "quarter": q["quarter_label"],
                        "signal": "YoY REVENUE GROWTH ACCELERATION",
                        "detail": f"YoY growth: {fmt_pct(yoy_curr)} vs prior quarter YoY: {fmt_pct(yoy_prev)} (accel: {fmt_pct(yoy_accel)})",
                        "severity": "HIGH" if yoy_accel > 0.15 else "MEDIUM",
                    })
                elif yoy_accel < -THRESHOLDS["rev_growth_accel"]:
                    signals.append({
                        "quarter": q["quarter_label"],
                        "signal": "YoY REVENUE GROWTH DECELERATION",
                        "detail": f"YoY growth: {fmt_pct(yoy_curr)} vs prior quarter YoY: {fmt_pct(yoy_prev)} (decel: {fmt_pct(yoy_accel)})",
                        "severity": "WARNING",
                    })

        # ── Signal 3: Gross Margin Inflection ─────────────────────
        if i >= 2:
            gm_curr = q.get("gross_margin")
            gm_prev = prev.get("gross_margin")
            gm_prev2 = quarters[i - 2].get("gross_margin")
            if gm_curr is not None and gm_prev is not None and gm_prev2 is not None:
                was_declining = gm_prev < gm_prev2
                now_improving = gm_curr > gm_prev
                improvement = gm_curr - gm_prev
                if was_declining and now_improving and improvement > THRESHOLDS["gross_margin_inflect"]:
                    signals.append({
                        "quarter": q["quarter_label"],
                        "signal": "GROSS MARGIN INFLECTION",
                        "detail": f"GM: {fmt_pct(gm_prev2)} -> {fmt_pct(gm_prev)} -> {fmt_pct(gm_curr)} (reversal {fmt_pct(improvement)})",
                        "severity": "HIGH" if improvement > 0.05 else "MEDIUM",
                    })

        # ── Signal 4: Operating Leverage ──────────────────────────
        if (rev is not None and prev_rev is not None and prev_rev != 0
                and q.get("opex") is not None and prev.get("opex") is not None and prev["opex"] != 0):
            rev_g = (rev - prev_rev) / prev_rev
            opex_g = (q["opex"] - prev["opex"]) / prev["opex"]
            if rev_g > 0.01:
                q["op_leverage_ratio"] = rev_g / opex_g if opex_g > 0.001 else float('inf')
                if opex_g > 0 and rev_g / opex_g > THRESHOLDS["op_leverage"]:
                    signals.append({
                        "quarter": q["quarter_label"],
                        "signal": "OPERATING LEVERAGE",
                        "detail": f"Rev growth {fmt_pct(rev_g)} vs OpEx growth {fmt_pct(opex_g)} (leverage: {rev_g/opex_g:.1f}x)",
                        "severity": "MEDIUM",
                    })

        # ── Signal 5: FCF Margin Improvement ──────────────────────
        fcf_m = q.get("fcf_margin")
        prev_fcf_m = prev.get("fcf_margin")
        if fcf_m is not None and prev_fcf_m is not None:
            fcf_improve = fcf_m - prev_fcf_m
            q["fcf_margin_change"] = fcf_improve

            if prev_fcf_m < 0 and fcf_m > 0:
                signals.append({
                    "quarter": q["quarter_label"],
                    "signal": "FCF TURNED POSITIVE",
                    "detail": f"FCF margin: {fmt_pct(prev_fcf_m)} -> {fmt_pct(fcf_m)}",
                    "severity": "HIGH",
                })
            elif fcf_improve > THRESHOLDS["fcf_margin_improve"]:
                signals.append({
                    "quarter": q["quarter_label"],
                    "signal": "FCF MARGIN EXPANSION",
                    "detail": f"FCF margin: {fmt_pct(prev_fcf_m)} -> {fmt_pct(fcf_m)} ({fmt_pct(fcf_improve)})",
                    "severity": "MEDIUM",
                })

    # Newest signals first
    signals.reverse()
    return signals


# ══════════════════════════════════════════════════════════════════════
#  Terminal output
# ══════════════════════════════════════════════════════════════════════

def print_company_report(company_data, signals, show_n=12):
    """Print formatted report for one company."""
    ticker = company_data["ticker"]
    name = company_data["name"]
    mcap = company_data.get("market_cap")
    quarters = company_data["quarters"]
    source = company_data.get("source", "?")

    mcap_str = fmt_b(mcap) if mcap else "N/A"
    print(f"\n{'='*80}")
    print(f"{BOLD}{CYAN} {ticker} -- {name}  (Mkt Cap: {mcap_str})  [{source}: {len(quarters)}Q]{RESET}")
    print(f"{'='*80}")

    # Data quality indicator
    quality = company_data.get("data_quality")
    if quality and quality.get("warnings"):
        print(f"  {YELLOW}[!] Data quality: {'; '.join(quality['warnings'])}{RESET}")

    # Show last N quarters
    display_q = quarters[-show_n:]

    print(f"\n{BOLD}Quarterly Data (last {len(display_q)} quarters):{RESET}")
    headers = ["Quarter", "Revenue", "Gross%", "OpMar%", "FCF%", "RevGr YoY", "RevAcc YoY", "IncrOPM YoY"]
    widths = [16, 12, 9, 9, 9, 12, 12, 13]
    header_line = "".join(h.rjust(w) for h, w in zip(headers, widths))
    print(f"{DIM}{header_line}{RESET}")
    print(f"{DIM}{'---'*sum(widths)}{RESET}")

    for q in display_q:
        cols = [
            q["quarter_label"],
            fmt_b(q["revenue"]),
            fmt_pct(q.get("gross_margin")) if q.get("gross_margin") is not None else "",
            fmt_pct(q.get("op_margin")) if q.get("op_margin") is not None else "",
            fmt_pct(q.get("fcf_margin")) if q.get("fcf_margin") is not None else "",
            fmt_pct(q.get("rev_growth_yoy")) if q.get("rev_growth_yoy") is not None else "",
            fmt_pct(q.get("rev_growth_yoy_accel")) if q.get("rev_growth_yoy_accel") is not None else "",
            fmt_pct(q.get("incr_op_margin_yoy")) if q.get("incr_op_margin_yoy") is not None else "",
        ]
        line = "".join(str(c).rjust(w) for c, w in zip(cols, widths))
        print(line)

    if signals:
        # Only show signals from displayed quarters
        display_dates = {q["quarter_label"] for q in display_q}
        visible_signals = [s for s in signals if s["quarter"] in display_dates]
        if visible_signals:
            print(f"\n{BOLD}{YELLOW}>> SIGNALS ({len(visible_signals)}):{RESET}")
            for s in visible_signals:
                sev = s["severity"]
                if sev == "HIGH":
                    color, icon = RED, "[!!]"
                elif sev == "WARNING":
                    color, icon = YELLOW, "[!]"
                else:
                    color, icon = GREEN, "[+]"
                print(f"  {icon} {color}[{s['quarter']}] {s['signal']}{RESET}")
                print(f"      {s['detail']}")
    else:
        print(f"\n{DIM}No significant signals detected.{RESET}")


def print_summary(all_results):
    """Print summary across all companies."""
    flagged = [(r, s) for r, s in all_results if s]
    if not flagged:
        print(f"\n{DIM}No signals detected across all companies.{RESET}")
        return

    print(f"\n{'='*80}")
    print(f"{BOLD}{CYAN} SIGNAL SUMMARY{RESET}")
    print(f"{'='*80}")

    def score(item):
        return sum(1 for s in item[1] if s["severity"] == "HIGH") * 10 + len(item[1])
    flagged.sort(key=score, reverse=True)

    for r, sigs in flagged:
        high = sum(1 for s in sigs if s["severity"] == "HIGH")
        med = sum(1 for s in sigs if s["severity"] == "MEDIUM")
        warn = sum(1 for s in sigs if s["severity"] == "WARNING")
        parts = []
        if high:
            parts.append(f"{RED}{high} HIGH{RESET}")
        if med:
            parts.append(f"{GREEN}{med} MEDIUM{RESET}")
        if warn:
            parts.append(f"{YELLOW}{warn} WARNING{RESET}")
        print(f"  {BOLD}{r['ticker']:<8}{RESET} {', '.join(parts)}")
        for s in sigs:
            if s["severity"] == "HIGH":
                print(f"           {RED}-> [{s['quarter']}] {s['signal']}: {s['detail']}{RESET}")


# ══════════════════════════════════════════════════════════════════════
#  Progress tracker (for scan/update modes)
# ══════════════════════════════════════════════════════════════════════

class ProgressTracker:
    """Progress bar with ETA, printed to stderr."""

    def __init__(self, total, label="Processing"):
        self.total = total
        self.label = label
        self.current = 0
        self.start_time = time.monotonic()
        self.skipped = 0
        self.passed = 0
        self.errors = 0

    def update(self, status="ok"):
        self.current += 1
        if status == "skip":
            self.skipped += 1
        elif status == "pass":
            self.passed += 1
        elif status == "error":
            self.errors += 1
        self._draw()

    def _draw(self):
        elapsed = time.monotonic() - self.start_time
        if self.current > 0 and elapsed > 0:
            rate = self.current / elapsed
            eta_sec = (self.total - self.current) / rate if rate > 0 else 0
            eta_str = f"ETA {int(eta_sec//60)}m{int(eta_sec%60):02d}s"
        else:
            eta_str = "ETA --"

        pct = self.current / self.total * 100 if self.total else 0
        bar_len = 30
        filled = int(bar_len * self.current / self.total) if self.total else 0
        bar = "#" * filled + "-" * (bar_len - filled)

        line = (f"\r  {self.label}: [{bar}] {pct:5.1f}% "
                f"({self.current}/{self.total}) "
                f"pass={self.passed} skip={self.skipped} err={self.errors} "
                f"{eta_str}    ")
        sys.stderr.write(line)
        sys.stderr.flush()

    def finish(self):
        elapsed = time.monotonic() - self.start_time
        sys.stderr.write(
            f"\r  {self.label}: Done. {self.current}/{self.total} processed "
            f"({self.passed} passed, {self.skipped} skipped, {self.errors} errors) "
            f"in {int(elapsed//60)}m{int(elapsed%60):02d}s\n"
        )
        sys.stderr.flush()


# ══════════════════════════════════════════════════════════════════════
#  Scan mode (Phase 3)
# ══════════════════════════════════════════════════════════════════════

def _process_company(cik, ticker, min_revenue, force_refresh=False):
    """Fetch, filter, compute signals for one company.

    Returns:
        (result_dict, status) where status is "pass", "skip", or "error"
        result_dict is None for skip/error.
    """
    try:
        company_data = fetch_data_sec(
            ticker=ticker, cik=cik, n_quarters=20,
            use_cache=True, force_refresh=force_refresh,
            skip_market_cap=True,
        )
        if company_data is None:
            return None, "skip"

        quarters = company_data["quarters"]
        if len(quarters) < 4:
            return None, "skip"

        # Revenue filter: check max recent quarterly revenue
        recent_revs = [q["revenue"] for q in quarters[-4:] if q.get("revenue") is not None]
        if not recent_revs:
            return None, "skip"
        max_rev = max(recent_revs)
        if max_rev < min_revenue:
            return None, "skip"

        signals = compute_signals(company_data)

        result = {
            "ticker": ticker,
            "name": company_data.get("name", ticker),
            "quarters": company_data["quarters"],
            "signals": signals,
            "data_quality": company_data.get("data_quality"),
            "market_cap": company_data.get("market_cap"),
            "source": company_data.get("source", "SEC EDGAR"),
            "last_updated": datetime.now().isoformat(),
        }

        _save_result(ticker, result)
        return result, "pass"

    except Exception as e:
        return None, "error"


def _run_scan_mode(args):
    """Full scan: iterate all SEC companies, filter, compute signals."""
    min_revenue = _parse_revenue_str(args.min_revenue)
    max_companies = args.max_companies

    print(f"{BOLD}{CYAN}Signal Screener -- Full Scan Mode{RESET}")
    print(f"  Min quarterly revenue: {fmt_b(min_revenue)}")
    if max_companies > 0:
        print(f"  Max companies: {max_companies} (testing mode)")
    print()

    cik_map = _load_ticker_cik_map()
    all_items = list(cik_map.items())  # [(ticker, cik), ...]
    if max_companies > 0:
        all_items = all_items[:max_companies]

    total = len(all_items)
    progress = ProgressTracker(total, label="Scanning")
    companies_with_signals = []
    total_passed = 0

    try:
        for idx, (ticker, cik) in enumerate(all_items):
            result, status = _process_company(cik, ticker, min_revenue,
                                              force_refresh=args.force_refresh)
            if status == "pass":
                total_passed += 1
                if result and result.get("signals"):
                    companies_with_signals.append({
                        "ticker": result["ticker"],
                        "name": result["name"],
                        "signals": _make_serializable(result["signals"]),
                        "data_quality": _make_serializable(result.get("data_quality")),
                    })
            progress.update(status)

    except KeyboardInterrupt:
        sys.stderr.write("\n  Interrupted! Saving partial results...\n")

    progress.finish()

    # Save scan history
    record = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "mode": "scan",
        "total_scanned": progress.current,
        "passed_filter": total_passed,
        "companies_with_signals": companies_with_signals,
    }
    _append_scan_history(record)

    # Print summary
    print(f"\n{BOLD}Scan Complete:{RESET}")
    print(f"  Scanned: {progress.current}/{total}")
    print(f"  Passed filter: {total_passed}")
    print(f"  Companies with signals: {len(companies_with_signals)}")

    if companies_with_signals:
        print(f"\n{BOLD}{YELLOW}Top companies with signals:{RESET}")
        # Sort by signal count (HIGH first)
        def _score(c):
            sigs = c.get("signals", [])
            return sum(1 for s in sigs if s.get("severity") == "HIGH") * 10 + len(sigs)
        companies_with_signals.sort(key=_score, reverse=True)
        for c in companies_with_signals[:20]:
            sigs = c.get("signals", [])
            high = sum(1 for s in sigs if s.get("severity") == "HIGH")
            med = sum(1 for s in sigs if s.get("severity") == "MEDIUM")
            parts = []
            if high:
                parts.append(f"{RED}{high}H{RESET}")
            if med:
                parts.append(f"{GREEN}{med}M{RESET}")
            print(f"  {BOLD}{c['ticker']:<8}{RESET} {c['name'][:40]:<40} {', '.join(parts)}")

    # Auto-generate master report unless --no-report
    if not args.no_report:
        print()
        _generate_master_report()


# ══════════════════════════════════════════════════════════════════════
#  Update mode (Phase 4)
# ══════════════════════════════════════════════════════════════════════

def _fetch_recent_filers(days=3):
    """Fetch recent 10-Q/10-K filers from SEC RSS feed.

    Returns list of {cik, ticker, form_type, filed_date}.
    """
    import re
    import xml.etree.ElementTree as ET

    cik_to_ticker = _build_cik_to_ticker_map()
    filers = []
    seen_ciks = set()
    target_forms = {"10-Q", "10-K", "10-Q/A", "10-K/A"}

    cutoff = datetime.now() - timedelta(days=days)

    # Fetch from SEC RSS feed for each form type
    for form_type in ["10-Q", "10-K"]:
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type={form_type}&company=&dateb=&owner=include&count=100&output=atom"

        try:
            _rate_limiter.wait()
            req = urllib.request.Request(url, headers={
                "User-Agent": SEC_UA,
                "Accept": "application/atom+xml,application/xml,text/xml,*/*"
            })
            with urllib.request.urlopen(req, timeout=20) as resp:
                xml_text = resp.read().decode("utf-8", errors="replace")
        except Exception:
            continue

        # Parse Atom XML
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            continue

        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):
            try:
                title = entry.find("atom:title", ns)
                if title is None or title.text is None:
                    continue

                # Title format: "10-Q - Company Name (CIK) (Filer)"
                title_text = title.text
                # Extract form type from title
                form_match = re.match(r"(10-[QK](/A)?)\s*-", title_text)
                if not form_match:
                    continue
                filing_form = form_match.group(1)
                if filing_form not in target_forms:
                    continue

                # Extract CIK from title (number in parentheses)
                cik_match = re.search(r"\((\d{7,10})\)", title_text)
                if not cik_match:
                    continue
                cik_raw = cik_match.group(1)
                cik_padded = cik_raw.zfill(10)

                if cik_padded in seen_ciks:
                    continue

                # Get filing date from <updated> element
                updated = entry.find("atom:updated", ns)
                if updated is not None and updated.text:
                    # Format: 2026-02-02T17:00:39-05:00
                    filed_date = updated.text[:10]
                    try:
                        filed_dt = datetime.strptime(filed_date, "%Y-%m-%d")
                        if filed_dt < cutoff:
                            continue
                    except ValueError:
                        pass
                else:
                    filed_date = datetime.now().strftime("%Y-%m-%d")

                ticker = cik_to_ticker.get(cik_padded, None)
                if ticker is None:
                    continue

                seen_ciks.add(cik_padded)
                filers.append({
                    "cik": cik_padded,
                    "ticker": ticker,
                    "form_type": filing_form,
                    "filed_date": filed_date,
                })

            except Exception:
                continue

    return filers


def _run_update_mode(args):
    """Incremental update: find recent filers, re-fetch, detect new signals."""
    days = args.days
    min_revenue = _parse_revenue_str(args.min_revenue)
    show_all = getattr(args, 'all', False)

    print(f"{BOLD}{CYAN}Signal Screener -- Incremental Update Mode{RESET}")
    print(f"  Looking back: {days} days")
    print(f"  Min quarterly revenue: {fmt_b(min_revenue)}")
    if show_all:
        print(f"  Mode: Show ALL signals from recent filers")
    print()

    print(f"{DIM}Fetching recent filers from SEC RSS feed...{RESET}")
    filers = _fetch_recent_filers(days)
    print(f"  Found {len(filers)} companies with recent 10-Q/10-K filings")

    if not filers:
        print(f"\n{DIM}No recent filings found.{RESET}")
        return

    progress = ProgressTracker(len(filers), label="Updating")
    signals_summary = []
    total_passed = 0

    try:
        for filer in filers:
            ticker = filer["ticker"]
            cik = filer["cik"]

            # Load old result for comparison (only needed if not --all)
            old_signal_keys = set()
            if not show_all:
                old_result = _load_result(ticker)
                if old_result and old_result.get("signals"):
                    for s in old_result["signals"]:
                        old_signal_keys.add((s.get("quarter", ""), s.get("signal", "")))

            # Re-fetch with force refresh
            result, status = _process_company(cik, ticker, min_revenue=min_revenue,
                                              force_refresh=True)
            if status == "pass" and result:
                total_passed += 1
                all_sigs = result.get("signals", [])

                if show_all:
                    # Show ALL signals from this company
                    if all_sigs:
                        signals_summary.append({
                            "ticker": ticker,
                            "name": result.get("name", ticker),
                            "form_type": filer["form_type"],
                            "filed_date": filer["filed_date"],
                            "display_signals": all_sigs,
                            "all_signals": all_sigs,
                        })
                else:
                    # Only show NEW signals (not in old cache)
                    new_sigs = []
                    for s in all_sigs:
                        key = (s.get("quarter", ""), s.get("signal", ""))
                        if key not in old_signal_keys:
                            new_sigs.append(s)

                    if new_sigs:
                        signals_summary.append({
                            "ticker": ticker,
                            "name": result.get("name", ticker),
                            "form_type": filer["form_type"],
                            "filed_date": filer["filed_date"],
                            "display_signals": new_sigs,
                            "all_signals": all_sigs,
                        })

            progress.update(status)

    except KeyboardInterrupt:
        sys.stderr.write("\n  Interrupted! Saving partial results...\n")

    progress.finish()

    # Save scan history
    companies_with_signals = []
    for item in signals_summary:
        companies_with_signals.append({
            "ticker": item["ticker"],
            "name": item["name"],
            "signals": _make_serializable(item["display_signals"]),
            "form_type": item["form_type"],
            "filed_date": item["filed_date"],
        })

    record = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "mode": "update" + (" --all" if show_all else ""),
        "days_back": days,
        "total_scanned": progress.current,
        "passed_filter": total_passed,
        "companies_with_signals": companies_with_signals,
    }
    _append_scan_history(record)

    # Print summary
    label = "ALL signals" if show_all else "NEW signals"
    print(f"\n{BOLD}Update Complete:{RESET}")
    print(f"  Processed: {progress.current}/{len(filers)}")
    print(f"  Successfully updated: {total_passed}")
    print(f"  Companies with {label}: {len(signals_summary)}")

    if signals_summary:
        print(f"\n{BOLD}{YELLOW}{'Signals' if show_all else 'New signals'} detected:{RESET}")
        for item in signals_summary:
            print(f"\n  {BOLD}{item['ticker']}{RESET} ({item['name'][:40]}) "
                  f"-- filed {item['form_type']} on {item['filed_date']}")
            for s in item["display_signals"]:
                sev = s.get("severity", "MEDIUM")
                if sev == "HIGH":
                    color, icon = RED, "[!!]"
                elif sev == "WARNING":
                    color, icon = YELLOW, "[!]"
                else:
                    color, icon = GREEN, "[+]"
                print(f"    {icon} {color}[{s['quarter']}] {s['signal']}{RESET}")
                print(f"        {s['detail']}")

    # Append to query history and update reports unless --no-report
    if not args.no_report and signals_summary:
        print()
        # Prepare data for query history
        query_companies = []
        for item in signals_summary:
            # Load full quarters data for the company
            result = _load_result(item["ticker"])
            quarters = result.get("quarters", [])[-8:] if result else []
            query_companies.append({
                "ticker": item["ticker"],
                "name": item["name"],
                "signals": item["display_signals"],
                "quarters": quarters,
                "filed_date": item.get("filed_date", ""),
            })

        cmd = f"update --days {days}" + (" --all" if show_all else "")
        _append_query(cmd, query_companies)

        # Also update master report
        _generate_master_report()


def _run_review_mode(args):
    """Review mode: show important signals from companies that filed in past N days.

    Approach A: Query recent filers, show their HIGH/MEDIUM signals (not just "new" ones).
    This gives a clear view of "what's interesting among companies that recently filed".
    """
    days = args.days
    min_revenue = _parse_revenue_str(args.min_revenue)
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    print(f"{BOLD}{CYAN}Signal Screener -- Review Mode{RESET}")
    print(f"  Looking back: {days} days (filed since {cutoff_str})")
    print(f"  Min quarterly revenue: {fmt_b(min_revenue)}")
    print(f"  Showing: All signals from recent quarters")
    print()

    # ── Step 1: Fetch recent filers from SEC ──
    print(f"{DIM}Fetching recent filers from SEC...{RESET}")
    filers = _fetch_recent_filers(days)
    print(f"  Found {len(filers)} companies with recent 10-Q/10-K filings")

    if not filers:
        print(f"\n{DIM}No recent filings found.{RESET}")
        return

    # ── Step 2: Process each filer, collect those with important signals ──
    progress = ProgressTracker(len(filers), label="Processing")
    companies_with_signals = []
    total_passed = 0

    try:
        for filer in filers:
            ticker = filer["ticker"]
            cik = filer["cik"]

            # Process company (uses cache if available, or fetches fresh)
            result, status = _process_company(cik, ticker, min_revenue=min_revenue,
                                              force_refresh=False)  # Use cache to speed up
            if status == "pass" and result:
                total_passed += 1
                all_sigs = result.get("signals", [])
                quarters = result.get("quarters", [])

                # Get the latest 2 quarter labels (the recently filed report should be one of these)
                recent_quarter_labels = set()
                if quarters:
                    for q in quarters[-2:]:  # Last 2 quarters
                        qlabel = q.get("quarter_label", "")
                        if qlabel:
                            recent_quarter_labels.add(qlabel)

                # Filter signals: only from the most recent 2 quarters
                # Include all severity levels (HIGH, MEDIUM, WARNING) for complete picture
                recent_sigs = [
                    s for s in all_sigs
                    if s.get("quarter", "") in recent_quarter_labels
                ]

                if recent_sigs:
                    companies_with_signals.append({
                        "ticker": ticker,
                        "name": result.get("name", ticker),
                        "form_type": filer["form_type"],
                        "filed_date": filer["filed_date"],
                        "signals": recent_sigs,
                        "quarters": quarters[-8:],  # Last 8 quarters for display
                    })

            progress.update(status)

    except KeyboardInterrupt:
        sys.stderr.write("\n  Interrupted! Showing partial results...\n")

    progress.finish()

    if not companies_with_signals:
        print(f"\n{DIM}No companies with signals (filed in past {days} days).{RESET}")
        return

    # ── Step 3: Display results ──
    print(f"\n{BOLD}{YELLOW}Companies with signals (filed in past {days} days): {len(companies_with_signals)}{RESET}")

    # Sort by filed_date (most recent first), then by signal count
    companies_with_signals.sort(
        key=lambda x: (x["filed_date"], len(x["signals"])),
        reverse=True
    )

    for company in companies_with_signals:
        ticker = company["ticker"]
        name = company["name"][:40]
        form_type = company["form_type"]
        filed_date = company["filed_date"]

        print(f"\n  {BOLD}{ticker}{RESET} ({name}) -- Filed {form_type} on {filed_date}")

        # Show signals
        print(f"    {CYAN}Signals:{RESET}")
        for s in company["signals"]:
            sev = s.get("severity", "MEDIUM")
            if sev == "HIGH":
                color, icon = RED, "[!!]"
            else:
                color, icon = GREEN, "[+]"
            print(f"      {icon} {color}[{s.get('quarter', '')}] {s.get('signal', '')}{RESET}")
            detail = s.get("detail", "")
            if detail:
                print(f"          {detail}")

        # Show quarterly data
        quarters = company.get("quarters", [])
        if quarters:
            print(f"    {CYAN}Quarterly Data (last {len(quarters)}Q):{RESET}")
            print(f"      {'Quarter':<12} {'Revenue':>10} {'RevGr%':>8} {'GM%':>7} {'OPM%':>7} {'IncrOPM':>8} {'FCF%':>7}")
            print(f"      {'-'*12} {'-'*10} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")
            for q in reversed(quarters):
                qlabel = q.get("quarter_label", "")[:12]
                rev = q.get("revenue")
                rev_str = fmt_b(rev) if rev else "-"
                rev_g = q.get("rev_growth_yoy")
                rev_g_str = f"{rev_g*100:+.1f}%" if rev_g is not None else "-"
                gm = q.get("gross_margin")
                gm_str = f"{gm*100:.1f}%" if gm is not None else "-"
                op_m = q.get("op_margin")
                op_m_str = f"{op_m*100:.1f}%" if op_m is not None else "-"
                incr_opm = q.get("incr_op_margin_yoy")
                incr_str = f"{incr_opm*100:.1f}%" if incr_opm is not None else "-"
                fcf_m = q.get("fcf_margin")
                fcf_str = f"{fcf_m*100:.1f}%" if fcf_m is not None else "-"
                print(f"      {qlabel:<12} {rev_str:>10} {rev_g_str:>8} {gm_str:>7} {op_m_str:>7} {incr_str:>8} {fcf_str:>7}")

    # Append to query history and update reports unless --no-report
    if not args.no_report and companies_with_signals:
        print()
        # Append to query history
        cmd = f"update --days {days} --review"
        _append_query(cmd, companies_with_signals)

        # Also keep the dedicated review report for backwards compatibility
        _generate_review_report_v2(days, companies_with_signals, cutoff_str)

        # Update master report
        _generate_master_report()


def _generate_review_report_v2(days, companies_with_signals, cutoff_str):
    """Generate HTML report for review mode (Approach A: important signals from recent filers)."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    output_path = os.path.join(REPORTS_DIR, f"signal_review_{days}d.html")

    # Build companies_list for HTML template
    companies_list = []
    for company in companies_with_signals:
        companies_list.append({
            "ticker": company["ticker"],
            "name": company["name"],
            "signals": _make_serializable(company["signals"]),
            "filed_date": company["filed_date"],
            "form_type": company["form_type"],
            "quarters": _make_serializable(company.get("quarters", [])),
        })

    # Create a pseudo scan_history entry for the HTML template
    review_entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "mode": f"review --days {days}",
        "days_back": days,
        "total_scanned": len(companies_with_signals),
        "passed_filter": len(companies_with_signals),
        "companies_with_signals": companies_list,
    }

    # Build all_results dict so HTML template can access quarters data
    all_results = {}
    for company in companies_list:
        ticker = company["ticker"]
        all_results[ticker] = {
            "name": company["name"],
            "signals": company["signals"],
            "quarters": company.get("quarters", []),
        }

    data = {
        "review_mode": True,
        "days_back": days,
        "cutoff_date": cutoff_str,
        "generated": datetime.now().isoformat(),
        "n_companies": len(companies_with_signals),
        "companies": companies_list,
        "scan_history": [review_entry],
        "all_results": all_results,
    }

    data_json = json.dumps(data, default=str)
    html = _HTML_TEMPLATE.replace("/*DATA_PLACEHOLDER*/", data_json)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"{GREEN}Review report generated: {output_path}{RESET}")

    try:
        webbrowser.open(f"file://{os.path.abspath(output_path)}")
        print(f"{DIM}Opened in browser.{RESET}")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
#  HTML Report (Phase 5)
# ══════════════════════════════════════════════════════════════════════

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Signal Screener Report</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #c9d1d9; --text-dim: #8b949e; --text-bright: #f0f6fc;
    --red: #f85149; --green: #3fb950; --yellow: #d29922; --blue: #58a6ff;
    --red-bg: #f8514920; --green-bg: #3fb95020; --yellow-bg: #d2992220;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 14px; }
  .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 16px 20px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; }
  header h1 { color: var(--text-bright); font-size: 20px; font-weight: 600; }
  header select { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; font-size: 14px; cursor: pointer; }
  .stats { display: flex; gap: 24px; padding: 12px 20px; background: var(--surface); border-bottom: 1px solid var(--border); flex-wrap: wrap; }
  .stat { display: flex; flex-direction: column; }
  .stat-value { font-size: 20px; font-weight: 600; color: var(--text-bright); }
  .stat-label { font-size: 12px; color: var(--text-dim); text-transform: uppercase; }
  .tabs { display: flex; border-bottom: 1px solid var(--border); background: var(--surface); }
  .tab { padding: 10px 20px; cursor: pointer; color: var(--text-dim); border-bottom: 2px solid transparent; transition: all 0.2s; }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--text-bright); border-bottom-color: var(--blue); }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin: 12px 0; overflow: hidden; }
  .card-header { padding: 12px 16px; display: flex; align-items: center; justify-content: space-between; cursor: pointer; user-select: none; }
  .card-header:hover { background: #1c2129; }
  .card-ticker { font-size: 16px; font-weight: 600; color: var(--text-bright); margin-right: 12px; }
  .card-name { color: var(--text-dim); font-size: 13px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .badges { display: flex; gap: 6px; flex-wrap: wrap; }
  .badge { padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
  .badge-high { background: var(--red-bg); color: var(--red); border: 1px solid var(--red); }
  .badge-medium { background: var(--green-bg); color: var(--green); border: 1px solid var(--green); }
  .badge-warning { background: var(--yellow-bg); color: var(--yellow); border: 1px solid var(--yellow); }
  .card-body { display: none; padding: 0 16px 16px; }
  .card.expanded .card-body { display: block; }
  .signal-list { list-style: none; margin: 8px 0; }
  .signal-list li { padding: 6px 0; border-bottom: 1px solid var(--border); display: flex; gap: 8px; align-items: baseline; }
  .signal-list li:last-child { border-bottom: none; }
  .signal-quarter { color: var(--blue); font-size: 12px; white-space: nowrap; min-width: 100px; }
  .signal-name { font-weight: 500; }
  .signal-detail { color: var(--text-dim); font-size: 12px; }
  table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }
  th { text-align: right; padding: 6px 10px; color: var(--text-dim); font-weight: 500; border-bottom: 1px solid var(--border); white-space: nowrap; }
  td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  .positive { color: var(--green); }
  .negative { color: var(--red); }
  .no-data { color: var(--text-dim); font-style: italic; text-align: center; padding: 40px; }
  .expand-icon { color: var(--text-dim); transition: transform 0.2s; }
  .card.expanded .expand-icon { transform: rotate(90deg); }
  .footer { text-align: center; padding: 20px; color: var(--text-dim); font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>Signal Screener Report</h1>
  <div>
    <label for="dateSelect" style="color:var(--text-dim);margin-right:8px;">Scan date:</label>
    <select id="dateSelect" onchange="onDateChange()"></select>
  </div>
</header>
<div class="stats" id="statsBar"></div>
<div class="tabs">
  <div class="tab active" onclick="switchTab('new')">New Signals</div>
  <div class="tab" onclick="switchTab('all')">All Signals</div>
</div>
<div class="container">
  <div id="newSignals" class="tab-content active"></div>
  <div id="allSignals" class="tab-content"></div>
</div>
<div class="footer">
  Generated: <span id="genTime"></span> | Signal Screener
</div>

<script>
const DATA = /*DATA_PLACEHOLDER*/;

function fmtPct(v) {
  if (v == null || isNaN(v)) return '';
  return (v >= 0 ? '+' : '') + (v * 100).toFixed(1) + '%';
}
function fmtB(v) {
  if (v == null || isNaN(v)) return '';
  let s = v < 0 ? '-' : '', a = Math.abs(v);
  if (a >= 1e9) return s + '$' + (a/1e9).toFixed(1) + 'B';
  if (a >= 1e6) return s + '$' + (a/1e6).toFixed(0) + 'M';
  return s + '$' + a.toLocaleString();
}
function valClass(v) { return v > 0 ? 'positive' : v < 0 ? 'negative' : ''; }
function badgeClass(sev) { return 'badge badge-' + sev.toLowerCase(); }

document.getElementById('genTime').textContent = DATA.generated;

// Populate date selector
const sel = document.getElementById('dateSelect');
DATA.scan_history.slice().reverse().forEach((h, i) => {
  const opt = document.createElement('option');
  opt.value = DATA.scan_history.length - 1 - i;
  opt.textContent = h.date + ' ' + (h.time || '') + ' (' + h.mode + ')';
  sel.appendChild(opt);
});
if (DATA.scan_history.length === 0) {
  const opt = document.createElement('option');
  opt.textContent = 'No scans yet';
  sel.appendChild(opt);
}

function onDateChange() { renderNewSignals(); updateStats(); }

function updateStats() {
  const idx = parseInt(sel.value) || 0;
  const h = DATA.scan_history[idx];
  const bar = document.getElementById('statsBar');
  if (!h) { bar.innerHTML = '<div class="no-data">No scan data</div>'; return; }
  bar.innerHTML = `
    <div class="stat"><span class="stat-value">${h.total_scanned||0}</span><span class="stat-label">Scanned</span></div>
    <div class="stat"><span class="stat-value">${h.passed_filter||0}</span><span class="stat-label">Passed Filter</span></div>
    <div class="stat"><span class="stat-value">${(h.companies_with_signals||[]).length}</span><span class="stat-label">With Signals</span></div>
    <div class="stat"><span class="stat-value">${h.mode||'?'}</span><span class="stat-label">Mode</span></div>
    <div class="stat"><span class="stat-value">${h.date}</span><span class="stat-label">Date</span></div>
  `;
}

function renderCard(ticker, name, signals, quarters) {
  const high = signals.filter(s => s.severity === 'HIGH').length;
  const med = signals.filter(s => s.severity === 'MEDIUM').length;
  const warn = signals.filter(s => s.severity === 'WARNING').length;
  let badges = '';
  if (high) badges += `<span class="badge badge-high">${high} HIGH</span>`;
  if (med) badges += `<span class="badge badge-medium">${med} MED</span>`;
  if (warn) badges += `<span class="badge badge-warning">${warn} WARN</span>`;

  let signalHtml = '<ul class="signal-list">';
  signals.forEach(s => {
    const c = s.severity === 'HIGH' ? 'var(--red)' : s.severity === 'WARNING' ? 'var(--yellow)' : 'var(--green)';
    signalHtml += `<li>
      <span class="signal-quarter">${s.quarter||''}</span>
      <span class="signal-name" style="color:${c}">${s.signal||''}</span>
      <span class="signal-detail">${s.detail||''}</span>
    </li>`;
  });
  signalHtml += '</ul>';

  let tableHtml = '';
  if (quarters && quarters.length > 0) {
    const recent = [...quarters].sort((a,b) => (a.end||a.quarter_end||'').localeCompare(b.end||b.quarter_end||'')).slice(-8);
    tableHtml = `<table>
      <tr>
        <th>Quarter</th><th>Revenue</th><th>RevGr YoY</th><th>Gross%</th>
        <th>OpMar%</th><th>Incr OPM</th><th>FCF%</th><th>Op Income</th>
      </tr>`;
    recent.forEach(q => {
      tableHtml += `<tr>
        <td style="text-align:left">${q.quarter_label||''}</td>
        <td>${fmtB(q.revenue)}</td>
        <td class="${valClass(q.rev_growth_yoy)}">${fmtPct(q.rev_growth_yoy)}</td>
        <td class="${valClass(q.gross_margin)}">${fmtPct(q.gross_margin)}</td>
        <td class="${valClass(q.op_margin)}">${fmtPct(q.op_margin)}</td>
        <td class="${valClass(q.incr_op_margin_yoy)}">${fmtPct(q.incr_op_margin_yoy)}</td>
        <td class="${valClass(q.fcf_margin)}">${fmtPct(q.fcf_margin)}</td>
        <td>${fmtB(q.operating_income)}</td>
      </tr>`;
    });
    tableHtml += '</table>';
  }

  return `<div class="card" onclick="this.classList.toggle('expanded')">
    <div class="card-header">
      <span class="card-ticker">${ticker}</span>
      <span class="card-name">${name||''}</span>
      <div class="badges">${badges}</div>
      <span class="expand-icon">&#9654;</span>
    </div>
    <div class="card-body">
      ${signalHtml}
      ${tableHtml}
    </div>
  </div>`;
}

function renderNewSignals() {
  const idx = parseInt(sel.value) || 0;
  const h = DATA.scan_history[idx];
  const container = document.getElementById('newSignals');
  if (!h || !h.companies_with_signals || h.companies_with_signals.length === 0) {
    container.innerHTML = '<div class="no-data">No new signals for this scan date.</div>';
    return;
  }
  let html = '';
  // Sort by signal severity/count
  const sorted = h.companies_with_signals.slice().sort((a, b) => {
    const scoreA = (a.signals||[]).filter(s=>s.severity==='HIGH').length * 10 + (a.signals||[]).length;
    const scoreB = (b.signals||[]).filter(s=>s.severity==='HIGH').length * 10 + (b.signals||[]).length;
    return scoreB - scoreA;
  });
  sorted.forEach(c => {
    const fullResult = DATA.all_results[c.ticker];
    const quarters = fullResult ? fullResult.quarters : [];
    html += renderCard(c.ticker, c.name, c.signals||[], quarters);
  });
  container.innerHTML = html;
}

function renderAllSignals() {
  const container = document.getElementById('allSignals');
  const tickers = Object.keys(DATA.all_results);
  if (tickers.length === 0) {
    container.innerHTML = '<div class="no-data">No cached results with signals.</div>';
    return;
  }
  // Sort by signal count
  tickers.sort((a, b) => {
    const sa = DATA.all_results[a].signals || [];
    const sb = DATA.all_results[b].signals || [];
    const scoreA = sa.filter(s=>s.severity==='HIGH').length * 10 + sa.length;
    const scoreB = sb.filter(s=>s.severity==='HIGH').length * 10 + sb.length;
    return scoreB - scoreA;
  });
  let html = '';
  tickers.forEach(t => {
    const r = DATA.all_results[t];
    html += renderCard(t, r.name, r.signals||[], r.quarters||[]);
  });
  container.innerHTML = html;
}

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  if (tab === 'new') {
    document.querySelectorAll('.tab')[0].classList.add('active');
    document.getElementById('newSignals').classList.add('active');
  } else {
    document.querySelectorAll('.tab')[1].classList.add('active');
    document.getElementById('allSignals').classList.add('active');
    renderAllSignals();
  }
}

// Initial render
updateStats();
renderNewSignals();
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════
#  Master Report Template (signal_master.html) - Complete cache view
# ══════════════════════════════════════════════════════════════════════

_MASTER_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Signal Master - Complete Cache</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #c9d1d9; --text-dim: #8b949e; --text-bright: #f0f6fc;
    --red: #f85149; --green: #3fb950; --yellow: #d29922; --blue: #58a6ff;
    --red-bg: #f8514920; --green-bg: #3fb95020; --yellow-bg: #d2992220;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 14px; }
  .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 16px 20px; }
  header h1 { color: var(--text-bright); font-size: 20px; font-weight: 600; margin-bottom: 12px; }
  .filter-bar { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
  .filter-bar input, .filter-bar select { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; font-size: 14px; }
  .filter-bar input { width: 180px; }
  .filter-bar label { color: var(--text-dim); font-size: 13px; margin-right: 4px; }
  .filter-group { display: flex; align-items: center; gap: 4px; }
  .stats { display: flex; gap: 24px; padding: 12px 20px; background: var(--surface); border-bottom: 1px solid var(--border); flex-wrap: wrap; }
  .stat { display: flex; flex-direction: column; }
  .stat-value { font-size: 20px; font-weight: 600; color: var(--text-bright); }
  .stat-label { font-size: 12px; color: var(--text-dim); text-transform: uppercase; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin: 12px 0; overflow: hidden; }
  .card-header { padding: 12px 16px; display: flex; align-items: center; justify-content: space-between; cursor: pointer; user-select: none; }
  .card-header:hover { background: #1c2129; }
  .card-ticker { font-size: 16px; font-weight: 600; color: var(--text-bright); margin-right: 12px; }
  .card-name { color: var(--text-dim); font-size: 13px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .badges { display: flex; gap: 6px; flex-wrap: wrap; }
  .badge { padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
  .badge-high { background: var(--red-bg); color: var(--red); border: 1px solid var(--red); }
  .badge-medium { background: var(--green-bg); color: var(--green); border: 1px solid var(--green); }
  .badge-warning { background: var(--yellow-bg); color: var(--yellow); border: 1px solid var(--yellow); }
  .card-body { display: none; padding: 0 16px 16px; }
  .card.expanded .card-body { display: block; }
  .signal-list { list-style: none; margin: 8px 0; }
  .signal-list li { padding: 6px 0; border-bottom: 1px solid var(--border); display: flex; gap: 8px; align-items: baseline; }
  .signal-list li:last-child { border-bottom: none; }
  .signal-quarter { color: var(--blue); font-size: 12px; white-space: nowrap; min-width: 100px; }
  .signal-name { font-weight: 500; }
  .signal-detail { color: var(--text-dim); font-size: 12px; }
  table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }
  th { text-align: right; padding: 6px 10px; color: var(--text-dim); font-weight: 500; border-bottom: 1px solid var(--border); white-space: nowrap; }
  td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  .positive { color: var(--green); }
  .negative { color: var(--red); }
  .no-data { color: var(--text-dim); font-style: italic; text-align: center; padding: 40px; }
  .expand-icon { color: var(--text-dim); transition: transform 0.2s; }
  .card.expanded .expand-icon { transform: rotate(90deg); }
  .footer { text-align: center; padding: 20px; color: var(--text-dim); font-size: 12px; }
  .hidden { display: none !important; }
</style>
</head>
<body>
<header>
  <h1>Signal Master - Complete Cache View</h1>
  <div class="filter-bar">
    <div class="filter-group">
      <label>Ticker:</label>
      <input type="text" id="tickerFilter" placeholder="Search ticker..." oninput="applyFilters()">
    </div>
    <div class="filter-group">
      <label>Severity:</label>
      <select id="severityFilter" onchange="applyFilters()">
        <option value="">All</option>
        <option value="HIGH">HIGH only</option>
        <option value="MEDIUM">MEDIUM+</option>
        <option value="WARNING">WARNING+</option>
      </select>
    </div>
    <div class="filter-group">
      <label>Signal:</label>
      <select id="signalFilter" onchange="applyFilters()">
        <option value="">All Types</option>
        <option value="TURNED POSITIVE">Turned Positive</option>
        <option value="INCREMENTAL">Incremental Margin</option>
        <option value="ACCELERATION">Rev Acceleration</option>
        <option value="INFLECTION">GM Inflection</option>
        <option value="LEVERAGE">Op Leverage</option>
        <option value="FCF">FCF Related</option>
      </select>
    </div>
    <div class="filter-group">
      <label>Data After:</label>
      <input type="text" id="afterFilter" placeholder="YYYY-MM" style="width:100px" oninput="applyFilters()">
    </div>
  </div>
</header>
<div class="stats" id="statsBar"></div>
<div class="container" id="mainContent"></div>
<div class="footer">
  Generated: <span id="genTime"></span> | Signal Master (Complete Cache)
</div>

<script>
const DATA = /*DATA_PLACEHOLDER*/;
const SEVERITY_RANK = {HIGH: 3, MEDIUM: 2, WARNING: 1};

function fmtPct(v) {
  if (v == null || isNaN(v)) return '';
  return (v >= 0 ? '+' : '') + (v * 100).toFixed(1) + '%';
}
function fmtB(v) {
  if (v == null || isNaN(v)) return '';
  let s = v < 0 ? '-' : '', a = Math.abs(v);
  if (a >= 1e9) return s + '$' + (a/1e9).toFixed(1) + 'B';
  if (a >= 1e6) return s + '$' + (a/1e6).toFixed(0) + 'M';
  return s + '$' + a.toLocaleString();
}
function valClass(v) { return v > 0 ? 'positive' : v < 0 ? 'negative' : ''; }

document.getElementById('genTime').textContent = DATA.generated;

function renderCard(ticker, data) {
  const signals = data.signals || [];
  const quarters = data.quarters || [];
  const name = data.name || '';

  const high = signals.filter(s => s.severity === 'HIGH').length;
  const med = signals.filter(s => s.severity === 'MEDIUM').length;
  const warn = signals.filter(s => s.severity === 'WARNING').length;
  let badges = '';
  if (high) badges += `<span class="badge badge-high">${high} HIGH</span>`;
  if (med) badges += `<span class="badge badge-medium">${med} MED</span>`;
  if (warn) badges += `<span class="badge badge-warning">${warn} WARN</span>`;

  let signalHtml = '<ul class="signal-list">';
  signals.forEach(s => {
    const c = s.severity === 'HIGH' ? 'var(--red)' : s.severity === 'WARNING' ? 'var(--yellow)' : 'var(--green)';
    signalHtml += `<li>
      <span class="signal-quarter">${s.quarter||''}</span>
      <span class="signal-name" style="color:${c}">${s.signal||''}</span>
      <span class="signal-detail">${s.detail||''}</span>
    </li>`;
  });
  signalHtml += '</ul>';

  let tableHtml = '';
  if (quarters.length > 0) {
    const recent = [...quarters].sort((a,b) => (a.end||a.quarter_end||'').localeCompare(b.end||b.quarter_end||'')).slice(-8);
    tableHtml = `<table>
      <tr>
        <th>Quarter</th><th>Revenue</th><th>RevGr YoY</th><th>Gross%</th>
        <th>OpMar%</th><th>Incr OPM</th><th>FCF%</th><th>Op Income</th>
      </tr>`;
    recent.forEach(q => {
      tableHtml += `<tr>
        <td style="text-align:left">${q.quarter_label||''}</td>
        <td>${fmtB(q.revenue)}</td>
        <td class="${valClass(q.rev_growth_yoy)}">${fmtPct(q.rev_growth_yoy)}</td>
        <td class="${valClass(q.gross_margin)}">${fmtPct(q.gross_margin)}</td>
        <td class="${valClass(q.op_margin)}">${fmtPct(q.op_margin)}</td>
        <td class="${valClass(q.incr_op_margin_yoy)}">${fmtPct(q.incr_op_margin_yoy)}</td>
        <td class="${valClass(q.fcf_margin)}">${fmtPct(q.fcf_margin)}</td>
        <td>${fmtB(q.operating_income)}</td>
      </tr>`;
    });
    tableHtml += '</table>';
  }

  return `<div class="card" data-ticker="${ticker}" onclick="this.classList.toggle('expanded')">
    <div class="card-header">
      <span class="card-ticker">${ticker}</span>
      <span class="card-name">${name}</span>
      <div class="badges">${badges}</div>
      <span class="expand-icon">&#9654;</span>
    </div>
    <div class="card-body">
      ${signalHtml}
      ${tableHtml}
    </div>
  </div>`;
}

function applyFilters() {
  const tickerQ = document.getElementById('tickerFilter').value.toUpperCase().trim();
  const severityQ = document.getElementById('severityFilter').value;
  const signalQ = document.getElementById('signalFilter').value.toUpperCase();
  const afterQ = document.getElementById('afterFilter').value.trim();

  const minRank = severityQ ? SEVERITY_RANK[severityQ] : 0;

  const tickers = Object.keys(DATA.all_results);
  let matched = 0, total = tickers.length;

  const container = document.getElementById('mainContent');
  let html = '';

  // Sort by signal score
  const sorted = tickers.slice().sort((a, b) => {
    const sa = DATA.all_results[a].signals || [];
    const sb = DATA.all_results[b].signals || [];
    const scoreA = sa.filter(s=>s.severity==='HIGH').length * 10 + sa.length;
    const scoreB = sb.filter(s=>s.severity==='HIGH').length * 10 + sb.length;
    return scoreB - scoreA;
  });

  sorted.forEach(ticker => {
    const data = DATA.all_results[ticker];
    let signals = data.signals || [];
    let quarters = data.quarters || [];

    // Ticker filter
    if (tickerQ && !ticker.includes(tickerQ)) return;

    // After date filter - filter BOTH quarters AND signals
    if (afterQ && /^\d{4}-\d{2}$/.test(afterQ)) {
      // Filter quarters: keep only those with date >= afterQ
      quarters = quarters.filter(q => {
        const qDate = (q.date || '').slice(0, 7);
        return qDate >= afterQ;
      });
      if (quarters.length === 0) return;

      // Get quarter labels for filtered quarters
      const validLabels = new Set(quarters.map(q => q.quarter_label));
      // Filter signals: keep only those in filtered quarters
      signals = signals.filter(s => validLabels.has(s.quarter));
    }

    // Severity filter
    let filteredSignals = signals;
    if (minRank > 0) {
      filteredSignals = signals.filter(s => (SEVERITY_RANK[s.severity] || 0) >= minRank);
      if (filteredSignals.length === 0) return;
    }

    // Signal type filter
    if (signalQ) {
      filteredSignals = filteredSignals.filter(s => (s.signal || '').toUpperCase().includes(signalQ));
      if (filteredSignals.length === 0) return;
    }

    matched++;
    // Render with filtered signals AND filtered quarters
    const renderData = {...data, signals: filteredSignals, quarters: quarters};
    html += renderCard(ticker, renderData);
  });

  container.innerHTML = html || '<div class="no-data">No companies match the filter criteria.</div>';

  // Update stats
  document.getElementById('statsBar').innerHTML = `
    <div class="stat"><span class="stat-value">${total}</span><span class="stat-label">Total Cached</span></div>
    <div class="stat"><span class="stat-value">${matched}</span><span class="stat-label">Matching Filter</span></div>
  `;
}

// Initial render
applyFilters();
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════
#  Queries Report Template (signal_queries.html) - Query history
# ══════════════════════════════════════════════════════════════════════

_QUERIES_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Signal Queries - History</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #c9d1d9; --text-dim: #8b949e; --text-bright: #f0f6fc;
    --red: #f85149; --green: #3fb950; --yellow: #d29922; --blue: #58a6ff;
    --red-bg: #f8514920; --green-bg: #3fb95020; --yellow-bg: #d2992220;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 14px; display: flex; height: 100vh; }
  .sidebar { width: 320px; background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; flex-shrink: 0; }
  .sidebar-header { padding: 16px; border-bottom: 1px solid var(--border); }
  .sidebar-header h1 { color: var(--text-bright); font-size: 18px; font-weight: 600; }
  .sidebar-header p { color: var(--text-dim); font-size: 12px; margin-top: 4px; }
  .query-list { flex: 1; overflow-y: auto; }
  .query-item { padding: 12px 16px; border-bottom: 1px solid var(--border); cursor: pointer; display: flex; flex-direction: column; gap: 4px; }
  .query-item:hover { background: #1c2129; }
  .query-item.selected { background: var(--blue); background-opacity: 0.2; border-left: 3px solid var(--blue); }
  .query-time { font-size: 12px; color: var(--text-dim); }
  .query-cmd { font-size: 13px; color: var(--text-bright); font-family: monospace; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .query-stats { font-size: 11px; color: var(--text-dim); }
  .query-delete { color: var(--red); font-size: 11px; margin-left: auto; opacity: 0; transition: opacity 0.2s; }
  .query-item:hover .query-delete { opacity: 1; }
  .main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .main-header { padding: 16px 20px; background: var(--surface); border-bottom: 1px solid var(--border); }
  .main-header h2 { color: var(--text-bright); font-size: 16px; font-weight: 600; }
  .main-content { flex: 1; overflow-y: auto; padding: 20px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin: 12px 0; overflow: hidden; }
  .card-header { padding: 12px 16px; display: flex; align-items: center; justify-content: space-between; cursor: pointer; user-select: none; }
  .card-header:hover { background: #1c2129; }
  .card-ticker { font-size: 16px; font-weight: 600; color: var(--text-bright); margin-right: 12px; }
  .card-name { color: var(--text-dim); font-size: 13px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .badges { display: flex; gap: 6px; flex-wrap: wrap; }
  .badge { padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
  .badge-high { background: var(--red-bg); color: var(--red); border: 1px solid var(--red); }
  .badge-medium { background: var(--green-bg); color: var(--green); border: 1px solid var(--green); }
  .badge-warning { background: var(--yellow-bg); color: var(--yellow); border: 1px solid var(--yellow); }
  .card-body { display: none; padding: 0 16px 16px; }
  .card.expanded .card-body { display: block; }
  .signal-list { list-style: none; margin: 8px 0; }
  .signal-list li { padding: 6px 0; border-bottom: 1px solid var(--border); display: flex; gap: 8px; align-items: baseline; }
  .signal-list li:last-child { border-bottom: none; }
  .signal-quarter { color: var(--blue); font-size: 12px; white-space: nowrap; min-width: 100px; }
  .signal-name { font-weight: 500; }
  .signal-detail { color: var(--text-dim); font-size: 12px; }
  table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }
  th { text-align: right; padding: 6px 10px; color: var(--text-dim); font-weight: 500; border-bottom: 1px solid var(--border); white-space: nowrap; }
  td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  .positive { color: var(--green); }
  .negative { color: var(--red); }
  .no-data { color: var(--text-dim); font-style: italic; text-align: center; padding: 40px; }
  .expand-icon { color: var(--text-dim); transition: transform 0.2s; }
  .card.expanded .expand-icon { transform: rotate(90deg); }
</style>
</head>
<body>
<div class="sidebar">
  <div class="sidebar-header">
    <h1>Query History</h1>
    <p id="queryCount">0 queries</p>
  </div>
  <div class="query-list" id="queryList"></div>
</div>
<div class="main">
  <div class="main-header">
    <h2 id="selectedQuery">Select a query from the list</h2>
  </div>
  <div class="main-content" id="mainContent">
    <div class="no-data">Select a query from the sidebar to view results.</div>
  </div>
</div>

<script>
const DATA = /*DATA_PLACEHOLDER*/;
let selectedIdx = -1;

function fmtPct(v) {
  if (v == null || isNaN(v)) return '';
  return (v >= 0 ? '+' : '') + (v * 100).toFixed(1) + '%';
}
function fmtB(v) {
  if (v == null || isNaN(v)) return '';
  let s = v < 0 ? '-' : '', a = Math.abs(v);
  if (a >= 1e9) return s + '$' + (a/1e9).toFixed(1) + 'B';
  if (a >= 1e6) return s + '$' + (a/1e6).toFixed(0) + 'M';
  return s + '$' + a.toLocaleString();
}
function valClass(v) { return v > 0 ? 'positive' : v < 0 ? 'negative' : ''; }

function renderCard(ticker, data) {
  const signals = data.signals || [];
  const quarters = data.quarters || [];
  const name = data.name || '';

  const high = signals.filter(s => s.severity === 'HIGH').length;
  const med = signals.filter(s => s.severity === 'MEDIUM').length;
  const warn = signals.filter(s => s.severity === 'WARNING').length;
  let badges = '';
  if (high) badges += `<span class="badge badge-high">${high} HIGH</span>`;
  if (med) badges += `<span class="badge badge-medium">${med} MED</span>`;
  if (warn) badges += `<span class="badge badge-warning">${warn} WARN</span>`;

  let signalHtml = '<ul class="signal-list">';
  signals.forEach(s => {
    const c = s.severity === 'HIGH' ? 'var(--red)' : s.severity === 'WARNING' ? 'var(--yellow)' : 'var(--green)';
    signalHtml += `<li>
      <span class="signal-quarter">${s.quarter||''}</span>
      <span class="signal-name" style="color:${c}">${s.signal||''}</span>
      <span class="signal-detail">${s.detail||''}</span>
    </li>`;
  });
  signalHtml += '</ul>';

  let tableHtml = '';
  if (quarters.length > 0) {
    const recent = [...quarters].sort((a,b) => (a.end||a.quarter_end||'').localeCompare(b.end||b.quarter_end||'')).slice(-8);
    tableHtml = `<table>
      <tr>
        <th>Quarter</th><th>Revenue</th><th>RevGr YoY</th><th>Gross%</th>
        <th>OpMar%</th><th>Incr OPM</th><th>FCF%</th><th>Op Income</th>
      </tr>`;
    recent.forEach(q => {
      tableHtml += `<tr>
        <td style="text-align:left">${q.quarter_label||''}</td>
        <td>${fmtB(q.revenue)}</td>
        <td class="${valClass(q.rev_growth_yoy)}">${fmtPct(q.rev_growth_yoy)}</td>
        <td class="${valClass(q.gross_margin)}">${fmtPct(q.gross_margin)}</td>
        <td class="${valClass(q.op_margin)}">${fmtPct(q.op_margin)}</td>
        <td class="${valClass(q.incr_op_margin_yoy)}">${fmtPct(q.incr_op_margin_yoy)}</td>
        <td class="${valClass(q.fcf_margin)}">${fmtPct(q.fcf_margin)}</td>
        <td>${fmtB(q.operating_income)}</td>
      </tr>`;
    });
    tableHtml += '</table>';
  }

  return `<div class="card" onclick="this.classList.toggle('expanded')">
    <div class="card-header">
      <span class="card-ticker">${ticker}</span>
      <span class="card-name">${name}</span>
      <div class="badges">${badges}</div>
      <span class="expand-icon">&#9654;</span>
    </div>
    <div class="card-body">
      ${signalHtml}
      ${tableHtml}
    </div>
  </div>`;
}

function renderQueryList() {
  const list = document.getElementById('queryList');
  const queries = DATA.queries || [];
  document.getElementById('queryCount').textContent = queries.length + ' queries';

  if (queries.length === 0) {
    list.innerHTML = '<div class="no-data" style="padding:20px;">No queries yet.</div>';
    return;
  }

  let html = '';
  queries.slice().reverse().forEach((q, i) => {
    const idx = queries.length - 1 - i;
    const signalCount = (q.companies || []).reduce((sum, c) => sum + (c.signals || []).length, 0);
    html += `<div class="query-item ${idx === selectedIdx ? 'selected' : ''}" onclick="selectQuery(${idx})">
      <span class="query-time">${q.timestamp || ''}</span>
      <span class="query-cmd">${q.command || ''}</span>
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span class="query-stats">${(q.companies || []).length} companies, ${signalCount} signals</span>
        <span class="query-delete" onclick="deleteQuery(${idx}, event)">[Delete]</span>
      </div>
    </div>`;
  });
  list.innerHTML = html;
}

function selectQuery(idx) {
  selectedIdx = idx;
  renderQueryList();

  const q = DATA.queries[idx];
  if (!q) return;

  document.getElementById('selectedQuery').textContent = q.command || 'Query Results';

  const container = document.getElementById('mainContent');
  const companies = q.companies || [];

  if (companies.length === 0) {
    container.innerHTML = '<div class="no-data">No companies with signals in this query.</div>';
    return;
  }

  // Sort by signal score
  companies.sort((a, b) => {
    const scoreA = (a.signals||[]).filter(s=>s.severity==='HIGH').length * 10 + (a.signals||[]).length;
    const scoreB = (b.signals||[]).filter(s=>s.severity==='HIGH').length * 10 + (b.signals||[]).length;
    return scoreB - scoreA;
  });

  let html = '';
  companies.forEach(c => {
    html += renderCard(c.ticker, c);
  });
  container.innerHTML = html;
}

function deleteQuery(idx, event) {
  event.stopPropagation();
  if (!confirm('Delete this query from history?')) return;

  DATA.queries.splice(idx, 1);
  if (selectedIdx === idx) selectedIdx = -1;
  else if (selectedIdx > idx) selectedIdx--;

  renderQueryList();
  if (selectedIdx === -1) {
    document.getElementById('selectedQuery').textContent = 'Select a query from the list';
    document.getElementById('mainContent').innerHTML = '<div class="no-data">Select a query from the sidebar to view results.</div>';
  }

  // Note: This only updates the in-memory data. To persist deletions,
  // you would need to regenerate the HTML file from Python.
}

// Initial render
renderQueryList();
if (DATA.queries && DATA.queries.length > 0) {
  selectQuery(DATA.queries.length - 1);
}
</script>
</body>
</html>"""


def _assemble_report_data():
    """Load scan history and all cached results with signals."""
    scan_history = _load_scan_history()

    # Load all cached results that have signals
    all_results = {}
    if os.path.isdir(RESULTS_CACHE_DIR):
        for fname in os.listdir(RESULTS_CACHE_DIR):
            if not fname.endswith(".json"):
                continue
            ticker = fname[:-5]
            result = _load_result(ticker)
            if result and result.get("signals"):
                all_results[ticker] = _make_serializable(result)

    return {
        "scan_history": scan_history,
        "all_results": all_results,
        "generated": datetime.now().isoformat(),
    }


def _generate_html_report(output_path=None):
    """Generate self-contained HTML report."""
    if output_path is None:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        output_path = os.path.join(REPORTS_DIR, "signal_report.html")

    data = _assemble_report_data()
    data_json = json.dumps(data, default=str)

    html = _HTML_TEMPLATE.replace("/*DATA_PLACEHOLDER*/", data_json)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"{GREEN}Report generated: {output_path}{RESET}")
    return output_path


def _run_report_mode(args):
    """Generate master report (complete cache) and optionally open in browser.

    New dual-file architecture:
    - signal_master.html: Complete cache view with client-side filtering
    - signal_queries.html: Query history (updated by update/filter commands)
    """
    print(f"{BOLD}{CYAN}Signal Screener -- Report Generation{RESET}")

    # Generate master report (always)
    master_path = _generate_master_report()

    # Also regenerate queries report if history exists
    queries = _load_query_history()
    if queries:
        _generate_queries_report(queries)
        print(f"  Query history entries: {len(queries)}")

    # Open master report in browser
    try:
        webbrowser.open(f"file://{os.path.abspath(master_path)}")
        print(f"{DIM}Opened master report in browser.{RESET}")
    except Exception:
        pass

    print(f"\n{DIM}Files:{RESET}")
    print(f"  Master (cache view): {MASTER_REPORT_FILE}")
    print(f"  Queries (history):   {QUERIES_REPORT_FILE}")


# ══════════════════════════════════════════════════════════════════════
#  Master Report & Query History (Dual-file architecture)
# ══════════════════════════════════════════════════════════════════════

MASTER_REPORT_FILE = os.path.join(REPORTS_DIR, "signal_master.html")
QUERIES_REPORT_FILE = os.path.join(REPORTS_DIR, "signal_queries.html")
QUERIES_JSON_FILE = os.path.join(CACHE_DIR, "query_history.json")


def _generate_master_report():
    """Generate signal_master.html - complete cache view with client-side filtering."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load all cached results (including those without signals for completeness)
    all_results = {}
    if os.path.isdir(RESULTS_CACHE_DIR):
        for fname in os.listdir(RESULTS_CACHE_DIR):
            if not fname.endswith(".json"):
                continue
            ticker = fname[:-5]
            result = _load_result(ticker)
            if result and result.get("signals"):  # Only include those with signals
                all_results[ticker] = _make_serializable(result)

    data = {
        "all_results": all_results,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    data_json = json.dumps(data, default=str)

    html = _MASTER_HTML_TEMPLATE.replace("/*DATA_PLACEHOLDER*/", data_json)

    with open(MASTER_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"{GREEN}Master report generated: {MASTER_REPORT_FILE}{RESET}")
    print(f"  Total companies with signals: {len(all_results)}")
    return MASTER_REPORT_FILE


def _load_query_history():
    """Load query history from JSON file."""
    if os.path.isfile(QUERIES_JSON_FILE):
        try:
            with open(QUERIES_JSON_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []


def _save_query_history(queries):
    """Save query history to JSON file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    _atomic_json_write(QUERIES_JSON_FILE, queries)


def _append_query(command, companies):
    """Append a query result to history and regenerate queries report.

    Args:
        command: str, the command that was run (e.g., "update --days 15 --review")
        companies: list of dicts, each with ticker, name, signals, quarters
    """
    queries = _load_query_history()

    query_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "command": command,
        "companies": _make_serializable(companies),
    }
    queries.append(query_entry)

    # Keep last 100 queries
    if len(queries) > 100:
        queries = queries[-100:]

    _save_query_history(queries)
    _generate_queries_report(queries)

    return query_entry


def _generate_queries_report(queries=None):
    """Generate signal_queries.html - query history with selectable results."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if queries is None:
        queries = _load_query_history()

    data = {
        "queries": queries,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    data_json = json.dumps(data, default=str)

    html = _QUERIES_HTML_TEMPLATE.replace("/*DATA_PLACEHOLDER*/", data_json)

    with open(QUERIES_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"{GREEN}Queries report updated: {QUERIES_REPORT_FILE}{RESET}")
    return QUERIES_REPORT_FILE


# ══════════════════════════════════════════════════════════════════════
#  Filter mode (Phase 6)
# ══════════════════════════════════════════════════════════════════════

_SEVERITY_RANK = {"HIGH": 3, "MEDIUM": 2, "WARNING": 1}


def _filter_company(result, criteria):
    """Apply filter criteria to one cached company result.

    Args:
        result: dict loaded from cache/results/<TICKER>.json
        criteria: dict with keys from argparse (recent_quarters, severity, signal, etc.)

    Returns:
        dict with matched info for display/sorting, or None if no match.
    """
    quarters = result.get("quarters", [])
    signals = result.get("signals", [])
    if not quarters:
        return None

    recent_n = criteria.get("recent_quarters", 4)
    recent_quarters = quarters[-recent_n:] if recent_n < len(quarters) else quarters
    recent_labels = {q["quarter_label"] for q in recent_quarters}

    # ── Filter by data freshness (--after) ──
    after_date = criteria.get("after")
    if after_date is not None:
        last_date_str = quarters[-1].get("date", "")
        if isinstance(last_date_str, str):
            last_date_str = last_date_str[:10]  # "2025-10-26T00:00:00" -> "2025-10-26"
        if last_date_str < after_date:
            return None

    # ── Filter signals by recency ──
    matched_signals = [s for s in signals if s.get("quarter") in recent_labels]

    # ── Filter by severity ──
    min_severity = criteria.get("severity")
    if min_severity:
        min_rank = _SEVERITY_RANK.get(min_severity, 0)
        matched_signals = [s for s in matched_signals
                           if _SEVERITY_RANK.get(s.get("severity"), 0) >= min_rank]

    # ── Filter by signal keyword ──
    signal_keyword = criteria.get("signal")
    if signal_keyword:
        kw = signal_keyword.upper()
        matched_signals = [s for s in matched_signals
                           if kw in s.get("signal", "").upper()]

    # ── Metric-based filters (use most recent quarter with data) ──
    last_q = recent_quarters[-1] if recent_quarters else {}

    # min_rev_growth: check most recent quarter with rev_growth_yoy
    min_rev_growth = criteria.get("min_rev_growth")
    if min_rev_growth is not None:
        rev_gr = None
        for q in reversed(recent_quarters):
            if q.get("rev_growth_yoy") is not None:
                rev_gr = q["rev_growth_yoy"]
                break
        if rev_gr is None or rev_gr < min_rev_growth:
            return None

    # min_op_margin: check most recent quarter with op_margin
    min_op_margin = criteria.get("min_op_margin")
    if min_op_margin is not None:
        opm = None
        for q in reversed(recent_quarters):
            if q.get("op_margin") is not None:
                opm = q["op_margin"]
                break
        if opm is None or opm < min_op_margin:
            return None

    # incr_margin_spread: find any recent quarter where incr_op_margin_yoy > op_margin + spread
    incr_spread = criteria.get("incr_margin_spread")
    if incr_spread is not None:
        spread = incr_spread / 100.0
        found_spread = False
        for q in recent_quarters:
            incr = q.get("incr_op_margin_yoy")
            opm = q.get("op_margin")
            if incr is not None and opm is not None:
                if incr > opm + spread:
                    found_spread = True
                    break
        if not found_spread:
            return None

    # If we have signal-related filters and no signals matched, skip
    if (min_severity or signal_keyword) and not matched_signals:
        return None

    # ── Collect summary metrics from latest quarter ──
    latest_rev = None
    latest_opm = None
    latest_incr_opm = None
    latest_rev_gr = None
    for q in reversed(recent_quarters):
        if latest_rev is None and q.get("revenue") is not None:
            latest_rev = q["revenue"]
        if latest_opm is None and q.get("op_margin") is not None:
            latest_opm = q["op_margin"]
        if latest_incr_opm is None and q.get("incr_op_margin_yoy") is not None:
            latest_incr_opm = q["incr_op_margin_yoy"]
        if latest_rev_gr is None and q.get("rev_growth_yoy") is not None:
            latest_rev_gr = q["rev_growth_yoy"]

    # Score for sorting
    high_count = sum(1 for s in matched_signals if s.get("severity") == "HIGH")
    med_count = sum(1 for s in matched_signals if s.get("severity") == "MEDIUM")
    warn_count = sum(1 for s in matched_signals if s.get("severity") == "WARNING")
    score = high_count * 10 + med_count * 3 + warn_count

    return {
        "ticker": result["ticker"],
        "name": result.get("name", result["ticker"]),
        "signals": matched_signals,
        "high": high_count,
        "medium": med_count,
        "warning": warn_count,
        "score": score,
        "revenue": latest_rev,
        "op_margin": latest_opm,
        "incr_op_margin": latest_incr_opm,
        "rev_growth": latest_rev_gr,
        "quarters": result.get("quarters", []),
    }


def _run_filter_mode(args):
    """Filter cached results by custom criteria."""
    criteria = {
        "recent_quarters": args.recent_quarters,
        "severity": args.severity,
        "signal": args.signal,
        "incr_margin_spread": args.incr_margin_spread,
        "min_rev_growth": args.min_rev_growth,
        "min_op_margin": args.min_op_margin,
        "after": args.after,
    }

    # Build description of active filters
    filter_parts = []
    filter_parts.append(f"recent_quarters={args.recent_quarters}")
    if args.severity:
        filter_parts.append(f"severity>={args.severity}")
    if args.signal:
        filter_parts.append(f'signal="{args.signal}"')
    if args.incr_margin_spread is not None:
        filter_parts.append(f"incr_margin_spread={args.incr_margin_spread:.0f}pp")
    if args.min_rev_growth is not None:
        filter_parts.append(f"min_rev_growth={args.min_rev_growth:.0%}")
    if args.min_op_margin is not None:
        filter_parts.append(f"min_op_margin={args.min_op_margin:.0%}")
    if args.after:
        filter_parts.append(f"after={args.after}")

    print(f"{BOLD}{CYAN}Signal Screener -- Filter Results{RESET}")
    print(f"  Filters: {', '.join(filter_parts)}")

    # Scan all cached results
    if not os.path.isdir(RESULTS_CACHE_DIR):
        print(f"\n{RED}No cached results found. Run 'scan' first.{RESET}")
        return

    files = [f for f in os.listdir(RESULTS_CACHE_DIR) if f.endswith(".json")]
    total_cached = len(files)
    matched = []

    for fname in files:
        ticker = fname[:-5]
        result = _load_result(ticker)
        if result is None:
            continue
        hit = _filter_company(result, criteria)
        if hit is not None:
            matched.append(hit)

    # Sort
    sort_key = args.sort
    if sort_key == "score":
        matched.sort(key=lambda x: x["score"], reverse=True)
    elif sort_key == "revenue":
        matched.sort(key=lambda x: x["revenue"] or 0, reverse=True)
    elif sort_key == "op_margin":
        matched.sort(key=lambda x: x["op_margin"] or -999, reverse=True)
    elif sort_key == "rev_growth":
        matched.sort(key=lambda x: x["rev_growth"] or -999, reverse=True)

    print(f"  Matched: {len(matched)} / {total_cached:,} cached companies")
    print()

    # Limit
    top_n = args.top
    display = matched[:top_n]

    if not display:
        print(f"  {DIM}No companies matched the filter criteria.{RESET}")
        return

    # Table header
    hdr = (f"  {'TICKER':<8} {'NAME':<30} {'SIGNALS':>8} {'Revenue':>10} "
           f"{'OPM':>9} {'IncrOPM':>9} {'RevGr YoY':>10}")
    print(f"{BOLD}{hdr}{RESET}")
    print(f"  {'-'*88}")

    for r in display:
        # Signal summary like "3H 2M"
        parts = []
        if r["high"]:
            parts.append(f"{r['high']}H")
        if r["medium"]:
            parts.append(f"{r['medium']}M")
        if r["warning"]:
            parts.append(f"{r['warning']}W")
        sig_str = " ".join(parts) if parts else "-"

        rev_str = fmt_b(r["revenue"]) if r["revenue"] is not None else ""
        opm_str = fmt_pct(r["op_margin"]) if r["op_margin"] is not None else ""
        incr_str = fmt_pct(r["incr_op_margin"]) if r["incr_op_margin"] is not None else ""
        rg_str = fmt_pct(r["rev_growth"]) if r["rev_growth"] is not None else ""

        name_trunc = r["name"][:28]
        line = (f"  {r['ticker']:<8} {name_trunc:<30} {sig_str:>8} {rev_str:>10} "
                f"{opm_str:>9} {incr_str:>9} {rg_str:>10}")
        print(line)

    if len(matched) > top_n:
        print(f"\n  {DIM}... and {len(matched) - top_n} more. Use --top to see more.{RESET}")

    print(f"\n  {DIM}Use --html to generate interactive report.{RESET}")

    # HTML report
    if args.html:
        print()
        _generate_filter_html_report(matched, filter_parts)


def _generate_filter_html_report(matched, filter_parts):
    """Generate HTML report for filter results and append to query history."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Build data for query history
    query_companies = []
    for r in matched:
        query_companies.append({
            "ticker": r["ticker"],
            "name": r["name"],
            "signals": _make_serializable(r["signals"]),
            "quarters": _make_serializable(r["quarters"]),
        })

    # Append to query history
    cmd = f"filter {', '.join(filter_parts)}"
    _append_query(cmd, query_companies)

    # Open queries report in browser
    try:
        webbrowser.open(f"file://{os.path.abspath(QUERIES_REPORT_FILE)}")
        print(f"{DIM}Opened queries report in browser.{RESET}")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
#  Serve mode (Local HTTP server)
# ══════════════════════════════════════════════════════════════════════

import http.server
import socketserver
import urllib.parse as urlparse

_SERVE_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Signal Screener</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #c9d1d9; --text-dim: #8b949e; --text-bright: #f0f6fc;
    --red: #f85149; --green: #3fb950; --yellow: #d29922; --blue: #58a6ff;
    --red-bg: #f8514920; --green-bg: #3fb95020; --yellow-bg: #d2992220;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 14px; }
  .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 16px 20px; }
  header h1 { color: var(--text-bright); font-size: 20px; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center; gap: 12px; }
  header h1 .status { font-size: 12px; padding: 4px 8px; border-radius: 4px; background: var(--green-bg); color: var(--green); }
  .filter-bar { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
  .filter-bar input, .filter-bar select { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; font-size: 14px; }
  .filter-bar input[type="text"] { width: 140px; }
  .filter-bar input[type="number"] { width: 80px; }
  .filter-bar label { color: var(--text-dim); font-size: 13px; margin-right: 4px; }
  .filter-group { display: flex; align-items: center; gap: 4px; }
  .btn { background: var(--blue); color: white; border: none; border-radius: 6px; padding: 6px 16px; font-size: 14px; cursor: pointer; transition: all 0.2s; }
  .btn:hover { opacity: 0.9; transform: translateY(-1px); }
  .btn-secondary { background: var(--surface); border: 1px solid var(--border); color: var(--text); transition: all 0.2s; }
  .btn-secondary:hover { background: var(--border); }
  .btn-clear { background: transparent; border: 1px solid var(--text-dim); color: var(--text-dim); border-radius: 6px; padding: 6px 14px; font-size: 13px; cursor: pointer; transition: all 0.2s; }
  .btn-clear:hover { border-color: var(--yellow); color: var(--yellow); background: var(--yellow-bg); }
  .btn-apply { background: var(--green); font-weight: 600; }
  .btn-apply:hover { background: #2ea043; }
  .btn-small { padding: 4px 10px; font-size: 12px; }
  .stats { display: flex; gap: 24px; padding: 12px 20px; background: var(--surface); border-bottom: 1px solid var(--border); flex-wrap: wrap; }
  .stat { display: flex; flex-direction: column; }
  .stat-value { font-size: 20px; font-weight: 600; color: var(--text-bright); }
  .stat-label { font-size: 12px; color: var(--text-dim); text-transform: uppercase; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin: 12px 0; overflow: hidden; }
  .card-header { padding: 12px 16px; display: flex; align-items: center; justify-content: space-between; cursor: pointer; user-select: none; }
  .card-header:hover { background: #1c2129; }
  .card-ticker { font-size: 16px; font-weight: 600; color: var(--text-bright); margin-right: 12px; }
  .card-name { color: var(--text-dim); font-size: 13px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .card-filed { color: var(--blue); font-size: 11px; margin-right: 12px; }
  .badges { display: flex; gap: 6px; flex-wrap: wrap; }
  .badge { padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
  .badge-high { background: var(--red-bg); color: var(--red); border: 1px solid var(--red); }
  .badge-medium { background: var(--green-bg); color: var(--green); border: 1px solid var(--green); }
  .badge-warning { background: var(--yellow-bg); color: var(--yellow); border: 1px solid var(--yellow); }
  .card-body { display: none; padding: 0 16px 16px; }
  .card.expanded .card-body { display: block; }
  .signal-list { list-style: none; margin: 8px 0; }
  .signal-list li { padding: 6px 0; border-bottom: 1px solid var(--border); display: flex; gap: 8px; align-items: baseline; }
  .signal-list li:last-child { border-bottom: none; }
  .signal-quarter { color: var(--blue); font-size: 12px; white-space: nowrap; min-width: 100px; }
  .signal-name { font-weight: 500; }
  .signal-detail { color: var(--text-dim); font-size: 12px; }
  table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }
  th { text-align: right; padding: 6px 10px; color: var(--text-dim); font-weight: 500; border-bottom: 1px solid var(--border); white-space: nowrap; }
  td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  .positive { color: var(--green); }
  .negative { color: var(--red); }
  .no-data { color: var(--text-dim); font-style: italic; text-align: center; padding: 40px; }
  .expand-icon { color: var(--text-dim); transition: transform 0.2s; }
  .card.expanded .expand-icon { transform: rotate(90deg); }
  .footer { text-align: center; padding: 20px; color: var(--text-dim); font-size: 12px; }
  .loading { text-align: center; padding: 40px; color: var(--text-dim); }
</style>
</head>
<body>
<header>
  <h1>Signal Screener <span class="status" id="status">Live</span></h1>
  <div class="filter-bar">
    <div class="filter-group">
      <label>Ticker:</label>
      <input type="text" id="tickerFilter" placeholder="Search..." onkeydown="if(event.key==='Enter')applyAllFilters()">
    </div>
    <div class="filter-group">
      <label>Severity:</label>
      <select id="severityFilter">
        <option value="">All</option>
        <option value="HIGH">HIGH only</option>
        <option value="MEDIUM">MEDIUM+</option>
        <option value="WARNING">WARNING+</option>
      </select>
    </div>
    <div class="filter-group">
      <label>Signal:</label>
      <select id="signalFilter">
        <option value="">All Types</option>
        <option value="TURNED POSITIVE">Turned Positive</option>
        <option value="INCREMENTAL">Incremental Margin</option>
        <option value="ACCELERATION">Rev Acceleration</option>
        <option value="DECELERATION">Rev Deceleration</option>
        <option value="INFLECTION">GM Inflection</option>
        <option value="LEVERAGE">Op Leverage</option>
        <option value="FCF">FCF Related</option>
      </select>
    </div>
    <div class="filter-group">
      <label>Quarter From:</label>
      <input type="text" id="fromFilter" placeholder="202501" style="width:80px" oninput="formatDateInput(this, event)" maxlength="7" onkeydown="if(event.key==='Enter')applyAllFilters()">
    </div>
    <div class="filter-group">
      <label>Quarter To:</label>
      <input type="text" id="toFilter" placeholder="202512" style="width:80px" oninput="formatDateInput(this, event)" maxlength="7" onkeydown="if(event.key==='Enter')applyAllFilters()">
    </div>
    <div class="filter-group">
      <label>Filed in last:</label>
      <input type="number" id="recentDays" placeholder="days" style="width:70px" min="1" value="" onkeydown="if(event.key==='Enter')applyAllFilters()">
      <span style="color:var(--text-dim);font-size:12px">days</span>
    </div>
    <button class="btn btn-apply" onclick="applyAllFilters()">Apply Filters</button>
    <button class="btn-clear" onclick="clearFilters()">&#x2715; Clear All</button>
    <button class="btn-secondary btn-small" onclick="refreshData()">Refresh Data</button>
  </div>
</header>
<div class="stats" id="statsBar">
  <div class="stat"><span class="stat-value" id="totalCount">-</span><span class="stat-label">Total Cached</span></div>
  <div class="stat"><span class="stat-value" id="matchedCount">-</span><span class="stat-label">Matching</span></div>
  <div class="stat"><span class="stat-value" id="lastUpdate">-</span><span class="stat-label">Last Update</span></div>
</div>
<div class="container" id="mainContent">
  <div class="loading">Loading data...</div>
</div>
<div class="footer">
  Signal Screener | Press Ctrl+C in terminal to stop server
</div>

<script>
let DATA = {all_results: {}, recent_filers: []};
const SEVERITY_RANK = {HIGH: 3, MEDIUM: 2, WARNING: 1};

function fmtPct(v) {
  if (v == null || isNaN(v)) return '';
  return (v >= 0 ? '+' : '') + (v * 100).toFixed(1) + '%';
}
function fmtB(v) {
  if (v == null || isNaN(v)) return '';
  let s = v < 0 ? '-' : '', a = Math.abs(v);
  if (a >= 1e9) return s + '$' + (a/1e9).toFixed(1) + 'B';
  if (a >= 1e6) return s + '$' + (a/1e6).toFixed(0) + 'M';
  return s + '$' + a.toLocaleString();
}
function valClass(v) { return v > 0 ? 'positive' : v < 0 ? 'negative' : ''; }

function formatDateInput(input, event) {
  // Check if user is deleting
  const isDeleting = event && event.inputType && event.inputType.startsWith('delete');

  // Remove all non-digits
  let v = input.value.replace(/\D/g, '');

  // Only auto-format when typing, not when deleting (YYYY/MM format)
  if (!isDeleting && v.length >= 4) {
    v = v.slice(0,4) + '/' + v.slice(4);
  }
  input.value = v.slice(0, 7);
}

function parseDateFilter(val) {
  // Convert YYYY/MM or YYYYMM to comparable format
  if (!val) return null;
  const clean = val.replace(/\D/g, '');
  if (clean.length >= 6) {
    return clean.slice(0,4) + '-' + clean.slice(4,6);
  }
  return null;
}

function renderCard(ticker, data, filedInfo) {
  const signals = data.signals || [];
  const quarters = data.quarters || [];
  const name = data.name || '';

  const high = signals.filter(s => s.severity === 'HIGH').length;
  const med = signals.filter(s => s.severity === 'MEDIUM').length;
  const warn = signals.filter(s => s.severity === 'WARNING').length;
  let badges = '';
  if (high) badges += `<span class="badge badge-high">${high} HIGH</span>`;
  if (med) badges += `<span class="badge badge-medium">${med} MED</span>`;
  if (warn) badges += `<span class="badge badge-warning">${warn} WARN</span>`;

  let filedHtml = '';
  if (filedInfo) {
    filedHtml = `<span class="card-filed">Filed ${filedInfo.form_type} on ${filedInfo.filed_date}</span>`;
  }

  let signalHtml = '<ul class="signal-list">';
  signals.forEach(s => {
    const c = s.severity === 'HIGH' ? 'var(--red)' : s.severity === 'WARNING' ? 'var(--yellow)' : 'var(--green)';
    signalHtml += `<li>
      <span class="signal-quarter">${s.quarter||''}</span>
      <span class="signal-name" style="color:${c}">${s.signal||''}</span>
      <span class="signal-detail">${s.detail||''}</span>
    </li>`;
  });
  signalHtml += '</ul>';

  let tableHtml = '';
  if (quarters.length > 0) {
    const recent = [...quarters].sort((a,b) => (a.end||a.quarter_end||'').localeCompare(b.end||b.quarter_end||'')).slice(-8);
    tableHtml = `<table>
      <tr>
        <th>Quarter</th><th>Revenue</th><th>RevGr YoY</th><th>Gross%</th>
        <th>OpMar%</th><th>Incr OPM</th><th>FCF%</th><th>Op Income</th>
      </tr>`;
    recent.forEach(q => {
      tableHtml += `<tr>
        <td style="text-align:left">${q.quarter_label||''}</td>
        <td>${fmtB(q.revenue)}</td>
        <td class="${valClass(q.rev_growth_yoy)}">${fmtPct(q.rev_growth_yoy)}</td>
        <td class="${valClass(q.gross_margin)}">${fmtPct(q.gross_margin)}</td>
        <td class="${valClass(q.op_margin)}">${fmtPct(q.op_margin)}</td>
        <td class="${valClass(q.incr_op_margin_yoy)}">${fmtPct(q.incr_op_margin_yoy)}</td>
        <td class="${valClass(q.fcf_margin)}">${fmtPct(q.fcf_margin)}</td>
        <td>${fmtB(q.operating_income)}</td>
      </tr>`;
    });
    tableHtml += '</table>';
  }

  return `<div class="card" onclick="this.classList.toggle('expanded')">
    <div class="card-header">
      <span class="card-ticker">${ticker}</span>
      <span class="card-name">${name}</span>
      ${filedHtml}
      <div class="badges">${badges}</div>
      <span class="expand-icon">&#9654;</span>
    </div>
    <div class="card-body">
      ${signalHtml}
      ${tableHtml}
    </div>
  </div>`;
}

function applyFilters() {
  const tickerQ = document.getElementById('tickerFilter').value.toUpperCase().trim();
  const severityQ = document.getElementById('severityFilter').value;
  const signalQ = document.getElementById('signalFilter').value.toUpperCase();
  const fromDate = parseDateFilter(document.getElementById('fromFilter').value);
  const toDate = parseDateFilter(document.getElementById('toFilter').value);

  const minRank = severityQ ? SEVERITY_RANK[severityQ] : 0;

  // Determine which tickers to show
  let tickersToShow = Object.keys(DATA.all_results);
  const recentFilerTickers = new Set(DATA.recent_filers.map(f => f.ticker));
  const recentDays = document.getElementById('recentDays').value;

  console.log(`[Filter] Total cached: ${tickersToShow.length}, Recent filers: ${DATA.recent_filers.length}, Days input: ${recentDays}`);

  // If recent filers filter is active
  let recentFilterActive = false;
  if (recentDays && DATA.recent_filers.length > 0) {
    recentFilterActive = true;
    const beforeFilter = tickersToShow.length;
    tickersToShow = tickersToShow.filter(t => recentFilerTickers.has(t));
    console.log(`[Filter] After recent filers filter: ${tickersToShow.length} (was ${beforeFilter})`);
  }

  let matched = 0;
  const total = Object.keys(DATA.all_results).length;
  const container = document.getElementById('mainContent');
  let html = '';

  // Sort by signal score
  const sorted = tickersToShow.slice().sort((a, b) => {
    const sa = DATA.all_results[a]?.signals || [];
    const sb = DATA.all_results[b]?.signals || [];
    const scoreA = sa.filter(s=>s.severity==='HIGH').length * 10 + sa.length;
    const scoreB = sb.filter(s=>s.severity==='HIGH').length * 10 + sb.length;
    return scoreB - scoreA;
  });

  sorted.forEach(ticker => {
    const data = DATA.all_results[ticker];
    if (!data) return;
    let signals = data.signals || [];
    const allQuarters = data.quarters || [];  // Keep full history for display

    // Ticker filter
    if (tickerQ && !ticker.includes(tickerQ)) return;

    // Date range filter - filter quarters to determine which signals to show
    let filteredQuarters = allQuarters;
    if (fromDate) {
      filteredQuarters = filteredQuarters.filter(q => (q.date || '').slice(0, 7) >= fromDate);
    }
    if (toDate) {
      filteredQuarters = filteredQuarters.filter(q => (q.date || '').slice(0, 7) <= toDate);
    }
    if (filteredQuarters.length === 0) return;  // Skip if no quarters in range

    // Filter signals to match filtered quarters (only signals within date range)
    const validLabels = new Set(filteredQuarters.map(q => q.quarter_label));
    signals = signals.filter(s => validLabels.has(s.quarter));

    // Severity filter - but still show company if it has data (just no signals)
    let filteredSignals = signals;
    if (minRank > 0) {
      filteredSignals = signals.filter(s => (SEVERITY_RANK[s.severity] || 0) >= minRank);
      if (filteredSignals.length === 0 && signals.length > 0) return; // Skip only if had signals but none matched
    }

    // Signal type filter
    if (signalQ) {
      filteredSignals = filteredSignals.filter(s => (s.signal || '').toUpperCase().includes(signalQ));
      if (filteredSignals.length === 0 && signals.length > 0) return; // Skip only if had signals but none matched
    }

    matched++;
    const filedInfo = DATA.recent_filers.find(f => f.ticker === ticker);
    // Pass filtered signals but FULL quarters history for display
    const renderData = {...data, signals: filteredSignals, quarters: allQuarters};
    html += renderCard(ticker, renderData, filedInfo);
  });

  // Show helpful message if no results
  if (!html) {
    if (total === 0) {
      html = '<div class="no-data">No cached data found. Run <code>python signal_screener.py scan</code> first to build the cache.</div>';
    } else if (recentFilterActive && tickersToShow.length === 0) {
      html = `<div class="no-data">None of the ${DATA.recent_filers.length} companies that filed in the last ${recentDays} days are in the cache.<br>Clear the "Filed in last" field to see all cached data, or the <code>--update</code> flag will fetch their data on next startup.</div>`;
    } else {
      html = '<div class="no-data">No companies match the filter criteria.</div>';
    }
  }

  container.innerHTML = html;
  document.getElementById('totalCount').textContent = total;
  document.getElementById('matchedCount').textContent = matched;
}

async function refreshData() {
  document.getElementById('status').textContent = 'Loading...';
  document.getElementById('status').style.background = 'var(--yellow-bg)';
  document.getElementById('status').style.color = 'var(--yellow)';
  console.log('[RefreshData] Fetching /api/data...');
  try {
    const resp = await fetch('/api/data');
    DATA = await resp.json();
    console.log(`[RefreshData] Loaded ${Object.keys(DATA.all_results).length} companies`);
    if (DATA.error) {
      console.error('[RefreshData] Server error:', DATA.error);
    }
    document.getElementById('lastUpdate').textContent = DATA.generated || 'Now';
    updateStatusLive();
    applyFilters();
  } catch (e) {
    document.getElementById('status').textContent = 'Error';
    document.getElementById('status').style.background = 'var(--red-bg)';
    document.getElementById('status').style.color = 'var(--red)';
    document.getElementById('mainContent').innerHTML = '<div class="no-data">Failed to load data. Check if server is running.</div>';
    console.error('[RefreshData] Error:', e);
  }
}

async function applyAllFilters() {
  const days = document.getElementById('recentDays').value;

  // If "Filed in last N days" is set, fetch filers first
  if (days && days > 0) {
    document.getElementById('status').textContent = 'Fetching filers...';
    document.getElementById('status').style.background = 'var(--yellow-bg)';
    document.getElementById('status').style.color = 'var(--yellow)';
    console.log(`[ApplyFilters] Fetching filers for last ${days} days...`);

    try {
      const resp = await fetch(`/api/recent-filers?days=${days}`);
      const result = await resp.json();
      DATA.recent_filers = result.filers || [];
      console.log(`[ApplyFilters] Got ${DATA.recent_filers.length} filers from SEC`);
    } catch (e) {
      document.getElementById('status').textContent = 'Error fetching filers';
      document.getElementById('status').style.background = 'var(--red-bg)';
      document.getElementById('status').style.color = 'var(--red)';
      console.error('[ApplyFilters] Error:', e);
      return;
    }
  } else {
    // Clear recent filers if no days specified
    DATA.recent_filers = [];
  }

  // Now apply all filters
  applyFilters();
  updateStatusLive();
}

function updateStatusLive() {
  const count = Object.keys(DATA.all_results).length;
  document.getElementById('status').textContent = count > 0 ? 'Live' : 'Empty Cache';
  document.getElementById('status').style.background = count > 0 ? 'var(--green-bg)' : 'var(--yellow-bg)';
  document.getElementById('status').style.color = count > 0 ? 'var(--green)' : 'var(--yellow)';
}

function clearFilters() {
  document.getElementById('tickerFilter').value = '';
  document.getElementById('severityFilter').value = '';
  document.getElementById('signalFilter').value = '';
  document.getElementById('fromFilter').value = '';
  document.getElementById('toFilter').value = '';
  document.getElementById('recentDays').value = '';
  DATA.recent_filers = [];
  updateStatusLive();
  applyFilters();
}

// Initial load
console.log('[Init] Starting Signal Screener...');
refreshData();
</script>
</body>
</html>"""


class SignalScreenerHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler for Signal Screener server."""

    def do_GET(self):
        parsed = urlparse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/data":
            self._serve_data()
        elif path.startswith("/api/recent-filers"):
            query = urlparse.parse_qs(parsed.query)
            days = int(query.get("days", [7])[0])
            self._serve_recent_filers(days)
        else:
            self.send_error(404, "Not Found")

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_SERVE_HTML_TEMPLATE.encode("utf-8"))

    def _serve_data(self):
        """Serve all cached results as JSON."""
        all_results = {}
        error_msg = None
        try:
            if os.path.isdir(RESULTS_CACHE_DIR):
                for fname in os.listdir(RESULTS_CACHE_DIR):
                    if not fname.endswith(".json"):
                        continue
                    ticker = fname[:-5]
                    try:
                        result = _load_result(ticker)
                        if result and result.get("quarters"):
                            all_results[ticker] = _make_serializable(result)
                    except Exception as e:
                        print(f"  Error loading {ticker}: {e}")
            print(f"  Serving {len(all_results)} companies")
        except Exception as e:
            error_msg = str(e)
            print(f"Error in _serve_data: {e}")

        data = {
            "all_results": all_results,
            "recent_filers": [],
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "error": error_msg,
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _serve_recent_filers(self, days):
        """Serve recent filers list."""
        try:
            filers = _fetch_recent_filers(days)
        except Exception as e:
            filers = []
            print(f"Error fetching recent filers: {e}")

        data = {"filers": filers, "days": days}

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format, *args):
        # Suppress default logging, use custom format
        pass


def _run_serve_mode(args):
    """Start local HTTP server for Signal Screener."""
    # Render/Heroku set PORT env var; use it if available
    port = int(os.environ.get("PORT", args.port))
    # Auto-disable browser on cloud (RENDER=true or PORT env set)
    no_browser = args.no_browser or os.environ.get("RENDER") or os.environ.get("PORT")
    do_update = args.update
    days = args.days

    print(f"{BOLD}{CYAN}Signal Screener -- Local Server Mode{RESET}")
    print(f"  Port: {port}")

    # Optional: run incremental update first
    if do_update:
        print(f"\n{DIM}Running incremental update (last {days} days)...{RESET}")
        # Create a minimal args object for update
        class UpdateArgs:
            pass
        update_args = UpdateArgs()
        update_args.days = days
        update_args.min_revenue = "5M"
        update_args.no_report = True
        _run_update_mode(update_args)
        print()

    # Count cached companies
    n_cached = 0
    if os.path.isdir(RESULTS_CACHE_DIR):
        n_cached = len([f for f in os.listdir(RESULTS_CACHE_DIR) if f.endswith(".json")])
    print(f"  Cached companies: {n_cached}")
    if n_cached == 0:
        print(f"\n{YELLOW}Warning: No cached data found!{RESET}")
        print(f"{DIM}Run 'python signal_screener.py scan' to build the initial cache.{RESET}")

    url = f"http://localhost:{port}"
    print(f"\n{GREEN}Server running at: {url}{RESET}")
    print(f"{DIM}Press Ctrl+C to stop{RESET}\n")

    # Open browser
    if not no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    # Start server
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", port), SignalScreenerHandler) as httpd:
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print(f"\n{DIM}Server stopped.{RESET}")
    except OSError as e:
        print(f"\n{RED}Error starting server: {e}{RESET}")
        print(f"{DIM}Port {port} may be in use. Try --port {port+1}{RESET}")


# ══════════════════════════════════════════════════════════════════════
#  CLI parsing
# ══════════════════════════════════════════════════════════════════════

def _parse_ticker_mode():
    """Parse args for ticker mode (backward compatible)."""
    parser = argparse.ArgumentParser(
        description="Signal Screener -- Detect financial inflection points"
    )
    parser.add_argument("tickers", nargs="*", default=["NVDA"],
                        help="Ticker symbols (default: NVDA)")
    parser.add_argument("--quarters", "-q", type=int, default=20,
                        help="Number of quarters to fetch (default: 20)")
    parser.add_argument("--show", "-s", type=int, default=12,
                        help="Number of quarters to display (default: 12)")
    parser.add_argument("--source", choices=["auto", "sec", "yf"], default="auto",
                        help="Data source (default: auto = SEC first, yfinance fallback)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override incremental margin threshold")
    parser.add_argument("--check-new", action="store_true",
                        help="Only show NEW signals (not in cache) from recent quarters")
    parser.add_argument("--days", type=int, default=30,
                        help="Days to look back for --check-new (default: 30)")
    return parser.parse_args()


def _parse_subcommand(mode):
    """Parse args for scan/update/report subcommands."""
    parser = argparse.ArgumentParser(
        description=f"Signal Screener -- {mode} mode"
    )
    parser.add_argument("mode", help="Subcommand (scan/update/report)")

    if mode == "scan":
        parser.add_argument("--min-revenue", default="5M",
                            help="Minimum quarterly revenue filter (default: 5M)")
        parser.add_argument("--max-companies", type=int, default=0,
                            help="Max companies to scan (0=all, for testing)")
        parser.add_argument("--force-refresh", action="store_true",
                            help="Force re-download of SEC data")
        parser.add_argument("--no-report", action="store_true",
                            help="Skip automatic HTML report generation")
    elif mode == "update":
        parser.add_argument("--days", type=int, default=3,
                            help="Number of days to look back (default: 3)")
        parser.add_argument("--min-revenue", default="5M",
                            help="Minimum quarterly revenue filter (default: 5M)")
        parser.add_argument("--all", action="store_true",
                            help="Show all signals from matched companies (not just new ones)")
        parser.add_argument("--review", action="store_true",
                            help="Review mode: show all new signals from scan_history within --days, no API calls")
        parser.add_argument("--no-report", action="store_true",
                            help="Skip automatic HTML report generation")
    elif mode == "report":
        parser.add_argument("--output", default=os.path.join(REPORTS_DIR, "signal_report.html"),
                            help="Output HTML file path")
    elif mode == "filter":
        parser.add_argument("--recent-quarters", type=int, default=4,
                            help="Only look at signals/metrics from last N quarters (default: 4)")
        parser.add_argument("--severity", default=None, choices=["HIGH", "MEDIUM", "WARNING"],
                            help="Minimum signal severity level")
        parser.add_argument("--signal", default=None,
                            help="Signal name keyword match (case-insensitive)")
        parser.add_argument("--incr-margin-spread", type=float, default=None,
                            help="Incr OPM must exceed quarter OPM by N pp (e.g. 20)")
        parser.add_argument("--min-rev-growth", type=float, default=None,
                            help="Min YoY revenue growth (0.3 = 30%%)")
        parser.add_argument("--min-op-margin", type=float, default=None,
                            help="Min operating margin (0.1 = 10%%)")
        parser.add_argument("--top", type=int, default=50,
                            help="Show top N results (default: 50)")
        parser.add_argument("--html", action="store_true",
                            help="Generate HTML report for filtered results")
        parser.add_argument("--after", default=None,
                            help="Only include companies whose latest quarter is after this date (e.g. 2025-01)")
        parser.add_argument("--sort", default="score",
                            choices=["score", "revenue", "op_margin", "rev_growth"],
                            help="Sort results by (default: score)")
    elif mode == "serve":
        parser.add_argument("--port", type=int, default=8000,
                            help="Port to serve on (default: 8000)")
        parser.add_argument("--no-browser", action="store_true",
                            help="Don't auto-open browser")
        parser.add_argument("--update", action="store_true",
                            help="Run incremental update before serving")
        parser.add_argument("--days", type=int, default=3,
                            help="Days for incremental update (default: 3)")

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
#  Mode runners
# ══════════════════════════════════════════════════════════════════════

def _run_ticker_mode(args):
    """Original ticker mode: fetch and display signals for given tickers."""
    if args.threshold is not None:
        THRESHOLDS["incr_margin_yoy"] = args.threshold

    tickers = [t.upper() for t in args.tickers]
    check_new = getattr(args, 'check_new', False)
    days_back = getattr(args, 'days', 30)

    if check_new:
        print(f"{BOLD}{CYAN}Signal Screener -- Check New Signals{RESET}")
        print(f"  Tickers: {', '.join(tickers)}")
        print(f"  Looking back: {days_back} days")
        print()
        cutoff = datetime.now() - timedelta(days=days_back)
        new_signals_found = []

    all_results = []

    for ticker in tickers:
        print(f"\n{DIM}Fetching {ticker}...{RESET}", end="", flush=True)
        try:
            # Load old cached signals if checking for new
            old_signal_keys = set()
            if check_new:
                old_result = _load_result(ticker)
                if old_result and old_result.get("signals"):
                    for s in old_result["signals"]:
                        old_signal_keys.add((s.get("quarter", ""), s.get("signal", "")))

            data = fetch_data(ticker, args.quarters, args.source)
            if data is None:
                print(f" {RED}FAILED (no data){RESET}")
                continue
            src = data.get("source", "?")
            print(f" {GREEN}OK{RESET} ({len(data['quarters'])}Q via {src})")

            signals = compute_signals(data)

            if check_new:
                # Filter to only NEW signals from recent quarters
                new_sigs = []
                for s in signals:
                    key = (s.get("quarter", ""), s.get("signal", ""))
                    if key in old_signal_keys:
                        continue
                    # Check if signal quarter is within the days range
                    q_label = s.get("quarter", "")
                    # quarter_label format: "2025-10 (Q4)" -> extract date
                    if q_label:
                        try:
                            q_date_str = q_label.split(" ")[0]  # "2025-10"
                            q_date = datetime.strptime(q_date_str + "-01", "%Y-%m-%d")
                            if q_date >= cutoff:
                                new_sigs.append(s)
                        except (ValueError, IndexError):
                            pass

                if new_sigs:
                    new_signals_found.append({
                        "ticker": ticker,
                        "name": data.get("name", ticker),
                        "signals": new_sigs,
                    })
                    # Print summary for this ticker
                    print(f"  {YELLOW}Found {len(new_sigs)} new signal(s){RESET}")
                else:
                    print(f"  {DIM}No new signals in last {days_back} days{RESET}")

                # Save updated result to cache
                _save_result(ticker, {
                    "ticker": ticker,
                    "name": data.get("name", ticker),
                    "quarters": data.get("quarters", []),
                    "signals": _make_serializable(signals),
                    "last_updated": datetime.now().isoformat(),
                })
            else:
                # Normal mode: show full report
                print_company_report(data, signals, args.show)

            all_results.append((data, signals))
        except Exception as e:
            print(f" {RED}ERROR: {e}{RESET}")
            import traceback
            traceback.print_exc()

        # SEC rate limiting (handled by RateLimiter now, but keep small gap for multi-ticker)
        if len(tickers) > 1:
            time.sleep(0.1)

    if check_new:
        # Print summary of new signals
        print(f"\n{BOLD}{'='*60}{RESET}")
        if new_signals_found:
            print(f"{BOLD}{YELLOW}New signals detected:{RESET}")
            for item in new_signals_found:
                print(f"\n  {BOLD}{item['ticker']}{RESET} ({item['name'][:40]})")
                for s in item["signals"]:
                    sev = s.get("severity", "MEDIUM")
                    if sev == "HIGH":
                        color, icon = RED, "[!!]"
                    elif sev == "WARNING":
                        color, icon = YELLOW, "[!]"
                    else:
                        color, icon = GREEN, "[+]"
                    print(f"    {icon} {color}[{s['quarter']}] {s['signal']}{RESET}")
                    print(f"        {s['detail']}")
        else:
            print(f"{DIM}No new signals found for the specified tickers in the last {days_back} days.{RESET}")
    elif len(tickers) > 1:
        print_summary(all_results)

    print()


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    # Detect mode: if first arg is a subcommand, route accordingly
    if len(sys.argv) > 1 and sys.argv[1] in SUBCOMMANDS:
        mode = sys.argv[1]
        args = _parse_subcommand(mode)

        if mode == "scan":
            _run_scan_mode(args)
        elif mode == "update":
            if getattr(args, 'review', False):
                _run_review_mode(args)
            else:
                _run_update_mode(args)
        elif mode == "report":
            _run_report_mode(args)
        elif mode == "filter":
            _run_filter_mode(args)
        elif mode == "serve":
            _run_serve_mode(args)
    else:
        args = _parse_ticker_mode()
        _run_ticker_mode(args)


if __name__ == "__main__":
    main()
