"""
Export Signal Screener data to a static HTML file for deployment.
Outputs a single index.html with embedded JSON data.

Usage:
  python export_static.py                    # Export to docs/index.html
  python export_static.py --output out.html  # Custom output path
  python export_static.py --days 365         # Only signals from last N days
"""

import os
import json
import argparse
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CACHE_DIR = os.path.join(SCRIPT_DIR, "cache", "results")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "docs", "index.html")


def export_compact_data(days=365, min_severity="MEDIUM"):
    """Export compact data from cache."""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m")
    severity_ok = {"HIGH", "MEDIUM"} if min_severity == "MEDIUM" else {"HIGH"}

    compact_data = {}

    for fname in os.listdir(RESULTS_CACHE_DIR):
        if not fname.endswith(".json"):
            continue
        ticker = fname[:-5]

        try:
            with open(os.path.join(RESULTS_CACHE_DIR, fname), "r") as f:
                data = json.load(f)
        except Exception:
            continue

        signals = data.get("signals", [])
        # Filter signals by severity and date
        recent_signals = [
            s for s in signals
            if s.get("severity") in severity_ok
            and s.get("quarter", "")[:7] >= cutoff
        ]

        if not recent_signals:
            continue

        # Keep last 6 quarters, minimal fields
        quarters = data.get("quarters", [])[-6:]
        compact_quarters = []
        for q in quarters:
            cq = {"q": q.get("quarter_label", "")}
            if q.get("revenue"):
                cq["r"] = q["revenue"]
            if q.get("gross_margin") is not None:
                cq["gm"] = round(q["gross_margin"] * 100, 1)
            if q.get("op_margin") is not None:
                cq["om"] = round(q["op_margin"] * 100, 1)
            if q.get("rev_growth_yoy") is not None:
                cq["rg"] = round(q["rev_growth_yoy"] * 100, 1)
            if q.get("incr_op_margin_yoy") is not None:
                cq["im"] = round(q["incr_op_margin_yoy"] * 100, 1)
            if q.get("fcf_margin") is not None:
                cq["fm"] = round(q["fcf_margin"] * 100, 1)
            if q.get("operating_income") is not None:
                cq["oi"] = q["operating_income"]
            compact_quarters.append(cq)

        # Compact signal format: q=quarter, t=type, d=detail, v=severity (H/M)
        compact_signals = [
            {
                "q": s["quarter"],
                "t": s["signal"],
                "d": s["detail"],
                "v": s["severity"][0]  # H or M
            }
            for s in recent_signals
        ]

        compact_data[ticker] = {
            "n": data.get("name", ""),
            "s": compact_signals,
            "qs": compact_quarters
        }

    return compact_data


def generate_static_html(data, generated_time):
    """Generate static HTML with embedded data."""

    json_data = json.dumps(data, separators=(",", ":"))

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Signal Screener</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #c9d1d9; --text-dim: #8b949e; --text-bright: #f0f6fc;
    --red: #f85149; --green: #3fb950; --yellow: #d29922; --blue: #58a6ff;
    --red-bg: #f8514920; --green-bg: #3fb95020; --yellow-bg: #d2992220;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 14px; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
  header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 16px 20px; }}
  header h1 {{ color: var(--text-bright); font-size: 20px; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center; gap: 12px; }}
  header h1 .status {{ font-size: 12px; padding: 4px 8px; border-radius: 4px; background: var(--blue); color: white; }}
  .filter-bar {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }}
  .filter-bar input, .filter-bar select {{ background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; font-size: 14px; }}
  .filter-bar input[type="text"] {{ width: 140px; }}
  .filter-bar label {{ color: var(--text-dim); font-size: 13px; margin-right: 4px; }}
  .filter-group {{ display: flex; align-items: center; gap: 4px; }}
  .btn {{ background: var(--green); color: white; border: none; border-radius: 6px; padding: 6px 16px; font-size: 14px; cursor: pointer; font-weight: 600; }}
  .btn:hover {{ opacity: 0.9; }}
  .btn-clear {{ background: transparent; border: 1px solid var(--text-dim); color: var(--text-dim); border-radius: 6px; padding: 6px 14px; font-size: 13px; cursor: pointer; }}
  .btn-clear:hover {{ border-color: var(--yellow); color: var(--yellow); }}
  .stats {{ display: flex; gap: 24px; padding: 12px 20px; background: var(--surface); border-bottom: 1px solid var(--border); flex-wrap: wrap; }}
  .stat {{ display: flex; flex-direction: column; }}
  .stat-value {{ font-size: 20px; font-weight: 600; color: var(--text-bright); }}
  .stat-label {{ font-size: 12px; color: var(--text-dim); text-transform: uppercase; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin: 12px 0; overflow: hidden; }}
  .card-header {{ padding: 12px 16px; display: flex; align-items: center; justify-content: space-between; cursor: pointer; user-select: none; }}
  .card-header:hover {{ background: #1c2129; }}
  .card-ticker {{ font-size: 16px; font-weight: 600; color: var(--text-bright); margin-right: 12px; }}
  .card-name {{ color: var(--text-dim); font-size: 13px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .badges {{ display: flex; gap: 6px; flex-wrap: wrap; }}
  .badge {{ padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }}
  .badge-high {{ background: var(--red-bg); color: var(--red); border: 1px solid var(--red); }}
  .badge-medium {{ background: var(--green-bg); color: var(--green); border: 1px solid var(--green); }}
  .card-body {{ display: none; padding: 0 16px 16px; }}
  .card.expanded .card-body {{ display: block; }}
  .signal-list {{ list-style: none; margin: 8px 0; }}
  .signal-list li {{ padding: 6px 0; border-bottom: 1px solid var(--border); display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }}
  .signal-list li:last-child {{ border-bottom: none; }}
  .signal-quarter {{ color: var(--blue); font-size: 12px; white-space: nowrap; min-width: 100px; }}
  .signal-name {{ font-weight: 500; }}
  .signal-detail {{ color: var(--text-dim); font-size: 12px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }}
  th {{ text-align: right; padding: 6px 10px; color: var(--text-dim); font-weight: 500; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  td {{ text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  th:first-child, td:first-child {{ text-align: left; }}
  .positive {{ color: var(--green); }}
  .negative {{ color: var(--red); }}
  .no-data {{ color: var(--text-dim); font-style: italic; text-align: center; padding: 40px; }}
  .expand-icon {{ color: var(--text-dim); transition: transform 0.2s; }}
  .card.expanded .expand-icon {{ transform: rotate(90deg); }}
  .footer {{ text-align: center; padding: 20px; color: var(--text-dim); font-size: 12px; }}
  @media (max-width: 768px) {{
    .filter-bar {{ flex-direction: column; align-items: stretch; }}
    .filter-group {{ width: 100%; }}
    .filter-bar input[type="text"] {{ width: 100%; }}
    table {{ font-size: 11px; }}
    th, td {{ padding: 4px 6px; }}
  }}
</style>
</head>
<body>
<header>
  <h1>Signal Screener <span class="status">Static</span></h1>
  <div class="filter-bar">
    <div class="filter-group">
      <label>Ticker:</label>
      <input type="text" id="tickerFilter" placeholder="Search..." onkeydown="if(event.key==='Enter')applyFilters()">
    </div>
    <div class="filter-group">
      <label>Severity:</label>
      <select id="severityFilter" onchange="applyFilters()">
        <option value="">All</option>
        <option value="H">HIGH only</option>
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
    <button class="btn" onclick="applyFilters()">Apply</button>
    <button class="btn-clear" onclick="clearFilters()">Clear</button>
  </div>
</header>
<div class="stats">
  <div class="stat"><span class="stat-value" id="totalCount">-</span><span class="stat-label">Total</span></div>
  <div class="stat"><span class="stat-value" id="matchedCount">-</span><span class="stat-label">Showing</span></div>
  <div class="stat"><span class="stat-value" id="lastUpdate">{generated_time}</span><span class="stat-label">Generated</span></div>
</div>
<div class="container" id="mainContent">
  <div class="no-data">Loading...</div>
</div>
<div class="footer">
  Signal Screener | Data from SEC EDGAR | Signals: HIGH (red) = major inflection, MEDIUM (green) = notable change
</div>

<script>
const DATA = {json_data};

function fmtPct(v) {{
  if (v == null || isNaN(v)) return '';
  return (v >= 0 ? '+' : '') + v.toFixed(1) + '%';
}}
function fmtB(v) {{
  if (v == null || isNaN(v)) return '';
  let s = v < 0 ? '-' : '', a = Math.abs(v);
  if (a >= 1e9) return s + '$' + (a/1e9).toFixed(1) + 'B';
  if (a >= 1e6) return s + '$' + (a/1e6).toFixed(0) + 'M';
  return s + '$' + a.toLocaleString();
}}
function valClass(v) {{ return v > 0 ? 'positive' : v < 0 ? 'negative' : ''; }}

function renderCard(ticker, data) {{
  const signals = data.s || [];
  const quarters = data.qs || [];
  const name = data.n || '';

  const high = signals.filter(s => s.v === 'H').length;
  const med = signals.filter(s => s.v === 'M').length;
  let badges = '';
  if (high) badges += `<span class="badge badge-high">${{high}} HIGH</span>`;
  if (med) badges += `<span class="badge badge-medium">${{med}} MED</span>`;

  let signalHtml = '<ul class="signal-list">';
  signals.forEach(s => {{
    const c = s.v === 'H' ? 'var(--red)' : 'var(--green)';
    signalHtml += `<li>
      <span class="signal-quarter">${{s.q||''}}</span>
      <span class="signal-name" style="color:${{c}}">${{s.t||''}}</span>
      <span class="signal-detail">${{s.d||''}}</span>
    </li>`;
  }});
  signalHtml += '</ul>';

  let tableHtml = '';
  if (quarters.length > 0) {{
    tableHtml = `<table>
      <tr>
        <th>Quarter</th><th>Revenue</th><th>RevGr%</th><th>GM%</th>
        <th>OPM%</th><th>Incr OPM</th><th>FCF%</th><th>Op Inc</th>
      </tr>`;
    quarters.forEach(q => {{
      tableHtml += `<tr>
        <td style="text-align:left">${{q.q||''}}</td>
        <td>${{fmtB(q.r)}}</td>
        <td class="${{valClass(q.rg)}}">${{fmtPct(q.rg)}}</td>
        <td class="${{valClass(q.gm)}}">${{fmtPct(q.gm)}}</td>
        <td class="${{valClass(q.om)}}">${{fmtPct(q.om)}}</td>
        <td class="${{valClass(q.im)}}">${{fmtPct(q.im)}}</td>
        <td class="${{valClass(q.fm)}}">${{fmtPct(q.fm)}}</td>
        <td>${{fmtB(q.oi)}}</td>
      </tr>`;
    }});
    tableHtml += '</table>';
  }}

  return `<div class="card" onclick="this.classList.toggle('expanded')">
    <div class="card-header">
      <span class="card-ticker">${{ticker}}</span>
      <span class="card-name">${{name}}</span>
      <div class="badges">${{badges}}</div>
      <span class="expand-icon">&#9654;</span>
    </div>
    <div class="card-body">
      ${{signalHtml}}
      ${{tableHtml}}
    </div>
  </div>`;
}}

function applyFilters() {{
  const tickerQ = document.getElementById('tickerFilter').value.toUpperCase().trim();
  const severityQ = document.getElementById('severityFilter').value;
  const signalQ = document.getElementById('signalFilter').value.toUpperCase();

  const total = Object.keys(DATA).length;
  let matched = 0;
  let html = '';

  // Sort by HIGH signal count
  const sorted = Object.keys(DATA).sort((a, b) => {{
    const sa = DATA[a]?.s || [];
    const sb = DATA[b]?.s || [];
    const scoreA = sa.filter(s=>s.v==='H').length * 10 + sa.length;
    const scoreB = sb.filter(s=>s.v==='H').length * 10 + sb.length;
    return scoreB - scoreA;
  }});

  sorted.forEach(ticker => {{
    const data = DATA[ticker];
    if (!data) return;
    let signals = data.s || [];

    // Ticker filter
    if (tickerQ && !ticker.includes(tickerQ)) return;

    // Severity filter
    if (severityQ) {{
      signals = signals.filter(s => s.v === severityQ);
      if (signals.length === 0) return;
    }}

    // Signal type filter
    if (signalQ) {{
      signals = signals.filter(s => (s.t || '').toUpperCase().includes(signalQ));
      if (signals.length === 0) return;
    }}

    matched++;
    const renderData = {{...data, s: signals}};
    html += renderCard(ticker, renderData);
  }});

  if (!html) {{
    html = '<div class="no-data">No companies match the filter criteria.</div>';
  }}

  document.getElementById('mainContent').innerHTML = html;
  document.getElementById('totalCount').textContent = total;
  document.getElementById('matchedCount').textContent = matched;
}}

function clearFilters() {{
  document.getElementById('tickerFilter').value = '';
  document.getElementById('severityFilter').value = '';
  document.getElementById('signalFilter').value = '';
  applyFilters();
}}

// Initialize
applyFilters();
</script>
</body>
</html>'''

    return html


def main():
    parser = argparse.ArgumentParser(description="Export Signal Screener to static HTML")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT,
                        help=f"Output file path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--days", "-d", type=int, default=365,
                        help="Include signals from last N days (default: 365)")
    args = parser.parse_args()

    print(f"Exporting Signal Screener data...")
    print(f"  Source: {RESULTS_CACHE_DIR}")
    print(f"  Days: {args.days}")

    # Export data
    data = export_compact_data(days=args.days)
    print(f"  Companies: {len(data)}")

    # Generate HTML
    generated_time = datetime.now().strftime("%Y-%m-%d")
    html = generate_static_html(data, generated_time)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Write file
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(args.output) / 1024
    print(f"  Output: {args.output}")
    print(f"  Size: {size_kb:.0f} KB")
    print(f"\nDone! Open {args.output} in a browser to view.")


if __name__ == "__main__":
    main()
