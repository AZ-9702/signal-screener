# Signal Screener 使用指南

检测美股公司财务拐点信号（增量利润率提升、FCF转正、收入加速等）。

---

## 五种模式

### 1. 单公司/多公司查看
```bash
python D:\claude-test\screener\signal_screener.py NVDA
python D:\claude-test\screener\signal_screener.py NVDA AAPL MSFT
```
终端输出季度数据表 + 信号列表。

**检查特定公司的新信号：**
```bash
# 查看 NVDA、AAPL 最近30天是否有新信号
python D:\claude-test\screener\signal_screener.py NVDA AAPL --check-new --days 30

# 查看最近90天的新信号
python D:\claude-test\screener\signal_screener.py TSLA META --check-new --days 90
```
只显示指定时间范围内、且之前缓存中没有的新信号。

### 2. 全量扫描
```bash
python D:\claude-test\screener\signal_screener.py scan
python D:\claude-test\screener\signal_screener.py scan --max-companies 100   # 测试用
python D:\claude-test\screener\signal_screener.py scan --min-revenue 10M     # 调高收入门槛
```
扫描所有 SEC 公司，过滤季度收入 < 5M 的，自动生成 HTML 报告。

### 3. 增量更新
```bash
python D:\claude-test\screener\signal_screener.py update
python D:\claude-test\screener\signal_screener.py update --days 5
python D:\claude-test\screener\signal_screener.py update --days 15 --all
```
只处理最近 N 天提交 10-Q/10-K 的公司。

| 参数 | 说明 |
|------|------|
| `--days N` | 回溯天数（默认 3） |
| `--all` | 显示所有信号（默认只显示新增信号） |

- 不加 `--all`：只显示缓存中没有的**新增信号**
- 加 `--all`：显示时间范围内所有公司的**全部信号**

### 4. 生成报告
```bash
python D:\claude-test\screener\signal_screener.py report
```
从缓存数据生成 HTML 报告并在浏览器中打开。

### 5. 自定义筛选 (filter)
```bash
python D:\claude-test\screener\signal_screener.py filter
```
直接查询本地缓存的公司数据，支持多种筛选条件组合，无需访问 SEC API。

**筛选参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--recent-quarters N` | 4 | 只看最近 N 个季度内的信号/指标 |
| `--severity LEVEL` | 全部 | 最低级别: `HIGH`, `MEDIUM`, `WARNING` |
| `--signal KEYWORD` | 全部 | 信号名称关键词匹配（模糊, 大小写不敏感） |
| `--incr-margin-spread N` | 关闭 | 增量 OPM 超过当季 OPM 至少 N 个百分点 |
| `--min-rev-growth X` | 关闭 | 最近季度 YoY 收入增速 >= X（0.3 = 30%） |
| `--min-op-margin X` | 关闭 | 最近季度经营利润率 >= X（0.1 = 10%） |
| `--after YYYY-MM` | 关闭 | 只看最新季度数据在该日期之后的公司（排除退市/停报） |
| `--top N` | 50 | 显示前 N 家 |
| `--sort KEY` | score | 排序方式: `score` / `revenue` / `op_margin` / `rev_growth` |
| `--html` | 关闭 | 生成筛选结果的 HTML 报告 |

**示例：**
```bash
# 最近2个季度有信号的公司
python D:\claude-test\screener\signal_screener.py filter --recent-quarters 2

# 增量 OPM 比当季 OPM 高 20pp 以上
python D:\claude-test\screener\signal_screener.py filter --incr-margin-spread 20

# 只看 HIGH 级别信号
python D:\claude-test\screener\signal_screener.py filter --severity HIGH

# OP margin 转正的公司
python D:\claude-test\screener\signal_screener.py filter --signal "TURNED POSITIVE"

# 收入加速 + 只看最近2季
python D:\claude-test\screener\signal_screener.py filter --signal "ACCELERATION" --recent-quarters 2

# 高增长 + 高利润率
python D:\claude-test\screener\signal_screener.py filter --min-rev-growth 0.3 --min-op-margin 0.1

# 排除退市/停报公司，只看2025年后仍有数据的
python D:\claude-test\screener\signal_screener.py filter --severity HIGH --after 2025-01

# 组合筛选 + HTML 报告
python D:\claude-test\screener\signal_screener.py filter --incr-margin-spread 20 --severity HIGH --html

# 按收入排序，看前20
python D:\claude-test\screener\signal_screener.py filter --sort revenue --top 20
```

---

## 输出文件

| 文件 | 说明 |
|------|------|
| `reports\signal_report.html` | HTML 报告（scan/update 后自动生成） |
| `reports\filter_report.html` | 筛选结果 HTML 报告（filter --html 生成） |
| `cache\facts\` | SEC 原始数据缓存（30天有效） |
| `cache\results\` | 每家公司的计算结果 |
| `cache\scan_history.json` | 扫描历史记录 |

---

## 典型用法

```bash
# 首次：全量扫描
python D:\claude-test\screener\signal_screener.py scan

# 之后每天：增量更新
python D:\claude-test\screener\signal_screener.py update

# 快速筛选：从缓存中找高质量信号
python D:\claude-test\screener\signal_screener.py filter --severity HIGH --recent-quarters 2

# 随时查看某公司详情
python D:\claude-test\screener\signal_screener.py NVDA
```

---

## 信号类型

| 信号 | 含义 | 级别 |
|------|------|------|
| OP MARGIN TURNED POSITIVE | 经营利润率从负转正 | HIGH |
| INCREMENTAL MARGIN (YoY) | 增量经营利润率显著高于去年同期 | HIGH/MEDIUM |
| YoY REVENUE GROWTH ACCELERATION | 收入同比增速加速 | HIGH/MEDIUM |
| GROSS MARGIN INFLECTION | 毛利率下降后反转回升 | HIGH/MEDIUM |
| OPERATING LEVERAGE | 收入增速远超费用增速 | MEDIUM |
| FCF TURNED POSITIVE | 自由现金流从负转正 | HIGH |
| FCF MARGIN EXPANSION | 自由现金流利润率显著改善 | MEDIUM |
