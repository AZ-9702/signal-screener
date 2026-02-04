# Signal Screener 使用指南

检测美股公司财务拐点信号（增量利润率提升、FCF转正、收入加速等）。

---

## 快速启动（双击 .bat 文件）

| 文件 | 功能 |
|------|------|
| `signal_serve.bat` | 启动本地服务器（推荐，自动更新 + 浏览器筛选） |
| `signal_report.bat` | 生成静态 HTML 报告 |
| `signal_update.bat` | 增量更新（最近 3 天新财报） |
| `signal_review.bat` | 查看最近 15 天提交财报的公司信号 |

---

## 六种模式

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
| `--review` | 回顾模式：从历史记录聚合显示，不调用 API |

- 不加 `--all`：只显示缓存中没有的**新增信号**
- 加 `--all`：显示时间范围内所有公司的**全部信号**
- 加 `--review`：从 scan_history 回顾过去 N 天所有新增信号（见下方说明）

### 3b. 回顾模式 (--review)
```bash
python D:\claude-test\screener\signal_screener.py update --days 30 --review
python D:\claude-test\screener\signal_screener.py update --days 15 --review
```

**用途**：查看过去 N 天提交财报的公司中，有哪些值得关注的信号。

**流程**：
1. 从 SEC 获取最近 N 天提交 10-Q/10-K 的公司列表
2. 加载每个公司的缓存数据
3. 筛选**最近 2 个季度**的所有 signals（HIGH/MEDIUM/WARNING 全部显示）
4. 显示 signals + 最近 8 季度财务数据（Revenue, RevGr%, GM%, OPM%, IncrOPM, FCF%）

**特点**：
- 不依赖 scan_history，直接查询最新的财报提交情况
- 显示全部级别信号，提供完整视角（正面 + 负面）
- 生成独立报告 `reports/signal_review_{days}d.html`

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

### 6. 本地服务器模式 (serve) ⭐推荐
```bash
python D:\claude-test\screener\signal_screener.py serve
python D:\claude-test\screener\signal_screener.py serve --update        # 启动前先增量更新
python D:\claude-test\screener\signal_screener.py serve --port 9000     # 使用其他端口
```

**特点**：
- 浏览器打开 `http://localhost:8000`
- 启动后显示全量缓存数据（~4000家公司）
- 设置筛选条件后点击 "Apply Filters" 一次性筛选
- 由多到少的筛选逻辑，不会自动触发

**界面功能**：
| 控件 | 说明 |
|------|------|
| Ticker | 按 ticker 搜索 |
| Severity | 筛选信号级别 (HIGH/MEDIUM/WARNING) |
| Signal | 筛选信号类型 |
| Quarter From / To | 季度结束年月范围（输入 `202501` 自动格式化为 `2025/01`） |
| Filed in last N days | 只显示最近 N 天提交财报的公司 |
| **Apply Filters** | 应用所有筛选条件（绿色按钮） |
| ✕ Clear All | 清除所有条件，显示全量数据 |
| Refresh Data | 重新从缓存加载数据 |

**参数**：
| 参数 | 默认 | 说明 |
|------|------|------|
| `--port N` | 8000 | 服务器端口 |
| `--update` | 关闭 | 启动前先运行增量更新 |
| `--days N` | 3 | 增量更新回溯天数 |
| `--no-browser` | 关闭 | 不自动打开浏览器 |

---

## 输出文件

| 文件 | 说明 |
|------|------|
| `reports\signal_master.html` | 完整缓存视图（report 命令生成） |
| `reports\signal_queries.html` | 查询历史报告（update/filter 追加） |
| `reports\signal_review_{N}d.html` | 回顾报告（update --review 生成） |
| `cache\facts\` | SEC 原始数据缓存（30天有效） |
| `cache\results\` | 每家公司的计算结果 |
| `cache\query_history.json` | 查询历史（JSON 格式） |

---

## 典型用法

```bash
# 首次：全量扫描（约 15-20 分钟）
python D:\claude-test\screener\signal_screener.py scan

# 之后每天：双击 signal_serve.bat 或运行
python D:\claude-test\screener\signal_screener.py serve --update

# 在浏览器中进行筛选，无需额外命令
```

**或者用命令行：**
```bash
# 增量更新
python D:\claude-test\screener\signal_screener.py update

# 快速筛选
python D:\claude-test\screener\signal_screener.py filter --severity HIGH --recent-quarters 2

# 查看某公司详情
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
