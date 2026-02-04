# Signal Screener 安装与使用指南

检测美股公司财务拐点信号的命令行工具。

---

## 快速启动（安装完成后）

双击以下 `.bat` 文件即可使用：

| 文件 | 功能 | 说明 |
|------|------|------|
| `signal_serve.bat` | **启动本地服务器** ⭐推荐 | 浏览器筛选界面，自动更新数据 |
| `signal_update.bat` | 增量更新 | 更新最近 3 天有新财报的公司 |
| `signal_review.bat` | 回顾信号 | 查看最近 15 天提交财报的公司 |
| `signal_report.bat` | 生成报告 | 生成静态 HTML 报告 |

---

## 第一步：安装 Python

1. 打开浏览器，访问 https://www.python.org/downloads/
2. 点击黄色按钮 **"Download Python 3.x.x"**
3. 运行下载的安装程序
4. **重要：勾选 "Add Python to PATH"**，然后点击 "Install Now"

验证安装成功：
- 按 `Win + R`，输入 `cmd`，回车打开命令行
- 输入以下命令并回车：
  ```
  python --version
  ```
- 如果显示 `Python 3.x.x` 就说明安装成功

---

## 第二步：下载工具

把整个 `screener` 文件夹复制到你的电脑上，比如放在 `D:\screener`。

文件夹里应该包含：
```
screener/
├── signal_screener.py   （主程序）
├── signal_serve.bat     （双击启动服务器）
├── signal_update.bat    （双击增量更新）
├── signal_review.bat    （双击查看近期信号）
├── signal_report.bat    （双击生成报告）
├── requirements.txt     （依赖列表）
├── USAGE.md             （详细使用说明）
└── README.md            （本文件）
```

---

## 第三步：安装依赖

打开命令行（按 `Win + R`，输入 `cmd`，回车），输入：

```
pip install -r D:\screener\requirements.txt
```

（把 `D:\screener` 换成你实际的路径）

或者直接运行：

```
pip install yfinance
```

等待安装完成即可。

---

## 第四步：首次运行（全量扫描）

首次使用需要扫描所有上市公司数据，这一步只需要做一次。

在命令行输入（根据你的实际路径调整）：

```
python D:\screener\signal_screener.py scan
```

这个过程大约需要 **15-20 分钟**，会：
- 从 SEC（美国证监会）下载所有上市公司的财务数据
- 计算各种财务信号
- 生成 HTML 报告并自动在浏览器中打开

完成后，数据会缓存在本地，之后的操作都会很快。

---

## 日常使用

### 方式一：本地服务器（推荐）⭐

双击 `signal_serve.bat` 或运行：
```bash
python D:\screener\signal_screener.py serve --update
```

浏览器会自动打开 `http://localhost:8000`，显示全量缓存数据（~4000家公司）。

**筛选控件：**
- **Ticker** — 搜索特定公司
- **Severity** — 筛选信号级别 (HIGH/MEDIUM/WARNING)
- **Signal** — 筛选信号类型
- **Quarter From/To** — 季度结束年月范围（输入 `202501` 自动格式化为 `2025/01`）
- **Filed in last N days** — 只显示最近 N 天提交财报的公司

**操作按钮：**
- **Apply Filters**（绿色）— 设置好条件后点击筛选
- **✕ Clear All** — 清除所有条件，回到全量数据
- **Refresh Data** — 重新从缓存加载

按 `Ctrl+C` 停止服务器。

### 方式二：命令行筛选

```bash
# 高信号 + 排除退市公司
python D:\screener\signal_screener.py filter --severity HIGH --after 2025-01

# 增量利润率超高的公司（增量OPM比当季OPM高20个百分点以上）
python D:\screener\signal_screener.py filter --incr-margin-spread 20 --after 2025-01

# 经营利润刚转正的公司
python D:\screener\signal_screener.py filter --signal "TURNED POSITIVE" --after 2025-01

# 收入加速增长
python D:\screener\signal_screener.py filter --signal "ACCELERATION" --after 2025-01

# 高增长 + 高利润率
python D:\screener\signal_screener.py filter --min-rev-growth 0.3 --min-op-margin 0.1 --after 2025-01
```

### 查看单个公司详情

```
python D:\screener\signal_screener.py NVDA
python D:\screener\signal_screener.py AAPL MSFT GOOGL
```

### 检查特定公司的新信号

```bash
# 查看 NVDA、AAPL 最近30天是否有新信号
python D:\screener\signal_screener.py NVDA AAPL --check-new --days 30
```

### 增量更新（每天运行）

只更新最近有新财报的公司，比全量扫描快很多：

```bash
# 只显示新增信号（缓存中没有的）
python D:\screener\signal_screener.py update

# 显示过去15天内申报公司的所有信号
python D:\screener\signal_screener.py update --days 15 --all
```

### 回顾重要信号

```bash
# 查看过去30天提交财报且有重要信号的公司
python D:\screener\signal_screener.py update --days 30 --review

# 查看过去15天
python D:\screener\signal_screener.py update --days 15 --review
```

`--review` 模式会：
1. 从 SEC 查询过去 N 天提交 10-Q/10-K 的公司
2. 显示每个公司的**重要信号**（HIGH/MEDIUM 级别）
3. 同时显示最近 8 季度的财务数据

适合快速了解"最近提交财报的公司中有哪些值得关注"。

---

## 筛选参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--severity` | 最低信号级别 | `--severity HIGH` |
| `--signal` | 信号关键词 | `--signal "TURNED POSITIVE"` |
| `--after` | 排除老数据 | `--after 2025-01` |
| `--incr-margin-spread` | 增量利润率门槛 | `--incr-margin-spread 20` |
| `--min-rev-growth` | 最低收入增速 | `--min-rev-growth 0.3`（30%） |
| `--min-op-margin` | 最低经营利润率 | `--min-op-margin 0.1`（10%） |
| `--recent-quarters` | 看最近几个季度 | `--recent-quarters 2` |
| `--top` | 显示前N家 | `--top 20` |
| `--sort` | 排序方式 | `--sort revenue` |
| `--html` | 生成网页报告 | `--html` |

---

## 信号类型

| 信号 | 含义 |
|------|------|
| OP MARGIN TURNED POSITIVE | 经营利润率从亏损转为盈利 |
| INCREMENTAL MARGIN | 新增收入带来的利润率很高 |
| REVENUE GROWTH ACCELERATION | 收入增速在加速 |
| GROSS MARGIN INFLECTION | 毛利率触底反弹 |
| FCF TURNED POSITIVE | 自由现金流转正 |

---

## 常见问题

### Q: 提示 "python 不是内部或外部命令"
A: Python 没有正确安装或没有加入 PATH。重新安装 Python，确保勾选 "Add Python to PATH"。

### Q: 提示 "No module named 'yfinance'"
A: 依赖没装。运行 `pip install yfinance`。

### Q: 扫描过程中报错或中断了怎么办？
A: 重新运行 `scan` 命令即可，已下载的数据会被缓存，不会重复下载。

### Q: 数据多久更新一次？
A: 建议每天运行一次 `update` 命令来获取最新财报数据。

---

## 文件说明

运行后会生成以下文件夹：

| 路径 | 说明 |
|------|------|
| `cache/facts/` | SEC 原始数据缓存（30天有效） |
| `cache/results/` | 计算结果缓存（约 4000 家公司） |
| `reports/signal_master.html` | 完整缓存视图 |
| `reports/signal_queries.html` | 查询历史报告 |

这些都是缓存文件，删除后重新扫描即可恢复。

---

## serve 模式参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--port N` | 8000 | 服务器端口 |
| `--update` | 关闭 | 启动前先运行增量更新 |
| `--days N` | 3 | 增量更新回溯天数 |
| `--no-browser` | 关闭 | 不自动打开浏览器 |
