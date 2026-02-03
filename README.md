# Signal Screener 安装与使用指南

检测美股公司财务拐点信号的命令行工具。

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

### 查看筛选结果

```
python D:\screener\signal_screener.py filter --severity HIGH --after 2025-01
```

这会显示：所有有高级别信号、且最近仍在正常申报的公司。

### 常用筛选命令

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

# 生成 HTML 报告（会自动打开浏览器）
python D:\screener\signal_screener.py filter --severity HIGH --after 2025-01 --html
```

### 查看单个公司详情

```
python D:\screener\signal_screener.py NVDA
python D:\screener\signal_screener.py AAPL MSFT GOOGL
```

### 增量更新（每天运行）

只更新最近有新财报的公司，比全量扫描快很多：

```
python D:\screener\signal_screener.py update
```

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
| `cache/facts/` | SEC 原始数据缓存 |
| `cache/results/` | 计算结果缓存（约 4000 家公司） |
| `reports/` | 生成的 HTML 报告 |

这些都是缓存文件，删除后重新扫描即可恢复。
