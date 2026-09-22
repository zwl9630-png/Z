import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import time
from datetime import datetime, timedelta

st.set_page_config(
    page_title="A股工程控制论 V2.1",
    page_icon="📈",
    layout="centered"
)

st.title("📈 A股工程控制论")
st.caption("V2.1 · 今日机会 · 三层控制 · 概率预测 · 模型验证")

# =========================
# 基础设置
# =========================

POOL = {
    "000977": ("浪潮信息", "AI算力"),
    "002475": ("立讯精密", "消费电子"),
    "002463": ("沪电股份", "PCB"),
    "002916": ("深南电路", "PCB"),
    "300308": ("中际旭创", "光模块"),
    "300502": ("新易盛", "光模块"),
    "600183": ("生益科技", "PCB"),
    "600584": ("长电科技", "半导体"),
    "600570": ("恒生电子", "金融科技"),
    "601138": ("工业富联", "AI算力"),
    "601360": ("三六零", "网络安全"),
    "603019": ("中科曙光", "AI算力"),
    "603986": ("兆易创新", "半导体"),
    "688008": ("澜起科技", "半导体"),
    "688041": ("海光信息", "AI算力"),
}

if "records" not in st.session_state:
    st.session_state.records = []


# =========================
# 数据
# =========================

def clean(df):
    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume"
    })

    need = ["date", "open", "high", "low", "close", "volume"]

    if not all(x in df.columns for x in need):
        raise ValueError("行情字段不完整")

    df = df[need].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for x in need[1:]:
        df[x] = pd.to_numeric(df[x], errors="coerce")

    df = df.dropna().sort_values("date").reset_index(drop=True)

    if len(df) < 70:
        raise ValueError("历史数据不足")

    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_em(code):
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=700)).strftime("%Y%m%d")

    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start,
        end_date=end,
        adjust="qfq"
    )

    return clean(df)


@st.cache_data(ttl=600, show_spinner=False)
def load_sina(code):
    prefix = "sh" if code.startswith(("5", "6", "9")) else "sz"

    df = ak.stock_zh_a_daily(
        symbol=prefix + code,
        adjust="qfq"
    )

    return clean(df)


def load_stock(code):
    try:
        return load_em(code), "东方财富"
    except Exception:
        time.sleep(0.3)

    try:
        return load_sina(code), "新浪财经"
    except Exception:
        raise RuntimeError("两个公开行情源均暂时不可用")


# =========================
# 指标
# =========================

def indicators(df):
    d = df.copy()

    for n in [5, 10, 20, 60]:
        d[f"MA{n}"] = d["close"].rolling(n).mean()

    e12 = d["close"].ewm(span=12, adjust=False).mean()
    e26 = d["close"].ewm(span=26, adjust=False).mean()

    d["DIF"] = e12 - e26
    d["DEA"] = d["DIF"].ewm(span=9, adjust=False).mean()
    d["MACD"] = 2 * (d["DIF"] - d["DEA"])

    change = d["close"].diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = -change.clip(upper=0).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - 100 / (1 + rs)

    d["VOL20"] = d["volume"].rolling(20).mean()
    d["RET20"] = d["close"].pct_change(20)
    d["RET60"] = d["close"].pct_change(60)

    pc = d["close"].shift(1)

    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - pc).abs(),
        (d["low"] - pc).abs()
    ], axis=1).max(axis=1)

    d["ATR"] = tr.rolling(14).mean()
    d["ATR_PCT"] = d["ATR"] / d["close"]

    return d


# =========================
# 个股控制器
# =========================

def stock_control(df):
    x = df.iloc[-1]

    score = 50
    good = []
    bad = []

    if x["close"] > x["MA20"]:
        score += 10
        good.append("价格站上MA20")
    else:
        score -= 10
        bad.append("价格位于MA20下方")

    if x["MA20"] > x["MA60"]:
        score += 12
        good.append("中期均线结构偏强")
    else:
        score -= 12
        bad.append("中期均线结构偏弱")

    if x["DIF"] > x["DEA"]:
        score += 10
        good.append("MACD偏强")
    else:
        score -= 10

    if x["RET20"] > 0:
        score += 8
        good.append("20日动量为正")
    else:
        score -= 8

    if x["RET60"] > 0:
        score += 8
        good.append("60日趋势为正")
    else:
        score -= 8

    if x["volume"] > x["VOL20"]:
        score += 5
        good.append("成交量高于20日均量")

    if 45 <= x["RSI"] <= 70:
        score += 5
        good.append("RSI处于趋势区")
    elif x["RSI"] > 80:
        score -= 5
        bad.append("RSI偏高")

    score = int(np.clip(score, 0, 100))

    atr = float(x["ATR_PCT"]) if pd.notna(x["ATR_PCT"]) else 0.03

    risk = "高" if atr > 0.06 else "中" if atr > 0.04 else "低"

    if score >= 75:
        state = "趋势强化"
    elif score >= 62:
        state = "趋势形成"
    elif score >= 45:
        state = "震荡观察"
    elif score >= 30:
        state = "趋势减弱"
    else:
        state = "风险释放"

    return score, risk, state, atr, good, bad


# =========================
# 大盘控制器
# =========================

@st.cache_data(ttl=600, show_spinner=False)
def market_control():
    try:
        d = ak.stock_zh_index_daily_em(symbol="sh000001")

        d["close"] = pd.to_numeric(d["close"], errors="coerce")
        d = d.dropna()

        ma20 = d["close"].rolling(20).mean().iloc[-1]
        ma60 = d["close"].rolling(60).mean().iloc[-1]
        close = d["close"].iloc[-1]

        score = 50
        score += 15 if close > ma20 else -15
        score += 15 if ma20 > ma60 else -15

        r20 = close / d["close"].iloc[-21] - 1
        score += 10 if r20 > 0 else -10

        score = int(np.clip(score, 0, 100))

        state = "偏强" if score >= 70 else "震荡" if score >= 45 else "偏弱"

        return score, state

    except Exception:
        return 50, "数据暂缺"


# =========================
# 行业层
# =========================

def industry_control(code, stock_score):
    industry = POOL.get(code, ("", "未分类"))[1]

    # 当前为代理行业层
    if stock_score >= 70:
        return industry, 70, "偏强"
    elif stock_score >= 50:
        return industry, 55, "中性"
    else:
        return industry, 40, "偏弱"


def total_score(market, industry, stock):
    return int(np.clip(
        market * 0.2 +
        industry * 0.2 +
        stock * 0.6,
        0,
        100
    ))


# =========================
# 概率情景
# =========================

def forecast(close, score, atr, r20, r60, days):
    strength = (score - 50) / 50
    scale = np.sqrt(days / 5)

    momentum = (
        np.clip(r20, -0.25, 0.25) * 0.6 +
        np.clip(r60, -0.40, 0.40) * 0.4
    )

    expected = (
        strength * atr * scale * 0.65 +
        momentum * min(days / 60, 1) * 0.35
    )

    center = close * (1 + expected)
    width = max(atr * scale * 1.25, 0.025)

    low = center * (1 - width)
    high = center * (1 + width)

    if score >= 60:
        direction = "偏强"
        probability = min(80, int(50 + (score - 50) * 0.75))
    elif score <= 40:
        direction = "偏弱"
        probability = min(80, int(50 + (50 - score) * 0.75))
    else:
        direction = "震荡"
        probability = 55

    return direction, probability, center, low, high


# =========================
# 基本面
# =========================

@st.cache_data(ttl=3600, show_spinner=False)
def basic_info(code):
    try:
        d = ak.stock_individual_info_em(symbol=code)

        if "item" not in d.columns or "value" not in d.columns:
            return {}

        return dict(zip(d["item"].astype(str), d["value"]))
    except Exception:
        return {}


# =========================
# 个股分析页
# =========================

def stock_page():
    st.subheader("🔎 个股分析")

    code = st.text_input(
        "输入6位股票代码",
        placeholder="例如：000977",
        max_chars=6
    ).strip()

    if not st.button("开始分析", type="primary", use_container_width=True):
        return

    if len(code) != 6 or not code.isdigit():
        st.error("请输入正确的6位股票代码")
        return

    try:
        with st.spinner("正在运行控制模型..."):
            df, source = load_stock(code)
            df = indicators(df)

            ss, risk, state, atr, good, bad = stock_control(df)
            ms, market_state = market_control()
            industry, ins, industry_state = industry_control(code, ss)
            total = total_score(ms, ins, ss)

            x = df.iloc[-1]
            close = float(x["close"])
            info = basic_info(code)

        name = info.get("股票简称", POOL.get(code, (code, ""))[0])

        st.success(f"分析完成 · {source}")
        st.subheader(f"{name} · {code}")
        st.caption(f"{x['date'].strftime('%Y-%m-%d')} ｜ 收盘 {close:.2f} 元")

        c1, c2, c3 = st.columns(3)
        c1.metric("综合评分", total)
        c2.metric("状态", state)
        c3.metric("风险", risk)

        st.subheader("🧠 三层控制")

        a, b, c = st.columns(3)
        a.metric("大盘", market_state, f"{ms}分")
        b.metric("行业", industry_state, industry)
        c.metric("个股", state, f"{ss}分")

        st.subheader("🔮 5 / 20 / 60日概率情景")

        rows = []

        for days in [5, 20, 60]:
            direction, prob, center, low, high = forecast(
                close,
                total,
                atr,
                float(x["RET20"]),
                float(x["RET60"]),
                days
            )

            rows.append({
                "周期": f"{days}日",
                "方向": direction,
                "概率": f"{prob}%",
                "中枢": round(center, 2),
                "区间": f"{low:.2f}～{high:.2f}"
            })

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True
        )

        st.caption("概率和区间是模型情景，不是确定性目标价。")

        support = float(df["low"].tail(20).min())
        resistance = float(df["high"].tail(20).max())

        st.subheader("🎯 关键位置")

        a, b = st.columns(2)
        a.metric("20日支撑", f"{support:.2f}")
        b.metric("20日压力", f"{resistance:.2f}")

        st.subheader("🏢 基本面")

        if info:
            for key in ["总市值", "流通市值", "行业", "上市时间"]:
                if key in info:
                    st.write(f"**{key}：** {info[key]}")
        else:
            st.info("基本面接口本次未返回数据。")

        st.caption(
            "缺失的PE、PB、ROE、营收、利润等数据不会被虚构并加入评分。"
        )

        st.subheader("🟢 正反馈")

        for item in good:
            st.write("• " + item)

        st.subheader("⚠️ 风险反馈")

        if bad:
            for item in bad:
                st.write("• " + item)
        else:
            st.write("暂未发现明显技术风险项")

        st.write(
            f"**失效观察：** 若有效跌破 {support:.2f} 元，"
            "重新评估当前状态。"
        )

        st.subheader("📉 近期走势")

        chart = df[
            ["date", "close", "MA20", "MA60"]
        ].tail(120).set_index("date")

        st.line_chart(chart)

        with st.expander("专业指标"):
            st.write({
                "MA5": round(float(x["MA5"]), 2),
                "MA20": round(float(x["MA20"]), 2),
                "MA60": round(float(x["MA60"]), 2),
                "RSI": round(float(x["RSI"]), 2),
                "MACD": round(float(x["MACD"]), 3),
                "20日收益": f"{float(x['RET20']) * 100:.2f}%",
                "60日收益": f"{float(x['RET60']) * 100:.2f}%",
                "ATR": f"{atr * 100:.2f}%"
            })

        if st.button("📝 保存本次预测", use_container_width=True):
            st.session_state.records.append({
                "时间": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "代码": code,
                "名称": name,
                "基准价": round(close, 2),
                "综合分": total,
                "状态": state,
                "风险": risk
            })

            st.success("已保存到预测验证记录")

    except Exception as e:
        st.error("数据获取失败，请稍后重试。")

        with st.expander("错误详情"):
            st.code(str(e))


# =========================
# 今日机会
# =========================

def opportunity_page():
    st.subheader("🎯 今日机会")
    st.caption("扫描控制状态较强的候选信号。候选不等于买入建议。")

    ms, market_state = market_control()

    a, b = st.columns(2)
    a.metric("大盘状态", market_state)
    b.metric("大盘评分", ms)

    count = st.slider("扫描数量", 5, len(POOL), 10)

    if not st.button("开始扫描", type="primary", use_container_width=True):
        return

    rows = []
    items = list(POOL.items())[:count]
    bar = st.progress(0)

    for i, (code, meta) in enumerate(items):
        name, industry = meta

        try:
            df, _ = load_stock(code)
            df = indicators(df)

            ss, risk, state, atr, good, bad = stock_control(df)
            _, ins, industry_state = industry_control(code, ss)
            total = total_score(ms, ins, ss)

            x = df.iloc[-1]

            rows.append({
                "代码": code,
                "名称": name,
                "行业": industry,
                "综合分": total,
                "状态": state,
                "风险": risk,
                "收盘": round(float(x["close"]), 2),
                "20日%": round(float(x["RET20"]) * 100, 2)
            })

        except Exception:
            pass

        bar.progress((i + 1) / len(items))
        time.sleep(0.1)

    bar.empty()

    if not rows:
        st.warning("本次没有取得有效扫描结果。")
        return

    result = pd.DataFrame(rows)

    result["风险序"] = result["风险"].map({
        "低": 0,
        "中": 1,
        "高": 2
    })

    result = result.sort_values(
        ["综合分", "风险序"],
        ascending=[False, True]
    ).drop(columns="风险序")

    st.subheader("📊 扫描结果")

    st.dataframe(
        result,
        hide_index=True,
        use_container_width=True
    )

    focus = result[
        (result["综合分"] >= 62) &
        (result["风险"] != "高")
    ]

    st.subheader("🔔 状态关注")

    if focus.empty:
        st.info("当前没有综合分≥62且风险非高的候选。")
    else:
        for _, r in focus.head(5).iterrows():
            st.write(
                f"**{r['名称']} {r['代码']}** ｜ "
                f"{r['行业']} ｜ {r['状态']} ｜ "
                f"{r['综合分']}分 ｜ 风险{r['风险']}"
            )


# =========================
# 自选股
# =========================

def favorites_page():
    st.subheader("⭐ 自选股")

    text = st.text_area(
        "股票代码，用逗号分隔",
        value="000977,600519,002475"
    )

    if not st.button("分析自选股", type="primary", use_container_width=True):
        return

    codes = [
        x.strip()
        for x in text.replace("，", ",").split(",")
        if x.strip().isdigit() and len(x.strip()) == 6
    ][:15]

    if not codes:
        st.warning("请输入正确股票代码")
        return

    ms, _ = market_control()
    rows = []
    bar = st.progress(0)

    for i, code in enumerate(codes):
        try:
            df, _ = load_stock(code)
            df = indicators(df)

            ss, risk, state, _, _, _ = stock_control(df)
            industry, ins, _ = industry_control(code, ss)
            total = total_score(ms, ins, ss)

            x = df.iloc[-1]
            name = POOL.get(code, (code, industry))[0]

            rows.append({
                "代码": code,
                "名称": name,
                "综合分": total,
                "状态": state,
                "风险": risk,
                "收盘": round(float(x["close"]), 2),
                "20日%": round(float(x["RET20"]) * 100, 2),
                "60日%": round(float(x["RET60"]) * 100, 2)
            })

        except Exception:
            pass

        bar.progress((i + 1) / len(codes))

    bar.empty()

    if rows:
        st.dataframe(
            pd.DataFrame(rows).sort_values(
                "综合分",
                ascending=False
            ),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("没有取得有效行情。")


# =========================
# 验证
# =========================

def validation_page():
    st.subheader("🧪 预测验证")

    if not st.session_state.records:
        st.info(
            "暂无记录。先进入个股分析，"
            "完成分析后点击“保存本次预测”。"
        )
        return

    df = pd.DataFrame(st.session_state.records)

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "当前为会话记录。下一版升级为持久化数据库，"
        "并自动验证5/20/60日结果。"
    )


# =========================
# 模型说明
# =========================

def model_page():
    st.subheader("⚙️ 模型")

    st.markdown("""
**当前控制链：**

大盘 → 行业 → 个股 → 动量/量价 → 风险反馈 → 概率情景 → 验证

**综合权重：**

- 大盘 20%
- 行业 20%
- 个股 60%

**技术变量：**

MA5 / MA10 / MA20 / MA60、MACD、RSI、成交量、
20日动量、60日动量、ATR、支撑与压力。

**下一阶段：**

真实行业指数、PE/PB/ROE、营收利润、资金流、
持久化预测数据库、自动回测与模型校准。
""")


# =========================
# 导航
# =========================

page = st.selectbox(
    "功能",
    [
        "🎯 今日机会",
        "🔎 个股分析",
        "⭐ 自选股",
        "🧪 预测验证",
        "⚙️ 模型"
    ]
)

st.divider()

if page == "🎯 今日机会":
    opportunity_page()
elif page == "🔎 个股分析":
    stock_page()
elif page == "⭐ 自选股":
    favorites_page()
elif page == "🧪 预测验证":
    validation_page()
else:
    model_page()

st.divider()

st.caption(
    "A股工程控制论 V2.1 ｜ "
    "模型结果用于研究与情景分析，不构成投资建议。"
)