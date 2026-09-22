import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import time
from datetime import datetime, timedelta

# =========================================================
# A股工程控制论 V2.2
# 市场温度 → 大盘 → 行业 → 个股 → 风险 → 概率预测
# =========================================================

st.set_page_config(
    page_title="A股工程控制论 V2.2",
    page_icon="🌡️",
    layout="centered"
)

st.title("📈 A股工程控制论")
st.caption("V2.2 · 市场温度计 · 三层控制 · 今日机会 · 概率预测")


# =========================================================
# 股票池
# =========================================================

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


# =========================================================
# 通用函数
# =========================================================

def clamp(x, a, b):
    return max(a, min(b, x))


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

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    for x in need[1:]:
        df[x] = pd.to_numeric(
            df[x],
            errors="coerce"
        )

    df = (
        df.dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(df) < 70:
        raise ValueError("历史数据不足")

    return df


# =========================================================
# 个股行情：双数据源
# =========================================================

@st.cache_data(ttl=600, show_spinner=False)
def load_em(code):

    end = datetime.now().strftime("%Y%m%d")

    start = (
        datetime.now() - timedelta(days=700)
    ).strftime("%Y%m%d")

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

    prefix = (
        "sh"
        if code.startswith(("5", "6", "9"))
        else "sz"
    )

    df = ak.stock_zh_a_daily(
        symbol=prefix + code,
        adjust="qfq"
    )

    return clean(df)


def load_stock(code):

    for _ in range(2):
        try:
            return load_em(code), "东方财富"
        except Exception:
            time.sleep(0.3)

    for _ in range(2):
        try:
            return load_sina(code), "新浪财经"
        except Exception:
            time.sleep(0.3)

    raise RuntimeError("公开行情源暂时不可用")


# =========================================================
# 技术指标
# =========================================================

def indicators(df):

    d = df.copy()

    for n in [5, 10, 20, 60]:
        d[f"MA{n}"] = (
            d["close"]
            .rolling(n)
            .mean()
        )

    e12 = (
        d["close"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    e26 = (
        d["close"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    d["DIF"] = e12 - e26

    d["DEA"] = (
        d["DIF"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    d["MACD"] = (
        2 * (d["DIF"] - d["DEA"])
    )

    change = d["close"].diff()

    gain = (
        change.clip(lower=0)
        .rolling(14)
        .mean()
    )

    loss = (
        -change.clip(upper=0)
        .rolling(14)
        .mean()
    )

    rs = gain / loss.replace(0, np.nan)

    d["RSI"] = (
        100 - 100 / (1 + rs)
    )

    d["VOL20"] = (
        d["volume"]
        .rolling(20)
        .mean()
    )

    d["RET5"] = d["close"].pct_change(5)
    d["RET20"] = d["close"].pct_change(20)
    d["RET60"] = d["close"].pct_change(60)

    pc = d["close"].shift(1)

    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - pc).abs(),
        (d["low"] - pc).abs()
    ], axis=1).max(axis=1)

    d["ATR"] = (
        tr.rolling(14).mean()
    )

    d["ATR_PCT"] = (
        d["ATR"] / d["close"]
    )

    return d


# =========================================================
# 🌡️ 市场温度计
# -100℃ ～ +100℃
# =========================================================

def temperature_label(temp):

    if temp <= -80:
        return "🥶 极度恐慌"

    if temp <= -50:
        return "❄️ 恐慌"

    if temp <= -20:
        return "🔵 偏冷"

    if temp < 20:
        return "⚪ 平衡"

    if temp < 50:
        return "🟡 回暖"

    if temp < 80:
        return "🔥 火热"

    return "🌋 过热"


@st.cache_data(ttl=300, show_spinner=False)
def market_temperature():

    try:

        spot = ak.stock_zh_a_spot_em()

        if spot is None or spot.empty:
            raise ValueError("市场快照为空")

        # -------------------------
        # 涨跌幅
        # -------------------------

        pct = pd.to_numeric(
            spot["涨跌幅"],
            errors="coerce"
        ).dropna()

        if len(pct) < 100:
            raise ValueError("市场样本不足")

        up = int((pct > 0).sum())
        down = int((pct < 0).sum())
        flat = int((pct == 0).sum())

        total = max(up + down + flat, 1)

        # 广度 -100 ~ +100
        breadth = (
            (up - down) / total
        ) * 100

        # -------------------------
        # 涨跌停近似
        # -------------------------

        limit_up = int(
            (pct >= 9.5).sum()
        )

        limit_down = int(
            (pct <= -9.5).sum()
        )

        extreme = clamp(
            (limit_up - limit_down) * 2.5,
            -100,
            100
        )

        # -------------------------
        # 赚钱效应
        # -------------------------

        median_pct = float(
            pct.median()
        )

        mean_pct = float(
            pct.mean()
        )

        profit_effect = clamp(
            median_pct * 22
            + mean_pct * 8,
            -100,
            100
        )

        # -------------------------
        # 成交活跃度
        # -------------------------

        amount = pd.to_numeric(
            spot["成交额"],
            errors="coerce"
        ).fillna(0)

        total_amount = float(
            amount.sum()
        )

        # 免费接口没有稳定的全市场历史成交额
        # 当前以当日成交活跃度做非线性映射
        trillion = (
            total_amount / 1_000_000_000_000
        )

        activity = clamp(
            (trillion - 1.0) * 70,
            -60,
            100
        )

        # -------------------------
        # 强势股比例
        # -------------------------

        strong_ratio = float(
            (pct >= 5).sum()
        ) / total

        weak_ratio = float(
            (pct <= -5).sum()
        ) / total

        risk_appetite = clamp(
            (strong_ratio - weak_ratio)
            * 600,
            -100,
            100
        )

        # -------------------------
        # 温度合成
        # -------------------------

        temp = (
            breadth * 0.30
            + extreme * 0.20
            + profit_effect * 0.25
            + activity * 0.10
            + risk_appetite * 0.15
        )

        temp = int(
            round(
                clamp(temp, -100, 100)
            )
        )

        label = temperature_label(temp)

        return {
            "temperature": temp,
            "label": label,
            "up": up,
            "down": down,
            "flat": flat,
            "limit_up": limit_up,
            "limit_down": limit_down,
            "amount": total_amount,
            "median_pct": median_pct,
            "mean_pct": mean_pct,
            "breadth": breadth,
            "profit_effect": profit_effect,
            "activity": activity,
            "risk_appetite": risk_appetite
        }

    except Exception as e:

        return {
            "temperature": 0,
            "label": "⚪ 数据暂缺",
            "error": str(e)
        }


# =========================================================
# 温度显示组件
# =========================================================

def show_temperature():

    data = market_temperature()

    temp = data["temperature"]
    label = data["label"]

    st.subheader("🌡️ 市场温度计")

    a, b = st.columns(2)

    a.metric(
        "市场温度",
        f"{temp:+d}℃"
    )

    b.metric(
        "市场情绪",
        label
    )

    # -100~100 映射到 0~100 的进度条
    progress = int(
        clamp(
            (temp + 100) / 2,
            0,
            100
        )
    )

    st.progress(progress)

    st.caption(
        "左端 -100℃ = 极度恐慌 ｜ "
        "0℃ = 多空平衡 ｜ "
        "+100℃ = 极度过热"
    )

    if "error" in data:

        st.warning(
            "市场全景数据暂时不可用，"
            "当前温度不参与综合评分。"
        )

        return data

    c1, c2 = st.columns(2)

    c1.metric(
        "上涨家数",
        data["up"]
    )

    c2.metric(
        "下跌家数",
        data["down"]
    )

    c3, c4 = st.columns(2)

    c3.metric(
        "涨停近似",
        data["limit_up"]
    )

    c4.metric(
        "跌停近似",
        data["limit_down"]
    )

    amount_yi = (
        data["amount"] / 100_000_000
    )

    st.metric(
        "全A成交额",
        f"{amount_yi:,.0f} 亿"
    )

    with st.expander(
        "查看温度构成"
    ):

        st.write(
            f"涨跌广度："
            f"{data['breadth']:+.1f}"
        )

        st.write(
            f"赚钱效应："
            f"{data['profit_effect']:+.1f}"
        )

        st.write(
            f"成交活跃："
            f"{data['activity']:+.1f}"
        )

        st.write(
            f"风险偏好："
            f"{data['risk_appetite']:+.1f}"
        )

        st.write(
            f"个股中位涨跌幅："
            f"{data['median_pct']:+.2f}%"
        )

    return data


# =========================================================
# 大盘状态
# =========================================================

@st.cache_data(ttl=600, show_spinner=False)
def market_control():

    try:

        d = ak.stock_zh_index_daily_em(
            symbol="sh000001"
        )

        d["close"] = pd.to_numeric(
            d["close"],
            errors="coerce"
        )

        d = d.dropna()

        close = d["close"].iloc[-1]

        ma20 = (
            d["close"]
            .rolling(20)
            .mean()
            .iloc[-1]
        )

        ma60 = (
            d["close"]
            .rolling(60)
            .mean()
            .iloc[-1]
        )

        score = 50

        score += (
            15
            if close > ma20
            else -15
        )

        score += (
            15
            if ma20 > ma60
            else -15
        )

        r20 = (
            close /
            d["close"].iloc[-21]
            - 1
        )

        score += (
            10
            if r20 > 0
            else -10
        )

        score = int(
            clamp(score, 0, 100)
        )

        if score >= 70:
            state = "偏强"

        elif score >= 45:
            state = "震荡"

        else:
            state = "偏弱"

        return score, state

    except Exception:

        return 50, "数据暂缺"


# =========================================================
# 个股控制
# =========================================================

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

    if (
        pd.notna(x["VOL20"])
        and x["volume"] > x["VOL20"]
    ):
        score += 5
        good.append("成交量高于20日均量")

    if pd.notna(x["RSI"]):

        if 45 <= x["RSI"] <= 70:
            score += 5
            good.append("RSI处于趋势区")

        elif x["RSI"] > 80:
            score -= 5
            bad.append("RSI较高")

    score = int(
        clamp(score, 0, 100)
    )

    atr = (
        float(x["ATR_PCT"])
        if pd.notna(x["ATR_PCT"])
        else 0.03
    )

    if atr > 0.06:
        risk = "高"

    elif atr > 0.04:
        risk = "中"

    else:
        risk = "低"

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

    return (
        score,
        risk,
        state,
        atr,
        good,
        bad
    )


# =========================================================
# 行业层
# 暂不伪造行业强度
# =========================================================

def industry_control(code):

    industry = POOL.get(
        code,
        ("", "未分类")
    )[1]

    return industry, None, "待接入"


# =========================================================
# 综合控制评分
# =========================================================

def total_score(
    market_score,
    stock_score,
    temperature
):

    # 温度转换成 0~100 环境分
    temp_score = (
        temperature + 100
    ) / 2

    # 过热风险修正
    if temperature >= 80:
        temp_score -= (
            temperature - 80
        ) * 1.5

    # 极寒环境不直接视为利好
    if temperature <= -80:
        temp_score = min(
            temp_score,
            20
        )

    score = (
        stock_score * 0.65
        + market_score * 0.20
        + temp_score * 0.15
    )

    return int(
        clamp(
            round(score),
            0,
            100
        )
    )


# =========================================================
# 概率情景 V2
# =========================================================

def forecast(
    close,
    score,
    atr,
    r20,
    r60,
    days
):

    strength = (
        score - 50
    ) / 50

    scale = np.sqrt(
        days / 5
    )

    momentum = (
        np.clip(
            r20,
            -0.25,
            0.25
        ) * 0.6
        +
        np.clip(
            r60,
            -0.40,
            0.40
        ) * 0.4
    )

    expected = (
        strength
        * atr
        * scale
        * 0.65
        +
        momentum
        * min(days / 60, 1)
        * 0.35
    )

    center = (
        close * (1 + expected)
    )

    width = max(
        atr * scale * 1.25,
        0.025
    )

    low = center * (1 - width)
    high = center * (1 + width)

    if score >= 60:

        direction = "偏强"

        probability = int(
            clamp(
                50
                + (score - 50)
                * 0.75,
                50,
                80
            )
        )

    elif score <= 40:

        direction = "偏弱"

        probability = int(
            clamp(
                50
                + (50 - score)
                * 0.75,
                50,
                80
            )
        )

    else:

        direction = "震荡"
        probability = 55

    return (
        direction,
        probability,
        center,
        low,
        high
    )


# =========================================================
# 基本面
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def basic_info(code):

    try:

        d = ak.stock_individual_info_em(
            symbol=code
        )

        if (
            "item" not in d.columns
            or "value" not in d.columns
        ):
            return {}

        return dict(
            zip(
                d["item"].astype(str),
                d["value"]
            )
        )

    except Exception:

        return {}


# =========================================================
# 今日机会
# =========================================================

def opportunity_page():

    temp_data = show_temperature()

    st.divider()

    st.subheader("🎯 今日机会")

    st.caption(
        "结合市场环境和个股状态寻找研究候选。"
    )

    ms, market_state = market_control()

    temp = temp_data["temperature"]

    a, b = st.columns(2)

    a.metric(
        "大盘状态",
        market_state
    )

    b.metric(
        "趋势评分",
        ms
    )

    count = st.slider(
        "扫描数量",
        5,
        len(POOL),
        10
    )

    if not st.button(
        "开始扫描",
        type="primary",
        use_container_width=True
    ):
        return

    rows = []

    items = list(
        POOL.items()
    )[:count]

    bar = st.progress(0)

    for i, (
        code,
        meta
    ) in enumerate(items):

        name, industry = meta

        try:

            df, _ = load_stock(code)

            df = indicators(df)

            (
                ss,
                risk,
                state,
                atr,
                good,
                bad
            ) = stock_control(df)

            total = total_score(
                ms,
                ss,
                temp
            )

            x = df.iloc[-1]

            rows.append({
                "代码": code,
                "名称": name,
                "行业": industry,
                "综合分": total,
                "状态": state,
                "风险": risk,
                "收盘":
                    round(
                        float(x["close"]),
                        2
                    ),
                "20日%":
                    round(
                        float(x["RET20"])
                        * 100,
                        2
                    ),
                "原因":
                    " / ".join(
                        good[:2]
                    )
            })

        except Exception:
            pass

        bar.progress(
            (i + 1)
            / len(items)
        )

        time.sleep(0.1)

    bar.empty()

    if not rows:

        st.warning(
            "本次没有取得有效扫描结果。"
        )

        return

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "综合分",
        ascending=False
    )

    st.subheader("📊 扫描结果")

    st.dataframe(
        result,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("📱 状态卡片")

    focus = result.head(5)

    for _, r in focus.iterrows():

        with st.container(border=True):

            st.markdown(
                f"### {r['名称']} · {r['代码']}"
            )

            st.write(
                f"**{r['行业']}** ｜ "
                f"{r['状态']}"
            )

            a, b, c = st.columns(3)

            a.metric(
                "综合分",
                r["综合分"]
            )

            b.metric(
                "风险",
                r["风险"]
            )

            c.metric(
                "20日",
                f"{r['20日%']:+.1f}%"
            )

            st.caption(
                f"主要反馈：{r['原因']}"
            )


# =========================================================
# 个股分析
# =========================================================

def stock_page():

    st.subheader("🔎 个股分析")

    code = st.text_input(
        "输入6位股票代码",
        placeholder="例如：000977",
        max_chars=6
    ).strip()

    if not st.button(
        "开始分析",
        type="primary",
        use_container_width=True
    ):
        return

    if (
        len(code) != 6
        or not code.isdigit()
    ):

        st.error(
            "请输入正确的6位股票代码"
        )

        return

    try:

        with st.spinner(
            "正在运行控制系统..."
        ):

            df, source = load_stock(code)

            df = indicators(df)

            (
                ss,
                risk,
                state,
                atr,
                good,
                bad
            ) = stock_control(df)

            ms, market_state = (
                market_control()
            )

            td = market_temperature()

            temp = td["temperature"]

            total = total_score(
                ms,
                ss,
                temp
            )

            industry, _, industry_state = (
                industry_control(code)
            )

            x = df.iloc[-1]

            close = float(
                x["close"]
            )

            info = basic_info(code)

        name = info.get(
            "股票简称",
            POOL.get(
                code,
                (code, "")
            )[0]
        )

        st.success(
            f"分析完成 · {source}"
        )

        st.subheader(
            f"{name} · {code}"
        )

        st.caption(
            f"{x['date'].strftime('%Y-%m-%d')} ｜ "
            f"收盘 {close:.2f} 元"
        )

        # 核心
        a, b, c = st.columns(3)

        a.metric(
            "综合评分",
            total
        )

        b.metric(
            "状态",
            state
        )

        c.metric(
            "风险",
            risk
        )

        # 市场温度
        st.subheader(
            "🌡️ 当前市场环境"
        )

        a, b = st.columns(2)

        a.metric(
            "市场温度",
            f"{temp:+d}℃"
        )

        b.metric(
            "市场情绪",
            td["label"]
        )

        # 三层
        st.subheader(
            "🧠 控制系统"
        )

        a, b, c = st.columns(3)

        a.metric(
            "大盘",
            market_state,
            f"{ms}分"
        )

        b.metric(
            "行业",
            industry_state,
            industry
        )

        c.metric(
            "个股",
            state,
            f"{ss}分"
        )

        st.caption(
            "行业真实强弱数据尚未接入，"
            "因此当前行业层不参与综合评分。"
        )

        # 预测
        st.subheader(
            "🔮 5 / 20 / 60日情景"
        )

        prediction_rows = []

        for days in [5, 20, 60]:

            (
                direction,
                prob,
                center,
                low,
                high
            ) = forecast(
                close,
                total,
                atr,
                float(x["RET20"]),
                float(x["RET60"]),
                days
            )

            prediction_rows.append({
                "周期": f"{days}日",
                "方向": direction,
                "概率": f"{prob}%",
                "中枢":
                    round(center, 2),
                "区间":
                    f"{low:.2f}～{high:.2f}"
            })

        st.dataframe(
            pd.DataFrame(
                prediction_rows
            ),
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "这是概率情景，不是确定性目标价。"
        )

        # 支撑压力
        support = float(
            df["low"]
            .tail(20)
            .min()
        )

        resistance = float(
            df["high"]
            .tail(20)
            .max()
        )

        st.subheader(
            "🎯 关键位置"
        )

        a, b = st.columns(2)

        a.metric(
            "20日支撑",
            f"{support:.2f}"
        )

        b.metric(
            "20日压力",
            f"{resistance:.2f}"
        )

        # 基本面
        st.subheader(
            "🏢 基本面"
        )

        if info:

            shown = False

            for key in [
                "总市值",
                "流通市值",
                "行业",
                "上市时间"
            ]:

                if key in info:

                    st.write(
                        f"**{key}：** "
                        f"{info[key]}"
                    )

                    shown = True

            if not shown:

                st.info(
                    "本次基本面接口没有返回可展示字段。"
                )

        else:

            st.info(
                "基本面接口暂时不可用。"
            )

        st.caption(
            "PE/PB/ROE、营收和净利润等数据"
            "只有可靠取得后才会参与后续版本评分。"
        )

        # 反馈
        st.subheader(
            "🟢 正反馈"
        )

        if good:

            for item in good:
                st.write(
                    "• " + item
                )

        else:

            st.write(
                "暂无明显正反馈"
            )

        st.subheader(
            "⚠️ 风险反馈"
        )

        if bad:

            for item in bad:
                st.write(
                    "• " + item
                )

        else:

            st.write(
                "暂无明显技术风险项"
            )

        st.write(
            f"**状态失效观察：** "
            f"若有效跌破 {support:.2f} 元，"
            f"重新评估当前状态。"
        )

        # 图
        st.subheader(
            "📉 近期走势"
        )

        chart = (
            df[
                [
                    "date",
                    "close",
                    "MA20",
                    "MA60"
                ]
            ]
            .tail(120)
            .set_index("date")
        )

        st.line_chart(chart)

        with st.expander(
            "专业指标"
        ):

            st.write({
                "MA5":
                    round(
                        float(x["MA5"]),
                        2
                    ),

                "MA20":
                    round(
                        float(x["MA20"]),
                        2
                    ),

                "MA60":
                    round(
                        float(x["MA60"]),
                        2
                    ),

                "RSI":
                    round(
                        float(x["RSI"]),
                        2
                    ),

                "MACD":
                    round(
                        float(x["MACD"]),
                        3
                    ),

                "20日收益":
                    f"{float(x['RET20']) * 100:.2f}%",

                "60日收益":
                    f"{float(x['RET60']) * 100:.2f}%",

                "ATR":
                    f"{atr * 100:.2f}%"
            })

        # 保存
        if st.button(
            "📝 保存本次预测",
            use_container_width=True
        ):

            record = {
                "时间":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                "代码": code,
                "名称": name,
                "基准价":
                    round(close, 2),
                "综合分": total,
                "市场温度":
                    temp,
                "状态": state,
                "风险": risk
            }

            for days in [
                5,
                20,
                60
            ]:

                (
                    direction,
                    prob,
                    center,
                    low,
                    high
                ) = forecast(
                    close,
                    total,
                    atr,
                    float(x["RET20"]),
                    float(x["RET60"]),
                    days
                )

                record[
                    f"{days}日方向"
                ] = direction

                record[
                    f"{days}日概率"
                ] = prob

                record[
                    f"{days}日下限"
                ] = round(low, 2)

                record[
                    f"{days}日上限"
                ] = round(high, 2)

            st.session_state.records.append(
                record
            )

            st.success(
                "预测已保存到验证中心"
            )

    except Exception as e:

        st.error(
            "数据获取失败，请稍后重试。"
        )

        with st.expander(
            "错误详情"
        ):
            st.code(str(e))


# =========================================================
# 自选股
# =========================================================

def favorites_page():

    st.subheader(
        "⭐ 自选股"
    )

    text = st.text_area(
        "股票代码，用逗号分隔",
        value="000977,600519,002475"
    )

    if not st.button(
        "分析自选股",
        type="primary",
        use_container_width=True
    ):
        return

    codes = [
        x.strip()
        for x in text.replace(
            "，",
            ","
        ).split(",")
        if (
            x.strip().isdigit()
            and len(x.strip()) == 6
        )
    ][:15]

    if not codes:

        st.warning(
            "请输入正确股票代码"
        )

        return

    ms, _ = market_control()

    td = market_temperature()

    temp = td["temperature"]

    rows = []

    bar = st.progress(0)

    for i, code in enumerate(codes):

        try:

            df, _ = load_stock(code)

            df = indicators(df)

            (
                ss,
                risk,
                state,
                atr,
                good,
                bad
            ) = stock_control(df)

            total = total_score(
                ms,
                ss,
                temp
            )

            x = df.iloc[-1]

            name = POOL.get(
                code,
                (code, "")
            )[0]

            rows.append({
                "代码": code,
                "名称": name,
                "综合分": total,
                "状态": state,
                "风险": risk,
                "收盘":
                    round(
                        float(x["close"]),
                        2
                    ),
                "20日%":
                    round(
                        float(x["RET20"])
                        * 100,
                        2
                    ),
                "60日%":
                    round(
                        float(x["RET60"])
                        * 100,
                        2
                    )
            })

        except Exception:
            pass

        bar.progress(
            (i + 1)
            / len(codes)
        )

    bar.empty()

    if rows:

        st.dataframe(
            pd.DataFrame(rows)
            .sort_values(
                "综合分",
                ascending=False
            ),
            hide_index=True,
            use_container_width=True
        )

    else:

        st.warning(
            "没有取得有效行情。"
        )


# =========================================================
# 预测验证
# =========================================================

def validation_page():

    st.subheader(
        "🧪 预测验证中心"
    )

    st.caption(
        "保存模型当时的判断，"
        "以后与真实走势进行比较。"
    )

    if not st.session_state.records:

        st.info(
            "暂无预测记录。\n\n"
            "先进入个股分析，"
            "完成分析后点击保存预测。"
        )

        return

    df = pd.DataFrame(
        st.session_state.records
    )

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True
    )

    st.metric(
        "预测记录",
        len(df)
    )

    st.warning(
        "目前记录仍为会话级保存。"
        "Streamlit重启可能清空。"
        "下一阶段升级为持久化数据库。"
    )


# =========================================================
# 独立市场温度页面
# =========================================================

def temperature_page():

    data = show_temperature()

    st.divider()

    st.subheader(
        "🧭 温度解释"
    )

    table = pd.DataFrame([
        ["-100～-80℃", "🥶 极度恐慌", "活跃度极低"],
        ["-80～-50℃", "❄️ 恐慌", "空方明显占优"],
        ["-50～-20℃", "🔵 偏冷", "情绪谨慎"],
        ["-20～+20℃", "⚪ 平衡", "多空相对均衡"],
        ["+20～+50℃", "🟡 回暖", "活跃度改善"],
        ["+50～+80℃", "🔥 火热", "赚钱效应较强"],
        ["+80～+100℃", "🌋 过热", "情绪亢奋/注意过热"]
    ], columns=[
        "温度",
        "市场情绪",
        "解释"
    ])

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "高温不等于继续上涨，"
        "低温也不等于立即反弹。"
        "温度描述的是市场活跃度、"
        "赚钱效应与风险偏好的综合状态。"
    )


# =========================================================
# 模型
# =========================================================

def model_page():

    st.subheader(
        "⚙️ 控制模型"
    )

    st.markdown("""
### V2.2 控制链

**市场温度**
↓  
**大盘趋势**
↓  
**行业状态**
↓  
**个股状态**
↓  
**量价 / 动量 / 波动**
↓  
**风险反馈**
↓  
**综合控制评分**
↓  
**5 / 20 / 60日概率情景**
↓  
**真实结果验证**

### 当前综合评分

- 个股状态：65%
- 大盘趋势：20%
- 市场温度：15%

行业真实强度尚未稳定接入，因此目前不伪造行业分数。

### 市场温度输入

- 上涨 / 下跌家数
- 涨跌广度
- 涨停 / 跌停近似
- 市场平均涨跌幅
- 市场中位涨跌幅
- 强势股 / 弱势股比例
- 全A成交活跃度

### 下一阶段

真实行业指数、PE/PB/ROE、
营收利润增长、资金流、
温度历史数据库、5/20日温度曲线、
预测自动验证和模型重新校准。
""")


# =========================================================
# 导航
# =========================================================

page = st.selectbox(
    "功能",
    [
        "🎯 今日机会",
        "🌡️ 市场温度计",
        "🔎 个股分析",
        "⭐ 自选股",
        "🧪 预测验证",
        "⚙️ 模型"
    ]
)

st.divider()

if page == "🎯 今日机会":
    opportunity_page()

elif page == "🌡️ 市场温度计":
    temperature_page()

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
    "A股工程控制论 V2.2 ｜ "
    "市场温度范围 -100℃～+100℃ ｜ "
    "模型结果用于研究与情景分析。"
)