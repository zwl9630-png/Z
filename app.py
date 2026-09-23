import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta

st.set_page_config(
    page_title="A股工程控制论 V2.3.2",
    page_icon="📈",
    layout="centered"
)

st.title("📈 A股工程控制论")
st.caption("V2.3.2 Mobile Fast · 一周验证版")


# =========================
# 基础工具
# =========================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def num(x):
    try:
        return float(str(x).replace(",", "").replace("%", ""))
    except:
        return np.nan


def money(x):
    try:
        x = float(x)

        if abs(x) >= 1e12:
            return f"{x / 1e12:.2f}万亿"

        if abs(x) >= 1e8:
            return f"{x / 1e8:.2f}亿"

        if abs(x) >= 1e4:
            return f"{x / 1e4:.2f}万"

        return f"{x:.2f}"

    except:
        return "--"


def pick(data, names):
    if not isinstance(data, dict):
        return None

    for name in names:
        if name in data:
            value = data[name]

            if pd.notna(value):
                return value

    return None


def prefix(code):
    if code.startswith(("5", "6", "9")):
        return "sh"

    return "sz"


# =========================
# 历史行情
# =========================

@st.cache_data(ttl=300, show_spinner=False)
def get_prices(code):

    end = datetime.now().strftime("%Y%m%d")

    start = (
        datetime.now() - timedelta(days=500)
    ).strftime("%Y%m%d")

    try:

        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq"
        )

        df = df.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume"
            }
        )

        df = df[
            [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        ].copy()

        df["date"] = pd.to_datetime(df["date"])

        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = (
            df.dropna()
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(df) >= 70:
            return df

    except:
        pass

    symbol = prefix(code) + code

    df = ak.stock_zh_a_daily(
        symbol=symbol,
        adjust="qfq"
    ).reset_index()

    df["date"] = pd.to_datetime(df["date"])

    df = df[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ].dropna()

    df = df.tail(500).reset_index(drop=True)

    if len(df) < 70:
        raise ValueError("历史行情不足")

    return df


# =========================
# 技术控制器
# =========================

def technical_engine(df):

    df = df.copy()

    for period in [5, 10, 20, 60]:
        df[f"MA{period}"] = (
            df["close"]
            .rolling(period)
            .mean()
        )

    ema12 = (
        df["close"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    df["DIF"] = ema12 - ema26

    df["DEA"] = (
        df["DIF"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    delta = df["close"].diff()

    gain = (
        delta.clip(lower=0)
        .rolling(14)
        .mean()
    )

    loss = (
        -delta.clip(upper=0)
        .rolling(14)
        .mean()
    )

    rs = gain / loss.replace(0, np.nan)

    df["RSI"] = 100 - 100 / (1 + rs)

    df["RET20"] = df["close"].pct_change(20)
    df["RET60"] = df["close"].pct_change(60)

    df["VOL20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    previous_close = df["close"].shift()

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()
    df["ATR_PCT"] = df["ATR"] / df["close"]

    row = df.iloc[-1]

    score = 50

    score += 10 if row["close"] > row["MA20"] else -10
    score += 12 if row["MA20"] > row["MA60"] else -12
    score += 10 if row["DIF"] > row["DEA"] else -10
    score += 8 if row["RET20"] > 0 else -8
    score += 8 if row["RET60"] > 0 else -8

    if (
        pd.notna(row["VOL20"])
        and row["volume"] > row["VOL20"]
    ):
        score += 5

    if pd.notna(row["RSI"]):

        if 45 <= row["RSI"] <= 70:
            score += 5

        elif row["RSI"] > 80:
            score -= 5

    score = int(clamp(score, 0, 100))

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

    if pd.notna(row["ATR_PCT"]):
        atr = float(row["ATR_PCT"])
    else:
        atr = 0.03

    if atr > 0.06:
        risk = "高"

    elif atr > 0.04:
        risk = "中"

    else:
        risk = "低"

    return df, score, state, risk, atr


# =========================
# 预测模型
# =========================

def forecast(
    close,
    score,
    atr,
    ret20,
    ret60,
    days
):

    strength = (score - 50) / 50

    momentum = (
        clamp(ret20, -0.25, 0.25) * 0.6
        +
        clamp(ret60, -0.40, 0.40) * 0.4
    )

    scale = np.sqrt(days / 5)

    expected = (
        strength * atr * scale * 0.6
        +
        momentum * min(days / 60, 1) * 0.35
    )

    center = close * (1 + expected)

    width = max(
        atr * scale * 1.15,
        0.025
    )

    low = center * (1 - width)
    high = center * (1 + width)

    if score >= 60:

        direction = "偏强"

        probability = int(
            clamp(
                50 + (score - 50) * 0.7,
                50,
                80
            )
        )

    elif score <= 40:

        direction = "偏弱"

        probability = int(
            clamp(
                50 + (50 - score) * 0.7,
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
        low,
        center,
        high
    )


# =========================
# 公司基本资料
# =========================

@st.cache_data(ttl=21600, show_spinner=False)
def get_basic(code):

    try:

        df = ak.stock_individual_info_em(
            symbol=code
        )

        if (
            df is not None
            and not df.empty
            and "item" in df.columns
            and "value" in df.columns
        ):

            return dict(
                zip(
                    df["item"],
                    df["value"]
                )
            )

    except:
        pass

    return {}


# =========================
# 财务指标
# =========================

@st.cache_data(ttl=21600, show_spinner=False)
def get_financial(code):

    result = {
        "report": "--",
        "roe": None,
        "gross": None,
        "net_margin": None,
        "debt": None
    }

    try:

        df = (
            ak.stock_financial_analysis_indicator(
                symbol=code
            )
        )

        if df is None or df.empty:
            return result

        if not isinstance(
            df.index,
            pd.RangeIndex
        ):
            df = df.reset_index()

        row = df.iloc[0].to_dict()

        result["report"] = str(
            pick(
                row,
                [
                    "日期",
                    "报告期",
                    "index"
                ]
            )
            or "--"
        )

        result["roe"] = pick(
            row,
            [
                "加权净资产收益率(%)",
                "摊薄净资产收益率(%)",
                "净资产收益率(%)"
            ]
        )

        result["gross"] = pick(
            row,
            [
                "销售毛利率(%)",
                "毛利率(%)"
            ]
        )

        result["net_margin"] = pick(
            row,
            [
                "销售净利率(%)",
                "净利率(%)"
            ]
        )

        result["debt"] = pick(
            row,
            ["资产负债率(%)"]
        )

    except:
        pass

    return result


# =========================
# 行业景气
# =========================

@st.cache_data(ttl=3600, show_spinner=False)
def get_industry(name):

    if not name:
        return None

    try:

        end = datetime.now().strftime("%Y%m%d")

        start = (
            datetime.now()
            - timedelta(days=160)
        ).strftime("%Y%m%d")

        df = ak.stock_board_industry_hist_em(
            symbol=name,
            start_date=start,
            end_date=end,
            period="日k",
            adjust=""
        )

        close = pd.to_numeric(
            df["收盘"],
            errors="coerce"
        ).dropna()

        if len(close) < 60:
            return None

        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]

        ret20 = (
            close.iloc[-1]
            / close.iloc[-21]
            - 1
        )

        score = 50

        score += 15 if close.iloc[-1] > ma20 else -15
        score += 15 if ma20 > ma60 else -15
        score += 15 if ret20 > 0 else -15

        score = int(clamp(score, 0, 100))

        if score >= 70:
            label = "🔥 高景气"

        elif score >= 55:
            label = "🟢 景气改善"

        elif score >= 40:
            label = "⚪ 中性"

        else:
            label = "🔵 景气偏弱"

        return {
            "score": score,
            "label": label,
            "ret20": ret20 * 100
        }

    except:
        return None


# =========================
# 资金流
# =========================

@st.cache_data(ttl=900, show_spinner=False)
def get_flow(code):

    try:

        df = ak.stock_individual_fund_flow(
            stock=code,
            market=prefix(code)
        )

        if df is None or df.empty:
            return None

        row = df.iloc[-1].to_dict()

        return {
            "main": pick(
                row,
                [
                    "主力净流入-净额",
                    "主力净流入净额"
                ]
            ),
            "ratio": pick(
                row,
                [
                    "主力净流入-净占比",
                    "主力净流入净占比"
                ]
            )
        }

    except:
        return None


# =========================
# 公告
# =========================

@st.cache_data(ttl=1800, show_spinner=False)
def get_notices(code):

    positive_words = [
        "中标",
        "合同",
        "订单",
        "回购",
        "增持",
        "预增",
        "扭亏",
        "重大项目",
        "战略合作",
        "获批"
    ]

    negative_words = [
        "减持",
        "亏损",
        "处罚",
        "立案",
        "风险提示",
        "诉讼",
        "终止",
        "下修",
        "预亏",
        "退市"
    ]

    try:

        df = ak.stock_individual_notice_report(
            symbol=code
        )

        result = []

        for _, row in df.head(40).iterrows():

            title = str(
                row.get(
                    "公告标题",
                    ""
                )
            )

            date = str(
                row.get(
                    "公告日期",
                    ""
                )
            )[:10]

            positive = any(
                word in title
                for word in positive_words
            )

            negative = any(
                word in title
                for word in negative_words
            )

            if not positive and not negative:
                continue

            if positive and not negative:
                kind = "🟢 潜在正反馈"

            elif negative and not positive:
                kind = "🔴 潜在风险"

            else:
                kind = "🟡 需判断"

            result.append(
                {
                    "date": date,
                    "type": kind,
                    "title": title
                }
            )

        return result[:8]

    except:
        return []


# =========================
# 市场温度
# =========================

@st.cache_data(ttl=600, show_spinner=False)
def get_temperature():

    try:

        df = ak.stock_zh_a_spot_em()

        pct = pd.to_numeric(
            df["涨跌幅"],
            errors="coerce"
        ).dropna()

        if len(pct) < 500:
            raise ValueError("样本不足")

        up = int((pct > 0).sum())
        down = int((pct < 0).sum())

        breadth = (
            (up - down)
            / len(pct)
            * 100
        )

        strong = (
            (
                (pct >= 5).sum()
                -
                (pct <= -5).sum()
            )
            / len(pct)
            * 100
        )

        extreme = clamp(
            (
                (pct >= 9.5).sum()
                -
                (pct <= -9.5).sum()
            ) * 2,
            -100,
            100
        )

        temp = (
            breadth * 0.40
            +
            clamp(
                pct.median() * 25,
                -100,
                100
            ) * 0.25
            +
            clamp(
                pct.mean() * 20,
                -100,
                100
            ) * 0.15
            +
            extreme * 0.10
            +
            clamp(
                strong * 8,
                -100,
                100
            ) * 0.10
        )

        temp = int(
            round(
                clamp(
                    temp,
                    -100,
                    100
                )
            )
        )

        if temp <= -80:
            label = "🥶 极度恐慌"

        elif temp <= -50:
            label = "❄️ 恐慌"

        elif temp <= -20:
            label = "🔵 偏冷"

        elif temp < 20:
            label = "⚪ 平衡"

        elif temp < 50:
            label = "🟡 回暖"

        elif temp < 80:
            label = "🔥 火热"

        else:
            label = "🌋 过热"

        return {
            "temp": temp,
            "label": label,
            "up": up,
            "down": down
        }

    except:
        return {
            "temp": None,
            "label": "数据暂缺"
        }


# =========================
# 个股页面
# =========================

def stock_page():

    code = st.text_input(
        "股票代码",
        placeholder="例如 000977",
        max_chars=6
    ).strip()

    analyse_button = st.button(
        "⚡ 开始快速分析",
        type="primary",
        use_container_width=True
    )

    if analyse_button:

        if len(code) != 6 or not code.isdigit():

            st.error(
                "请输入6位股票代码"
            )

        else:

            try:

                with st.spinner(
                    "正在读取行情..."
                ):

                    df = get_prices(code)

                    (
                        df,
                        score,
                        state,
                        risk,
                        atr
                    ) = technical_engine(df)

                st.session_state["fast"] = {
                    "code": code,
                    "df": df,
                    "score": score,
                    "state": state,
                    "risk": risk,
                    "atr": atr
                }

            except Exception as error:

                st.error(
                    f"行情获取失败：{error}"
                )

    result = st.session_state.get("fast")

    if not result:
        return

    if result["code"] != code:
        return

    df = result["df"]
    row = df.iloc[-1]

    basic = get_basic(code)

    stock_name = (
        pick(
            basic,
            [
                "股票简称",
                "名称"
            ]
        )
        or code
    )

    st.success(
        "⚡ 快速分析完成"
    )

    st.header(
        f"{stock_name} · {code}"
    )

    st.caption(
        f"行情日期："
        f"{row['date'].strftime('%Y-%m-%d')}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "技术分",
        result["score"]
    )

    col2.metric(
        "状态",
        result["state"]
    )

    col3.metric(
        "风险",
        result["risk"]
    )

    st.metric(
        "最新收盘",
        f"{row['close']:.2f}"
    )

    ret20 = (
        float(row["RET20"])
        if pd.notna(row["RET20"])
        else 0
    )

    ret60 = (
        float(row["RET60"])
        if pd.notna(row["RET60"])
        else 0
    )

    st.subheader(
        "🔮 5 / 20 / 60日概率情景"
    )

    forecast_rows = []

    for days in [5, 20, 60]:

        (
            direction,
            probability,
            low,
            center,
            high
        ) = forecast(
            float(row["close"]),
            result["score"],
            result["atr"],
            ret20,
            ret60,
            days
        )

        forecast_rows.append(
            {
                "周期": f"{days}日",
                "情景": direction,
                "概率": f"{probability}%",
                "中枢": round(center, 2),
                "参考区间":
                    f"{low:.2f}～{high:.2f}"
            }
        )

    st.dataframe(
        pd.DataFrame(forecast_rows),
        hide_index=True,
        use_container_width=True
    )

    support = df["low"].tail(20).min()
    resistance = df["high"].tail(20).max()

    col1, col2 = st.columns(2)

    col1.metric(
        "20日支撑",
        f"{support:.2f}"
    )

    col2.metric(
        "20日压力",
        f"{resistance:.2f}"
    )

    with st.expander(
        "📊 技术趋势"
    ):

        st.line_chart(
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

        if pd.notna(row["RSI"]):
            st.write(
                f"RSI14：{row['RSI']:.1f}"
            )

        st.write(
            f"ATR波动率："
            f"{result['atr'] * 100:.2f}%"
        )

        st.write(
            f"20日涨跌："
            f"{ret20 * 100:+.2f}%"
        )

        st.write(
            f"60日涨跌："
            f"{ret60 * 100:+.2f}%"
        )

    st.divider()

    st.subheader(
        "🧩 深度数据"
    )

    st.caption(
        "以下数据按需加载，不拖慢首屏。"
    )

    # 基本面
    if st.button(
        "🏢 加载基本面 / 估值",
        use_container_width=True
    ):

        with st.spinner(
            "读取财务数据..."
        ):

            financial = get_financial(code)

        st.session_state["financial"] = {
            "code": code,
            "data": financial
        }

    financial_result = (
        st.session_state.get(
            "financial"
        )
    )

    if (
        financial_result
        and financial_result["code"] == code
    ):

        f = financial_result["data"]

        st.markdown(
            "### 🏢 基本面 / 估值"
        )

        pe = pick(
            basic,
            [
                "市盈率(TTM)",
                "市盈率-动态"
            ]
        )

        pb = pick(
            basic,
            ["市净率"]
        )

        market_cap = pick(
            basic,
            ["总市值"]
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "PE",
            pe if pe is not None else "--"
        )

        col2.metric(
            "PB",
            pb if pb is not None else "--"
        )

        st.write(
            "总市值：",
            money(market_cap)
        )

        st.caption(
            "财务报告期："
            +
            str(f["report"])
        )

        roe = num(f["roe"])
        gross = num(f["gross"])

        col1, col2 = st.columns(2)

        col1.metric(
            "ROE",
            "--"
            if pd.isna(roe)
            else f"{roe:.2f}%"
        )

        col2.metric(
            "毛利率",
            "--"
            if pd.isna(gross)
            else f"{gross:.2f}%"
        )

        net_margin = num(
            f["net_margin"]
        )

        debt = num(
            f["debt"]
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "净利率",
            "--"
            if pd.isna(net_margin)
            else f"{net_margin:.2f}%"
        )

        col2.metric(
            "资产负债率",
            "--"
            if pd.isna(debt)
            else f"{debt:.2f}%"
        )

    # 行业
    if st.button(
        "🏭 加载行业景气",
        use_container_width=True
    ):

        industry_name = pick(
            basic,
            [
                "行业",
                "所属行业"
            ]
        )

        with st.spinner(
            "读取行业数据..."
        ):

            industry = get_industry(
                industry_name
            )

        st.session_state["industry"] = {
            "code": code,
            "name": industry_name,
            "data": industry
        }

    industry_result = (
        st.session_state.get(
            "industry"
        )
    )

    if (
        industry_result
        and industry_result["code"] == code
    ):

        st.markdown(
            "### 🏭 行业景气"
        )

        st.write(
            "所属行业：",
            industry_result["name"]
            or "--"
        )

        industry = (
            industry_result["data"]
        )

        if industry:

            col1, col2 = st.columns(2)

            col1.metric(
                "景气分",
                industry["score"]
            )

            col2.metric(
                "状态",
                industry["label"]
            )

            st.write(
                f"行业20日走势："
                f"{industry['ret20']:+.2f}%"
            )

        else:

            st.info(
                "本次行业数据暂缺。"
            )

    # 资金
    if st.button(
        "💵 加载资金流",
        use_container_width=True
    ):

        with st.spinner(
            "读取资金数据..."
        ):

            flow = get_flow(code)

        st.session_state["flow"] = {
            "code": code,
            "data": flow
        }

    flow_result = (
        st.session_state.get("flow")
    )

    if (
        flow_result
        and flow_result["code"] == code
    ):

        st.markdown(
            "### 💵 资金流"
        )

        flow = flow_result["data"]

        if flow:

            col1, col2 = st.columns(2)

            col1.metric(
                "主力净流入",
                money(flow["main"])
            )

            col2.metric(
                "主力净占比",
                flow["ratio"]
                if flow["ratio"] is not None
                else "--"
            )

        else:

            st.info(
                "本次资金流数据暂缺。"
            )

    # 公告
    if st.button(
        "📦 加载订单 / 公告",
        use_container_width=True
    ):

        with st.spinner(
            "读取近期公告..."
        ):

            notice_data = get_notices(
                code
            )

        st.session_state["notices"] = {
            "code": code,
            "data": notice_data
        }

    notice_result = (
        st.session_state.get(
            "notices"
        )
    )

    if (
        notice_result
        and notice_result["code"] == code
    ):

        st.markdown(
            "### 📦 订单 · 催化 · 风险"
        )

        notice_data = (
            notice_result["data"]
        )

        if not notice_data:

            st.info(
                "近期未识别到相关公告。"
                "这不代表公司不存在相关事项。"
            )

        for notice in notice_data:

            with st.container(
                border=True
            ):

                st.write(
                    f"**{notice['type']}**"
                )

                st.write(
                    notice["title"]
                )

                st.caption(
                    notice["date"]
                    +
                    " ｜ 公司公开公告"
                )

        st.caption(
            "公告只进行关键词初筛，"
            "不自动认定为确定性利好或利空。"
        )


# =========================
# 温度页面
# =========================

def temperature_page():

    st.subheader(
        "🌡️ 市场温度计"
    )

    st.caption(
        "-100℃ ～ +100℃"
    )

    if st.button(
        "🌡️ 测量市场温度",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "正在扫描全A..."
        ):

            result = get_temperature()

        st.session_state["temperature"] = result

    result = (
        st.session_state.get(
            "temperature"
        )
    )

    if not result:

        st.info(
            "温度计按需运行，"
            "因此不会拖慢个股分析。"
        )

        return

    if result["temp"] is None:

        st.metric(
            "市场温度",
            "--℃"
        )

        st.warning(
            "当前全A数据接口暂时不可用。"
        )

        return

    col1, col2 = st.columns(2)

    col1.metric(
        "市场温度",
        f"{result['temp']:+d}℃"
    )

    col2.metric(
        "市场情绪",
        result["label"]
    )

    st.progress(
        int(
            (result["temp"] + 100)
            / 2
        )
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "上涨股票",
        result["up"]
    )

    col2.metric(
        "下跌股票",
        result["down"]
    )

    st.caption(
        "高温不代表一定继续上涨；"
        "低温也不代表立即见底。"
    )


# =========================
# 导航
# =========================

page = st.selectbox(
    "功能",
    [
        "🔎 个股分析",
        "🌡️ 市场温度计",
        "⚙️ 模型说明"
    ]
)

st.divider()

if page == "🔎 个股分析":

    stock_page()

elif page == "🌡️ 市场温度计":

    temperature_page()

else:

    st.subheader(
        "⚙️ V2.3.2 Mobile Fast"
    )

    st.write(
        """
**快速层**

行情 → MA → MACD → RSI → 动量 → ATR → 风险 → 5/20/60日情景

**按需层**

基本面 / 估值  
行业景气  
资金流  
订单 / 公告  
市场温度

**数据原则**

数据缺失显示“-- / 暂缺”，不当作0分。

**实验原则**

本版本先运行一周，再根据实际预测结果调整参数。
"""
    )

st.divider()

st.caption(
    "A股工程控制论 V2.3.2 Mobile Fast"
)