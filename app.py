import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta

# ============================================================
# A股工程控制论 V2.3.1 FAST
# ============================================================

st.set_page_config(
    page_title="A股工程控制论 V2.3.1 Fast",
    page_icon="📈",
    layout="centered"
)

st.title("📈 A股工程控制论")
st.caption("V2.3.1 FAST · 一周验证版")

# ============================================================
# 基础工具
# ============================================================

def clamp(x, low, high):
    return max(low, min(high, x))


def safe_num(x):
    try:
        return float(
            str(x)
            .replace(",", "")
            .replace("%", "")
            .replace("--", "")
        )
    except:
        return np.nan


def fmt_num(x, digits=2):
    try:
        x = float(x)
        if np.isnan(x):
            return "--"
        return f"{x:.{digits}f}"
    except:
        return "--"


def fmt_money(x):
    try:
        x = float(x)

        if np.isnan(x):
            return "--"

        if abs(x) >= 1e12:
            return f"{x/1e12:.2f}万亿"

        if abs(x) >= 1e8:
            return f"{x/1e8:.2f}亿"

        if abs(x) >= 1e4:
            return f"{x/1e4:.2f}万"

        return f"{x:.2f}"

    except:
        return "--"


def get_item(data, names):

    if data is None:
        return None

    for name in names:

        try:
            if isinstance(data, dict):
                if name in data:
                    v = data[name]

                    if pd.notna(v):
                        return v
        except:
            pass

    return None


def market_prefix(code):

    if code.startswith(("5", "6", "9")):
        return "sh"

    return "sz"


def exchange_code(code):

    if code.startswith(("5", "6", "9")):
        return "SH" + code

    return "SZ" + code


# ============================================================
# ① FAST 行情
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_price_data(code):

    end = datetime.now().strftime("%Y%m%d")

    start = (
        datetime.now() -
        timedelta(days=500)
    ).strftime("%Y%m%d")

    # 主接口
    try:

        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq"
        )

        if df is not None and len(df) >= 70:

            df = df.rename(
                columns={
                    "日期": "date",
                    "开盘": "open",
                    "最高": "high",
                    "最低": "low",
                    "收盘": "close",
                    "成交量": "volume",
                    "成交额": "amount"
                }
            )

            needed = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]

            df = df[needed].copy()

            df["date"] = pd.to_datetime(
                df["date"]
            )

            for c in needed[1:]:

                df[c] = pd.to_numeric(
                    df[c],
                    errors="coerce"
                )

            df = (
                df.dropna()
                .sort_values("date")
                .reset_index(drop=True)
            )

            return df

    except:
        pass

    # 备用接口
    try:

        symbol = market_prefix(code) + code

        df = ak.stock_zh_a_daily(
            symbol=symbol,
            adjust="qfq"
        )

        if df is None or len(df) < 70:
            raise ValueError("历史行情不足")

        df = df.reset_index()

        df["date"] = pd.to_datetime(
            df["date"]
        )

        return (
            df[
                [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume"
                ]
            ]
            .dropna()
            .tail(500)
            .reset_index(drop=True)
        )

    except Exception as e:

        raise RuntimeError(
            f"行情接口暂时不可用：{e}"
        )


# ============================================================
# ② 技术控制器
# ============================================================

def technical_engine(df):

    d = df.copy()

    for n in [5, 10, 20, 60]:

        d[f"MA{n}"] = (
            d["close"]
            .rolling(n)
            .mean()
        )

    # MACD
    ema12 = (
        d["close"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    ema26 = (
        d["close"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    d["DIF"] = ema12 - ema26

    d["DEA"] = (
        d["DIF"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    d["MACD"] = (
        (d["DIF"] - d["DEA"]) * 2
    )

    # RSI
    delta = d["close"].diff()

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

    d["RSI"] = (
        100 - 100 / (1 + rs)
    )

    # 收益率
    d["RET5"] = (
        d["close"]
        .pct_change(5)
    )

    d["RET20"] = (
        d["close"]
        .pct_change(20)
    )

    d["RET60"] = (
        d["close"]
        .pct_change(60)
    )

    # 成交量
    d["VOL20"] = (
        d["volume"]
        .rolling(20)
        .mean()
    )

    # ATR
    previous_close = (
        d["close"].shift(1)
    )

    tr = pd.concat(
        [
            d["high"] - d["low"],

            (
                d["high"] -
                previous_close
            ).abs(),

            (
                d["low"] -
                previous_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    d["ATR"] = (
        tr.rolling(14).mean()
    )

    d["ATR_PCT"] = (
        d["ATR"] /
        d["close"]
    )

    x = d.iloc[-1]

    score = 50

    # 价格趋势
    score += (
        10
        if x["close"] > x["MA20"]
        else -10
    )

    score += (
        12
        if x["MA20"] > x["MA60"]
        else -12
    )

    # MACD
    score += (
        10
        if x["DIF"] > x["DEA"]
        else -10
    )

    # 动量
    score += (
        8
        if x["RET20"] > 0
        else -8
    )

    score += (
        8
        if x["RET60"] > 0
        else -8
    )

    # 成交量
    if (
        pd.notna(x["VOL20"])
        and x["volume"] > x["VOL20"]
    ):
        score += 5

    # RSI
    if pd.notna(x["RSI"]):

        if 45 <= x["RSI"] <= 70:
            score += 5

        elif x["RSI"] >= 80:
            score -= 6

    score = int(
        clamp(score, 0, 100)
    )

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

    atr_pct = (
        float(x["ATR_PCT"])
        if pd.notna(x["ATR_PCT"])
        else 0.03
    )

    if atr_pct >= 0.06:

        risk = "高"

    elif atr_pct >= 0.04:

        risk = "中"

    else:

        risk = "低"

    return (
        d,
        score,
        state,
        risk,
        atr_pct
    )


# ============================================================
# ③ FAST预测
# ============================================================

def forecast(
    close,
    score,
    atr,
    ret20,
    ret60,
    days
):

    strength = (
        score - 50
    ) / 50

    momentum = (
        clamp(ret20, -0.25, 0.25)
        * 0.60
        +
        clamp(ret60, -0.40, 0.40)
        * 0.40
    )

    scale = np.sqrt(
        days / 5
    )

    expected = (
        strength
        * atr
        * scale
        * 0.60
        +
        momentum
        * min(days / 60, 1)
        * 0.35
    )

    center = (
        close *
        (1 + expected)
    )

    width = max(
        atr * scale * 1.15,
        0.025
    )

    low = (
        center *
        (1 - width)
    )

    high = (
        center *
        (1 + width)
    )

    if score >= 60:

        direction = "偏强"

        probability = clamp(
            50 +
            (score - 50) * 0.7,
            50,
            80
        )

    elif score <= 40:

        direction = "偏弱"

        probability = clamp(
            50 +
            (50 - score) * 0.7,
            50,
            80
        )

    else:

        direction = "震荡"
        probability = 55

    return (
        direction,
        int(probability),
        low,
        center,
        high
    )


# ============================================================
# ④ 基本资料
# 缓存6小时
# ============================================================

@st.cache_data(
    ttl=21600,
    show_spinner=False
)
def get_basic_info(code):

    result = {}

    try:

        df = (
            ak.stock_individual_info_em(
                symbol=code
            )
        )

        if (
            df is not None
            and not df.empty
            and "item" in df.columns
            and "value" in df.columns
        ):

            result = dict(
                zip(
                    df["item"],
                    df["value"]
                )
            )

    except:
        pass

    return result


# ============================================================
# ⑤ 财务指标
# 按需加载
# ============================================================

@st.cache_data(
    ttl=21600,
    show_spinner=False
)
def get_financial(code):

    result = {
        "report": "--",
        "roe": None,
        "gross": None,
        "net_margin": None,
        "debt": None,
        "revenue": None,
        "revenue_yoy": None,
        "profit": None,
        "profit_yoy": None
    }

    # 财务分析指标
    try:

        df = (
            ak.stock_financial_analysis_indicator(
                symbol=code
            )
        )

        if (
            df is not None
            and not df.empty
        ):

            if not isinstance(
                df.index,
                pd.RangeIndex
            ):

                df = df.reset_index()

            row = (
                df.iloc[0]
                .to_dict()
            )

            result["report"] = (
                get_item(
                    row,
                    [
                        "日期",
                        "报告期",
                        "index"
                    ]
                )
                or "--"
            )

            result["roe"] = (
                get_item(
                    row,
                    [
                        "加权净资产收益率(%)",
                        "摊薄净资产收益率(%)",
                        "净资产收益率(%)"
                    ]
                )
            )

            result["gross"] = (
                get_item(
                    row,
                    [
                        "销售毛利率(%)",
                        "毛利率(%)"
                    ]
                )
            )

            result["net_margin"] = (
                get_item(
                    row,
                    [
                        "销售净利率(%)",
                        "净利率(%)"
                    ]
                )
            )

            result["debt"] = (
                get_item(
                    row,
                    ["资产负债率(%)"]
                )
            )

    except:
        pass

    # 利润表
    try:

        df = (
            ak.stock_profit_sheet_by_report_em(
                symbol=exchange_code(code)
            )
        )

        if (
            df is not None
            and not df.empty
        ):

            row = (
                df.iloc[0]
                .to_dict()
            )

            report = get_item(
                row,
                [
                    "REPORT_DATE",
                    "报告日期",
                    "报告期"
                ]
            )

            if report is not None:

                result["report"] = (
                    str(report)[:10]
                )

            result["revenue"] = (
                get_item(
                    row,
                    [
                        "TOTAL_OPERATE_INCOME",
                        "OPERATE_INCOME",
                        "营业总收入",
                        "营业收入"
                    ]
                )
            )

            result["revenue_yoy"] = (
                get_item(
                    row,
                    [
                        "TOTAL_OPERATE_INCOME_YOY",
                        "OPERATE_INCOME_YOY",
                        "营业收入同比"
                    ]
                )
            )

            result["profit"] = (
                get_item(
                    row,
                    [
                        "PARENT_NETPROFIT",
                        "NETPROFIT",
                        "归属于母公司股东的净利润",
                        "净利润"
                    ]
                )
            )

            result["profit_yoy"] = (
                get_item(
                    row,
                    [
                        "PARENT_NETPROFIT_YOY",
                        "NETPROFIT_YOY",
                        "净利润同比"
                    ]
                )
            )

    except:
        pass

    return result


# ============================================================
# ⑥ 行业景气
# 按需加载
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def get_industry(industry):

    if not industry:
        return None

    try:

        end = (
            datetime.now()
            .strftime("%Y%m%d")
        )

        start = (
            datetime.now()
            - timedelta(days=160)
        ).strftime("%Y%m%d")

        df = (
            ak.stock_board_industry_hist_em(
                symbol=industry,
                start_date=start,
                end_date=end,
                period="日k",
                adjust=""
            )
        )

        if (
            df is None
            or len(df) < 60
        ):

            return None

        close = pd.to_numeric(
            df["收盘"],
            errors="coerce"
        ).dropna()

        if len(close) < 60:
            return None

        ma20 = (
            close
            .rolling(20)
            .mean()
            .iloc[-1]
        )

        ma60 = (
            close
            .rolling(60)
            .mean()
            .iloc[-1]
        )

        ret20 = (
            close.iloc[-1]
            / close.iloc[-21]
            - 1
        )

        score = 50

        score += (
            15
            if close.iloc[-1] > ma20
            else -15
        )

        score += (
            15
            if ma20 > ma60
            else -15
        )

        score += (
            15
            if ret20 > 0
            else -15
        )

        score = int(
            clamp(score, 0, 100)
        )

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


# ============================================================
# ⑦ 资金流
# 按需加载
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def get_fund_flow(code):

    try:

        market = market_prefix(code)

        df = (
            ak.stock_individual_fund_flow(
                stock=code,
                market=market
            )
        )

        if (
            df is None
            or df.empty
        ):

            return None

        row = (
            df.iloc[-1]
            .to_dict()
        )

        return {
            "main": get_item(
                row,
                [
                    "主力净流入-净额",
                    "主力净流入净额"
                ]
            ),
            "ratio": get_item(
                row,
                [
                    "主力净流入-净占比",
                    "主力净流入净占比"
                ]
            )
        }

    except:

        return None


# ============================================================
# ⑧ 公告 / 订单
# 按需加载
# ============================================================

POSITIVE_WORDS = [
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

NEGATIVE_WORDS = [
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


@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def get_notices(code):

    try:

        df = (
            ak.stock_individual_notice_report(
                symbol=code
            )
        )

        if (
            df is None
            or df.empty
        ):

            return []

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
                for word in POSITIVE_WORDS
            )

            negative = any(
                word in title
                for word in NEGATIVE_WORDS
            )

            if not (
                positive
                or negative
            ):
                continue

            if (
                positive
                and not negative
            ):

                kind = (
                    "🟢 潜在正反馈"
                )

            elif (
                negative
                and not positive
            ):

                kind = (
                    "🔴 潜在风险"
                )

            else:

                kind = (
                    "🟡 需人工判断"
                )

            result.append(
                {
                    "date": date,
                    "title": title,
                    "type": kind
                }
            )

        return result[:10]

    except:

        return []


# ============================================================
# ⑨ 市场温度
# 10分钟缓存
# ============================================================

def temperature_label(t):

    if t is None:
        return "数据暂缺"

    if t <= -80:
        return "🥶 极度恐慌"

    if t <= -50:
        return "❄️ 恐慌"

    if t <= -20:
        return "🔵 偏冷"

    if t < 20:
        return "⚪ 平衡"

    if t < 50:
        return "🟡 回暖"

    if t < 80:
        return "🔥 火热"

    return "🌋 过热"


@st.cache_data(
    ttl=600,
    show_spinner=False
)
def get_market_temperature():

    try:

        df = (
            ak.stock_zh_a_spot_em()
        )

        pct = pd.to_numeric(
            df["涨跌幅"],
            errors="coerce"
        ).dropna()

        if len(pct) < 500:

            raise ValueError(
                "市场样本不足"
            )

        total = len(pct)

        up = int(
            (pct > 0).sum()
        )

        down = int(
            (pct < 0).sum()
        )

        breadth = (
            (up - down)
            / total
            * 100
        )

        median = float(
            pct.median()
        )

        mean = float(
            pct.mean()
        )

        strong = (
            (
                (pct >= 5).sum()
                -
                (pct <= -5).sum()
            )
            / total
            * 100
        )

        strong_up = int(
            (pct >= 9.5).sum()
        )

        strong_down = int(
            (pct <= -9.5).sum()
        )

        extreme = clamp(
            (
                strong_up
                -
                strong_down
            )
            * 2,
            -100,
            100
        )

        temp = (
            breadth * 0.35
            +
            clamp(
                median * 25,
                -100,
                100
            ) * 0.25
            +
            clamp(
                mean * 20,
                -100,
                100
            ) * 0.15
            +
            extreme * 0.15
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

        return {
            "temp": temp,
            "label":
                temperature_label(temp),
            "up": up,
            "down": down,
            "strong_up": strong_up,
            "strong_down":
                strong_down,
            "median": median,
            "status": "正常"
        }

    except Exception as e:

        return {
            "temp": None,
            "label": "数据暂缺",
            "status": "降级",
            "error": str(e)
        }


# ============================================================
# ⑩ 首屏 FAST 分析
# ============================================================

def fast_analyse(code):

    df = get_price_data(code)

    (
        df,
        score,
        state,
        risk,
        atr
    ) = technical_engine(df)

    x = df.iloc[-1]

    return {
        "df": df,
        "x": x,
        "score": score,
        "state": state,
        "risk": risk,
        "atr": atr
    }


# ============================================================
# 综合评分
# ============================================================

def weighted_score(
    tech,
    fundamental=None,
    industry=None,
    temperature=None
):

    components = [
        (tech, 0.50)
    ]

    if fundamental is not None:

        components.append(
            (
                fundamental,
                0.25
            )
        )

    if industry is not None:

        components.append(
            (
                industry,
                0.15
            )
        )

    if temperature is not None:

        market_score = (
            temperature + 100
        ) / 2

        # 过热风险反馈
        if temperature > 80:

            market_score -= (
                temperature - 80
            ) * 1.5

        components.append(
            (
                clamp(
                    market_score,
                    0,
                    100
                ),
                0.10
            )
        )

    total_weight = sum(
        w
        for _, w
        in components
    )

    return int(
        sum(
            value * weight
            for value, weight
            in components
        )
        / total_weight
    )


# ============================================================
# 基本面评分
# ============================================================

def fundamental_score(fin):

    scores = []

    roe = safe_num(
        fin.get("roe")
    )

    revenue_yoy = safe_num(
        fin.get("revenue_yoy")
    )

    profit_yoy = safe_num(
        fin.get("profit_yoy")
    )

    gross = safe_num(
        fin.get("gross")
    )

    debt = safe_num(
        fin.get("debt")
    )

    if pd.notna(roe):

        scores.append(
            clamp(
                50
                +
                (roe - 10) * 2,
                0,
                100
            )
        )

    if pd.notna(revenue_yoy):

        scores.append(
            clamp(
                50
                +
                revenue_yoy,
                0,
                100
            )
        )

    if pd.notna(profit_yoy):

        scores.append(
            clamp(
                50
                +
                profit_yoy * 0.6,
                0,
                100
            )
        )

    if pd.notna(gross):

        scores.append(
            clamp(
                40 + gross,
                0,
                100
            )
        )

    if pd.notna(debt):

        scores.append(
            clamp(
                100 - debt,
                0,
                100
            )
        )

    if not scores:

        return None

    return int(
        np.mean(scores)
    )


# ============================================================
# 个股页面
# ============================================================

def stock_page():

    st.subheader(
        "🔎 个股控制台"
    )

    code = (
        st.text_input(
            "股票代码",
            placeholder="例如 000977",
            max_chars=6
        )
        .strip()
    )

    if st.button(
        "⚡ 开始快速分析",
        type="primary",
        use_container_width=True
    ):

        if (
            len(code) != 6
            or not code.isdigit()
        ):

            st.error(
                "请输入6位股票代码"
            )

        else:

            try:

                with st.spinner(
                    "正在读取行情..."
                ):

                    result = (
                        fast_analyse(code)
                    )

                st.session_state[
                    "fast_result"
                ] = result

                st.session_state[
                    "current_code"
                ] = code

            except Exception as e:

                st.error(
                    "行情获取失败"
                )

                st.caption(
                    str(e)
                )

    if (
        "fast_result"
        not in st.session_state
        or
        st.session_state.get(
            "current_code"
        ) != code
    ):

        return

    r = (
        st.session_state[
            "fast_result"
        ]
    )

    x = r["x"]

    # --------------------------------------------------------
    # FAST首屏
    # --------------------------------------------------------

    st.success(
        "⚡ 快速数据已完成"
    )

    st.header(
        f"{code}"
    )

    st.caption(
        "行情日期："
        +
        x["date"]
        .strftime("%Y-%m-%d")
    )

    a, b, c = st.columns(3)

    a.metric(
        "技术评分",
        r["score"]
    )

    b.metric(
        "状态",
        r["state"]
    )

    c.metric(
        "风险",
        r["risk"]
    )

    st.metric(
        "最新收盘",
        f"{x['close']:.2f}"
    )

    # --------------------------------------------------------
    # 预测
    # --------------------------------------------------------

    st.subheader(
        "🔮 概率情景"
    )

    forecasts = []

    ret20 = (
        float(x["RET20"])
        if pd.notna(x["RET20"])
        else 0
    )

    ret60 = (
        float(x["RET60"])
        if pd.notna(x["RET60"])
        else 0
    )

    for days in [
        5,
        20,
        60
    ]:

        (
            direction,
            probability,
            low,
            center,
            high
        ) = forecast(
            float(x["close"]),
            r["score"],
            r["atr"],
            ret20,
            ret60,
            days
        )

        forecasts.append(
            {
                "周期":
                    f"{days}日",

                "情景":
                    direction,

                "概率":
                    f"{probability}%",

                "中枢":
                    round(
                        center,
                        2
                    ),

                "参考区间":
                    (
                        f"{low:.2f}"
                        f"～"
                        f"{high:.2f}"
                    )
            }
        )

    st.dataframe(
        pd.DataFrame(
            forecasts
        ),
        hide_index=True,
        use_container_width=True
    )

    # --------------------------------------------------------
    # 支撑压力
    # --------------------------------------------------------

    support = (
        r["df"]["low"]
        .tail(20)
        .min()
    )

    resistance = (
        r["df"]["high"]
        .tail(20)
        .max()
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

    # --------------------------------------------------------
    # 技术图
    # --------------------------------------------------------

    with st.expander(
        "📊 查看技术趋势"
    ):

        st.line_chart(
            r["df"][
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

        a, b = st.columns(2)

        a.metric(
            "RSI14",
            fmt_num(
                x["RSI"]
            )
        )

        b.metric(
            "ATR%",
            (
                f"{r['atr']*100:.2f}%"
            )
        )

        st.write(
            "MA20：",
            fmt_num(
                x["MA20"]
            )
        )

        st.write(
            "MA60：",
            fmt_num(
                x["MA60"]
            )
        )

        st.write(
            "20日涨跌：",
            f"{ret20*100:+.2f}%"
        )

        st.write(
            "60日涨跌：",
            f"{ret60*100:+.2f}%"
        )

    st.divider()

    st.subheader(
        "🧩 深度数据"
    )

    st.caption(
        "下面模块只有点击后才联网，"
        "因此不会拖慢首屏。"
    )

    # ========================================================
    # 基本面按钮
    # ========================================================

    if st.button(
        "🏢 加载基本面 / 估值",
        use_container_width=True
    ):

        with st.spinner(
            "读取基本面..."
        ):

            basic = (
                get_basic_info(code)
            )

            fin = (
                get_financial(code)
            )

        st.session_state[
            "basic_data"
        ] = basic

        st.session_state[
            "financial_data"
        ] = fin

        st.session_state[
            "basic_code"
        ] = code

    if (
        st.session_state.get(
            "basic_code"
        ) == code
    ):

        basic = (
            st.session_state.get(
                "basic_data",
                {}
            )
        )

        fin = (
            st.session_state.get(
                "financial_data",
                {}
            )
        )

        st.markdown(
            "### 🏢 基本面"
        )

        name = (
            get_item(
                basic,
                [
                    "股票简称",
                    "名称"
                ]
            )
            or "--"
        )

        industry = (
            get_item(
                basic,
                [
                    "行业",
                    "所属行业"
                ]
            )
            or "--"
        )

        st.write(
            f"**名称：** {name}"
        )

        st.write(
            f"**行业：** {industry}"
        )

        st.caption(
            "报告期："
            +
            str(
                fin.get(
                    "report",
                    "--"
                )
            )
        )

        fs = (
            fundamental_score(fin)
        )

        st.metric(
            "基本面评分",
            "--"
            if fs is None
            else fs
        )

        a, b = st.columns(2)

        roe = safe_num(
            fin.get("roe")
        )

        gross = safe_num(
            fin.get("gross")
        )

        a.metric(
            "ROE",
            "--"
            if pd.isna(roe)
            else f"{roe:.2f}%"
        )

        b.metric(
            "毛利率",
            "--"
            if pd.isna(gross)
            else f"{gross:.2f}%"
        )

        a, b = st.columns(2)

        margin = safe_num(
            fin.get(
                "net_margin"
            )
        )

        debt = safe_num(
            fin.get("debt")
        )

        a.metric(
            "净利率",
            "--"
            if pd.isna(margin)
            else f"{margin:.2f}%"
        )

        b.metric(
            "资产负债率",
            "--"
            if pd.isna(debt)
            else f"{debt:.2f}%"
        )

        st.write(
            "**营业收入：**",
            fmt_money(
                fin.get(
                    "revenue"
                )
            )
        )

        rg = safe_num(
            fin.get(
                "revenue_yoy"
            )
        )

        st.write(
            "**营收同比：**",
            "--"
            if pd.isna(rg)
            else f"{rg:+.2f}%"
        )

        st.write(
            "**归母净利润：**",
            fmt_money(
                fin.get(
                    "profit"
                )
            )
        )

        pg = safe_num(
            fin.get(
                "profit_yoy"
            )
        )

        st.write(
            "**净利润同比：**",
            "--"
            if pd.isna(pg)
            else f"{pg:+.2f}%"
        )

        pe = get_item(
            basic,
            [
                "市盈率(TTM)",
                "市盈率-动态",
                "市盈率(动)"
            ]
        )

        pb = get_item(
            basic,
            ["市净率"]
        )

        cap = get_item(
            basic,
            ["总市值"]
        )

        a, b = st.columns(2)

        a.metric(
            "PE",
            "--"
            if pe is None
            else pe
        )

        b.metric(
            "PB",
            "--"
            if pb is None
            else pb
        )

        st.write(
            "**总市值：**",
            fmt_money(cap)
        )

    # ========================================================
    # 行业按钮
    # ========================================================

    if st.button(
        "🏭 加载行业景气",
        use_container_width=True
    ):

        with st.spinner(
            "读取行业..."
        ):

            basic = (
                get_basic_info(code)
            )

            industry = (
                get_item(
                    basic,
                    [
                        "行业",
                        "所属行业"
                    ]
                )
            )

            industry_result = (
                get_industry(
                    industry
                )
                if industry
                else None
            )

        st.session_state[
            "industry_name"
        ] = industry

        st.session_state[
            "industry_result"
        ] = industry_result

        st.session_state[
            "industry_code"
        ] = code

    if (
        st.session_state.get(
            "industry_code"
        ) == code
    ):

        st.markdown(
            "### 🏭 行业景气"
        )

        industry = (
            st.session_state.get(
                "industry_name"
            )
        )

        ind = (
            st.session_state.get(
                "industry_result"
            )
        )

        st.write(
            "**所属行业：**",
            industry or "--"
        )

        if ind:

            a, b = st.columns(2)

            a.metric(
                "景气分",
                ind["score"]
            )

            b.metric(
                "行业状态",
                ind["label"]
            )

            st.write(
                "20日行业走势：",
                f"{ind['ret20']:+.2f}%"
            )

        else:

            st.info(
                "本次未取得可靠行业行情。"
            )

    # ========================================================
    # 资金按钮
    # ========================================================

    if st.button(
        "💵 加载资金流",
        use_container_width=True
    ):

        with st.spinner(
            "读取资金流..."
        ):

            flow = (
                get_fund_flow(code)
            )

        st.session_state[
            "flow_data"
        ] = flow

        st.session_state[
            "flow_code"
        ] = code

    if (
        st.session_state.get(
            "flow_code"
        ) == code
    ):

        st.markdown(
            "### 💵 资金"
        )

        flow = (
            st.session_state.get(
                "flow_data"
            )
        )

        if flow:

            a, b = st.columns(2)

            a.metric(
                "主力净流入",
                fmt_money(
                    flow.get(
                        "main"
                    )
                )
            )

            b.metric(
                "主力净占比",
                (
                    "--"
                    if flow.get(
                        "ratio"
                    ) is None
                    else str(
                        flow.get(
                            "ratio"
                        )
                    )
                )
            )

        else:

            st.info(
                "本次未取得可靠资金流数据。"
            )

    # ========================================================
    # 公告按钮
    # ========================================================

    if st.button(
        "📦 加载订单 / 利好 / 风险",
        use_container_width=True
    ):

        with st.spinner(
            "读取近期公告..."
        ):

            notice_data = (
                get_notices(code)
            )

        st.session_state[
            "notice_data"
        ] = notice_data

        st.session_state[
            "notice_code"
        ] = code

    if (
        st.session_state.get(
            "notice_code"
        ) == code
    ):

       