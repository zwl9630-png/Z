import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import time
from datetime import datetime, timedelta

# =========================
# 页面设置
# =========================

st.set_page_config(
    page_title="A股工程控制论 V2.0",
    page_icon="📈",
    layout="centered"
)

st.title("📈 A股工程控制论")
st.caption("V2.0 · 市场扫描 · 个股控制 · 概率情景")


# =========================
# 数据层
# =========================

@st.cache_data(ttl=600, show_spinner=False)
def get_em_data(code):

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (
        datetime.now() - timedelta(days=650)
    ).strftime("%Y%m%d")

    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"
    )

    return normalize_df(df)


@st.cache_data(ttl=600, show_spinner=False)
def get_sina_data(code):

    symbol = (
        "sh" + code
        if code.startswith(("5", "6", "9"))
        else "sz" + code
    )

    df = ak.stock_zh_a_daily(
        symbol=symbol,
        adjust="qfq"
    )

    return normalize_df(df)


def normalize_df(df):

    if df is None or df.empty:
        raise ValueError("行情数据为空")

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume"
    }

    df = df.rename(columns=rename_map)

    needed = [
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume"
    ]

    missing = [
        x for x in needed
        if x not in df.columns
    ]

    if missing:
        raise ValueError("行情字段不完整")

    df = df[needed].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    for col in [
        "open",
        "close",
        "high",
        "low",
        "volume"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = (
        df.dropna()
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    if len(df) < 70:
        raise ValueError("历史数据不足")

    return df


def get_data(code):

    errors = []

    # 东方财富
    for attempt in range(2):

        try:
            df = get_em_data(code)
            return df, "东方财富"

        except Exception as e:
            errors.append(str(e))
            time.sleep(0.5)

    # 新浪
    for attempt in range(2):

        try:
            df = get_sina_data(code)
            return df, "新浪财经"

        except Exception as e:
            errors.append(str(e))
            time.sleep(0.5)

    raise RuntimeError(
        "公开行情源暂时无法连接"
    )


# =========================
# 指标计算
# =========================

def calculate(df):

    df = df.copy()

    df["MA5"] = (
        df["close"]
        .rolling(5)
        .mean()
    )

    df["MA10"] = (
        df["close"]
        .rolling(10)
        .mean()
    )

    df["MA20"] = (
        df["close"]
        .rolling(20)
        .mean()
    )

    df["MA60"] = (
        df["close"]
        .rolling(60)
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

    df["MACD"] = (
        2 * (df["DIF"] - df["DEA"])
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

    df["RSI"] = (
        100 - 100 / (1 + rs)
    )

    df["VOL20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["RET5"] = (
        df["close"]
        .pct_change(5)
    )

    df["RET20"] = (
        df["close"]
        .pct_change(20)
    )

    df["RET60"] = (
        df["close"]
        .pct_change(60)
    )

    prev_close = (
        df["close"].shift(1)
    )

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"] - prev_close
            ).abs(),
            (
                df["low"] - prev_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = (
        tr.rolling(14).mean()
    )

    df["ATR_PCT"] = (
        df["ATR"] / df["close"]
    )

    return df


# =========================
# 控制论评分
# =========================

def control_model(df):

    x = df.iloc[-1]

    score = 50
    reasons = []

    if x["close"] > x["MA20"]:
        score += 10
        reasons.append("价格站上MA20")
    else:
        score -= 10

    if x["MA20"] > x["MA60"]:
        score += 12
        reasons.append("中期均线多头")
    else:
        score -= 12

    if x["DIF"] > x["DEA"]:
        score += 10
        reasons.append("MACD偏强")
    else:
        score -= 10

    if x["RET20"] > 0:
        score += 8
        reasons.append("20日动量为正")
    else:
        score -= 8

    if x["RET60"] > 0:
        score += 8
        reasons.append("60日趋势为正")
    else:
        score -= 8

    if (
        pd.notna(x["VOL20"])
        and x["volume"] > x["VOL20"]
    ):
        score += 5
        reasons.append("成交量高于20日均量")

    if pd.notna(x["RSI"]):

        if 45 <= x["RSI"] <= 70:
            score += 5
            reasons.append("RSI处于趋势区")

        elif x["RSI"] > 80:
            score -= 5

    score = int(
        np.clip(score, 0, 100)
    )

    atr_pct = (
        float(x["ATR_PCT"])
        if pd.notna(x["ATR_PCT"])
        else 0.03
    )

    if atr_pct > 0.06:
        risk = "高"

    elif atr_pct > 0.04:
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
        reasons
    )


# =========================
# 概率情景
# =========================

def forecast(
    close,
    score,
    atr_pct,
    days
):

    strength = (
        score - 50
    ) / 50

    scale = np.sqrt(
        days / 5
    )

    expected = (
        strength
        * atr_pct
        * scale
        * 0.8
    )

    center = (
        close * (1 + expected)
    )

    width = max(
        atr_pct
        * scale
        * 1.3,
        0.025
    )

    low = (
        center * (1 - width)
    )

    high = (
        center * (1 + width)
    )

    probability = int(
        np.clip(
            50
            + abs(score - 50) * 0.8,
            50,
            82
        )
    )

    if score >= 60:
        direction = "偏强"

    elif score <= 40:
        direction = "偏弱"

    else:
        direction = "震荡"

    return (
        direction,
        probability,
        low,
        high
    )


# =========================
# 单股分析页面
# =========================

def stock_page():

    st.subheader("🔎 个股分析")

    code = st.text_input(
        "输入6位股票代码",
        placeholder="例如：000977",
        max_chars=6
    )

    if st.button(
        "开始分析",
        type="primary",
        use_container_width=True
    ):

        code = code.strip()

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
                "正在运行控制模型..."
            ):

                df, source = get_data(code)

                df = calculate(df)

                (
                    score,
                    risk,
                    state,
                    reasons
                ) = control_model(df)

                x = df.iloc[-1]

                close = float(
                    x["close"]
                )

                atr_pct = (
                    float(x["ATR_PCT"])
                    if pd.notna(
                        x["ATR_PCT"]
                    )
                    else 0.03
                )

            st.success(
                f"分析完成 · {source}"
            )

            st.subheader(
                f"股票代码：{code}"
            )

            st.caption(
                f"{x['date'].strftime('%Y-%m-%d')}"
                f" ｜ 收盘 {close:.2f} 元"
            )

            c1, c2, c3 = (
                st.columns(3)
            )

            c1.metric(
                "状态",
                state
            )

            c2.metric(
                "评分",
                f"{score}"
            )

            c3.metric(
                "风险",
                risk
            )

            st.divider()

            st.subheader(
                "🔮 未来情景"
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
                    high
                ) = forecast(
                    close,
                    score,
                    atr_pct,
                    days
                )

                st.markdown(
                    f"""
**{days}个交易日**

方向：**{direction}**  
参考概率：**{probability}%**  
区间：**{low:.2f} ～ {high:.2f} 元**
"""
                )

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
                "参考支撑",
                f"{support:.2f}"
            )

            b.metric(
                "参考压力",
                f"{resistance:.2f}"
            )

            st.subheader(
                "⚙️ 控制反馈"
            )

            if reasons:

                for reason in reasons:
                    st.write(
                        "• " + reason
                    )

            st.write(
                f"**失效观察：** "
                f"若价格有效跌破 "
                f"{support:.2f}，"
                f"需要重新评估当前状态。"
            )

            st.subheader(
                "📉 近期走势"
            )

            chart = (
                df[
                    [
                        "date",
                        "close"
                    ]
                ]
                .tail(120)
                .set_index("date")
            )

            st.line_chart(chart)

            with st.expander(
                "查看专业指标"
            ):

                st.write(
                    {
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
                        f"{atr_pct * 100:.2f}%"
                    }
                )

        except Exception as e:

            st.error(
                "行情源暂时连接失败，"
                "请稍后重试。"
            )

            with st.expander(
                "错误详情"
            ):
                st.code(str(e))


# =========================
# 今日机会
# =========================

WATCHLIST = {

    "000977": "浪潮信息",
    "002475": "立讯精密",
    "002463": "沪电股份",
    "002916": "深南电路",
    "300308": "中际旭创",
    "300502": "新易盛",
    "600183": "生益科技",
    "600584": "长电科技",
    "600570": "恒生电子",
    "600588": "用友网络",
    "600745": "闻泰科技",
    "601138": "工业富联",
    "601360": "三六零",
    "603019": "中科曙光",
    "603986": "兆易创新",
    "688008": "澜起科技",
    "688041": "海光信息",
    "688111": "金山办公"
}


def opportunity_page():

    st.subheader(
        "🎯 今日机会"
    )

    st.caption(
        "寻找模型状态较强、"
        "风险可控的候选信号。"
        "候选≠买入建议。"
    )

    st.info(
        "当前版本采用精选候选池扫描。"
        "后续版本升级为服务器端全A扫描。"
    )

    max_scan = st.slider(
        "本次扫描数量",
        5,
        len(WATCHLIST),
        10
    )

    if st.button(
        "开始扫描",
        type="primary",
        use_container_width=True
    ):

        results = []

        progress = st.progress(0)

        items = list(
            WATCHLIST.items()
        )[:max_scan]

        for i, (
            code,
            name
        ) in enumerate(items):

            try:

                df, source = (
                    get_data(code)
                )

                df = calculate(df)

                (
                    score,
                    risk,
                    state,
                    reasons
                ) = control_model(df)

                x = df.iloc[-1]

                close = float(
                    x["close"]
                )

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

                results.append(
                    {
                        "代码": code,
                        "名称": name,
                        "状态": state,
                        "评分": score,
                        "风险": risk,
                        "收盘": round(
                            close,
                            2
                        ),
                        "支撑": round(
                            support,
                            2
                        ),
                        "压力": round(
                            resistance,
                            2
                        ),
                        "触发因素":
                        " / ".join(
                            reasons[:3]
                        )
                    }
                )

            except Exception:
                pass

            progress.progress(
                (i + 1)
                / len(items)
            )

            time.sleep(0.15)

        progress.empty()

        if not results:

            st.warning(
                "本次未取得有效扫描结果。"
                "可能是公开行情源暂时限制访问。"
            )

            return

        result_df = pd.DataFrame(
            results
        )

        risk_order = {
            "低": 0,
            "中": 1,
            "高": 2
        }

        result_df[
            "_risk"
        ] = (
            result_df["风险"]
            .map(risk_order)
        )

        result_df = (
            result_df
            .sort_values(
                [
                    "评分",
                    "_risk"
                ],
                ascending=[
                    False,
                    True
                ]
            )
            .drop(
                columns=["_risk"]
            )
        )

        st.subheader(
            "模型候选"
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        strong = result_df[
            (
                result_df["评分"] >= 62
            )
            &
            (
                result_df["风险"]
                != "高"
            )
        ]

        if not strong.empty:

            st.subheader(
                "🔔 状态关注"
            )

            for _, row in (
                strong.head(5)
                .iterrows()
            ):

                st.write(
                    f"**{row['名称']} "
                    f"{row['代码']}** ｜ "
                    f"{row['状态']} ｜ "
                    f"评分 {row['评分']} ｜ "
                    f"风险 {row['风险']}"
                )

        else:

            st.info(
                "本次扫描没有出现"
                "评分≥62且风险非高的候选。"
            )


# =========================
# 自选股
# =========================

def watchlist_page():

    st.subheader(
        "⭐ 自选股"
    )

    st.caption(
        "V2.0 第一阶段先提供"
        "快速自选池。"
    )

    default_codes = (
        "000977,600519,002475"
    )

    text = st.text_area(
        "输入股票代码，用逗号分隔",
        value=default_codes
    )

    if st.button(
        "分析自选股",
        type="primary",
        use_container_width=True
    ):

        codes = [
            x.strip()
            for x in text.split(",")
            if (
                x.strip().isdigit()
                and len(
                    x.strip()
                ) == 6
            )
        ]

        if not codes:

            st.warning(
                "请至少输入一个"
                "正确的6位股票代码。"
            )

            return

        rows = []

        progress = st.progress(0)

        for i, code in enumerate(
            codes[:15]
        ):

            try:

                df, source = (
                    get_data(code)
                )

                df = calculate(df)

                (
                    score,
                    risk,
                    state,
                    reasons
                ) = control_model(df)

                x = df.iloc[-1]

                rows.append(
                    {
                        "代码": code,
                        "状态": state,
                        "评分": score,
                        "风险": risk,
                        "收盘":
                        round(
                            float(
                                x["close"]
                            ),
                            2
                        ),
                        "20日%":
                        round(
                            float(
                                x["RET20"]
                            ) * 100,
                            2
                        ),
                        "60日%":
                        round(
                            float(
                                x["RET60"]
                            ) * 100,
                            2
                        )
                    }
                )

            except Exception:
                pass

            progress.progress(
                (i + 1)
                / min(
                    len(codes),
                    15
                )
            )

        progress.empty()

        if rows:

            df_result = (
                pd.DataFrame(rows)
                .sort_values(
                    "评分",
                    ascending=False
                )
            )

            st.dataframe(
                df_result,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "暂时没有取得有效数据。"
            )


# =========================
# 主导航
# =========================

page = st.radio(
    "功能",
    [
        "🎯 今日机会",
        "🔎 个股分析",
        "⭐ 自选股"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()

if page == "🎯 今日机会":

    opportunity_page()

elif page == "🔎 个股分析":

    stock_page()

else:

    watchlist_page()


st.divider()

st.caption(
    "A股工程控制论 V2.0 ｜ "
    "模型输出用于研究与情景分析，"
    "不构成投资建议。"
)