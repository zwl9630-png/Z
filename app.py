import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import time
from datetime import datetime, timedelta

st.set_page_config(
    page_title="A股工程控制论",
    page_icon="📈",
    layout="centered"
)

st.title("📈 A股工程控制论")
st.caption("A股公开数据 · 免 Token · 多数据源自动容错")

code = st.text_input(
    "股票代码",
    value="",
    placeholder="例如：600519",
    max_chars=6
)


def normalize_df(df):
    """统一不同数据源的字段"""
    if df is None or df.empty:
        raise ValueError("行情数据为空")

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "date": "date",
        "open": "open",
        "close": "close",
        "high": "high",
        "low": "low",
        "volume": "volume",
    }

    df = df.rename(columns=rename_map)

    needed = ["date", "open", "close", "high", "low", "volume"]

    missing = [x for x in needed if x not in df.columns]

    if missing:
        raise ValueError("行情字段不完整")

    df = df[needed].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["open", "close", "high", "low", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna()
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    if len(df) < 70:
        raise ValueError("有效历史数据不足70个交易日")

    return df


def get_em_data(code):
    """数据源1：东方财富"""
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (
        datetime.now() - timedelta(days=600)
    ).strftime("%Y%m%d")

    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"
    )

    return normalize_df(df)


def get_sina_data(code):
    """数据源2：新浪"""
    symbol = (
        "sh" + code
        if code.startswith(("5", "6", "9"))
        else "sz" + code
    )

    df = ak.stock_zh_a_daily(
        symbol=symbol,
        adjust="qfq"
    )

    df = normalize_df(df)

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=600)

    df = df[df["date"] >= cutoff].reset_index(drop=True)

    return df


def get_data(code):
    """自动重试 + 自动切换数据源"""

    errors = []

    # 东方财富尝试2次
    for attempt in range(2):
        try:
            df = get_em_data(code)
            return df, "东方财富"
        except Exception as e:
            errors.append(f"东方财富第{attempt + 1}次：{e}")
            time.sleep(1.2)

    # 新浪尝试2次
    for attempt in range(2):
        try:
            df = get_sina_data(code)
            return df, "新浪财经"
        except Exception as e:
            errors.append(f"新浪第{attempt + 1}次：{e}")
            time.sleep(1.2)

    raise RuntimeError(
        "两个公开行情源暂时都无法连接。"
        "请稍后再试。"
    )


def calculate(df):
    df = df.copy()

    df["MA5"] = df["close"].rolling(5).mean()
    df["MA10"] = df["close"].rolling(10).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA60"] = df["close"].rolling(60).mean()

    ema12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["DIF"] = ema12 - ema26

    df["DEA"] = df["DIF"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD"] = 2 * (df["DIF"] - df["DEA"])

    delta = df["close"].diff()

    gain = delta.clip(lower=0).rolling(14).mean()

    loss = (
        -delta.clip(upper=0)
    ).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )

    df["VOL20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["RET20"] = df["close"].pct_change(20)
    df["RET60"] = df["close"].pct_change(60)

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.rolling(14).mean()

    df["ATR_PCT"] = (
        df["ATR"] / df["close"]
    )

    return df


def control_model(df):
    x = df.iloc[-1]

    score = 50

    if x["close"] > x["MA20"]:
        score += 10
    else:
        score -= 10

    if x["MA20"] > x["MA60"]:
        score += 12
    else:
        score -= 12

    if x["DIF"] > x["DEA"]:
        score += 10
    else:
        score -= 10

    if x["RET20"] > 0:
        score += 8
    else:
        score -= 8

    if x["RET60"] > 0:
        score += 8
    else:
        score -= 8

    if x["volume"] > x["VOL20"]:
        score += 5

    if pd.notna(x["RSI"]):
        if 45 <= x["RSI"] <= 70:
            score += 5

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

    return score, risk, state


def forecast(close, score, atr_pct, days):

    strength = (
        score - 50
    ) / 50

    scale = np.sqrt(days / 5)

    expected = (
        strength
        * atr_pct
        * scale
        * 0.8
    )

    center = close * (
        1 + expected
    )

    width = max(
        atr_pct * scale * 1.3,
        0.025
    )

    low = center * (
        1 - width
    )

    high = center * (
        1 + width
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
            "请输入正确的6位A股股票代码"
        )

    else:

        try:

            with st.spinner(
                "正在连接A股行情源..."
            ):

                df, source = get_data(code)

                df = calculate(df)

                score, risk, state = (
                    control_model(df)
                )

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
                f"分析完成 · 当前数据源：{source}"
            )

            st.subheader(
                f"股票代码：{code}"
            )

            st.caption(
                f"数据日期："
                f"{x['date'].strftime('%Y-%m-%d')}"
                f" ｜ 最新收盘："
                f"{close:.2f} 元"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "当前状态",
                state
            )

            c2.metric(
                "控制评分",
                f"{score}/100"
            )

            c3.metric(
                "风险等级",
                risk
            )

            st.divider()

            st.subheader(
                "🔮 未来情景分析"
            )

            for days in [5, 20, 60]:

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

情景区间：**{low:.2f} ～ {high:.2f} 元**
"""
                )

            st.divider()

            support20 = float(
                df["low"]
                .tail(20)
                .min()
            )

            resistance20 = float(
                df["high"]
                .tail(20)
                .max()
            )

            st.subheader(
                "🎯 关键位置"
            )

            a, b = st.columns(2)

            a.metric(
                "20日参考支撑",
                f"{support20:.2f}"
            )

            b.metric(
                "20日参考压力",
                f"{resistance20:.2f}"
            )

            st.subheader(
                "🧠 模型反馈"
            )

            if score >= 75:

                st.success(
                    "趋势结构较强，"
                    "均线、动量和MACD"
                    "形成较明显正反馈。"
                )

            elif score >= 62:

                st.info(
                    "趋势正在形成，"
                    "重点观察成交量以及"
                    "20日均线能否继续强化。"
                )

            elif score >= 45:

                st.warning(
                    "目前处于震荡观察阶段，"
                    "方向性暂时不够明确。"
                )

            elif score >= 30:

                st.warning(
                    "趋势出现减弱迹象，"
                    "重点观察支撑位及"
                    "风险反馈。"
                )

            else:

                st.error(
                    "当前处于风险释放阶段，"
                    "趋势和动量信号整体偏弱。"
                )

            with st.expander(
                "查看详细技术指标"
            ):

                st.write(
                    {
                        "MA5":
                        round(
                            float(x["MA5"]),
                            2
                        ),

                        "MA10":
                        round(
                            float(x["MA10"]),
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

                        "RSI14":
                        round(
                            float(x["RSI"]),
                            2
                        ),

                        "DIF":
                        round(
                            float(x["DIF"]),
                            3
                        ),

                        "DEA":
                        round(
                            float(x["DEA"]),
                            3
                        ),

                        "MACD":
                        round(
                            float(x["MACD"]),
                            3
                        ),

                        "20日涨跌幅":
                        f"{float(x['RET20']) * 100:.2f}%",

                        "60日涨跌幅":
                        f"{float(x['RET60']) * 100:.2f}%",

                        "ATR波动率":
                        f"{atr_pct * 100:.2f}%"
                    }
                )

            st.subheader(
                "📉 近期价格走势"
            )

            chart = (
                df[
                    ["date", "close"]
                ]
                .tail(120)
                .set_index("date")
            )

            st.line_chart(chart)

            st.caption(
                "模型基于历史价格、趋势、"
                "动量及波动率生成情景分析。"
                "概率和价格区间并非确定性预测，"
                "不构成投资建议。"
            )

        except Exception as e:

            st.error(
                "公开行情源当前连接失败。"
                "这通常是数据供应方限制"
                "云服务器访问导致的，"
                "并非股票代码错误。"
            )

            with st.expander(
                "查看错误信息"
            ):
                st.code(str(e))