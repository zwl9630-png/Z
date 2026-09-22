import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta

st.set_page_config(
    page_title="A股工程控制论",
    page_icon="📈",
    layout="centered"
)

st.title("📈 A股工程控制论")
st.caption("真实A股数据 · 手机网页版")

token = st.text_input(
    "Tushare Token",
    type="password",
    placeholder="请输入你的 Tushare Token"
)

code = st.text_input(
    "股票代码",
    placeholder="例如：600519",
    max_chars=6
)

def ts_code(c):
    if c.startswith(("5", "6", "9")):
        return c + ".SH"
    return c + ".SZ"


def calculate(df):
    close = df["close"]

    df["MA5"] = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()

    delta = close.diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)

    df["VOL20"] = df["volume"].rolling(20).mean()
    df["RET20"] = close.pct_change(20)
    df["RET60"] = close.pct_change(60)

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()
    df["ATR_PCT"] = df["ATR"] / close * 100

    score = np.zeros(len(df))

    score += np.where(close > df["MA20"], 15, 0)
    score += np.where(df["MA5"] > df["MA20"], 10, 0)
    score += np.where(df["MA20"] > df["MA60"], 20, 0)
    score += np.where(df["DIF"] > df["DEA"], 15, 0)

    score += np.where(
        (df["RSI"] >= 50) & (df["RSI"] <= 70),
        15,
        np.where(df["RSI"] > 70, 5, 0),
    )

    score += np.where(
        df["volume"] > df["VOL20"],
        10,
        3,
    )

    score += np.where(df["RET20"] > 0, 10, 0)
    score += np.where(df["RET60"] > 0, 5, 0)

    df["SCORE"] = score

    risk = (
        np.clip((df["ATR_PCT"].fillna(0) - 1) * 14, 0, 55)
        + np.clip((df["RSI"].fillna(50) - 68) * 2, 0, 25)
    )

    df["RISK"] = np.clip(risk, 0, 100)

    return df


def market_state(score, risk):

    if risk >= 70:
        return "🔴 风险释放"

    if score >= 75:
        return "🟢 趋势强化"

    if score >= 62:
        return "🟢 趋势形成"

    if score >= 45:
        return "🟡 震荡观察"

    return "🔴 趋势减弱"


def forecast(row):

    atr = row["ATR_PCT"]

    if pd.isna(atr):
        atr = 2

    volatility = max(float(atr) / 100, 0.012)

    momentum = row["RET20"]

    if pd.isna(momentum):
        momentum = 0

    bias = (
        0.7 * (row["SCORE"] - 50) / 50
        + 0.3 * np.clip(momentum * 3, -1, 1)
    )

    bias = np.clip(bias, -1, 1)

    bias *= (1 - 0.4 * row["RISK"] / 100)

    result = []

    for days in [5, 20, 60]:

        root = np.sqrt(days)

        drift = bias * volatility * root * 0.52
        band = volatility * root * 1.12

        probability = np.clip(
            50 + bias * 28,
            20,
            80
        )

        if bias > 0.14:
            direction = "偏强"

        elif bias < -0.14:
            direction = "偏弱"

        else:
            direction = "震荡"

        low = row["close"] * (1 + drift - band)
        high = row["close"] * (1 + drift + band)

        result.append(
            (days, direction, probability, low, high)
        )

    return result


if st.button(
    "开始分析",
    type="primary",
    use_container_width=True
):

    try:

        if not token:
            raise ValueError("请先输入 Tushare Token")

        if len(code) != 6 or not code.isdigit():
            raise ValueError("请输入正确的6位股票代码")

        with st.spinner("正在读取A股真实数据并分析…"):

            pro = ts.pro_api(token)

            end = datetime.now().strftime("%Y%m%d")

            start = (
                datetime.now()
                - timedelta(days=750)
            ).strftime("%Y%m%d")

            data = pro.daily(
                ts_code=ts_code(code),
                start_date=start,
                end_date=end
            )

            if data.empty:
                raise ValueError("没有取得该股票行情")

            data = data.rename(
                columns={
                    "trade_date": "date",
                    "vol": "volume"
                }
            )

            data["date"] = pd.to_datetime(data["date"])

            data = (
                data
                .sort_values("date")
                .reset_index(drop=True)
            )

            data = calculate(
                data[
                    [
                        "date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume"
                    ]
                ].copy()
            )

            if len(data) < 80:
                raise ValueError("历史数据不足")

            row = data.iloc[-1]

            info = pro.stock_basic(
                ts_code=ts_code(code),
                fields="ts_code,name,industry"
            )

            if len(info):

                name = info.iloc[0]["name"]
                industry = info.iloc[0]["industry"]

            else:

                name = code
                industry = "—"


        st.subheader(
            f"{name} · {code}"
        )

        st.caption(
            f"{industry} ｜ "
            f"数据日期 {row['date'].date()} ｜ "
            f"收盘 ¥{row['close']:.2f}"
        )


        state = market_state(
            row["SCORE"],
            row["RISK"]
        )


        c1, c2, c3 = st.columns(3)

        c1.metric(
            "当前状态",
            state
        )

        c2.metric(
            "控制分",
            f"{row['SCORE']:.0f}"
        )

        c3.metric(
            "风险",
            f"{row['RISK']:.0f}"
        )


        st.progress(
            int(
                np.clip(
                    row["SCORE"],
                    0,
                    100
                )
            )
        )


        st.markdown("### 🔮 未来情景")


        for (
            days,
            direction,
            probability,
            low,
            high
        ) in forecast(row):

            st.markdown(
                f"""
**{days}交易日：{direction}**

偏强概率：**{probability:.0f}%**

参考区间：
**¥{low:.2f} ～ ¥{high:.2f}**
"""
            )


        support = float(
            data.tail(20)["low"].min()
        )

        if (
            pd.notna(row["MA20"])
            and row["MA20"] < row["close"]
        ):

            support = max(
                support,
                float(row["MA20"])
            )


        resistance = float(
            data.tail(20)["high"].max()
        )


        st.markdown("### 🎯 关键位置")


        c1, c2 = st.columns(2)

        c1.metric(
            "参考支撑",
            f"¥{support:.2f}"
        )

        c2.metric(
            "参考压力",
            f"¥{resistance:.2f}"
        )


        st.markdown("### 📈 趋势")


        chart = (
            data[
                [
                    "date",
                    "close",
                    "MA20",
                    "MA60"
                ]
            ]
            .set_index("date")
        )

        st.line_chart(chart)


        with st.expander("查看详细指标"):

            st.write(
                "MA5：",
                round(row["MA5"], 2)
            )

            st.write(
                "MA20：",
                round(row["MA20"], 2)
            )

            st.write(
                "MA60：",
                round(row["MA60"], 2)
            )

            st.write(
                "RSI：",
                round(row["RSI"], 2)
            )

            st.write(
                "ATR%：",
                round(row["ATR_PCT"], 2)
            )

            st.write(
                "20日涨跌幅：",
                f"{row['RET20'] * 100:.2f}%"
            )

            st.write(
                "60日涨跌幅：",
                f"{row['RET60'] * 100:.2f}%"
            )


        st.info(
            "预测结果属于概率情景推演，"
            "不是确定性目标价。"
            "当前版本使用A股日线数据，"
            "不是Level-2实时盘口。"
        )


    except Exception as e:

        st.error(
            "分析失败：" + str(e)
        )
