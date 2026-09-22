import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta

st.set_page_config(
    page_title="A股工程控制论",
    page_icon="📈",
    layout="centered"
)

st.title("📈 A股工程控制论")
st.caption("A股公开数据 · 免 Token · 手机网页版")

code = st.text_input(
    "股票代码",
    placeholder="例如：600519",
    max_chars=6
)

def get_data(code):
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=500)).strftime("%Y%m%d")

    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"
    )

    if df is None or df.empty:
        raise ValueError("没有获取到该股票的数据")

    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume"
    })

    df["date"] = pd.to_datetime(df["date"])

    for col in ["open", "close", "high", "low", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df


def calculate(df):
    df = df.copy()

    df["MA5"] = df["close"].rolling(5).mean()
    df["MA10"] = df["close"].rolling(10).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA60"] = df["close"].rolling(60).mean()

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()

    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD"] = 2 * (df["DIF"] - df["DEA"])

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["VOL20"] = df["volume"].rolling(20).mean()
    df["RET20"] = df["close"].pct_change(20)
    df["RET60"] = df["close"].pct_change(60)

    prev_close = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()
    df["ATR_PCT"] = df["ATR"] / df["close"]

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

    if 45 <= x["RSI"] <= 70:
        score += 5

    score = int(np.clip(score, 0, 100))

    atr_pct = float(x["ATR_PCT"]) if pd.notna(x["ATR_PCT"]) else 0.03

    risk = "低"
    if atr_pct > 0.06:
        risk = "高"
    elif atr_pct > 0.04:
        risk = "中"

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
    strength = (score - 50) / 50

    scale = np.sqrt(days / 5)

    expected = strength * atr_pct * scale * 0.8

    center = close * (1 + expected)

    width = max(atr_pct * scale * 1.3, 0.025)

    low = center * (1 - width)
    high = center * (1 + width)

    probability = int(
        np.clip(
            50 + abs(score - 50) * 0.8,
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

    return direction, probability, low, high


if st.button("开始分析", type="primary", use_container_width=True):

    if not code.isdigit() or len(code) != 6:
        st.error("请输入正确的6位A股股票代码")

    else:
        try:
            with st.spinner("正在获取A股数据并运行模型..."):
                df = get_data(code)
                df = calculate(df)

                if len(df) < 70:
                    raise ValueError("历史数据不足，暂时无法完成模型分析")

                score, risk, state = control_model(df)

                x = df.iloc[-1]

                close = float(x["close"])
                atr_pct = (
                    float(x["ATR_PCT"])
                    if pd.notna(x["ATR_PCT"])
                    else 0.03
                )

            st.success("分析完成")

            st.subheader(f"股票代码：{code}")

            st.caption(
                f"数据日期：{x['date'].strftime('%Y-%m-%d')} ｜ "
                f"最新收盘：{close:.2f} 元"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric("当前状态", state)
            c2.metric("控制评分", f"{score}/100")
            c3.metric("风险等级", risk)

            st.divider()

            st.subheader("🔮 未来情景分析")

            for days in [5, 20, 60]:
                direction, probability, low, high = forecast(
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

            support20 = float(df["low"].tail(20).min())
            resistance20 = float(df["high"].tail(20).max())

            st.subheader("🎯 关键位置")

            a, b = st.columns(2)

            a.metric("20日参考支撑", f"{support20:.2f}")
            b.metric("20日参考压力", f"{resistance20:.2f}")

            st.subheader("🧠 模型反馈")

            if score >= 75:
                st.success(
                    "趋势结构较强，均线、动量和MACD形成较明显的正反馈。"
                )
            elif score >= 62:
                st.info(
                    "趋势正在形成，后续重点观察成交量和20日均线是否继续强化。"
                )
            elif score >= 45:
                st.warning(
                    "当前主要处于震荡观察阶段，方向性暂时不够明确。"
                )
            elif score >= 30:
                st.warning(
                    "趋势出现减弱迹象，应重点观察支撑位以及风险反馈。"
                )
            else:
                st.error(
                    "当前处于风险释放阶段，趋势和动量信号整体偏弱。"
                )

            with st.expander("查看详细技术指标"):
                st.write({
                    "MA5": round(float(x["MA5"]), 2),
                    "MA10": round(float(x["MA10"]), 2),
                    "MA20": round(float(x["MA20"]), 2),
                    "MA60": round(float(x["MA60"]), 2),
                    "RSI14": round(float(x["RSI"]), 2),
                    "DIF": round(float(x["DIF"]), 3),
                    "DEA": round(float(x["DEA"]), 3),
                    "MACD": round(float(x["MACD"]), 3),
                    "20日涨跌幅": f"{float(x['RET20']) * 100:.2f}%",
                    "60日涨跌幅": f"{float(x['RET60']) * 100:.2f}%",
                    "ATR波动率": f"{atr_pct * 100:.2f}%"
                })

            st.subheader("📉 近期价格走势")

            chart = (
                df[["date", "close"]]
                .tail(120)
                .set_index("date")
            )

            st.line_chart(chart)

            st.caption(
                "说明：预测为基于历史价格、趋势、动量和波动率的情景模型，"
                "不是确定性目标价，也不构成投资建议。"
            )

        except Exception as e:
            st.error(f"数据获取或分析失败：{e}")