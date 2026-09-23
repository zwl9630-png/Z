# -*- coding: utf-8 -*-
"""
A股工程控制论 V1.1
新增：趋势变化 / 趋势加速度 ΔScore
数据源：AKShare（免 Token）
运行：
    pip install -r requirements.txt
    streamlit run app.py
"""

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import akshare as ak


st.set_page_config(
    page_title="A股工程控制论",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# 基础工具
# -----------------------------
def clamp(x, lo=0.0, hi=100.0):
    try:
        return float(max(lo, min(hi, x)))
    except Exception:
        return float(lo)


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        if isinstance(x, str):
            x = x.replace(",", "").replace("%", "").strip()
        return float(x)
    except Exception:
        return default


def normalize_code(code: str) -> str:
    code = str(code).strip().upper().replace("SH", "").replace("SZ", "").replace(".", "")
    code = "".join(ch for ch in code if ch.isdigit())
    return code.zfill(6)[-6:]


def market_symbol(code: str) -> str:
    return normalize_code(code)


def fmt_num(x, digits=2, empty="—"):
    try:
        if pd.isna(x):
            return empty
        return f"{float(x):.{digits}f}"
    except Exception:
        return empty


def pct(x, digits=1, empty="—"):
    try:
        if pd.isna(x):
            return empty
        return f"{float(x) * 100:.{digits}f}%"
    except Exception:
        return empty


# -----------------------------
# 数据获取
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def get_hist(code: str, days: int = 260) -> pd.DataFrame:
    code = market_symbol(code)
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=max(days * 2, 500))).strftime("%Y%m%d")

    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start,
        end_date=end,
        adjust="qfq",
    )

    if df is None or df.empty:
        raise ValueError("没有获取到行情数据")

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_chg",
        "涨跌额": "change",
        "换手率": "turnover",
    }

    df = df.rename(columns=rename_map).copy()
    df["date"] = pd.to_datetime(df["date"])

    for c in [
        "open", "close", "high", "low",
        "volume", "amount", "pct_chg", "turnover"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.sort_values("date")
        .dropna(subset=["close"])
        .tail(days)
        .reset_index(drop=True)
    )

    return df


@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_info(code: str) -> dict:
    code = normalize_code(code)

    out = {
        "name": code,
        "industry": "—",
        "pe": np.nan,
        "pb": np.nan,
        "market_cap": np.nan,
    }

    try:
        info = ak.stock_individual_info_em(symbol=code)

        if info is not None and not info.empty:
            if {"item", "value"}.issubset(set(info.columns)):
                kv = dict(zip(info["item"], info["value"]))
            elif len(info.columns) >= 2:
                kv = dict(zip(info.iloc[:, 0], info.iloc[:, 1]))
            else:
                kv = {}

            out["name"] = str(
                kv.get("股票简称", kv.get("名称", code))
            )

            out["industry"] = str(
                kv.get("行业", "—")
            )

            out["pe"] = safe_float(
                kv.get(
                    "市盈率-动态",
                    kv.get("市盈率", np.nan)
                )
            )

            out["pb"] = safe_float(
                kv.get("市净率", np.nan)
            )

            out["market_cap"] = safe_float(
                kv.get("总市值", np.nan)
            )

    except Exception:
        pass

    try:
        spot = ak.stock_zh_a_spot_em()
        row = spot[spot["代码"].astype(str) == code]

        if not row.empty:
            r = row.iloc[0]

            out["name"] = str(
                r.get("名称", out["name"])
            )

            out["pe"] = safe_float(
                r.get("市盈率-动态", out["pe"])
            )

            out["pb"] = safe_float(
                r.get("市净率", out["pb"])
            )

            out["market_cap"] = safe_float(
                r.get("总市值", out["market_cap"])
            )

    except Exception:
        pass

    return out


# -----------------------------
# 技术指标
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    for n in [5, 10, 20, 60]:
        d[f"ma{n}"] = d["close"].rolling(n).mean()

    # MACD
    d["ema12"] = d["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    d["ema26"] = d["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    d["dif"] = d["ema12"] - d["ema26"]

    d["dea"] = d["dif"].ewm(
        span=9,
        adjust=False
    ).mean()

    d["macd"] = 2 * (d["dif"] - d["dea"])

    # RSI
    delta = d["close"].diff()

    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)

    roll_up = up.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    roll_down = down.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = roll_up / roll_down.replace(0, np.nan)

    d["rsi14"] = 100 - (100 / (1 + rs))
    d["rsi14"] = d["rsi14"].fillna(50)

    # ATR
    prev_close = d["close"].shift(1)

    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - prev_close).abs(),
            (d["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    d["atr14"] = tr.rolling(14).mean()
    d["atr_pct"] = d["atr14"] / d["close"]

    # 成交量
    d["vol_ma5"] = d["volume"].rolling(5).mean()
    d["vol_ma20"] = d["volume"].rolling(20).mean()

    d["vol_ratio"] = (
        d["vol_ma5"]
        / d["vol_ma20"].replace(0, np.nan)
    )

    # 动量
    d["ret5"] = d["close"].pct_change(5)
    d["ret10"] = d["close"].pct_change(10)
    d["ret20"] = d["close"].pct_change(20)
    d["ret60"] = d["close"].pct_change(60)

    # 高低点
    d["high20"] = d["high"].rolling(20).max()
    d["low20"] = d["low"].rolling(20).min()

    d["high60"] = d["high"].rolling(60).max()
    d["low60"] = d["low"].rolling(60).min()

    # 偏离均线
    d["dist_ma20"] = (
        d["close"] / d["ma20"] - 1
    )

    d["dist_ma60"] = (
        d["close"] / d["ma60"] - 1
    )

    # 上影线比例
    body_top = d[["open", "close"]].max(axis=1)

    d["upper_shadow_pct"] = (
        (d["high"] - body_top)
        .clip(lower=0)
        / d["close"]
    )

    # 20日回撤
    rolling_peak = d["close"].rolling(
        20,
        min_periods=1
    ).max()

    d["dd20"] = (
        d["close"]
        / rolling_peak
        - 1
    )

    return d


# -----------------------------
# 技术评分
# -----------------------------
def technical_score_row(row) -> float:
    """
    技术分 0-100
    """

    score = 50.0

    close = safe_float(row.get("close"))
    ma5 = safe_float(row.get("ma5"))
    ma10 = safe_float(row.get("ma10"))
    ma20 = safe_float(row.get("ma20"))
    ma60 = safe_float(row.get("ma60"))

    # -------------------------
    # 1. 均线结构
    # -------------------------
    if all(
        np.isfinite(x)
        for x in [close, ma5, ma10, ma20]
    ):

        if close > ma5 > ma10 > ma20:
            score += 20

        elif close > ma10 > ma20:
            score += 13

        elif close > ma20:
            score += 7

        elif close < ma20:
            score -= 8

    if np.isfinite(ma20) and np.isfinite(ma60):

        if ma20 > ma60:
            score += 8

        else:
            score -= 4

    # -------------------------
    # 2. MACD
    # -------------------------
    dif = safe_float(row.get("dif"))
    dea = safe_float(row.get("dea"))
    macd = safe_float(row.get("macd"))

    if np.isfinite(dif) and np.isfinite(dea):

        if dif > dea:
            score += 8
        else:
            score -= 6

    if np.isfinite(macd):

        if macd > 0:
            score += 4
        else:
            score -= 3

    # -------------------------
    # 3. RSI
    # -------------------------
    rsi = safe_float(
        row.get("rsi14"),
        50
    )

    if 55 <= rsi <= 72:
        score += 10

    elif 50 <= rsi < 55:
        score += 5

    elif 72 < rsi <= 80:
        score += 2

    elif rsi > 80:
        score -= 8

    elif rsi < 40:
        score -= 8

    # -------------------------
    # 4. 动量
    # -------------------------
    ret5 = safe_float(
        row.get("ret5"),
        0
    )

    ret20 = safe_float(
        row.get("ret20"),
        0
    )

    if ret5 > 0:
        score += min(
            ret5 * 100 * 1.2,
            6
        )

    else:
        score += max(
            ret5 * 100,
            -6
        )

    if ret20 > 0:
        score += min(
            ret20 * 100 * 0.7,
            8
        )

    else:
        score += max(
            ret20 * 100 * 0.6,
            -8
        )

    # -------------------------
    # 5. 量价
    # -------------------------
    vr = safe_float(
        row.get("vol_ratio"),
        1.0
    )

    if 1.0 <= vr <= 1.8 and ret5 > 0:

        score += 6

    elif vr > 2.5:

        score -= 3

    # -------------------------
    # 6. 过热惩罚
    # -------------------------
    dist20 = safe_float(
        row.get("dist_ma20"),
        0
    )

    if dist20 > 0.18:

        score -= 15

    elif dist20 > 0.12:

        score -= 8

    elif dist20 > 0.08:

        score -= 3

    # -------------------------
    # 7. 上影线惩罚
    # -------------------------
    upper = safe_float(
        row.get("upper_shadow_pct"),
        0
    )

    if upper > 0.06:

        score -= 8

    elif upper > 0.04:

        score -= 4

    # -------------------------
    # 8. 回撤惩罚
    # -------------------------
    dd20 = safe_float(
        row.get("dd20"),
        0
    )

    if dd20 < -0.12:

        score -= 10

    elif dd20 < -0.08:

        score -= 6

    elif dd20 < -0.04:

        score -= 3

    return clamp(
        round(score, 1)
    )


# -----------------------------
# 构建趋势变化
# -----------------------------
def build_score_history(
    df: pd.DataFrame
) -> pd.DataFrame:

    d = add_indicators(df)

    d["tech_score"] = d.apply(
        technical_score_row,
        axis=1
    )

    # 平滑分
    d["score_smooth"] = (
        d["tech_score"]
        .ewm(
            span=3,
            adjust=False
        )
        .mean()
        .round(1)
    )

    # -------------------------
    # ΔScore
    # -------------------------
    d["delta1"] = (
        d["score_smooth"]
        .diff(1)
    )

    d["delta3"] = (
        d["score_smooth"]
        .diff(3)
    )

    d["delta5"] = (
        d["score_smooth"]
        .diff(5)
    )

    # 趋势速度
    d["velocity"] = d["delta3"]

    # -------------------------
    # 趋势加速度
    # -------------------------
    prev_velocity = (
        d["score_smooth"].shift(3)
        - d["score_smooth"].shift(6)
    )

    d["acceleration"] = (
        d["velocity"]
        - prev_velocity
    )

    return d


# -----------------------------
# 状态机
# -----------------------------
def get_state(
    score: float,
    row: pd.Series
) -> str:

    close = safe_float(
        row.get("close")
    )

    ma20 = safe_float(
        row.get("ma20")
    )

    dif = safe_float(
        row.get("dif")
    )

    dea = safe_float(
        row.get("dea")
    )

    above20 = (
        np.isfinite(close)
        and np.isfinite(ma20)
        and close >= ma20
    )

    macd_ok = (
        np.isfinite(dif)
        and np.isfinite(dea)
        and dif >= dea
    )

    if (
        score >= 85
        and above20
        and macd_ok
    ):
        return "趋势强化"

    if (
        score >= 60
        and above20
    ):
        return "趋势形成"

    if score >= 40:
        return "震荡观察"

    return "风险释放"


# -----------------------------
# 趋势变化状态
# -----------------------------
def trend_change_label(
    delta3: float,
    acceleration: float,
    score: float
):

    d3 = (
        0
        if not np.isfinite(delta3)
        else float(delta3)
    )

    acc = (
        0
        if not np.isfinite(acceleration)
        else float(acceleration)
    )

    if (
        d3 >= 12
        and acc >= 4
    ):
        return (
            "🚀 加速强化",
            "技术分连续抬升，且上涨速度继续增加"
        )

    if d3 >= 6:

        return (
            "📈 趋势增强",
            "技术分持续上升，正反馈正在加强"
        )

    if d3 >= 2:

        return (
            "↗ 温和改善",
            "技术状态改善，但加速度不强"
        )

    if (
        d3 <= -12
        and acc <= -4
    ):

        return (
            "🔻 加速转弱",
            "技术分下滑且下降速度扩大"
        )

    if d3 <= -6:

        return (
            "📉 趋势减速",
            "技术分明显回落，原趋势正在削弱"
        )

    if d3 <= -2:

        return (
            "↘ 小幅转弱",
            "技术分轻度下降，需观察是否继续恶化"
        )

    return (
        "➡️ 趋势平稳",
        "近3日技术分变化较小"
    )


# -----------------------------
# 风险等级
# -----------------------------
def risk_level(row: pd.Series):

    risk = 0.0

    atrp = safe_float(
        row.get("atr_pct"),
        0
    )

    rsi = safe_float(
        row.get("rsi14"),
        50
    )

    dist20 = abs(
        safe_float(
            row.get("dist_ma20"),
            0
        )
    )

    dd20 = abs(
        min(
            0,
            safe_float(
                row.get("dd20"),
                0
            )
        )
    )

    upper = safe_float(
        row.get("upper_shadow_pct"),
        0
    )

    risk += min(
        atrp / 0.05 * 25,
        25
    )

    risk += min(
        dist20 / 0.15 * 25,
        25
    )

    risk += min(
        dd20 / 0.12 * 30,
        30
    )

    risk += min(
        upper / 0.06 * 10,
        10
    )

    if rsi > 80:
        risk += 10

    elif rsi < 35:
        risk += 8

    risk = clamp(risk)

    if risk < 38:
        return "低", risk

    if risk < 66:
        return "中", risk

    return "高", risk


# -----------------------------
# 5 / 20 / 60日情景
# -----------------------------
def forecast_scenarios(
    df: pd.DataFrame,
    current_score: float
):

    d = df.copy()

    close = float(
        d["close"].iloc[-1]
    )

    daily_ret = (
        d["close"]
        .pct_change()
        .dropna()
    )

    vol = (
        daily_ret
        .tail(60)
        .std()
    )

    if (
        not np.isfinite(vol)
        or vol <= 0
    ):
        vol = 0.02

    trend_bias = (
        current_score - 50
    ) / 50.0

    mom20 = safe_float(
        d["close"]
        .pct_change(20)
        .iloc[-1],
        0
    )

    rows = []

    for horizon in [
        5,
        20,
        60
    ]:

        drift = np.clip(
            0.35
            * mom20
            * (horizon / 20)
            + 0.025
            * trend_bias
            * math.sqrt(
                horizon / 20
            ),
            -0.22,
            0.30,
        )

        sigma = (
            vol
            * math.sqrt(horizon)
        )

        low = (
            close
            * (
                1
                + drift
                - 0.80 * sigma
            )
        )

        high = (
            close
            * (
                1
                + drift
                + 0.80 * sigma
            )
        )

        mid = (
            close
            * (
                1
                + drift
            )
        )

        prob_up = clamp(
            50
            + trend_bias * 25
            + np.clip(
                mom20 * 100,
                -15,
                15
            ) * 0.8,
            20,
            80,
        )

        if prob_up >= 58:

            scene = "偏强"

        elif prob_up <= 42:

            scene = "偏弱"

        else:

            scene = "震荡"

        rows.append(
            {
                "周期": f"{horizon}日",
                "情景": scene,
                "概率": f"{prob_up:.0f}%",
                "中枢": round(
                    mid,
                    2
                ),
                "参考区间": (
                    f"{low:.2f}～{high:.2f}"
                ),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# 趋势温度计
# -----------------------------
def stock_temperature(
    score,
    delta3,
    risk_score
):

    base = (
        score - 50
    ) * 1.35

    delta = (
        np.clip(
            delta3
            if np.isfinite(delta3)
            else 0,
            -20,
            20
        )
        * 1.2
    )

    risk_penalty = (
        max(
            0,
            risk_score - 50
        )
        * 0.5
    )

    temp = int(
        round(
            np.clip(
                base
                + delta
                - risk_penalty,
                -100,
                100
            )
        )
    )

    if temp >= 75:

        label = "极热 / 强趋势"

    elif temp >= 45:

        label = "活跃 / 偏强"

    elif temp >= 15:

        label = "温和偏强"

    elif temp > -15:

        label = "中性"

    elif temp > -45:

        label = "偏冷 / 偏弱"

    elif temp > -75:

        label = "低迷 / 弱势"

    else:

        label = "极冷 / 风险释放"

    return temp, label


# -----------------------------
# 页面
# -----------------------------
st.title(
    "📈 A股工程控制论 V1.1"
)

st.caption(
    "新增：趋势变化 ΔScore + 趋势速度 + 趋势加速度。"
    "核心不是只看“现在多少分”，而是看分数正在向哪个方向变化。"
)

code = st.text_input(
    "股票代码",
    value="002332",
    max_chars=12
)

analyze = st.button(
    "⚡ 开始快速分析",
    type="primary",
    use_container_width=True
)


if analyze or code:

    code = normalize_code(code)

    try:

        with st.spinner(
            "正在获取行情并运行工程控制模型..."
        ):

            hist_raw = get_hist(
                code,
                280
            )

            d = build_score_history(
                hist_raw
            )

            info = get_stock_info(
                code
            )

        if len(d) < 65:

            st.warning(
                "历史数据不足65个交易日，部分指标可能不完整。"
            )

        latest = d.iloc[-1]

        current_score = float(
            latest["score_smooth"]
        )

        delta1 = safe_float(
            latest.get("delta1"),
            0
        )

        delta3 = safe_float(
            latest.get("delta3"),
            0
        )

        delta5 = safe_float(
            latest.get("delta5"),
            0
        )

        acceleration = safe_float(
            latest.get("acceleration"),
            0
        )

        state = get_state(
            current_score,
            latest
        )

        risk_text, risk_score = (
            risk_level(latest)
        )

        trend_label, trend_desc = (
            trend_change_label(
                delta3,
                acceleration,
                current_score
            )
        )

        temp, temp_label = (
            stock_temperature(
                current_score,
                delta3,
                risk_score
            )
        )

        name = info.get(
            "name",
            code
        )

        date_text = (
            pd.to_datetime(
                latest["date"]
            )
            .strftime(
                "%Y-%m-%d"
            )
        )

        st.success(
            "⚡ 快速分析完成"
        )

        st.markdown(
            f"## {name} · {code}"
        )

        st.caption(
            f"行情日期：{date_text}"
        )

        # -------------------------
        # 顶部核心指标
        # -------------------------
        c1, c2, c3, c4 = (
            st.columns(4)
        )

        c1.metric(
            "技术分",
            f"{current_score:.0f}",
            f"{delta1:+.1f} / 1日"
        )

        c2.metric(
            "状态",
            state
        )

        c3.metric(
            "风险",
            risk_text,
            f"风险分 {risk_score:.0f}"
        )

        c4.metric(
            "最新收盘",
            f"{latest['close']:.2f}"
        )

        st.divider()

        # -------------------------
        # 趋势变化
        # -------------------------
        st.subheader(
            "🚦 趋势变化 / ΔScore"
        )

        a1, a2, a3, a4 = (
            st.columns(4)
        )

        a1.metric(
            "1日 ΔScore",
            f"{delta1:+.1f}"
        )

        a2.metric(
            "3日 ΔScore",
            f"{delta3:+.1f}"
        )

        a3.metric(
            "5日 ΔScore",
            f"{delta5:+.1f}"
        )

        a4.metric(
            "趋势加速度",
            f"{acceleration:+.1f}"
        )

        st.info(
            f"**{trend_label}** ｜ {trend_desc}"
        )

        # -------------------------
        # 自动解释
        # -------------------------
        if (
            current_score >= 85
            and delta3 > 0
        ):

            st.write(
                "当前属于："
                "**高分 + 趋势继续增强**。"
                "趋势本身强，且控制力仍在提高。"
            )

        elif (
            current_score >= 85
            and delta3 < -5
        ):

            st.write(
                "当前属于："
                "**高分但开始减速**。"
                "绝对分数仍高，但边际趋势已经转弱，"
                "需要防止“高分见顶”。"
            )

        elif (
            60 <= current_score < 85
            and delta3 >= 6
        ):

            st.write(
                "当前属于："
                "**趋势形成并快速改善**。"
                "这是模型重点观察的由弱转强阶段。"
            )

        elif (
            current_score < 40
            and delta3 >= 6
        ):

            st.write(
                "当前属于："
                "**风险释放后的修复早期**。"
                "分数仍低，但趋势方向已经开始改善。"
            )

        elif (
            current_score < 40
            and delta3 <= 0
        ):

            st.write(
                "当前属于："
                "**风险释放且尚未出现明显修复**。"
            )

        else:

            st.write(
                "当前趋势变化处于中间状态，"
                "重点观察未来2～5个交易日ΔScore是否持续同方向。"
            )

        # -------------------------
        # 技术分轨迹
        # -------------------------
        chart_df = (
            d[
                [
                    "date",
                    "score_smooth"
                ]
            ]
            .tail(30)
            .set_index("date")
        )

        st.subheader(
            "📉 近30个交易日技术分轨迹"
        )

        st.line_chart(
            chart_df,
            height=260
        )

        st.caption(
            "判读重点：不是100分一定更好，"
            "而是观察60→70→80→90的持续抬升，"
            "以及100→95→88这类减速信号。"
        )

        # -------------------------
        # 温度计
        # -------------------------
        st.subheader(
            "🌡️ 个股趋势温度计"
        )

        st.metric(
            "温度",
            f"{temp} ℃",
            temp_label
        )

        st.progress(
            int(
                (temp + 100) / 2
            )
        )

        st.markdown(
            """
**温度解释：**

- +75～+100：极热 / 强趋势
- +45～+74：活跃 / 偏强
- +15～+44：温和偏强
- -14～+14：中性
- -44～-15：偏冷 / 偏弱
- -74～-45：低迷 / 弱势
- -100～-75：极冷 / 风险释放
            """
        )

        # -------------------------
        # 技术结构
        # -------------------------
        st.subheader(
            "🧭 技术结构"
        )

        t1, t2, t3, t4 = (
            st.columns(4)
        )

        t1.metric(
            "MA5",
            fmt_num(
                latest.get("ma5")
            )
        )

        t2.metric(
            "MA10",
            fmt_num(
                latest.get("ma10")
            )
        )

        t3.metric(
            "MA20",
            fmt_num(
                latest.get("ma20")
            )
        )

        t4.metric(
            "MA60",
            fmt_num(
                latest.get("ma60")
            )
        )

        t5, t6, t7, t8 = (
            st.columns(4)
        )

        t5.metric(
            "RSI14",
            fmt_num(
                latest.get("rsi14"),
                1
            )
        )

        t6.metric(
            "MACD",
            fmt_num(
                latest.get("macd"),
                3
            )
        )

        t7.metric(
            "20日涨跌",
            pct(
                latest.get("ret20")
            )
        )

        t8.metric(
            "量比(5/20)",
            fmt_num(
                latest.get("vol_ratio"),
                2
            )
        )

        # -------------------------
        # 概率情景
        # -------------------------
        st.subheader(
            "🔮 5 / 20 / 60日概率情景"
        )

        scenario_df = (
            forecast_scenarios(
                d,
                current_score
            )
        )

        st.dataframe(
            scenario_df,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------
        # 基本信息
        # -------------------------
        st.subheader(
            "🏢 基本信息"
        )

        f1, f2, f3, f4 = (
            st.columns(4)
        )

        f1.metric(
            "行业",
            str(
                info.get(
                    "industry",
                    "—"
                )
            )
        )

        f2.metric(
            "PE",
            fmt_num(
                info.get("pe")
            )
        )

        f3.metric(
            "PB",
            fmt_num(
                info.get("pb")
            )
        )

        mc = info.get(
            "market_cap",
            np.nan
        )

        if np.isfinite(mc):

            if mc > 1e7:

                mc_text = (
                    f"{mc / 1e8:.1f} 亿"
                )

            else:

                mc_text = fmt_num(mc)

        else:

            mc_text = "—"

        f4.metric(
            "总市值",
            mc_text
        )

        # -------------------------
        # 状态机说明
        # -------------------------
        st.subheader(
            "🧠 工程控制论状态机"
        )

        state_df = pd.DataFrame(
            [
                [
                    "趋势强化",
                    "高技术分 + 多头结构 + 动量确认",
                    "趋势已建立并强化"
                ],
                [
                    "趋势形成",
                    "技术分进入中高区 + 站上中期均线",
                    "正反馈正在建立"
                ],
                [
                    "震荡观察",
                    "多空信号混合",
                    "等待控制方向确认"
                ],
                [
                    "风险释放",
                    "技术分低 / 中期结构偏弱",
                    "负反馈占优，先释放风险"
                ],
            ],
            columns=[
                "状态",
                "典型形式",
                "系统含义"
            ]
        )

        st.dataframe(
            state_df,
            hide_index=True,
            use_container_width=True
        )

        # -------------------------
        # 趋势变化规则
        # -------------------------
        st.subheader(
            "⚙️ 趋势变化规则"
        )

        delta_rules = pd.DataFrame(
            [
                [
                    "🚀 加速强化",
                    "3日ΔScore ≥ +12 且加速度 ≥ +4",
                    "趋势和趋势速度同时增强"
                ],
                [
                    "📈 趋势增强",
                    "3日ΔScore ≥ +6",
                    "正反馈增强"
                ],
                [
                    "↗ 温和改善",
                    "3日ΔScore ≥ +2",
                    "状态边际改善"
                ],
                [
                    "➡️ 趋势平稳",
                    "-2 < 3日ΔScore < +2",
                    "变化不明显"
                ],
                [
                    "↘ 小幅转弱",
                    "3日ΔScore ≤ -2",
                    "出现边际走弱"
                ],
                [
                    "📉 趋势减速",
                    "3日ΔScore ≤ -6",
                    "原趋势明显削弱"
                ],
                [
                    "🔻 加速转弱",
                    "3日ΔScore ≤ -12 且加速度 ≤ -4",
                    "下降速度扩大"
                ],
            ],
            columns=[
                "趋势变化",
                "触发条件",
                "含义"
            ]
        )

        st.dataframe(
            delta_rules,
            hide_index=True,
            use_container_width=True
        )

        st.warning(
            "说明：技术分、趋势变化和5/20/60日情景"
            "属于量化研究模型输出，不等于确定性涨跌预测。"
            "建议重点验证状态转换和ΔScore方向"
            "是否与实际K线演化一致。"
        )

    except Exception as e:

        st.error(
            f"分析失败：{e}"
        )

        st.caption(
            "AKShare数据接口偶尔会临时波动。"
            "可以稍后重试，或检查股票代码是否为6位A股代码。"
        )