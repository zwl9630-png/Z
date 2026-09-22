import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import time
from datetime import datetime, timedelta

st.set_page_config(
    page_title="A股工程控制论 V2.3",
    page_icon="📈",
    layout="centered"
)

st.title("📈 A股工程控制论")
st.caption("V2.3 · 一周验证版")

# =========================================================
# 工具
# =========================================================

def clamp(x, a, b):
    return max(a, min(b, x))

def num(x):
    try:
        return float(
            str(x).replace(",", "").replace("%", "")
        )
    except:
        return np.nan

def exchange_code(code):
    if code.startswith(("5", "6", "9")):
        return "SH" + code
    return "SZ" + code

def fmt_money(v):
    try:
        v = float(v)
        if abs(v) >= 1e8:
            return f"{v/1e8:.2f}亿"
        if abs(v) >= 1e4:
            return f"{v/1e4:.2f}万"
        return f"{v:.2f}"
    except:
        return "--"

# =========================================================
# 股票行情
# =========================================================

@st.cache_data(ttl=600, show_spinner=False)
def get_stock(code):
    end = datetime.now().strftime("%Y%m%d")
    start = (
        datetime.now() - timedelta(days=800)
    ).strftime("%Y%m%d")

    try:
        d = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq"
        )

        d = d.rename(columns={
            "日期":"date",
            "开盘":"open",
            "最高":"high",
            "最低":"low",
            "收盘":"close",
            "成交量":"volume"
        })

        cols = [
            "date","open","high",
            "low","close","volume"
        ]

        d = d[cols].copy()
        d["date"] = pd.to_datetime(d["date"])

        for c in cols[1:]:
            d[c] = pd.to_numeric(
                d[c], errors="coerce"
            )

        d = d.dropna().sort_values("date")

        if len(d) < 70:
            raise ValueError("历史数据不足")

        return d.reset_index(drop=True)

    except:
        prefix = (
            "sh"
            if code.startswith(("5","6","9"))
            else "sz"
        )

        d = ak.stock_zh_a_daily(
            symbol=prefix + code,
            adjust="qfq"
        )

        d = d.reset_index()

        d = d.rename(columns={
            "date":"date",
            "open":"open",
            "high":"high",
            "low":"low",
            "close":"close",
            "volume":"volume"
        })

        d["date"] = pd.to_datetime(d["date"])

        return d[
            ["date","open","high",
             "low","close","volume"]
        ].dropna()

# =========================================================
# 技术指标
# =========================================================

def technical(d):
    d = d.copy()

    for n in [5,10,20,60]:
        d[f"MA{n}"] = (
            d.close.rolling(n).mean()
        )

    ema12 = d.close.ewm(
        span=12, adjust=False
    ).mean()

    ema26 = d.close.ewm(
        span=26, adjust=False
    ).mean()

    d["DIF"] = ema12 - ema26
    d["DEA"] = d.DIF.ewm(
        span=9, adjust=False
    ).mean()

    change = d.close.diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = (-change.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - 100/(1+rs)

    d["RET20"] = d.close.pct_change(20)
    d["RET60"] = d.close.pct_change(60)
    d["VOL20"] = d.volume.rolling(20).mean()

    pc = d.close.shift()

    tr = pd.concat([
        d.high-d.low,
        (d.high-pc).abs(),
        (d.low-pc).abs()
    ], axis=1).max(axis=1)

    d["ATR"] = tr.rolling(14).mean()
    d["ATR_PCT"] = d.ATR/d.close

    x = d.iloc[-1]
    s = 50

    s += 10 if x.close > x.MA20 else -10
    s += 12 if x.MA20 > x.MA60 else -12
    s += 10 if x.DIF > x.DEA else -10
    s += 8 if x.RET20 > 0 else -8
    s += 8 if x.RET60 > 0 else -8

    if x.volume > x.VOL20:
        s += 5

    if 45 <= x.RSI <= 70:
        s += 5
    elif x.RSI > 80:
        s -= 5

    s = int(clamp(s,0,100))

    if s >= 75:
        state = "趋势强化"
    elif s >= 62:
        state = "趋势形成"
    elif s >= 45:
        state = "震荡观察"
    elif s >= 30:
        state = "趋势减弱"
    else:
        state = "风险释放"

    atr = (
        float(x.ATR_PCT)
        if pd.notna(x.ATR_PCT)
        else .03
    )

    risk = (
        "高" if atr > .06
        else "中" if atr > .04
        else "低"
    )

    return d, s, state, risk, atr

# =========================================================
# 🌡️ 市场温度
# =========================================================

def temp_label(t):
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

@st.cache_data(ttl=300, show_spinner=False)
def market_temp():
    try:
        d = ak.stock_zh_a_spot_em()

        pct = pd.to_numeric(
            d["涨跌幅"], errors="coerce"
        ).dropna()

        if len(pct) < 500:
            raise ValueError("样本不足")

        up = int((pct>0).sum())
        down = int((pct<0).sum())
        total = len(pct)

        breadth = (
            (up-down)/total*100
        )

        median = float(pct.median())
        mean = float(pct.mean())

        strong = (
            ((pct>=5).sum()-(pct<=-5).sum())
            / total * 100
        )

        limit_up = int((pct>=9.5).sum())
        limit_down = int((pct<=-9.5).sum())

        extreme = clamp(
            (limit_up-limit_down)*2,
            -100,100
        )

        amount = pd.to_numeric(
            d["成交额"], errors="coerce"
        ).fillna(0).sum()

        # 成交额只作为辅助活跃度
        trillion = amount/1e12
        activity = clamp(
            (trillion-1.0)*45,
            -50,100
        )

        t = (
            breadth*.30 +
            clamp(median*25,-100,100)*.25 +
            clamp(mean*15,-100,100)*.10 +
            extreme*.20 +
            clamp(strong*8,-100,100)*.10 +
            activity*.05
        )

        t = int(round(clamp(t,-100,100)))

        return {
            "temp":t,
            "label":temp_label(t),
            "up":up,
            "down":down,
            "limit_up":limit_up,
            "limit_down":limit_down,
            "median":median,
            "amount":amount,
            "status":"正常"
        }

    except Exception as e:
        return {
            "temp":None,
            "label":"数据暂缺",
            "status":"降级",
            "error":str(e)
        }

def show_temp():
    m = market_temp()

    st.subheader("🌡️ 市场温度计")

    if m["temp"] is None:
        st.metric("市场温度","--℃")
        st.warning("市场全景数据暂时不可用")
        return m

    a,b = st.columns(2)
    a.metric("市场温度",f"{m['temp']:+d}℃")
    b.metric("市场情绪",m["label"])

    st.progress(
        int((m["temp"]+100)/2)
    )

    a,b = st.columns(2)
    a.metric("上涨",m["up"])
    b.metric("下跌",m["down"])

    a,b = st.columns(2)
    a.metric("强涨近似",m["limit_up"])
    b.metric("强跌近似",m["limit_down"])

    st.caption(
        f"个股中位涨跌：{m['median']:+.2f}% ｜ "
        f"全A成交额：{m['amount']/1e8:,.0f}亿"
    )

    return m

# =========================================================
# 基本资料 + PE/PB
# =========================================================

@st.cache_data(ttl=1800, show_spinner=False)
def stock_snapshot(code):
    out = {}

    try:
        info = ak.stock_individual_info_em(
            symbol=code
        )

        if "item" in info and "value" in info:
            out.update(
                dict(zip(info.item,info.value))
            )
    except:
        pass

    # 雪球估值接口备用
    try:
        d = ak.stock_individual_spot_xq(
            symbol=exchange_code(code)
        )

        if not d.empty:
            # 兼容两种返回格式
            if "item" in d.columns and "value" in d.columns:
                out.update(
                    dict(zip(d["item"],d["value"]))
                )
            elif len(d) == 1:
                out.update(d.iloc[0].to_dict())
    except:
        pass

    return out

def pick(data, names):
    for n in names:
        if n in data:
            v = data[n]
            if pd.notna(v):
                return v
    return None

# =========================================================
# 财务控制器
# =========================================================

@st.cache_data(ttl=21600, show_spinner=False)
def financial_data(code):
    result = {
        "report":"--",
        "roe":None,
        "gross":None,
        "net_margin":None,
        "debt":None,
        "revenue":None,
        "revenue_yoy":None,
        "profit":None,
        "profit_yoy":None
    }

    symbol = exchange_code(code)

    # 财务指标
    try:
        f = ak.stock_financial_analysis_indicator(
            symbol=code
        )

        if f is not None and not f.empty:
            f = f.copy()

            # 有些版本日期在 index
            if not isinstance(f.index, pd.RangeIndex):
                f = f.reset_index()

            row = f.iloc[0]

            result["report"] = str(
                pick(
                    row.to_dict(),
                    ["日期","报告期","index"]
                ) or "--"
            )

            result["roe"] = pick(
                row.to_dict(),
                [
                    "加权净资产收益率(%)",
                    "摊薄净资产收益率(%)",
                    "净资产收益率(%)"
                ]
            )

            result["gross"] = pick(
                row.to_dict(),
                ["销售毛利率(%)","毛利率(%)"]
            )

            result["net_margin"] = pick(
                row.to_dict(),
                ["销售净利率(%)","净利率(%)"]
            )

            result["debt"] = pick(
                row.to_dict(),
                ["资产负债率(%)"]
            )
    except:
        pass

    # 利润表
    try:
        p = ak.stock_profit_sheet_by_report_em(
            symbol=symbol
        )

        if p is not None and not p.empty:
            row = p.iloc[0].to_dict()

            result["report"] = str(
                pick(
                    row,
                    ["REPORT_DATE","报告日期","报告期"]
                ) or result["report"]
            )[:10]

            result["revenue"] = pick(
                row,
                [
                    "TOTAL_OPERATE_INCOME",
                    "OPERATE_INCOME",
                    "营业总收入",
                    "营业收入"
                ]
            )

            result["revenue_yoy"] = pick(
                row,
                [
                    "TOTAL_OPERATE_INCOME_YOY",
                    "OPERATE_INCOME_YOY",
                    "营业收入同比"
                ]
            )

            result["profit"] = pick(
                row,
                [
                    "PARENT_NETPROFIT",
                    "NETPROFIT",
                    "归属于母公司股东的净利润",
                    "净利润"
                ]
            )

            result["profit_yoy"] = pick(
                row,
                [
                    "PARENT_NETPROFIT_YOY",
                    "NETPROFIT_YOY",
                    "净利润同比"
                ]
            )
    except:
        pass

    return result

def fundamental_score(f):
    points = []
    reasons = []

    roe = num(f["roe"])
    gross = num(f["gross"])
    debt = num(f["debt"])
    rg = num(f["revenue_yoy"])
    pg = num(f["profit_yoy"])

    if pd.notna(roe):
        points.append(
            clamp(50+(roe-10)*2,0,100)
        )
        reasons.append(
            f"ROE {roe:.1f}%"
        )

    if pd.notna(rg):
        points.append(
            clamp(50+rg,0,100)
        )
        reasons.append(
            f"营收同比 {rg:+.1f}%"
        )

    if pd.notna(pg):
        points.append(
            clamp(50+pg*.6,0,100)
        )
        reasons.append(
            f"净利润同比 {pg:+.1f}%"
        )

    if pd.notna(gross):
        points.append(
            clamp(40+gross,0,100)
        )

    if pd.notna(debt):
        points.append(
            clamp(100-debt,0,100)
        )

    if not points:
        return None, reasons

    return int(np.mean(points)), reasons

# =========================================================
# 行业景气度
# =========================================================

@st.cache_data(ttl=1800, show_spinner=False)
def industry_state(industry):
    if not industry or industry == "--":
        return None

    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (
            datetime.now()-timedelta(days=150)
        ).strftime("%Y%m%d")

        d = ak.stock_board_industry_hist_em(
            symbol=industry,
            start_date=start,
            end_date=end,
            period="日k",
            adjust=""
        )

        close = pd.to_numeric(
            d["收盘"], errors="coerce"
        ).dropna()

        if len(close) < 60:
            return None

        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]

        r20 = close.iloc[-1]/close.iloc[-21]-1

        score = 50
        score += 15 if close.iloc[-1]>ma20 else -15
        score += 15 if ma20>ma60 else -15
        score += 15 if r20>0 else -15

        score = int(clamp(score,0,100))

        if score >= 70:
            label = "🔥 高景气/偏强"
        elif score >= 55:
            label = "🟢 景气改善"
        elif score >= 40:
            label = "⚪ 中性"
        else:
            label = "🔵 景气偏弱"

        return {
            "score":score,
            "label":label,
            "ret20":r20*100
        }

    except:
        return None

# =========================================================
# 资金流
# =========================================================

@st.cache_data(ttl=900, show_spinner=False)
def fund_flow(code):
    try:
        market = (
            "sh"
            if code.startswith(("5","6","9"))
            else "sz"
        )

        d = ak.stock_individual_fund_flow(
            stock=code,
            market=market
        )

        if d is None or d.empty:
            return None

        row = d.iloc[-1].to_dict()

        main = pick(
            row,
            [
                "主力净流入-净额",
                "主力净流入净额"
            ]
        )

        ratio = pick(
            row,
            [
                "主力净流入-净占比",
                "主力净流入净占比"
            ]
        )

        return {
            "main":main,
            "ratio":ratio
        }

    except:
        return None

# =========================================================
# 公告：订单/利好/利空
# =========================================================

POS_WORDS = [
    "中标","合同","订单","回购",
    "增持","预增","扭亏","重大项目",
    "战略合作","获批"
]

NEG_WORDS = [
    "减持","亏损","处罚","立案",
    "风险提示","诉讼","终止",
    "下修","预亏","退市"
]

@st.cache_data(ttl=1800, show_spinner=False)
def notices(code):
    try:
        d = ak.stock_individual_notice_report(
            symbol=code
        )

        if d is None or d.empty:
            return []

        rows = []

        for _,r in d.head(30).iterrows():
            title = str(
                r.get("公告标题","")
            )

            date = str(
                r.get("公告日期","")
            )[:10]

            pos = any(
                w in title for w in POS_WORDS
            )

            neg = any(
                w in title for w in NEG_WORDS
            )

            if pos or neg:
                rows.append({
                    "date":date,
                    "title":title,
                    "type":
                        "🟢 潜在正反馈"
                        if pos and not neg
                        else "🔴 潜在风险"
                        if neg and not pos
                        else "🟡 需人工判断"
                })

        return rows[:8]

    except:
        return []

# =========================================================
# 综合评分
# 缺失模块自动重新分配权重
# =========================================================

def composite(tech, fund, industry, temp):
    items = [
        (tech,.45)
    ]

    if fund is not None:
        items.append((fund,.25))

    if industry is not None:
        items.append((industry,.15))

    if temp is not None:
        env = (temp+100)/2

        # 高温过热惩罚
        if temp > 80:
            env -= (temp-80)*1.5

        items.append(
            (clamp(env,0,100),.15)
        )

    total_w = sum(w for _,w in items)

    return int(
        sum(v*w for v,w in items)
        / total_w
    )

# =========================================================
# 预测
# =========================================================

def forecast(close,score,atr,r20,r60,days):
    strength = (score-50)/50

    momentum = (
        clamp(r20,-.25,.25)*.6 +
        clamp(r60,-.40,.40)*.4
    )

    scale = np.sqrt(days/5)

    expected = (
        strength*atr*scale*.6 +
        momentum*min(days/60,1)*.35
    )

    center = close*(1+expected)

    width = max(
        atr*scale*1.2,
        .025
    )

    low = center*(1-width)
    high = center*(1+width)

    if score >= 60:
        direction = "偏强"
        prob = clamp(
            50+(score-50)*.7,
            50,80
        )
    elif score <= 40:
        direction = "偏弱"
        prob = clamp(
            50+(50-score)*.7,
            50,80
        )
    else:
        direction = "震荡"
        prob = 55

    return (
        direction,
        int(prob),
        low,
        center,
        high
    )

# =========================================================
# 个股分析
# =========================================================

def analyse(code):
    d = get_stock(code)
    d,tech,state,risk,atr = technical(d)

    snap = stock_snapshot(code)
    fin = financial_data(code)
    fscore, freasons = fundamental_score(fin)

    industry = pick(
        snap,
        ["行业","所属行业"]
    ) or "--"

    ind = industry_state(str(industry))

    temp = market_temp()
    t = temp["temp"]

    iscore = (
        ind["score"]
        if ind else None
    )

    score = composite(
        tech,
        fscore,
        iscore,
        t
    )

    return {
        "df":d,
        "x":d.iloc[-1],
        "tech":tech,
        "state":state,
        "risk":risk,
        "atr":atr,
        "snap":snap,
        "fin":fin,
        "fund_score":fscore,
        "industry":industry,
        "industry_data":ind,
        "temperature":temp,
        "score":score,
        "flow":fund_flow(code),
        "notices":notices(code)
    }

# =========================================================
# 个股页面
# =========================================================

def stock_page():
    st.subheader("🔎 个股控制台")

    code = st.text_input(
        "股票代码",
        placeholder="例如 000977",
        max_chars=6
    ).strip()

    if not st.button(
        "开始分析",
        type="primary",
        use_container_width=True
    ):
        return

    if len(code)!=6 or not code.isdigit():
        st.error("请输入6位股票代码")
        return

    try:
        with st.spinner("正在运行控制系统..."):
            r = analyse(code)

        x = r["x"]
        snap = r["snap"]

        name = (
            pick(snap,["股票简称","名称"])
            or code
        )

        st.header(f"{name} · {code}")
        st.caption(
            f"数据日期：{x['date'].strftime('%Y-%m-%d')}"
        )

        a,b,c = st.columns(3)
        a.metric("综合控制分",r["score"])
        b.metric("技术状态",r["state"])
        c.metric("风险",r["risk"])

        # 温度
        st.subheader("🌡️ 市场环境")
        t = r["temperature"]

        a,b = st.columns(2)

        if t["temp"] is None:
            a.metric("市场温度","--℃")
            b.metric("市场情绪","数据暂缺")
        else:
            a.metric(
                "市场温度",
                f"{t['temp']:+d}℃"
            )
            b.metric(
                "市场情绪",
                t["label"]
            )

        # 行业
        st.subheader("🏭 行业景气度")

        st.write(
            f"**所属行业：{r['industry']}**"
        )

        if r["industry_data"]:
            ind = r["industry_data"]

            a,b = st.columns(2)
            a.metric(
                "行业景气分",
                ind["score"]
            )
            b.metric(
                "行业状态",
                ind["label"]
            )

            st.caption(
                f"行业20日走势："
                f"{ind['ret20']:+.2f}%"
            )
        else:
            st.info(
                "当前行业行情接口未取得有效数据"
            )

        # 基本面
        st.subheader("🏢 基本面控制器")

        f = r["fin"]

        if r["fund_score"] is not None:
            st.metric(
                "基本面评分",
                r["fund_score"]
            )
        else:
            st.metric(
                "基本面评分",
                "--"
            )

        st.caption(
            f"财务报告期：{f['report']}"
        )

        a,b = st.columns(2)
        a.metric(
            "ROE",
            "--" if f["roe"] is None
            else f"{num(f['roe']):.2f}%"
        )
        b.metric(
            "毛利率",
            "--" if f["gross"] is None
            else f"{num(f['gross']):.2f}%"
        )

        a,b = st.columns(2)
        a.metric(
            "净利率",
            "--" if f["net_margin"] is None
            else f"{num(f['net_margin']):.2f}%"
        )
        b.metric(
            "资产负债率",
            "--" if f["debt"] is None
            else f"{num(f['debt']):.2f}%"
        )

        st.write(
            "**营业收入：**",
            fmt_money(f["revenue"])
        )

        st.write(
            "**营收同比：**",
            "--"
            if f["revenue_yoy"] is None
            else f"{num(f['revenue_yoy']):+.2f}%"
        )

        st.write(
            "**归母净利润：**",
            fmt_money(f["profit"])
        )

        st.write(
            "**净利润同比：**",
            "--"
            if f["profit_yoy"] is None
            else f"{num(f['profit_yoy']):+.2f}%"
        )

        # 估值
        st.subheader("💰 估值")

        pe = pick(
            snap,
            [
                "市盈率(TTM)",
                "市盈率-动态",
                "市盈率(动)"
            ]
        )

        pb = pick(
            snap,
            ["市净率"]
        )

        cap = pick(
            snap,
            ["总市值"]
        )

        a,b = st.columns(2)
        a.metric(
            "PE",
            "--" if pe is None else pe
        )
        b.metric(
            "PB",
            "--" if pb is None else pb
        )

        st.write(
            "**总市值：**",
            fmt_money(cap)
        )

        # 资金
        st.subheader("💵 资金/量价")

        flow = r["flow"]

        if flow:
            a,b = st.columns(2)
            a.metric(
                "主力净流入",
                fmt_money(flow["main"])
            )
            b.metric(
                "主力净占比",
                "--"
                if flow["ratio"] is None
                else str(flow["ratio"])
            )
        else:
            st.info(
                "本次未取得可靠资金流数据"
            )

        # 订单/利好利空
        st.subheader("📦 订单 · 催化 · 风险")

        ns = r["notices"]

        if not ns:
            st.info(
                "近期公告中未识别到可靠的"
                "订单/重大催化/风险关键词。"
                "这不代表公司没有相关事项。"
            )
        else:
            for n in ns:
                with st.container(border=True):
                    st.write(
                        f"**{n['type']}**"
                    )
                    st.write(n["title"])
                    st.caption(
                        f"{n['date']} ｜ "
                        "来源：公开公司公告"
                    )

        st.caption(
            "⚠️ 当前公告分类属于关键词初筛，"
            "不会仅凭标题自动认定为真实利好或利空。"
        )

        # 技术
        st.subheader("📊 技术控制器")

        a,b = st.columns(2)
        a.metric("技术评分",r["tech"])
        b.metric("当前状态",r["state"])

        st.line_chart(
            r["df"][
                ["date","close","MA20","MA60"]
            ].tail(120).set_index("date")
        )

        # 预测
        st.subheader("🔮 5 / 20 / 60日概率情景")

        rows=[]

        for days in [5,20,60]:
            direction,prob,low,center,high = forecast(
                float(x.close),
                r["score"],
                r["atr"],
                float(x.RET20),
                float(x.RET60),
                days
            )

            rows.append({
                "周期":f"{days}日",
                "方向":direction,
                "概率":f"{prob}%",
                "中枢":round(center,2),
                "参考区间":
                    f"{low:.2f}～{high:.2f}"
            })

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True
        )

        support = r["df"].low.tail(20).min()
        resistance = r["df"].high.tail(20).max()

        a,b = st.columns(2)
        a.metric(
            "20日支撑",
            f"{support:.2f}"
        )
        b.metric(
            "20日压力",
            f"{resistance:.2f}"
        )

        st.caption(
            "概率与价格区间是模型情景，"
            "不是确定性目标价。"
        )

    except Exception as e:
        st.error("本次分析失败")
        with st.expander("错误详情"):
            st.code(str(e))

# =========================================================
# 温度页面
# =========================================================

def temperature_page():
    m = show_temp()

    st.subheader("市场情绪定义")

    table = pd.DataFrame([
        ["-100～-80℃","🥶 极度恐慌"],
        ["-80～-50℃","❄️ 恐慌"],
        ["-50～-20℃","🔵 偏冷"],
        ["-20～+20℃","⚪ 平衡"],
        ["+20～+50℃","🟡 回暖"],
        ["+50～+80℃","🔥 火热"],
        ["+80～+100℃","🌋 过热"],
    ],columns=["温度","市场情绪"])

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "高温并不等于一定上涨，"
        "低温也不等于立即反弹。"
    )

# =========================================================
# 首页
# =========================================================

def home():
    m = show_temp()

    st.divider()

    st.subheader("🧠 当前控制系统")

    st.write(
        "🌡️ 市场情绪 → 📈 市场趋势 → "
        "🏭 行业景气 → 📦 订单/催化 → "
        "🏢 基本面 → 💰 估值 → "
        "💵 资金 → 📊 技术 → "
        "⚠️ 风险 → 🔮 概率情景"
    )

    st.info(
        "V2.3 为一周验证版本。"
        "数据缺失项不按0分处理，"
        "而是退出对应评分并重新分配权重。"
    )

# =========================================================
# 导航
# =========================================================

page = st.selectbox(
    "功能",
    [
        "🏠 控制中心",
        "🔎 个股分析",
        "🌡️ 市场温度计",
        "⚙️ 模型说明"
    ]
)

st.divider()

if page == "🏠 控制中心":
    home()

elif page == "🔎 个股分析":
    stock_page()

elif page == "🌡️ 市场温度计":
    temperature_page()

else:
    st.subheader("⚙️ V2.3 模型")
    st.markdown("""
**当前主要控制器**

- 市场温度：-100℃ ～ +100℃
- 行业景气度
- 公司基本面
- PE / PB估值
- ROE / 毛利率 / 净利率
- 营收 / 净利润 / 同比变化
- 资产负债率
- 主力资金流
- 技术趋势
- 订单 / 公告 / 催化 / 风险
- 5 / 20 / 60日概率情景

**重要原则**

缺失数据不等于0。

公告标题只进行信息初筛，
不自动把关键词认定为真实利好或利空。

本版本先运行一周，
再根据真实结果校准各控制器权重。
""")

st.divider()
st.caption(
    "A股工程控制论 V2.3 · 一周验证版"
)