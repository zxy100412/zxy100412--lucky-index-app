# -*- coding: utf-8 -*-
"""
幸运日预测网页应用
用户输入日期，输出"幸运日"或"不幸日"
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ephem
from dateutil.easter import easter

# ========== 熵权法固定权重 ==========
OMEGA_WEEKDAY = 0.0392382881   # 星期权重
OMEGA_DAY = 0.0964037618       # 日期权重
OMEGA_MOON = 0.1747364794      # 月相权重
OMEGA_HOLIDAY = 0.6896214708   # 节假日权重

# ========== 加载概率表 ==========
@st.cache_data
def load_prob_tables():
    # 直接读取同目录下的Excel文件（部署后和py文件在同一文件夹）
    weekday_df = pd.read_excel("星期好日子概率.xlsx")
    day_df = pd.read_excel("日期好日子概率.xlsx")
    moon_df = pd.read_excel("月相好日子概率.xlsx")
    holiday_df = pd.read_excel("节假日好日子概率.xlsx")
    return weekday_df, day_df, moon_df, holiday_df# ========== 辅助函数 ==========
def get_moon_phase(date):
    """根据日期计算月相"""
    obs = ephem.Observer()
    obs.date = date.strftime('%Y/%m/%d')
    moon = ephem.Moon(obs)
    phase = moon.phase / 100
    
    if 0.48 <= phase <= 0.52:
        return '满月'
    elif 0.23 <= phase < 0.27:
        return '上弦月'
    elif 0.73 <= phase < 0.77:
        return '下弦月'
    elif phase < 0.1 or phase > 0.9:
        return '残月'
    else:
        return '其他'

def get_thanksgiving(year):
    """计算感恩节日期（11月第四个周四）"""
    nov1 = datetime(year, 11, 1)
    days_until_thursday = (3 - nov1.weekday()) % 7
    first_thursday = nov1 + timedelta(days=days_until_thursday)
    return (first_thursday + timedelta(weeks=3)).date()

def get_holiday_status(date):
    """判断节假日状态"""
    year = date.year
    date_only = date.date() if hasattr(date, 'date') else date
    
    holidays = [
        datetime(year, 12, 25).date(),  # 圣诞节
        datetime(year, 1, 1).date(),     # 元旦
        easter(year),                     # 复活节
        get_thanksgiving(year)            # 感恩节
    ]
    
    if date_only in holidays:
        return '节假日当天'
    for h in holidays:
        if date_only == h - timedelta(days=1):
            return '节假日前一天'
    for h in holidays:
        if date_only == h + timedelta(days=1):
            return '节假日后一天'
    return '其他'

def prob_to_logodds(p):
    """概率转换为log-odds"""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1 - p))

def logodds_to_prob(logodds):
    """log-odds转换为概率"""
    return np.exp(logodds) / (1 + np.exp(logodds))

# ========== 主预测函数 ==========
def predict_good_day(date, weekday_df, day_df, moon_df, holiday_df):
    """预测指定日期是否为幸运日"""
    
    # 1. 获取各维度的类别
    weekday = date.weekday()  # 0-6
    weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    weekday_name = weekday_names[weekday]
    
    day = date.day  # 1-31
    day_name = f'{day}号'
    
    moon_phase = get_moon_phase(date)
    
    holiday_status = get_holiday_status(date)
    
    # 2. 查表获取概率
    p_weekday = weekday_df[weekday_df['星期'] == weekday_name]['好日子概率'].values[0]
    p_day = day_df[day_df['日期'] == day_name]['好日子概率'].values[0]
    p_moon = moon_df[moon_df['月相'] == moon_phase]['好日子概率'].values[0]
    p_holiday = holiday_df[holiday_df['节假日状态'] == holiday_status]['好日子概率'].values[0]
    
    # 3. 转换为log-odds
    logodds_weekday = prob_to_logodds(p_weekday)
    logodds_day = prob_to_logodds(p_day)
    logodds_moon = prob_to_logodds(p_moon)
    logodds_holiday = prob_to_logodds(p_holiday)
    
    # 4. 加权求和
    logodds_all = (OMEGA_WEEKDAY * logodds_weekday + 
                   OMEGA_DAY * logodds_day + 
                   OMEGA_MOON * logodds_moon + 
                   OMEGA_HOLIDAY * logodds_holiday)
    
    # 5. 转换回概率
    p_all = logodds_to_prob(logodds_all)
    
    # 6. 判断结果
    is_good_day = p_all >= 0.5
    
    return {
        'date': date,
        'weekday': weekday_name,
        'day': day_name,
        'moon_phase': moon_phase,
        'holiday_status': holiday_status,
        'p_weekday': p_weekday,
        'p_day': p_day,
        'p_moon': p_moon,
        'p_holiday': p_holiday,
        'logodds_all': logodds_all,
        'p_all': p_all,
        'is_good_day': is_good_day
    }

# ========== Streamlit 界面 ==========
st.set_page_config(page_title="幸运日预测器", page_icon="🔮", layout="centered")

# 自定义CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .result-good {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .result-bad {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🔮 幸运日预测器</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#666;">基于2000-2025年历史数据，预测任意日期的运气指数</p>', unsafe_allow_html=True)

# 加载数据
weekday_df, day_df, moon_df, holiday_df = load_prob_tables()

# 日期输入（限制范围：2026-2050）
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    from datetime import date
    input_date = st.date_input("📅 请选择日期", 
                               value=datetime(2027, 3, 13),
                               min_value=date(2026, 1, 1),
                               max_value=date(2050, 12, 31))

# 预测按钮
if st.button("🔮 开始预测", type="primary", use_container_width=True):
    result = predict_good_day(input_date, weekday_df, day_df, moon_df, holiday_df)
    
    st.markdown("---")
    
    # 显示结果
    if result['is_good_day']:
        st.markdown(f'<div class="result-good">✨ {input_date.strftime("%Y年%m月%d日")} 是幸运日！</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-bad">⚠️ {input_date.strftime("%Y年%m月%d日")} 是不幸日</div>', unsafe_allow_html=True)
    
    # 显示详细信息
    st.markdown("### 📊 详细分析")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="info-card">
        <b>📆 日期属性</b><br>
        星期: {result['weekday']}<br>
        日期: {result['day']}<br>
        月相: {result['moon_phase']}<br>
        节假日: {result['holiday_status']}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-card">
        <b>📈 各维度概率</b><br>
        星期概率: {result['p_weekday']:.2%}<br>
        日期概率: {result['p_day']:.2%}<br>
        月相概率: {result['p_moon']:.2%}<br>
        节假日概率: {result['p_holiday']:.2%}
        </div>
        """, unsafe_allow_html=True)
    
    # 显示最终概率
    st.markdown(f"""
    <div style="text-align:center; padding:1rem; background:#e8f4fd; border-radius:0.5rem; margin-top:1rem;">
    <b>综合幸运概率: {result['p_all']:.2%}</b>
    <br><small>判断标准: ≥50% 为幸运日</small>
    </div>
    """, unsafe_allow_html=True)
    
    # 显示权重信息
    with st.expander("📐 查看权重信息"):
        st.write(f"星期权重 (ω₁): {OMEGA_WEEKDAY:.4f} ({OMEGA_WEEKDAY*100:.2f}%)")
        st.write(f"日期权重 (ω₂): {OMEGA_DAY:.4f} ({OMEGA_DAY*100:.2f}%)")
        st.write(f"月相权重 (ω₃): {OMEGA_MOON:.4f} ({OMEGA_MOON*100:.2f}%)")
        st.write(f"节假日权重 (ω₄): {OMEGA_HOLIDAY:.4f} ({OMEGA_HOLIDAY*100:.2f}%)")

st.markdown("---")
st.markdown('<p style="text-align:center; color:#999; font-size:0.8rem;">基于熵权法和Log-Odds模型构建 | 数据来源: 2000-2025年历史数据</p>', unsafe_allow_html=True)
