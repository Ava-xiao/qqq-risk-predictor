import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# 设置页面配置
st.set_page_config(page_title="QQQ 风险预测", layout="centered")

st.title("📊 QQQ 下周大回撤风险预测")
st.markdown("基于逻辑回归模型（召回率 92.3%，AUC 0.802）")

# 加载模型和特征（缓存，避免重复加载）
@st.cache_resource
def load_model():
    model = joblib.load('final_model.pkl')
    features = joblib.load('features.pkl')
    return model, features

@st.cache_data
def load_base():
    return joblib.load('max_drawdown_week_features.pkl')

try:
    model, features = load_model()
    base = load_base()
except FileNotFoundError as e:
    st.error(f"❌ 模型文件未找到: {e}\n请确保 final_model.pkl, features.pkl, max_drawdown_week_features.pkl 在当前目录。")
    st.stop()

# 提取基准特征值（用于滑块默认值）
base_vals = base.iloc[0]

# 创建两列布局
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔧 调整特征值")
    vix = st.slider(
        "VIX 趋势 (VIX_Trend)",
        min_value=-2.0, max_value=30.0, value=float(base_vals['VIX_Trend']),
        step=0.5, help="VIX 与20周均线的差值，反映市场恐慌程度"
    )
    sentiment = st.slider(
        "情绪分数 (Sentiment_Level)",
        min_value=0.0, max_value=1.0, value=float(base_vals['Sentiment_Level']),
        step=0.01, help="Reddit 情绪分数，0=悲观，1=乐观"
    )
    ma_bias = st.slider(
        "价格偏离均线 (MA_Bias)",
        min_value=-0.3, max_value=0.3, value=float(base_vals['MA_Bias']),
        step=0.01, help="当前价格与20周均线的相对偏差"
    )
    volume_spike = st.slider(
        "讨论量暴增 (Volume_Spike)",
        min_value=0.5, max_value=3.0, value=float(base_vals['Volume_Spike']),
        step=0.05, help="本周帖子数 / 过去4周平均帖子数"
    )
    # 高级选项
    with st.expander("📈 更多特征"):
        yield_spread = st.slider(
            "债券利差 (yield_spread)",
            min_value=-1.0, max_value=1.0, value=float(base_vals['yield_spread']),
            step=0.05
        )
        rsi = st.slider(
            "RSI",
            min_value=0, max_value=100, value=int(base_vals['RSI']),
            step=1
        )
        # 其他特征保持不变（取基准值）
        # 注意：为了简化，只让用户调整以上特征，其余保持基准

    # 构造特征向量
    X = base.copy()
    X['VIX_Trend'] = vix
    X['Sentiment_Level'] = sentiment
    X['MA_Bias'] = ma_bias
    X['Volume_Spike'] = volume_spike
    X['yield_spread'] = yield_spread
    X['RSI'] = rsi

# 预测概率
prob = model.predict_proba(X)[0][1]
threshold = 0.41
risk = "高风险 (建议避险)" if prob >= threshold else "低风险 (正常持仓)"
color = "red" if prob >= threshold else "green"

with col2:
    st.subheader("📈 预测结果")
    # 使用 Plotly 仪表盘
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={'text': "风险概率 (%)"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, threshold*100], 'color': "lightgreen"},
                {'range': [threshold*100, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold*100
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"<h3 style='color:{color};'>风险等级: {risk}</h3>", unsafe_allow_html=True)
    st.caption(f"决策阈值: {threshold:.0%} (成本法优化)")

# 显示当前特征值表
with st.expander("📋 当前使用的全部特征值"):
    df_show = pd.DataFrame(X.iloc[0].to_dict(), index=["值"]).T
    st.dataframe(df_show)

# 模型说明
st.markdown("---")
st.markdown("""
### 💡 模型简介
- **训练期间**: 2023-01-06 至 2026-01-02 (157周)
- **验证方法**: 滚动窗口 (训练100周，预测57次)
- **核心指标**: 召回率 92.3% | AUC 0.802 | 最优阈值 0.41
- **最强特征**: VIX_Trend (系数 +0.66)，情绪分数系数 +0.12
""")