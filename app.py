import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="QQQ Risk Predictor", layout="centered", page_icon="📈")

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    .stSlider label {
        font-weight: 500;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: white;
    }
    .risk-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .risk-low {
        color: #2e7d32;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 QQQ Next-Week Large Drawdown Risk Predictor")
st.markdown("*Logistic Regression Model | Recall 92.3% | AUC 0.802*")

# Load model and features
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
    base_vals = base.iloc[0].to_dict()
except FileNotFoundError as e:
    st.error(f"❌ Model files not found: {e}\nEnsure final_model.pkl, features.pkl, max_drawdown_week_features.pkl are in the current directory.")
    st.stop()

# Define feature groups for better organization
tech_features = ['MA_Bias', 'ATR', 'RSI', 'Volume_Change']
sentiment_features = ['Sentiment_Level', 'Sentiment_Uncertainty', 'Volume_Spike']
macro_features = ['VIX_Trend', 'yield_spread']
interaction_features = ['Risk_Resonance', 'Price_Sentiment_Divergence']

# Sidebar for feature adjustment (or main column)
st.sidebar.header("⚙️ Feature Adjustments")
st.sidebar.markdown("Adjust the values below to see how risk probability changes. Default values are from the historical max-drawdown week (2025-04-04).")

# Create a dictionary to store user-adjusted values
adjusted = base_vals.copy()

# Technical indicators
with st.sidebar.expander("📉 Technical Indicators", expanded=True):
    adjusted['MA_Bias'] = st.slider(
        "MA_Bias (Price vs 20-week MA)",
        min_value=-0.3, max_value=0.3, value=float(base_vals['MA_Bias']), step=0.01,
        help="(Close - MA20)/MA20"
    )
    adjusted['ATR'] = st.slider(
        "ATR (Average True Range)",
        min_value=0.0, max_value=50.0, value=float(base_vals['ATR']), step=0.5,
        help="14-week average true range"
    )
    adjusted['RSI'] = st.slider(
        "RSI (Relative Strength Index)",
        min_value=0, max_value=100, value=int(base_vals['RSI']), step=1
    )
    adjusted['Volume_Change'] = st.slider(
        "Volume Change (%)",
        min_value=-0.5, max_value=1.0, value=float(base_vals['Volume_Change']), step=0.05,
        help="Weekly volume percentage change"
    )

with st.sidebar.expander("😊 Sentiment Indicators", expanded=True):
    adjusted['Sentiment_Level'] = st.slider(
        "Sentiment Level",
        min_value=0.0, max_value=1.0, value=float(base_vals['Sentiment_Level']), step=0.01,
        help="Average sentiment score (0=pessimistic, 1=optimistic)"
    )
    adjusted['Sentiment_Uncertainty'] = st.slider(
        "Sentiment Uncertainty",
        min_value=0.0, max_value=0.3, value=float(base_vals['Sentiment_Uncertainty']), step=0.01,
        help="Standard deviation of sentiment over 4 weeks"
    )
    adjusted['Volume_Spike'] = st.slider(
        "Volume Spike (Post count ratio)",
        min_value=0.5, max_value=3.0, value=float(base_vals['Volume_Spike']), step=0.05,
        help="Current week post count / 4-week average"
    )

with st.sidebar.expander("🏦 Macro & Interaction", expanded=True):
    adjusted['VIX_Trend'] = st.slider(
        "VIX Trend",
        min_value=-2.0, max_value=30.0, value=float(base_vals['VIX_Trend']), step=0.5,
        help="Current VIX - 20-week MA"
    )
    adjusted['yield_spread'] = st.slider(
        "Yield Spread (10Y-2Y)",
        min_value=-1.0, max_value=1.0, value=float(base_vals['yield_spread']), step=0.05
    )
    adjusted['Risk_Resonance'] = st.selectbox(
        "Risk Resonance",
        options=[0, 1],
        index=int(base_vals['Risk_Resonance']),
        help="1 if Sentiment_Uncertainty > median and VIX_Trend > 0"
    )
    adjusted['Price_Sentiment_Divergence'] = st.selectbox(
        "Price-Sentiment Divergence",
        options=[0, 1],
        index=int(base_vals['Price_Sentiment_Divergence']),
        help="1 if price direction differs from sentiment direction"
    )

# Construct feature vector in correct order
X = pd.DataFrame([ [adjusted[f] for f in features] ], columns=features)

# Predict
prob = model.predict_proba(X)[0][1]
threshold = 0.41
risk_text = "High Risk (Recommended to hedge)" if prob >= threshold else "Low Risk (Normal position)"
risk_class = "risk-high" if prob >= threshold else "risk-low"
color = "#d32f2f" if prob >= threshold else "#2e7d32"

# Main column: Display gauge and risk
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📈 Prediction Result")
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prob * 100,
        title = {'text': "Risk Probability (%)", 'font': {'size': 20}},
        delta = {'reference': threshold*100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold*100], 'color': '#c8e6c9'},
                {'range': [threshold*100, 100], 'color': '#ffcdd2'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold*100
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(t=50, b=20, l=20, r=20), paper_bgcolor="#f5f7fa", font=dict(color="black"))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"<h3 class='{risk_class}'>Risk Level: {risk_text}</h3>", unsafe_allow_html=True)
    st.caption(f"Decision threshold: {threshold:.0%} (cost-based optimization)")

with col2:
    st.subheader("🔍 Model Interpretation")
    st.markdown("""
    - **VIX Trend** is the strongest positive factor (coefficient +0.66)
    - **Sentiment Level** also increases risk (coefficient +0.12)
    - **MA_Bias** has negative coefficient (bull market effect)
    - **Risk Resonance** amplifies risk when sentiment uncertainty meets high VIX
    """)
    st.metric("Model Recall (Rolling Window)", "92.3%")
    st.metric("AUC", "0.802")
    st.metric("Optimal Threshold", "0.41")

# Show current feature values in a table
with st.expander("📋 Current Feature Vector (All 11 Features)"):
    df_show = pd.DataFrame([adjusted]).T
    df_show.columns = ["Value"]
    st.dataframe(df_show, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
### 💡 Model Summary
- **Training period**: 2023-01-06 to 2026-01-02 (157 weeks)
- **Validation**: Rolling window (100 weeks train, 1 week test, 57 predictions)
- **Performance**: Recall 92.3% | AUC 0.802 | F1 0.522
- **Features**: 11 features including technicals, sentiment, macro, and interactions
""")
