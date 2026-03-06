import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import google.generativeai as genai

st.set_page_config(page_title="Store Ops", layout="wide", initial_sidebar_state="expanded")

# --- CSS Injection ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #000000; color: #F3F2F1; }
    [data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #333333; }
    .copilot-title {
        background: linear-gradient(90deg, #2870EA 0%, #E362F8 50%, #FFB6B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem; font-weight: 700; margin-bottom: 0.5rem;
    }
    .sub-header { color: #939393; font-size: 1.2rem; margin-top: -10px; margin-bottom: 25px; }
    [data-testid="stMetricValue"] { color: #E362F8; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] { color: #8A929A; font-weight: 600; font-size: 1.05rem; padding: 12px 20px; border-radius: 6px 6px 0 0; }
    .stTabs [aria-selected="true"] { color: #F3F2F1 !important; border-bottom: 3px solid #2870EA !important; background-color: rgba(40, 112, 234, 0.15) !important; }
</style>
""", unsafe_allow_html=True)

# --- Data Generation ---
np.random.seed(42)
n = 5000
today = datetime.now()
dates = [today - timedelta(days=np.random.randint(0, 30)) for _ in range(n)]
vendors = ["Internal Engineering", "Salesforce", "Atlassian", "Independent", "SAP"]

df = pd.DataFrame({
    "Agent_ID": [f"AGNT-{i:05d}" for i in range(n)],
    "Name": [f"Copilot_Ext_{i}" for i in range(n)],
    "Vendor": np.random.choice(vendors, n, p=[0.4, 0.15, 0.15, 0.2, 0.1]),
    "Status": np.random.choice(["Published", "In Review", "Flagged"], n, p=[0.7, 0.2, 0.1]),
    "Publish_Date": dates,
    "Time_to_Publish": np.random.uniform(1, 14, n),
    "Pass_Rate": np.random.uniform(50, 99, n),
    "Compliance_Rate": np.random.uniform(80, 100, n),
    "Hallucination_Acc": np.random.uniform(75, 99, n),
    "Rejection_Reason": np.random.choice(["None", "Data Privacy", "Auth Timeout", "Manifest Bug", "Token Limit"], n, p=[0.7, 0.1, 0.08, 0.07, 0.05]),
    "Action_Latency": np.random.normal(150, 40, n).clip(50, 800),
    "Installs": np.random.exponential(1000, n).astype(int),
    "Bias_Score": np.random.normal(92, 5, n).clip(0, 100),
    "Malware_Score": np.random.normal(99, 1, n).clip(0, 100),
    "Inclusivity_Score": np.random.normal(88, 8, n).clip(0, 100)
})

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h3 style='color: #2870EA;'>🔐 Session</h3>", unsafe_allow_html=True)
    current_role = st.selectbox("Role:", ["Admin (Global)"] + [f"Vendor: {v}" for v in vendors if v != "Internal Engineering"])

if current_role == "Admin (Global)":
    view_df = df
else:
    active_vendor = current_role.split(": ")[1]
    view_df = df[df['Vendor'] == active_vendor]

# --- Main UI ---
st.markdown('<div class="copilot-title">Store ops : Agent performance metrics</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Access: {current_role}</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Velocity", "Quality", "Dev Exp", "Marketplace", "💬 Support Copilot"])

with tab1:
    st.metric("Avg Time to Publish", f"{view_df['Time_to_Publish'].mean():.1f} Days")
    daily_vol = view_df.groupby(view_df['Publish_Date'].dt.date).size().reset_index(name='Volume')
    fig = px.area(daily_vol, x="Publish_Date", y="Volume", template="plotly_dark", color_discrete_sequence=['#2870EA'])
    st.plotly_chart(fig, width="stretch")

# --- RAG Copilot Logic ---
with tab5:
    st.subheader("🤖 Developer Support Copilot")
    if "gemini_key" in st.secrets:
        genai.configure(api_key=st.secrets["gemini_key"])
        if "chat_session" not in st.session_state:
            model = genai.GenerativeModel('gemini-2.5-flash', system_instruction="Analyze agent data for rejections.")
            st.session_state.chat_session = model.start_chat(history=[])
            st.session_state.messages = [{"role": "assistant", "content": "Connected to Secrets. How can I help?"}]
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about your agents..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            failing = view_df[view_df['Status'] != 'Published'].head(5).to_csv(index=False)
            response = st.session_state.chat_session.send_message(f"Context:\n{failing}\n\nQuestion: {prompt}")
            with st.chat_message("assistant"): st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
    else:
        st.error("Missing 'gemini_key' in Streamlit Secrets.")

st.divider()
st.subheader("🔍 Inspector")
search = st.text_input("Search Agent:")
if search:
    res = view_df[view_df["Name"].str.contains(search, case=False)]
    if not res.empty: st.json(res.iloc[0].to_dict())
