import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from google import genai

# --- 1. Page Config & CSS ---
st.set_page_config(page_title="Store Ops", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #000000; color: #F3F2F1; }
    [data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #333333; }
    .copilot-title {
        background: linear-gradient(90deg, #2870EA 0%, #E362F8 50%, #FFB6B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem;
    }
    .sub-header { color: #939393; font-size: 1.1rem; margin-top: -10px; margin-bottom: 25px; }
    [data-testid="stMetricValue"] { color: #E362F8; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- 2. Data Generation ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 5000
    vendors = ["Internal Engineering", "Salesforce", "Atlassian", "Independent", "SAP"]
    today = datetime.now()
    return pd.DataFrame({
        "Agent_ID": [f"AGNT-{i:05d}" for i in range(n)],
        "Name": [f"Copilot_Ext_{i}" for i in range(n)],
        "Vendor": np.random.choice(vendors, n, p=[0.4, 0.15, 0.15, 0.2, 0.1]),
        "Status": np.random.choice(["Published", "In Review", "Flagged"], n, p=[0.7, 0.2, 0.1]),
        "Publish_Date": [today - timedelta(days=np.random.randint(0, 30)) for _ in range(n)],
        "Time_to_Publish": np.random.uniform(1, 14, n),
        "Rejection_Reason": np.random.choice(["None", "Data Privacy", "Auth Timeout", "Manifest Bug"], n, p=[0.7, 0.1, 0.1, 0.1]),
        "Action_Latency": np.random.normal(150, 40, n).clip(50, 800),
        "Installs": np.random.exponential(1000, n).astype(int),
        "Bias_Score": np.random.normal(92, 5, n).clip(0, 100),
        "Malware_Score": np.random.normal(99, 1, n).clip(0, 100),
        "Inclusivity_Score": np.random.normal(88, 8, n).clip(0, 100)
    })

df = load_data()

# --- 3. Sidebar RBAC ---
with st.sidebar:
    st.markdown("<h3 style='color: #2870EA;'>🔐 Session</h3>", unsafe_allow_html=True)
    role_choice = st.selectbox("Current Role:", ["Admin (Global)"] + [f"Vendor: {v}" for v in ["Salesforce", "Atlassian", "Independent", "SAP"]])

if "Admin" in role_choice:
    view_df = df
else:
    v_name = role_choice.split(": ")[1]
    view_df = df[df["Vendor"] == v_name]

# --- 4. Main UI ---
st.markdown('<div class="copilot-title">Store ops : Agent performance metrics</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Access: {role_choice}</div>', unsafe_allow_html=True)

t1, t2, t3, t4, t5 = st.tabs(["Velocity", "Quality", "Dev Exp", "Marketplace", "💬 Support Copilot"])

with t1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Days to Publish", f"{view_df['Time_to_Publish'].mean():.1f}")
    c2.metric("Active Submissions", f"{len(view_df):,}")
    c3.metric("Deployment Success", "94.2%")
    fig = px.area(view_df.groupby(view_df['Publish_Date'].dt.date).size().reset_index(name='V'), x='Publish_Date', y='V', template="plotly_dark", color_discrete_sequence=['#2870EA'])
    st.plotly_chart(fig, use_container_width=True)

with t5:
    st.subheader("🤖 Developer Support Copilot (Hybrid RAG)")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "I am connected to your portfolio. How can I help?"}]

    with st.form("chat_form", clear_on_submit=True):
        f_cols = st.columns([8, 1])
        prompt = f_cols[0].text_input("Msg", label_visibility="collapsed", placeholder="Ask about AGNT-00007...")
        sub = f_cols[1].form_submit_button("Send")

    if sub and prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        # RAG Retrieval Logic
        match = view_df[view_df['Agent_ID'].str.contains(prompt.upper()) | view_df['Name'].str.contains(prompt, case=False)].head(1)
        
        # API Call with Local Fallback
        answer = ""
        if "gemini_key" in st.secrets:
            try:
                client = genai.Client(api_key=st.secrets["gemini_key"])
                ctx = match.to_csv() if not match.empty else "No agent found."
                resp = client.models.generate_content(model="gemini-2.0-flash", contents=f"Data: {ctx}\n\nUser: {prompt}")
                answer = resp.text
            except Exception as e:
                if not match.empty:
                    m = match.iloc[0]
                    answer = f"**[Local RAG Fallback]** I've retrieved data for **{m['Name']}**. Status: **{m['Status']}**. Rejection Reason: **{m['Rejection_Reason']}**."
                else:
                    answer = "I'm experiencing heavy traffic. Please try again in 60s."
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

# --- 5. Inspector Section ---
st.divider()
st.subheader("🔍 Agent Timeline & Diagnostics")
search = st.text_input("Enter Agent ID to Inspect (e.g., AGNT-00005):")
if search:
    res = view_df[view_df["Agent_ID"].str.contains(search.upper())]
    if not res.empty:
        a = res.iloc[0]
        st.success(f"Diagnostics for {a['Name']}")
        
        # Timeline
        p_date = a['Publish_Date']
        s_date = p_date - timedelta(days=a['Time_to_Publish'])
        r_date = p_date - timedelta(days=a['Time_to_Publish']/2)
        
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Submitted", s_date.strftime("%b %d"))
        tc2.metric("Review Started", r_date.strftime("%b %d"))
        tc3.metric(f"Status: {a['Status']}", p_date.strftime("%b %d"))
        
        st.json(a.to_dict())
    else:
        st.warning("Agent not found.")
