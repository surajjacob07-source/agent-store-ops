import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from openai import OpenAI

# --- 1. Page Config & Professional CSS ---
st.set_page_config(page_title="Store Ops", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #000000; color: #F3F2F1; }
    [data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #333333; }
    .copilot-title {
        background: linear-gradient(90deg, #2870EA 0%, #E362F8 50%, #FFB6B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem; font-weight: 700; margin-bottom: 0.5rem;
    }
    .sub-header { color: #939393; font-size: 1.1rem; margin-top: -10px; margin-bottom: 25px; }
    [data-testid="stMetricValue"] { color: #E362F8; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- 2. Data Generation Engine ---
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
        "Rejection_Reason": np.random.choice(["None", "Data Privacy", "Auth Timeout", "Manifest Bug", "Token Limit"], n, p=[0.7, 0.1, 0.08, 0.07, 0.05]),
        "Action_Latency": np.random.normal(150, 40, n).clip(50, 800),
        "Installs": np.random.exponential(1000, n).astype(int),
        "Bias_Score": np.random.normal(92, 5, n).clip(0, 100),
        "Malware_Score": np.random.normal(99, 1, n).clip(0, 100),
        "Inclusivity_Score": np.random.normal(88, 8, n).clip(0, 100)
    })

df = load_data()

# --- 3. Sidebar & RBAC ---
with st.sidebar:
    st.markdown("<h3 style='color: #2870EA;'>🔐 Active Session</h3>", unsafe_allow_html=True)
    role_choice = st.selectbox("Simulate Login Role:", ["Admin (Global View)"] + [f"Vendor: {v}" for v in ["Salesforce", "Atlassian", "Independent", "SAP"]])

if "Admin" in role_choice:
    view_df = df
    role_title = "Global Ecosystem View"
else:
    v_name = role_choice.split(": ")[1]
    view_df = df[df["Vendor"] == v_name]
    role_title = f"{v_name} Portfolio View"

# --- 4. Main UI Header ---
st.markdown('<div class="copilot-title">Store ops : Agent performance metrics</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Current Access Level: {role_title}</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Velocity & Throughput", "Quality & Trust", "Dev Experience", "Marketplace Success", "💬 Support Copilot"
])

# --- TAB 1: Velocity ---
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Time to Publish", f"{view_df['Time_to_Publish'].mean():.1f} Days")
    c2.metric("Total Submissions (30d)", f"{len(view_df):,}")
    c3.metric("Pass Rate", f"{(len(view_df[view_df['Status']=='Published'])/len(view_df)*100):.1f}%")
    daily_vol = view_df.groupby(view_df['Publish_Date'].dt.date).size().reset_index(name='Volume')
    fig_area = px.area(daily_vol, x="Publish_Date", y="Volume", title="Publishing Trend", template="plotly_dark", color_discrete_sequence=['#2870EA'])
    st.plotly_chart(fig_area, use_container_width=True)

# --- TAB 2: Quality ---
with tab2:
    rejections = view_df[view_df['Rejection_Reason'] != 'None']['Rejection_Reason'].value_counts().reset_index()
    if not rejections.empty:
        rejections.columns = ['Reason', 'Count']
        fig_bar = px.bar(rejections, x='Count', y='Reason', orientation='h', title="Top Rejection Reasons", template="plotly_dark", color_discrete_sequence=['#E362F8'])
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.write("No rejections found for this view.")

# --- TAB 3: Dev Exp ---
with tab3:
    fig_box = px.box(view_df, x="Vendor" if "Admin" in role_choice else "Status", y="Action_Latency", title="Action Latency Distribution (ms)", template="plotly_dark")
    st.plotly_chart(fig_box, use_container_width=True)

# --- TAB 4: Marketplace ---
with tab4:
    st.markdown("##### Top Performing Agents by Installs")
    st.dataframe(view_df.nlargest(15, 'Installs')[['Name', 'Vendor', 'Status', 'Installs']], use_container_width=True)

# --- TAB 5: OpenAI RAG Copilot ---
with tab5:
    st.subheader("🤖 Developer Support Copilot (OpenAI Powered)")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "I am connected to your portfolio via OpenAI. How can I help?"}]

    with st.form("chat_form", clear_on_submit=True):
        f_cols = st.columns([8, 1])
        prompt = f_cols[0].text_input("Msg", label_visibility="collapsed", placeholder="Ask about AGNT-00007...")
        if f_cols[1].form_submit_button("Send") and prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            match = view_df[view_df['Agent_ID'].str.contains(prompt.upper()) | view_df['Name'].str.contains(prompt, case=False)].head(1)
            
            try:
                client = OpenAI(api_key=st.secrets["openai_key"])
                ctx = match.to_csv(index=False) if not match.empty else "No specific agent found in current context. Please answer generally based on your knowledge."
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": f"You are a helpful Microsoft Store Ops Copilot. Use this context data to answer the user: {ctx}"},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"⚠️ Setup Required: Please ensure your OpenAI API key is added to Streamlit Secrets. Error: {e}"
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

# --- 5. Inspector Section ---
st.divider()
st.subheader("🔍 Agent Lifecycle & Security Inspector")
search = st.text_input("Enter Agent ID to Inspect (e.g., AGNT-00007):")

if search:
    res = view_df[view_df["Agent_ID"].str.contains(search.upper())]
    if not res.empty:
        a = res.iloc[0]
        st.markdown(f"### Diagnostic Report: {a['Name']}")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Status", a['Status'])
        m2.metric("Latency", f"{a['Action_Latency']:.1f}ms")
        m3.metric("Installs", f"{a['Installs']:,}")
        m4.metric("Flag", a['Rejection_Reason'])

        st.markdown("#### 🛡️ AI Security Scorecard")
        sc1, sc2, sc3 = st.columns(3)
        sc1.progress(a['Bias_Score']/100, text=f"Fairness: {a['Bias_Score']:.1f}%")
        sc2.progress(a['Malware_Score']/100, text=f"Safety: {a['Malware_Score']:.1f}%")
        sc3.progress(a['Inclusivity_Score']/100, text=f"Inclusivity: {a['Inclusivity_Score']:.1f}%")

        p_date = a['Publish_Date']
        s_date = p_date - timedelta(days=a['Time_to_Publish'])
        r_date = p_date - timedelta(days=a['Time_to_Publish']/2)
        st.markdown("#### 📅 Lifecycle Timeline")
        st.write(f"✅ **Submitted:** {s_date.strftime('%b %d, %Y')} | 🔍 **Review:** {r_date.strftime('%b %d, %Y')} | 🏁 **Action:** {p_date.strftime('%b %d, %Y')}")
    else:
        st.warning("Agent ID not found.")
