import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from google import genai

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

# --- 1. Data Generation ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 5000
    today = datetime.now()
    dates = [today - timedelta(days=np.random.randint(0, 30)) for _ in range(n)]
    vendors = ["Internal Engineering", "Salesforce", "Atlassian", "Independent", "SAP"]

    return pd.DataFrame({
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

df = load_data()

# --- 2. Sidebar RBAC ---
with st.sidebar:
    st.markdown("<h3 style='color: #2870EA;'>🔐 Active Session</h3>", unsafe_allow_html=True)
    current_role = st.selectbox("Simulate Login Role:", ["Admin (Global View)"] + [f"Vendor: {v}" for v in df["Vendor"].unique() if v != "Internal Engineering"])

if current_role == "Admin (Global View)":
    view_df = df
    role_title = "Global Ecosystem View"
else:
    active_vendor = current_role.split(": ")[1]
    view_df = df[df['Vendor'] == active_vendor]
    role_title = f"{active_vendor} Portfolio View"

# --- 3. Main UI ---
st.markdown('<div class="copilot-title">Store ops : Agent performance metrics</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Current Access Level: {role_title}</div>', unsafe_allow_html=True)

plotly_template = "plotly_dark"

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Velocity & Throughput", "Quality & Trust", "Dev Experience", "Marketplace Success", "💬 Support Copilot"
])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Time to Publish", f"{view_df['Time_to_Publish'].mean():.1f} Days")
    c2.metric("Total Submissions (30d)", f"{len(view_df):,}")
    c3.metric("Pass Rate", f"{view_df['Pass_Rate'].mean():.1f}%")
    daily_vol = view_df.groupby(view_df['Publish_Date'].dt.date).size().reset_index(name='Volume')
    fig_area = px.area(daily_vol, x="Publish_Date", y="Volume", title="Publishing Trend", template=plotly_template, color_discrete_sequence=['#2870EA'])
    st.plotly_chart(fig_area, width="stretch")

with tab2:
    rejections = view_df[view_df['Rejection_Reason'] != 'None']['Rejection_Reason'].value_counts().reset_index()
    if not rejections.empty:
        rejections.columns = ['Reason', 'Count']
        fig_bar = px.bar(rejections, x='Count', y='Reason', orientation='h', title="Top Rejection Reasons", template=plotly_template, color_discrete_sequence=['#E362F8'])
        st.plotly_chart(fig_bar, width="stretch")

with tab3:
    fig_box = px.box(view_df, x="Vendor" if current_role=="Admin (Global View)" else "Status", y="Action_Latency", template=plotly_template)
    st.plotly_chart(fig_box, width="stretch")

with tab4:
    st.dataframe(view_df.nlargest(10, 'Installs')[['Name', 'Status', 'Installs']], width="stretch")

# --- 4. RAG AI Integration Tab ---
with tab5:
    st.subheader("🤖 Developer Support Copilot (RAG Enabled)")
    if "gemini_key" in st.secrets:
        client = genai.Client(api_key=st.secrets["gemini_key"])
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Authentication successful. I am connected to your portfolio data. How can I help?"}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("E.g., Which of my agents are currently failing and why?"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Scanning your portfolio data..."):
                try:
                    # Look for specific agent mentions, otherwise grab general portfolio stats
                    mentioned_agents = view_df[view_df['Name'].apply(lambda x: x.lower() in prompt.lower()) | view_df['Agent_ID'].apply(lambda x: x.lower() in prompt.lower())]
                    
                    if not mentioned_agents.empty:
                        context_str = mentioned_agents[['Agent_ID', 'Name', 'Status', 'Rejection_Reason', 'Action_Latency']].to_csv(index=False)
                        rag_prompt = f"Context Data for requested agents:\n{context_str}\n\nUser Question: {prompt}"
                    else:
                        failing_agents = view_df[view_df['Status'] != 'Published'].head(10)
                        context_str = failing_agents[['Agent_ID', 'Name', 'Status', 'Rejection_Reason']].to_csv(index=False)
                        rag_prompt = f"Portfolio Summary Context (Top flagged agents):\n{context_str}\n\nUser Question: {prompt}"
                    
                    response = client.models.generate_content(model="gemini-2.0-flash", contents=rag_prompt)
                    with st.chat_message("assistant"):
                        st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"API Error: {e}")
    else:
        st.error("Missing Gemini API Key. Please add 'gemini_key' to your Streamlit Secrets.")

st.divider()
st.subheader("🔍 Agent Inspector")
search = st.text_input("Enter Agent Name or ID (must belong to your active session):")

if search:
    agent = view_df[(view_df["Name"].str.contains(search, case=False)) | (view_df["Agent_ID"].str.contains(search, case=False))]
    if not agent.empty:
        a = agent.iloc[0]
        st.success(f"Diagnostics loaded for: {a['Name']}")
        p1, p2, p3 = st.columns(3)
        p1.metric("Bias Check", f"{a['Bias_Score']:.1f}/100")
        p2.metric("Code Malware", f"{a['Malware_Score']:.1f}/100")
        p3.metric("AI Inclusivity", f"{a['Inclusivity_Score']:.1f}/100")
    else:
        st.error("Agent not found or you do not have permission to view it.")
