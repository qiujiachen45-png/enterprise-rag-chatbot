import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

# ===============================
# 1️⃣ 初始化配置
# ===============================
st.set_page_config(page_title="企业 RAG 智能问答系统", layout="wide")
client = OpenAI() # 自动使用环境变量中的 API_KEY

@st.cache_resource
def load_resources():
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("enterprise_classified.index")
    with open("enterprise_classified_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return st_model, index, metadata

st_model, index, metadata = load_resources()

# ===============================
# 2️⃣ 核心功能函数
# ===============================
def get_retrieval(query, k=3):
    query_vec = st_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)
    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx])
    return results

def generate_answer(query, context_list):
    context_text = "\n\n".join([
        f"【证据类型: {c['Evidence_Type']}】\n内容: {c['Original_Text']}" 
        for c in context_list
    ])
    
    prompt = f"""你是一个专业的企业咨询顾问。请根据以下提供的参考资料回答学生的问题。
要求：
1. 必须基于参考资料回答，不要编造。
2. 返回格式必须包含：[标准化证据总结] 和 [原文引用]。
3. 语言专业、客观。

参考资料：
{context_text}

学生问题：{query}
"""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ===============================
# 3️⃣ Streamlit UI 界面
# ===============================
st.title("🏢 企业数字化转型 RAG 智能问答系统")
st.markdown("---")

# 侧边栏
with st.sidebar:
    st.header("系统状态")
    st.success("✅ 向量库已加载")
    st.info(f"📊 当前索引块数量: {len(metadata)}")
    st.write("支持 30+ 学生并发访问")

# 主界面
query = st.text_input("🔍 请输入您的问题（例如：当前的流程自动化水平如何？）：", placeholder="输入问题并回车...")

if query:
    with st.spinner("正在检索并生成回答..."):
        # 1. 检索
        retrieved_results = get_retrieval(query)
        
        if not retrieved_results:
            st.warning("未找到相关参考资料。")
        else:
            # 2. 生成回答
            answer = generate_answer(query, retrieved_results)
            
            # 3. 展示结果
            st.subheader("🤖 系统回答")
            st.markdown(answer)
            
            st.markdown("---")
            st.subheader("📄 检索到的原始证据")
            
            cols = st.columns(len(retrieved_results))
            for i, res in enumerate(retrieved_results):
                with cols[i]:
                    with st.expander(f"证据 {i+1}: {res['Claim_Category']}", expanded=True):
                        st.write(f"**数据类型:** {res['Data_Type']}")
                        st.write(f"**逻辑角色:** {res['Logic_Role']}")
                        st.info(res['Original_Text'][:200] + "...")

# 页脚
st.markdown("---")
st.caption("© 2024 企业 RAG 智能问答系统 - 基于 Streamlit + FAISS + GPT-4.1-mini")
