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

# 直接配置API密钥
client = OpenAI(
    api_key="sk-51ba3b3e428d407bb7f41de7de4c371f",  # 直接使用你的密钥
    base_url="https://api.deepseek.com"
)

@st.cache_resource
def load_resources():
    """加载模型和向量库"""
    try:
        with st.spinner("正在加载模型..."):
            st_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        with st.spinner("正在加载向量索引..."):
            index = faiss.read_index("enterprise_classified.index")
        
        with st.spinner("正在加载元数据..."):
            with open("enterprise_classified_metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
        
        return st_model, index, metadata
    except FileNotFoundError as e:
        st.error(f"文件未找到: {e}")
        st.info("请确保以下文件存在：\n- enterprise_classified.index\n- enterprise_classified_metadata.pkl")
        st.stop()
    except Exception as e:
        st.error(f"加载资源失败: {e}")
        st.stop()

st_model, index, metadata = load_resources()

# ===============================
# 2️⃣ 核心功能函数
# ===============================
def get_retrieval(query, k=3):
    """检索相关文档"""
    try:
        query_vec = st_model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_vec, k)
        results = []
        for idx in indices[0]:
            if idx < len(metadata):
                results.append(metadata[idx])
        return results
    except Exception as e:
        st.error(f"检索失败: {e}")
        return []

def generate_answer(query, context_list):
    """生成回答"""
    # 限制上下文长度，避免超出token限制
    context_text = "\n\n".join([
        f"【证据类型: {c.get('Evidence_Type', '未知')}】\n内容: {c.get('Original_Text', '')[:500]}" 
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
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # DeepSeek 模型名称
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"生成回答失败: {e}")
        return f"抱歉，生成回答时出现错误：{str(e)}"

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
    
    # 添加使用说明
    with st.expander("使用说明"):
        st.markdown("""
        1. 在输入框中输入您的问题
        2. 系统会检索相关文档
        3. AI会基于检索结果生成回答
        4. 可以查看原始证据来源
        
        **示例问题：**
        - 当前的流程自动化水平如何？
        - 财务自动化水平怎么样？
        - 数字化转型有哪些挑战？
        """)

# 主界面
query = st.text_input(
    "🔍 请输入您的问题：", 
    placeholder="例如：当前的流程自动化水平如何？",
    key="query_input"
)

if query:
    with st.spinner("🔍 正在检索相关文档..."):
        # 1. 检索
        retrieved_results = get_retrieval(query)
        
        if not retrieved_results:
            st.warning("未找到相关参考资料。")
        else:
            st.success(f"找到 {len(retrieved_results)} 条相关证据")
            
            with st.spinner("🤖 AI正在生成回答..."):
                # 2. 生成回答
                answer = generate_answer(query, retrieved_results)
            
            # 3. 展示结果
            st.subheader("🤖 系统回答")
            st.markdown(answer)
            
            st.markdown("---")
            st.subheader("📄 检索到的原始证据")
            
            # 使用展开方式显示证据
            for i, res in enumerate(retrieved_results):
                with st.expander(f"证据 {i+1}: {res.get('Claim_Category', '未分类')}", expanded=False):
                    st.write(f"**数据类型:** {res.get('Data_Type', '未知')}")
                    st.write(f"**逻辑角色:** {res.get('Logic_Role', '未知')}")
                    st.write(f"**证据类型:** {res.get('Evidence_Type', '未知')}")
                    st.write(f"**原始内容:**")
                    st.info(res.get('Original_Text', '无内容'))

# 页脚
st.markdown("---")
st.caption("© 2026 workshop 企业 RAG 智能问答系统 - 基于 Streamlit + FAISS + DeepSeek | Qiujiachen ACCT大一")
