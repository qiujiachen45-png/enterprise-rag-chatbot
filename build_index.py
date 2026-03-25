import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ===============================
# 1️⃣ 数据预处理 & 分块
# ===============================
def split_file(file_path, chunk_size=1000):
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# 多源文件路径
files = ['访谈.txt', '系统日志.txt', 'KPI报表.txt', '行业数据.txt']
all_blocks = []
for file in files:
    all_blocks.extend(split_file(file))
print(f"✅ 总文本块数量: {len(all_blocks)}")

# ===============================
# 2️⃣ 小型文本分类模型训练
# ===============================
train_texts = []
train_claim_labels = []
train_data_type_labels = []
train_logic_labels = []

training_file = "training_100.txt"
if os.path.exists(training_file):
    with open(training_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            text, claim, data_type, logic = parts
            train_texts.append(text)
            train_claim_labels.append(claim)
            train_data_type_labels.append(data_type)
            train_logic_labels.append(logic)
    print(f"✅ 成功读取训练集，总样本数：{len(train_texts)}")

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    claim_model = LogisticRegression(max_iter=1000).fit(X_train, train_claim_labels)
    data_type_model = LogisticRegression(max_iter=1000).fit(X_train, train_data_type_labels)
    logic_model = LogisticRegression(max_iter=1000).fit(X_train, train_logic_labels)

    # 保存模型
    with open("template_vectorizer.pkl", "wb") as f: pickle.dump(vectorizer, f)
    with open("claim_model.pkl", "wb") as f: pickle.dump(claim_model, f)
    with open("data_type_model.pkl", "wb") as f: pickle.dump(data_type_model, f)
    with open("logic_model.pkl", "wb") as f: pickle.dump(logic_model, f)
    print("✅ 小型分类模型训练完成")
else:
    print(f"❌ 训练文件 {training_file} 不存在")

# ===============================
# 3️⃣ 分类函数
# ===============================
def classify_block(block_text):
    try:
        with open("template_vectorizer.pkl", "rb") as f: vectorizer = pickle.load(f)
        with open("claim_model.pkl", "rb") as f: claim_model = pickle.load(f)
        with open("data_type_model.pkl", "rb") as f: data_type_model = pickle.load(f)
        with open("logic_model.pkl", "rb") as f: logic_model = pickle.load(f)

        X_vec = vectorizer.transform([block_text])
        claim = claim_model.predict(X_vec)[0]
        data_type = data_type_model.predict(X_vec)[0]
        logic = logic_model.predict(X_vec)[0]

        if claim == "流程自动化水平":
            evidence_type = "KPI报表 / 系统数据" if "报表" in block_text or "系统" in block_text else "员工访谈 / 操作观察"
        elif claim == "数据治理水平中":
            evidence_type = "报表缺失 / 数据完整性" if "报表" in block_text else "系统编码 / 数据冲突"
        elif claim == "决策智能水平中低":
            evidence_type = "KPI报表 / 预测结果" if any(c.isdigit() for c in block_text) else "高管访谈 / 使用情况"
        else:
            evidence_type = "未知证据"
        return claim, data_type, evidence_type, logic
    except Exception as e:
        return "未知", "未知", "未知", "未知"

# ===============================
# 4️⃣ 向量化 & 建立FAISS索引
# ===============================
print("🚀 正在加载 SentenceTransformer 模型...")
st_model = SentenceTransformer('all-MiniLM-L6-v2')
vectors = []
metadata_list = []

print("🚀 正在处理文本块并生成向量...")
for block in all_blocks:
    vec = st_model.encode(block, convert_to_numpy=True)
    vectors.append(vec)
    claim, data_type, evidence, logic = classify_block(block)
    metadata_list.append({
        "Claim_Category": claim,
        "Data_Type": data_type,
        "Evidence_Type": evidence,
        "Logic_Role": logic,
        "Original_Text": block
    })

if vectors:
    vectors = np.stack(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # 保存索引和 metadata
    faiss.write_index(index, "enterprise_classified.index")
    with open("enterprise_classified_metadata.pkl", "wb") as f:
        pickle.dump(metadata_list, f)
    print("✅ 自动分类向量库创建完成")
else:
    print("❌ 没有可处理的文本块")
