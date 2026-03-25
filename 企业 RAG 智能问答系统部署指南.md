# 企业 RAG 智能问答系统部署指南

本指南将帮助您将“企业 RAG 智能问答系统”部署到 **Streamlit Community Cloud**，使其成为一个永久在线的网站。

## 🚀 部署步骤

### 1. 准备您的 GitHub 仓库

Streamlit Community Cloud 要求您的应用代码托管在 GitHub 仓库中。请按照以下步骤操作：

1.  **创建新的 GitHub 仓库**：
    *   访问 [GitHub](https://github.com/) 并登录您的账户。
    *   点击右上角的 `+` 号，选择 `New repository`。
    *   为您的仓库命名（例如：`enterprise-rag-chatbot`），选择 `Public` 或 `Private`（如果选择 Private，部署时需要授权），然后点击 `Create repository`。

2.  **上传项目文件**：
    将以下所有文件上传到您新创建的 GitHub 仓库的根目录中：
    *   `app.py` (Streamlit 应用主文件)
    *   `build_index.py` (构建向量索引和分类模型的脚本)
    *   `requirements.txt` (Python 依赖列表)
    *   `enterprise_classified.index` (FAISS 向量索引文件)
    *   `enterprise_classified_metadata.pkl` (向量索引的元数据文件)
    *   `template_vectorizer.pkl` (TF-IDF Vectorizer 模型)
    *   `claim_model.pkl` (分类模型)
    *   `data_type_model.pkl` (分类模型)
    *   `logic_model.pkl` (分类模型)
    *   `KPI报表.txt` (原始数据文件)
    *   `系统日志.txt` (原始数据文件)
    *   `行业数据.txt` (原始数据文件)
    *   `访谈.txt` (原始数据文件)
    *   `training_100.txt` (训练数据文件)

    **重要提示**：在上传之前，请确保您已在本地运行 `build_index.py` 脚本，以生成 `enterprise_classified.index`、`enterprise_classified_metadata.pkl`、`template_vectorizer.pkl`、`claim_model.pkl`、`data_type_model.pkl` 和 `logic_model.pkl` 这些文件。这些文件是 RAG 系统正常运行所必需的。

### 2. 配置 Deepseek API Key

您的应用需要访问 Deepseek API 来生成回答。请将您的 `OPENAI_API_KEY` 作为 Secret 变量添加到 Streamlit Cloud 中。

1.  **获取 OpenAI API Key**：
    *   访问 [OpenAI API Keys](https://platform.openai.com/account/api-keys) 并登录。
    *   创建一个新的 Secret Key 并复制它。

2.  **在 Streamlit Cloud 中添加 Secret**：
    *   登录 [Streamlit Community Cloud](https://share.streamlit.io/)。
    *   点击右上角的 `New app` 或进入您已部署的应用设置。
    *   在部署应用时，找到 `Advanced settings` -> `Secrets` 部分。
    *   添加一个新的 Secret，`Key` 为 `OPENAI_API_KEY`，`Value` 为您从 OpenAI 复制的 API Key。

### 3. 部署到 Streamlit Community Cloud

1.  **登录 Streamlit Community Cloud**：
    *   访问 [Streamlit Community Cloud](https://share.streamlit.io/) 并使用您的 GitHub 账户登录。

2.  **部署新应用**：
    *   点击 `New app` 按钮。
    *   选择您刚刚创建的 GitHub 仓库。
    *   `Branch` 选择 `main` (或您上传代码的分支)。
    *   `Main file path` 填写 `app.py`。
    *   点击 `Deploy!`。

Streamlit Cloud 将会自动安装 `requirements.txt` 中列出的依赖，并启动您的应用。部署完成后，您将获得一个永久的 URL 来访问您的智能问答系统。

## 疑难解答

*   **部署失败**：请检查 Streamlit Cloud 的部署日志，通常会显示具体的错误信息。常见问题包括 `requirements.txt` 依赖安装失败或 API Key 配置错误。
*   **模型加载失败**：确保 `enterprise_classified.index` 和 `enterprise_classified_metadata.pkl` 等模型文件已正确上传到 GitHub 仓库，并且 `build_index.py` 脚本在本地成功运行生成了这些文件。

如果您在部署过程中遇到任何问题，请随时联系支持。
