import streamlit as st
import json
import os
from tqdm import tqdm
from openai import OpenAI
import tempfile
from dotenv import load_dotenv

from day4 import extract_text_from_pdf, split_text_with_overlap
from day9 import MultiDocRAGBot

load_dotenv()

st.set_page_config(
    page_title="论文阅读助手",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stChatMessage {font-family: 'sans-serif';}
    .stButton button {width: 100%; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

st.title("🎓 ScholarMind Pro")
st.caption("🚀 基于GLM 的本地化 RAG 知识引擎")

# ================= 状态管理 (Session State) =================
# 用于在网页刷新时保留数据
#聊天记录
if "messages" not in st.session_state:
    st.session_state["messages"] = []

#RAGbot实例
if "bot" not in st.session_state:
    st.session_state.bot = None

#当前文件名
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# ================= 辅助函数：实时 ETL 流水线 =================
def process_uploaded_file(uploaded_file, api_key, base_url):
    """
    实时处理上传文件，完成之前流程：切片、向量化、存JSON
    """
    #创建临时文件来存PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    #进度条
    progress_text = "正在启动嵌入引擎..."
    my_bar = st.progress(0, text=progress_text)

    try:
        my_bar.progress(20, text="正在解析PDF文本...")
        raw_text = extract_text_from_pdf(tmp_path)

        my_bar.progress(40, text="正在进行语义切片...")
        chunks = split_text_with_overlap(raw_text, chunk_size=500, overlap=50)

        my_bar.progress(50, text=f"正在并行向量化{len(chunks)} 个切片...")
        client = OpenAI(api_key=api_key, base_url=base_url)

        knowledge_base = []
        step = 50 / len(chunks)
        current_progress = 50

        for i,chunk in enumerate(chunks):
            try:
                response = client.embeddings.create(
                    input=chunk.replace("\n", " "),
                    model="embedding-2",
                )
                vector = response.data[0].embedding
                knowledge_base.append({
                    "id": i,
                    "text": chunk,
                    "vector": vector,
                    "source": uploaded_file.name
                })
            except Exception as e:
                st.error(f"向量化失败：{e}")
            
            current_progress += step
            if current_progress > 95: current_progress = 95
            my_bar.progress(int(current_progress), text="正在向量化...")
        my_bar.progress(98, text="正在构建索引...")
        json_path = f"kb_{uploaded_file.name}.json"
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(knowledge_base, f, ensure_ascii=False)

        my_bar.progress(100, text="构建完成！")
        return json_path

    except Exception as e:
        st.error(f"处理过程出错：{e}")
        return None
    finally:
        os.remove(tmp_path)
        my_bar.empty()

# ================= 侧边栏：控制台 =================
with st.sidebar:
    st.header("⚙️ 引擎配置")

    #默认从环境变量读取
    default_key = os.getenv("API Key", "")
    default_url = os.getenv("Base URL", "https://open.bigmodel.cn/api/paas/v4/")

    api_key = st.text_input("API Key", type="password", value=default_key)
    base_url = st.text_input("Base URL", value=default_url)

    # 只有输入了 Key 才能初始化数据库连接
    if api_key and not st.session_state.bot:
        st.session_state.bot = MultiDocRAGBot(api_key, base_url)

    st.divider()

    st.header("📂 多文档知识库")
    uploaded_files = st.file_uploader("上传论文 (支持多选)", type=["pdf"], accept_multiple_files=True)

    if st.button("🚀 初始化知识库", type="primary"):
        if not api_key:
            st.warning("请先输入 API Key！")
        elif not uploaded_files:
            st.warning("请先上传文件！")
        else:
            with st.spinner("正在构建多文档索引..."):
                for uploaded_file in uploaded_files:
                    # 进度提示
                    st.toast(f"正在处理: {uploaded_file.name}")

                    # 1. 存临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # 2. ETL 提取与切片
                    raw_text = extract_text_from_pdf(tmp_path)
                    chunks = split_text_with_overlap(raw_text)

                    # 3. 注入 ChromaDB (自带查重)
                    st.session_state.bot.add_document(uploaded_file.name, chunks)

                    os.remove(tmp_path)

                st.success("✅ 所有文档已成功注入知识库！")

    if st.session_state.bot:
        doc_count = st.session_state.bot.collection.count()
        st.info(f"🟢 数据库在线 | 当前拥有 {doc_count} 个知识切片。")

# ================= 主聊天区域 =================
if not st.session_state.messages:
    st.info("👋 欢迎使用！请在左侧上传 PDF 并初始化，然后开始提问。")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("想问什么？(例如：这篇论文的创新点是什么？)"):#:= 先赋值后判断
    # 用户消息上屏
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):#会自动帮你画一个用户头像，并把背景变成浅灰色
        st.markdown(prompt)

    if st.session_state.bot:
        with st.chat_message("assistant"):
            # 这是一个占位符，用于显示流式输出或思考状态
            message_placeholder = st.empty()

            with st.spinner("正在检索并生成..."):
                try:
                    # 调用 Day 6 的 chat 方法
                    full_response = st.session_state.bot.chat(prompt)
                    answer = full_response["answer"]
                    sources = full_response["sources"]

                    # 模拟打字机效果 (可选，如果 RAGBot 支持 yield 可以做真正的流式)
                    message_placeholder.markdown(full_response)

                    #渲染参考来源
                    if sources:
                        with st.expander("📚 查看参考来源 (Source Documents)"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**[doc {i+1}] 相关度评分:** `{source['score']:.4f}`")
                                # 用 caption 显示小字体的原文
                                st.caption(source['text'])
                                st.divider()  # 画一条分割线

                except Exception as e:
                    st.error(f"生成失败: {e}")
                    full_response = "系统遇到错误，请检查日志。"
    st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.warning("⚠️ 请先在左侧初始化知识库！")


