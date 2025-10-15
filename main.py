import re
import json
import logging
from pathlib import Path
import streamlit as st
from llama_index.core import PromptTemplate
from configs.embedding import configure_embedding  # 配置BGE-M3
from configs.llm import configure_llm
from modules.data_loader import process_uploaded_files, check_data_updates, load_markdown_documents
from modules.indexer import IndexManager

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 初始化配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "upload"
INDEX_DIR = BASE_DIR / "storage"

QA_PROMPT_TEMPLATE = PromptTemplate("""\
你是一个专业的陶瓷知识专家，请根据提供的上下文信息直接给出明确的答案。
请严格遵守以下要求：
1. 只返回最终答案。
2. 只回答与陶瓷相关的问题，其他问题请回答：
"对不起，我是一个智能陶瓷知识问答机器人，无法回答这个问题。如果你有其他与陶瓷相关的问题，我非常乐意为你提供帮助。"
3. 使用简洁的书面化中文。
--------------------
{context_str}
--------------------
问题：{query_str}
答案：""")

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

@st.cache_resource
def init_system():
    """初始化系统组件，只初始化一次"""
    try:
        configure_embedding()  # 初始化BGE-M3
        llm = configure_llm()
        st.success("系统组件初始化完成")
        return llm
    except Exception as e:
        st.error(f"系统初始化失败: {str(e)}")
        raise

def display_sidebar_status():
    """显示侧边栏状态"""
    with st.sidebar:
        st.header("📊 系统状态")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("待处理文件", len(list(UPLOAD_DIR.glob("*"))))
        
        with col2:
            st.metric("知识文档", len(list(DATA_DIR.glob("*.md"))))
        
        st.subheader("最近操作日志")

@st.cache_data
def load_documents():
    """加载和处理文档"""
    try:
        documents = load_markdown_documents(DATA_DIR)
        return documents
    except Exception as e:
        st.error(f"文档加载失败: {str(e)}")
        return []

def render_chat_messages():
    """渲染聊天消息"""
    st.markdown("""<style>
    .chat-message { padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .user-message { background-color: #e3f2fd; margin-left: 20%; }
    .bot-message { background-color: #f5f5f5; margin-right: 20%; }
    </style>""", unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    for role, msg in st.session_state.history:
        css_class = "user-message" if role == "user" else "bot-message"
        st.markdown(f"""
        <div class="chat-message {css_class}">
            <strong>{'👤 用户' if role == 'user' else '🤖 助手'}:</strong>
            <div>{msg}</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    configure_logging()
    st.set_page_config(page_title="智能陶瓷问答系统", page_icon="📚")
    
    # 初始化目录
    for d in [UPLOAD_DIR, DATA_DIR, INDEX_DIR]:
        d.mkdir(exist_ok=True)

    # 系统初始化（只初始化一次）
    try:
        llm = init_system()
        index_manager = IndexManager(INDEX_DIR)
    except Exception:
        st.stop()

    # 处理文件上传
    try:
        process_uploaded_files(UPLOAD_DIR, DATA_DIR)
        updated_files, deleted_files = check_data_updates(DATA_DIR, INDEX_DIR)
    except Exception as e:
        st.error(f"文件处理失败: {str(e)}")
        return

    # 索引管理
    try:
        if not index_manager.load_existing_index():
            # 如果索引不存在或数据发生更改，重新初始化索引
            documents = load_documents()
            index_manager.initialize_index(documents)
        else:
            # 仅在文件更新时更新索引
            if updated_files or deleted_files:
                documents = load_documents()
                index_manager.update_index(documents, deleted_files)  # 增量更新
    except Exception as e:
        st.error(f"索引操作失败: {str(e)}")
        return

    # 界面布局
    display_sidebar_status()
    st.title("📚 智能陶瓷问答系统")
    
    # 聊天界面
    render_chat_messages()
    
    # 用户输入
    if query := st.chat_input("请输入您的问题..."):
        st.session_state.history.append(("user", query))
        try:
            # 使用已经初始化好的索引进行查询
            query_engine = index_manager.index.as_query_engine(
                text_qa_template=QA_PROMPT_TEMPLATE, 
                similarity_top_k=3,
                streaming=False
            )
            response = query_engine.query(query)
            answer = re.sub(r"<think>.*?</think>", "", response.response, flags=re.DOTALL).strip()
        except Exception as e:
            answer = f"⚠️ 处理出错: {str(e)}"
            logging.error(f"查询失败: {str(e)}", exc_info=True)  # 添加详细错误日志
        
        st.session_state.history.append(("bot", answer))
        st.rerun()

if __name__ == "__main__":
    main()
