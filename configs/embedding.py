from langchain_ollama import OllamaEmbeddings
from llama_index.core import Settings

def configure_embedding():
    """配置本地 Ollama 的嵌入模型"""
    model = OllamaEmbeddings(
        model="BGE-M3"  # 本地 Ollama 中的嵌入模型名称
    )
    # 设置全局 Embedding
    Settings.embed_model = model
    return model

