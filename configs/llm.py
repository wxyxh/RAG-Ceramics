from langchain_ollama import OllamaLLM
from llama_index.core import Settings

def configure_llm():
    """配置本地 Ollama 的 LLM 并设置为 LlamaIndex 的全局 LLM"""
    llm = OllamaLLM(
        model="deepseek-r1:32b", 
    )
    # 设置为全局使用的 LLM
    Settings.llm = llm
    return llm

