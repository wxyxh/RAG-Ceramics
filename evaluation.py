import json
import logging
import os
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from ragas import evaluate, RunConfig, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from httpx import Client

# 清除所有系统代理设置，确保直接网络连接
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

# 配置日志系统，设置日志级别为INFO，并定义输出格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 创建自定义HTTP客户端，明确禁用代理设置
custom_http_client = Client(trust_env=False)

# 初始化语言模型和嵌入模型
logger.info("正在初始化语言模型和嵌入模型...")
try:
    # 初始化ChatOpenAI模型，使用本地部署的deepseek-r1模型
    llm = ChatOpenAI(
        model="deepseek-r1:14b",
        base_url='http://127.0.0.1:11434/v1/',
        api_key="not-needed",
        http_client=custom_http_client
    )
    evaluator_llm = LangchainLLMWrapper(llm)

    # 初始化Ollama嵌入模型，使用本地部署的BGE模型
    embed_model = OllamaEmbeddings(
        model="quentinz/bge-large-zh-v1.5:latest",
        base_url="http://localhost:11434"
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(embed_model)
    logger.info("语言模型和嵌入模型初始化完成。")
except Exception as e:
    logger.error(f"初始化模型失败：{e}")
    exit(1)

# 定义评估指标数组，包含四个核心指标
logger.info("正在加载评估指标...")
try:
    metrics = [
        Faithfulness(llm=evaluator_llm),           # 评估回答的忠实度
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),  # 评估回答的相关性
        ContextPrecision(llm=evaluator_llm),       # 评估上下文的精确度
        ContextRecall(llm=evaluator_llm)           # 评估上下文的召回率
    ]
    logger.info("评估指标加载完成。")
except Exception as e:
    logger.error(f"加载评估指标失败：{e}")
    exit(1)

# 主程序入口
if __name__ == '__main__':
    # 定义评估数据文件路径
    data_file_path = 'evaluation_results.json'
    logger.info(f"开始读取评估数据，路径为 {data_file_path}...")
    
    # 读取评估数据JSON文件
    try:
        with open(data_file_path, 'r', encoding='utf-8') as fr:
            data = json.load(fr)
        logger.info(f"成功读取评估数据，共 {len(data)} 条记录。")
    except Exception as e:
        logger.error(f"读取评估数据失败：{e}")
        exit(1)

    # 转换数据格式，准备评估数据集
    eval_data = []
    for idx, item in enumerate(data):
        eval_item = {
            'user_input': item['question'],        # 用户输入的问题
            'response': item['answer'],            # 模型的回答
            'reference': item['reference'],        # 参考答案
            'retrieved_contexts': item['retrieved_contexts']  # 检索到的上下文
        }
        eval_data.append(eval_item)
        # 每处理10条数据打印一次进度
        if (idx + 1) % 10 == 0:
            logger.info(f"已处理 {idx + 1}/{len(data)} 条评估数据。")

    # 创建评估数据集对象
    eval_dataset = EvaluationDataset.from_list(eval_data)
    logger.info(f"评估数据集准备完成，包含 {len(eval_dataset)} 条记录。")

    # 执行评估过程
    logger.info("开始进行评估，请稍候...")
    try:
        results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            batch_size=1,              # 设置批处理大小为1
            run_config=RunConfig(timeout=900)  # 设置超时时间为900秒
        )
        logger.info("评估完成。")
    except Exception as e:
        logger.error(f"评估过程中出现错误：{e}")
        exit(1)

    # 将评估结果保存为CSV文件
    logger.info("正在保存评估结果到CSV文件...")
    try:
        df = results.to_pandas()
        output_file = "result.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"评估结果已成功保存到 {output_file}。")
    except Exception as e:
        logger.error(f"保存评估结果失败：{e}")