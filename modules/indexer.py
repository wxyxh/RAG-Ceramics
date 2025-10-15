import logging
import faiss
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import Document
from llama_index.core.node_parser import SimpleNodeParser
from rank_bm25 import BM25Okapi
from configs.embedding import configure_embedding
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化NLTK资源
try:
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))


class IndexManager:
    def __init__(self, index_dir: Path, faiss_nlist: int = 256):
        self.index_dir = index_dir
        self.faiss_nlist = faiss_nlist
        self.storage_context = None
        self.index = None
        self.faiss_index = None
        self.bm25_index = None
        self.id_map = {}
        self.current_max_id = 0
        self.is_index_loaded = False

    # region 核心接口方法
    def initialize_index(self, documents: List[Document], batch_size: int = 512) -> VectorStoreIndex:
        """初始化混合索引"""
        try:
            nodes = self._parse_documents(documents)
            embeddings = self._batch_embed([node.text for node in nodes], batch_size)
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                faiss_future = executor.submit(self._build_faiss_index, embeddings, nodes)
                bm25_future = executor.submit(self._rebuild_bm25_index, nodes)
                faiss_future.result(), bm25_future.result()

            self._init_llama_index(nodes)  
            self.is_index_loaded = True
            return self.index
        except Exception as e:
            logger.error(f"索引初始化失败: {str(e)}")
            raise

    def query(self, query: str, top_k: int = 3, alpha: float = 0.7) -> List[Document]:
        """混合检索"""
        if not self.is_index_loaded:
            raise RuntimeError("索引尚未加载")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            vec_future = executor.submit(self._vector_search, query, top_k * 3)
            bm25_future = executor.submit(self._bm25_search, query, top_k * 3)
            vec_ids, bm25_ids = vec_future.result(), bm25_future.result()

        combined = self._weighted_rank_fusion(vec_ids, bm25_ids, alpha)
        return [self._get_node(faiss_id) for faiss_id in combined[:top_k] if faiss_id in self.id_map]
    # endregion

    # region 索引初始化方法
    def _init_llama_index(self, nodes: List[Document]):
        """LlamaIndex存储初始化（新增方法）"""
        self.storage_context = StorageContext.from_defaults()
        self.index = VectorStoreIndex(
            nodes, 
            storage_context=self.storage_context,
            show_progress=True
        )
        self._persist_index()

    def _build_faiss_index(self, embeddings: np.ndarray, nodes: List[Document]):
        """FAISS索引构建"""
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        dimension = embeddings.shape[1]
        
        ids = np.arange(1, len(embeddings) + 1, dtype=np.int64)
        self.current_max_id = len(ids)
        
        if len(embeddings) >= self.faiss_nlist * 39:
            quantizer = faiss.IndexFlatL2(dimension)
            base_index = faiss.IndexIVFPQ(quantizer, dimension, self.faiss_nlist, 16, 8)
            base_index.train(embeddings)
            logger.info("使用IVFPQ索引")
        else:
            base_index = faiss.IndexFlatL2(dimension)
            logger.warning("使用FlatL2索引")
        
        self.faiss_index = faiss.IndexIDMap(base_index)
        self.faiss_index.add_with_ids(embeddings, ids)
        self.id_map = {faiss_id: node.node_id for faiss_id, node in zip(ids, nodes)}

    def _rebuild_bm25_index(self, nodes: List[Document]):
        """BM25索引重建"""
        tokenized_docs = [self._tokenize_text(node.text) for node in nodes]
        self.bm25_index = BM25Okapi(tokenized_docs)
        logger.info(f"BM25索引已重建，包含{len(tokenized_docs)}个文档")
    # endregion

    # region 检索方法
    def _vector_search(self, query: str, top_k: int) -> List[int]:
        """向量检索"""
        query_embed = self.encode_query(query)
        distances, indices = self.faiss_index.search(query_embed, top_k)
        return indices[0].tolist()

    def _bm25_search(self, query: str, top_k: int) -> List[int]:
        """关键词检索"""
        tokenized_query = self._tokenize_text(query)
        doc_scores = self.bm25_index.get_scores(tokenized_query)
        sorted_indices = np.argsort(doc_scores)[::-1][:top_k].tolist()
        return [list(self.id_map.keys())[i] for i in sorted_indices]

    def _weighted_rank_fusion(self, vec_ids: list, bm25_ids: list, alpha: float) -> list:
        """加权排序融合算法"""
        combined = {}
        for rank, vid in enumerate(vec_ids):
            combined[vid] = combined.get(vid, 0) + alpha * (1 / (rank + 1))
        for rank, bid in enumerate(bm25_ids):
            combined[bid] = combined.get(bid, 0) + (1 - alpha) * (1 / (rank + 1))
        return sorted(combined.keys(), key=lambda x: combined[x], reverse=True)

    # region 辅助方法
    def _parse_documents(self, documents: List[Document]) -> List[Document]:
        """文档预处理"""
        parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=64)
        nodes = parser.get_nodes_from_documents(documents)
        for node in nodes:
            node.text = self._preprocess_text(node.text)
        return nodes

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """文本清洗"""
        text = ''.join([c for c in text if c.isalnum() or c.isspace()])
        words = [stemmer.stem(word) for word in text.lower().split() 
                if word not in stop_words and len(word) > 2]
        return ' '.join(words)

    def _tokenize_text(self, text: str) -> List[str]:
        """BM25分词"""
        return [stemmer.stem(word) for word in text.lower().split() 
               if word not in stop_words and len(word) > 2]

    def _batch_embed(self, texts: List[str], batch_size: int) -> np.ndarray:
        """批量生成嵌入"""
        embed_model = configure_embedding()
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings.append(embed_model.embed_documents(batch))
        return np.vstack(embeddings)

    @lru_cache(maxsize=1000)
    def encode_query(self, query: str) -> np.ndarray:
        """查询编码"""
        return configure_embedding().embed_documents([query])[0].reshape(1, -1)

    def _get_node(self, faiss_id: int) -> Optional[Document]:
        """安全获取节点"""
        return self.storage_context.docstore.get_node(self.id_map[faiss_id])
    # endregion

    # region 持久化方法
    def _persist_index(self):
        """持久化索引"""
        self.index_dir.mkdir(exist_ok=True, parents=True)
        
        # 持久化LlamaIndex
        self.storage_context.persist(persist_dir=self.index_dir)
        
        # 持久化FAISS
        faiss.write_index(self.faiss_index, str(self.index_dir / "faiss.index"))
        
        # 持久化元数据
        with open(self.index_dir / "metadata.pkl", "wb") as f:
            pickle.dump({
                "id_map": self.id_map,
                "current_max_id": self.current_max_id,
                "faiss_nlist": self.faiss_nlist
            }, f)
        logger.info(f"索引已持久化到 {self.index_dir}")

    def load_existing_index(self) -> bool:
        """加载索引"""
        try:
            # 加载LlamaIndex
            self.storage_context = StorageContext.from_defaults(persist_dir=self.index_dir)
            self.index = load_index_from_storage(self.storage_context)
            
            # 加载FAISS
            faiss_path = self.index_dir / "faiss.index"
            if faiss_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_path))
                
                # 加载元数据
                with open(self.index_dir / "metadata.pkl", "rb") as f:
                    meta = pickle.load(f)
                    self.id_map = meta["id_map"]
                    self.current_max_id = meta["current_max_id"]
                    self.faiss_nlist = meta.get("faiss_nlist", 256)
                
                # 重建BM25
                nodes = [self.storage_context.docstore.get_node(nid) for nid in self.id_map.values()]
                self._rebuild_bm25_index(nodes)
                
                self.is_index_loaded = True
                return True
            return False
        except Exception as e:
            logger.error(f"索引加载失败: {str(e)}")
            return False

