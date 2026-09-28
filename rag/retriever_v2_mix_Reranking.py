import sys
import argparse
import logging
import jieba
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder

# 路径管理
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
logger = logging.getLogger("RAG")

class QwenEmbeddingFunction:
    def __init__(self, model_path):
        self.model = SentenceTransformer(str(model_path), device="cpu")
    
    def name(self):
        return "qwen3-embedding"
        
    # 给 ChromaDB 调用，用于文档（列表）向量化
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.model.encode(input, normalize_embeddings=True, show_progress_bar=False).tolist()
    
    # 给 ChromaDB 调用，专门处理单个查询字符串
    def embed_query(self, input: str) -> List[float]:
        # Qwen3 查询侧需要 query prompt
        return self.model.encode(input, prompt_name="query", normalize_embeddings=True, show_progress_bar=False).tolist()
    
    # 给 ChromaDB 调用，处理文档列表 (通常和__call__逻辑一致)
    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self.model.encode(input, normalize_embeddings=True, show_progress_bar=False).tolist()

class MuseumRetriever:
    def __init__(self):
        # 1. 先定义所有必要的路径和变量
        model_candidates = [
            project_root / "models" / "qwen3-embedding",
            project_root / "models" / "Qwen3-Embedding-0.6B",
        ]
        self.model_path = next((path for path in model_candidates if path.exists()), model_candidates[0])
        self.reranker_path = project_root / "models" / "bge-reranker-base"
        self.chroma_path = project_root / "data" / "chroma_db_qwen3"
        self.collection_name = "works"
        self.reranker: Optional[CrossEncoder] = None
        self.reranker_name = ""
        self.reranker_enabled = False
        
        # 2. 然后加载模型
        embedding_id = str(self.model_path) if self.model_path.exists() else "Qwen/Qwen3-Embedding-0.6B"
        if not self.model_path.exists():
            logger.warning("本地 Embedding 模型未找到，将尝试从 HuggingFace 自动下载: %s", embedding_id)
        else:
            logger.info("已加载本地 Embedding 模型: %s", embedding_id)
        reranker_id = str(self.reranker_path)
        if not self.reranker_path.exists():
            logger.warning("本地 Reranker 模型未找到，当前将跳过重排序，仅使用向量 + BM25 混合召回。")
        else:
            logger.info("已加载本地 Reranker 模型: %s", reranker_id)

        logger.info("正在加载 Embedding: %s", embedding_id)
        self.qwen_func = QwenEmbeddingFunction(embedding_id)
        
        self.reranker_name = reranker_id if self.reranker_path.exists() else ""
        if self.reranker_path.exists():
            try:
                logger.info("正在加载重排序模型: %s", reranker_id)
                self.reranker = CrossEncoder(reranker_id, max_length=512)
                self.reranker_enabled = True
            except Exception as exc:
                self.reranker = None
                self.reranker_enabled = False
                logger.warning("重排序模型加载失败，降级为混合召回直出: %s", exc)
        
        # 3. 最后初始化数据库引擎
        self._init_vector_engine()
        self._init_bm25_engine()

    # --- 1. 原子召回 (支持指定集合) ---
    def vector_search(self, query: str, collection_name: str = "works", n_results: int = 5) -> List[Dict]:
        target_collection = self.collection
        if collection_name != self.collection_name:
            target_collection = self.client.get_collection(name=collection_name, embedding_function=self.qwen_func)

        res = target_collection.query(
            query_embeddings=[self.qwen_func.embed_query(query)],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "content": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "vector_score": float(res["distances"][0][i]) if res.get("distances") else 0.0,
            }
            for i in range(len(res["ids"][0]))
        ]

    # --- 2. 统计接口 (更详细) ---
    def get_stats(self) -> Dict:
        # 获取当前集合的文档总数
        count = self.collection.count()
        return {
            "total_docs": count, 
            "collection": self.collection_name,
            "engine": "Qwen3-Embedding + BM25 + Reranker" if self.reranker_enabled else "Qwen3-Embedding + BM25",
            "reranker_enabled": self.reranker_enabled,
            "database_path": str(self.chroma_path),
            "embedding_model": str(self.model_path if self.model_path.exists() else "Qwen/Qwen3-Embedding-0.6B"),
            "reranker_model": self.reranker_name,
        }

    # --- 3. 索引维护 ---
    def rebuild_index(self):
        # 建议这里直接调用你的 VectorBuilder，或者重新载入 BM25 索引
        logger.info("正在触发索引重建流程...")
        self._init_bm25_engine()
        logger.info("索引已刷新。")

    def bm25_search(self, query: str, n_results: int = 5) -> List[Dict]:
        if not self.all_docs:
            return []
        tokenized_query = jieba.lcut(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_n_idx = np.argsort(scores)[::-1][:n_results]
        return [
            {
                "content": self.all_docs[idx],
                "metadata": self.all_metadatas[idx],
                "bm25_score": float(scores[idx]),
            }
            for idx in top_n_idx
            if idx < len(self.all_docs)
        ]

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return []
        if not self.reranker_enabled or self.reranker is None:
            for index, cand in enumerate(candidates):
                cand["score"] = float(cand.get("bm25_score", 0.0) or 0.0) + float(cand.get("vector_score", 0.0) or 0.0)
                cand["rank_source"] = "hybrid_recall"
                cand["rank_order"] = index
            return candidates

        pairs = [(query, cand['content']) for cand in candidates]
        scores = self.reranker.predict(pairs)
        for i, cand in enumerate(candidates):
            cand['score'] = float(scores[i])
            cand["rank_source"] = "reranker"
        return sorted(candidates, key=lambda x: x['score'], reverse=True)

    def fuse_knowledge(self, chunks: List[Dict]) -> str:
        if not chunks:
            return "（未找到相关信息）"
        blocks = [f"【相关度 {round(float(c.get('score', 0.0))*100, 1)}%】\n{c['content']}" for c in chunks]
        return "\n\n".join(blocks)

    def hybrid_query(self, query: str, top_k: int = 2) -> Tuple[List[Dict], Dict[str, float]]:
        """
        执行混合检索流程：向量召回 + BM25 召回 -> 候选集融合 -> BGE重排 -> Top-K 截断。
        
        Args:
            query (str): 用户输入的查询指令。
            top_k (int): 返回给大模型参考的最相关结果数量，默认取前3。
            
        Returns:
            tuple: (results, metrics) 
                   results: 包含内容和 metadata 的列表。
                   metrics: 包含 recall/rerank/total 的耗时指标。
        """
        start_total = time.time()
        t0 = time.time()
        vec_cands = self.vector_search(query)
        bm25_cands = self.bm25_search(query)
        
        seen = set()
        candidates = []
        for c in vec_cands + bm25_cands:
            if c['content'] not in seen:
                candidates.append(c)
                seen.add(c['content'])
        t_recall = time.time() - t0
        
        t1 = time.time()
        candidates = candidates[:5]
        final_results = self.rerank(query, candidates)[:top_k]
        t_rerank = time.time() - t1
        
        return final_results, {"recall": t_recall, "rerank": t_rerank, "total": time.time() - start_total}

    def retrieve(self, query: str, top_k: int = 3) -> str:
        results, _metrics = self.hybrid_query(query, top_k=top_k)
        return self.fuse_knowledge(results)

    def _init_vector_engine(self):
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, 
            embedding_function=self.qwen_func
        )

    def _init_bm25_engine(self):
        all_data = self.collection.get(include=["documents", "metadatas"])
        self.all_docs, self.all_metadatas = all_data['documents'], all_data['metadatas']
        tokenized_docs = [jieba.lcut(doc) for doc in self.all_docs] if self.all_docs else [[]]
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info("BM25 引擎初始化完成，索引 %d 条数据。", len(self.all_docs))


class Retriever:
    """兼容旧接口。"""

    def __init__(self, persist_dir: str = "./data/chroma_db"):
        del persist_dir
        self.exhibition_retriever = MuseumRetriever()

    def retrieve(self, query: str, top_k: int = 3) -> str:
        return self.exhibition_retriever.retrieve(query, top_k)

    def get_stats(self) -> Dict[str, Any]:
        return self.exhibition_retriever.get_stats()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', '-q', type=str, required=True)
    args = parser.parse_args()
    retriever = MuseumRetriever()
    results, metrics = retriever.hybrid_query(args.query)
    print(f"\n🔍 查询: {args.query} (耗时: {metrics['total']:.3f}s)\n" + "="*50)
    print(retriever.fuse_knowledge(results))
