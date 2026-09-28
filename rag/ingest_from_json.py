import json
import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Any
import chromadb

# 确保能导入 rag 包
root_path = Path(__file__).resolve().parent
project_root = root_path.parent
sys.path.append(str(root_path))

# 导入在 retriever 中定义的 Embedding 函数，确保一致性
from rag.retriever_v2_mix_Reranking import QwenEmbeddingFunction

# ============================================================
# 配置区域
# ============================================================
CONFIG = {
    "json_path": "data/processed/standard.json",
    "chroma_path": "data/chroma_db_qwen3",         # 新数据库路径
    "model_path": "models/qwen3-embedding",       # 优先使用新的命名
    "collection_name": "works",
    "batch_size": 50,
}

class VectorBuilder:
    def __init__(self, force_rebuild: bool = True):
        self.force_rebuild = force_rebuild
        self.model_path = self._resolve_model_path()

    def _resolve_model_path(self) -> str:
        candidates = [
            project_root / "models" / "qwen3-embedding",
            project_root / "models" / "Qwen3-Embedding-0.6B",
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return CONFIG["model_path"]

    def extract_text_with_keys(self, obj, prefix: str = "") -> List[str]:
        texts = []
        if isinstance(obj, str):
            if obj.strip():
                val = obj.strip()
                if len(val) > 200: val = val[:200] + "..."
                texts.append(f"{prefix}：{val}" if prefix else val)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if key in ["id", "来源工作表", "source_sheet", "source_row"]: continue
                texts.extend(self.extract_text_with_keys(value, key))
        elif isinstance(obj, list):
            for item in obj:
                texts.extend(self.extract_text_with_keys(item, prefix))
        elif isinstance(obj, (int, float, bool)):
            texts.append(f"{prefix}：{str(obj)}" if prefix else str(obj))
        return texts

    def build_document(self, work: Dict) -> str:
        exclude_keys = ["id", "来源工作表", "source_sheet", "source_row", "metadata"] 
        raw_texts = self.extract_text_with_keys(work)
        final_parts = [line for line in raw_texts if not any(line.startswith(f"{ex_k}：") for ex_k in exclude_keys)]
        return "\n".join(final_parts)

    def build_metadata(self, work: Dict) -> Dict:
        flat_meta = {}
        for k, v in work.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, (str, int, float, bool)): flat_meta[sk] = sv
            elif isinstance(v, list):
                flat_meta[k] = ", ".join([str(i) for i in v])
            elif isinstance(v, (str, int, float, bool)):
                flat_meta[k] = v
        return flat_meta

    def build(self):
        print("🚀 正在使用 Qwen3 模型构建向量数据库...")
        chroma_path = Path(CONFIG["chroma_path"])
        
        # 1. 强制清理旧库
        if self.force_rebuild and chroma_path.exists():
            print(f"🗑️ 清理旧数据: {chroma_path}")
            shutil.rmtree(chroma_path, ignore_errors=True)

        # 2. 初始化客户端与模型
        client = chromadb.PersistentClient(path=str(chroma_path))
        # 使用 retriever 中定义的类来初始化 embedding_fn
        embedding_fn = QwenEmbeddingFunction(self.model_path)

        # 3. 创建集合
        collection = client.create_collection(
            name=CONFIG["collection_name"],
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        # 4. 导入数据
        json_path = Path(CONFIG["json_path"])
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        works = data.get("works", [])

        print(f"📤 准备导入 {len(works)} 条数据...")
        for i in range(0, len(works), CONFIG["batch_size"]):
            batch = works[i:i+CONFIG["batch_size"]]
            collection.add(
                ids=[str(w.get("id", idx)) for idx, w in enumerate(batch, i)],
                documents=[self.build_document(w) for w in batch],
                metadatas=[self.build_metadata(w) for w in batch]
            )
            print(f"  已完成: {min(i + CONFIG['batch_size'], len(works))}/{len(works)}")

        print(f"\n✅ 数据入库成功！位置: {chroma_path}")

if __name__ == "__main__":
    builder = VectorBuilder(force_rebuild=True)
    builder.build()
