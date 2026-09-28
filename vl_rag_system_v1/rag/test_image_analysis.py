# backend/rag/ingest_descriptions.py
import os
import re
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

# 路径配置
project_root = Path(__file__).parent.parent.parent
TXT_PATH = project_root / "data" / "raw_docs" / "无标题.txt"
DB_PATH = project_root / "data" / "image_analysis_db"
MODEL_PATH = project_root / "models" / "bge-small-zh-v1.5"

def ingest_descriptions():
    # 1. 初始化 Embedding
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=str(MODEL_PATH),
        device="cpu"
    )

    # 2. 初始化 ChromaDB
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(
        name="industrial_design_assets", # 保持与之前分析脚本一致的名称
        embedding_function=emb_fn
    )

    # 3. 解析 TXT 文件
    if not TXT_PATH.exists():
        print(f"❌ 找不到文件: {TXT_PATH}")
        return

    content = TXT_PATH.read_text(encoding='utf-8')
    # 简单的按“作品名称与分类”进行分割
    sections = re.split(r'作品名称与分类|作品名称：', content)
    
    for section in sections:
        if "分类：" not in section: continue
        
        # 提取作品名作为 ID
        try:
            name_match = re.search(r'作品名称：(.*?)\n', section)
            if not name_match: 
                name_match = re.search(r'^(.*?)\s+分类：', section.strip())
            
            exhibit_name = name_match.group(1).strip() if name_match else "未知作品"
            
            print(f"📦 正在存入描述库: {exhibit_name}")
            
            collection.add(
                ids=[exhibit_name],
                documents=[section.strip()],
                metadatas=[{"exhibit_name": exhibit_name}]
            )
        except Exception as e:
            print(f"⚠️ 处理片段失败: {e}")

    print(f"✅ 描述信息已存入: {DB_PATH}")

if __name__ == "__main__":
    ingest_descriptions()