# backend/rag/ingest_descriptions.py
import os
import re
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

# 路径配置
project_root = Path(__file__).parent.parent
TXT_PATH = project_root / "data" / "raw_docs" / "industrial_design.txt"
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
        name="industrial_design_assets", 
        embedding_function=emb_fn
    )

    # 3. 解析 TXT 文件
    if not TXT_PATH.exists():
        print(f"❌ 找不到文件: {TXT_PATH}")
        return

    content = TXT_PATH.read_text(encoding='utf-8')

    # --- 核心切分逻辑修改 ---
    # 使用正则表达式匹配“名称：”作为切分点，同时保留匹配到的名称部分
    # \n\s*名称： 确保从新行开始匹配，避免误伤正文中的文字
    pattern = re.compile(r'(?=\n\s*名称：|^名称：)')
    sections = pattern.split(content)
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        # 提取作品名：匹配“名称：”之后到行尾的内容
        # 例如：“名称：咚迦-萨满鼓主题虚拟偶像” -> “咚迦-萨满鼓主题虚拟偶像”
        name_match = re.search(r'名称：\s*(.*)', section)
        
        if name_match:
            exhibit_name = name_match.group(1).strip()
            # 提取主类别（如果存在），存入 metadata 方便后续过滤检索
            category_match = re.search(r'主类别：\s*(.*)', section)
            main_category = category_match.group(1).strip() if category_match else "未分类"

            print(f"📦 正在存入描述库: {exhibit_name} [{main_category}]")
            
            try:
                collection.add(
                    ids=[exhibit_name],
                    documents=[section],
                    metadatas=[{
                        "exhibit_name": exhibit_name,
                        "main_category": main_category
                    }]
                )
            except Exception as e:
                # 处理 ID 重复或其他写入错误
                print(f"⚠️ 跳过重复或错误条目 [{exhibit_name}]: {e}")
        else:
            print(f"🔍 忽略非作品格式片段")

    print(f"✅ 描述信息处理完成。当前库内条目总数: {collection.count()}")

if __name__ == "__main__":
    ingest_descriptions()