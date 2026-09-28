"""
ingest.py - 构建向量数据库
"""
# --- 关键修复 1: 必须在所有导入前预加载 torch 以解决 DLL 1114 错误 ---
try:
    import torch
    torch.empty(1)
except:
    pass
# ----------------------------------------------------------------

import os
import sys
from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def build_database():
    print("=" * 60)
    print(" 构建向量数据库")
    print("=" * 60)

    # 路径配置
    data_dir = project_root / "data"
    excel_path = data_dir / "raw_docs" / "艺术与科技展览数据.xlsx"
    model_path = project_root / "models" / "bge-small-zh-v1.5"
    chroma_path = data_dir / "chroma_db_local_model"

    # 检查文件
    if not excel_path.exists():
        print(f"Excel文件不存在: {excel_path}")
        return False

    if not model_path.exists():
        print(f"模型不存在: {model_path}")
        return False

    print(f"Excel文件: {excel_path}")
    print(f"模型路径: {model_path}")
    print(f"向量库: {chroma_path}")

    # 检查是否已存在
    if chroma_path.exists():
        print(f"\n向量库已存在，自动重新构建...")
        import shutil
        print("删除旧的向量库...")
        shutil.rmtree(chroma_path)

    print("\n开始构建向量数据库...")

    # 初始化ChromaDB
    os.makedirs(chroma_path, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))

    # 使用本地BGE模型
    print("加载本地BGE模型...")
    try:
        # 1. 先加载原始的嵌入函数
        from chromadb.utils import embedding_functions
        base_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=str(model_path),
            device="cpu"
        )
        # 这里是关键：定义一个显式的包装类，强制符合 ChromaDB 的签名要求
        class EmbeddingWrapper:
            def __call__(self, input):
                # 内部调用原始类的逻辑，但参数名映射正确
                return base_fn(input)
        # 实例化这个包装类
        embedding_fn = EmbeddingWrapper()
            
        print("本地模型加载成功")
    except Exception as e:
        print(f"本地模型加载失败: {e}")
        return False

    # 创建集合 (现在 embedding_fn 的签名会被识别为 ['self', 'input'])
    print("创建集合...")
    collection = client.create_collection(
        name="museum_local",
        embedding_function=embedding_fn,
        metadata={"description": "艺术与科技展览数据库"}
    )

    # 导入数据
    print("从Excel导入数据...")

    try:
        excel_file = pd.ExcelFile(excel_path)
        sheet_names = excel_file.sheet_names
        print(f"找到 {len(sheet_names)} 个工作表: {sheet_names}")

        all_docs = []
        all_metas = []
        all_ids = []
        total_records = 0

        for sheet_name in sheet_names:
            print(f"\n  处理: {sheet_name}")

            df = pd.read_excel(excel_path, sheet_name=sheet_name, header=1)
            df.columns = [str(col).strip().replace('\n', '').replace('（简）', '') for col in df.columns]

            print(f"    数据行数: {len(df)}")

            for idx, row in df.iterrows():
                if pd.isna(row.get('作品名称', '')) or str(row.get('作品名称', '')).strip() == '':
                    continue

                # 构建完整的文档内容
                text_parts = []

                fields = [
                    ('作品名称', '作品名称：《{}》'),
                    ('设计作者', '设计作者：{}'),
                    ('指导老师', '指导老师：{}'),
                    ('类别标签', '类别标签：{}'),
                    ('呈现形式', '呈现形式：{}'),
                    ('作品描述', '作品描述：{}'),
                    ('创作时间', '创作时间：{}'),
                    ('设计动机', '设计动机：{}'),
                    ('灵感来源', '灵感来源：{}'),
                    ('设计目的/意义', '设计目的：{}'),
                    ('设计理念/风格', '设计理念：{}'),
                    ('视觉形式语言', '视觉形式语言：{}'),
                    ('技术特点', '技术特点：{}'),
                    ('预期效果', '预期效果：{}'),
                    ('创作历程', '创作历程：{}'),
                    ('面临的困难', '面临的困难：{}')
                ]

                for field_name, template in fields:
                    value = row.get(field_name, '')
                    if pd.notna(value) and str(value).strip():
                        text_parts.append(template.format(str(value).strip()))

                text_parts.append(f"所属展区：{sheet_name}")
                text = '\n'.join(text_parts)

                all_docs.append(text)
                all_metas.append({
                    "作品名称": str(row.get('作品名称', '')),
                    "设计作者": str(row.get('设计作者', '')),
                    "指导老师": str(row.get('指导老师', '')),
                    "类别标签": str(row.get('类别标签', '')),
                    "呈现形式": str(row.get('呈现形式', '')),
                    "创作时间": str(row.get('创作时间', '')),
                    "所属展区": sheet_name
                })
                all_ids.append(f"{sheet_name}_{idx}")
                total_records += 1

        # 批量导入
        if all_docs:
            print(f"\n导入 {total_records} 条记录...")
            batch_size = 100
            for i in range(0, len(all_docs), batch_size):
                end_idx = min(i + batch_size, len(all_docs))
                collection.add(
                    documents=all_docs[i:end_idx],
                    metadatas=all_metas[i:end_idx],
                    ids=all_ids[i:end_idx]
                )
                print(f"    已导入 {end_idx}/{len(all_docs)} 条记录")

            print(f"\n导入完成!")
            print(f"统计信息:")
            print(f"    - 总作品数: {total_records}")
            print(f"    - 展区数量: {len(sheet_names)}")
            print(f"    - 平均文档长度: {sum(len(d) for d in all_docs) / len(all_docs):.0f} 字符")
            print(f"    - 向量库位置: {chroma_path}")
        else:
            print("没有数据")
            return False

    except Exception as e:
        print(f"导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n向量数据库构建完成！")
    return True

if __name__ == "__main__":
    build_database()