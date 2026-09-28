# 展品知识检索系统 (RAG) - 完整数据链路版

本系统是一套基于混合检索（向量 + BM25）及深度语义重排序（Reranking）的知识检索架构，旨在为人形机器人或其他智能终端提供精准、高效的展品知识检索服务。



## 1. 核心架构职责

| 文件/路径 | 核心职责 |
| :--- | :--- |
| `rag/excel_to_json.py` | **数据 ETL 层**：将原始 Excel 清洗、标准化为 `standard.json`。 |
| `rag/ingest_from_json.py` | **索引构建层**：调用 `Qwen3-Embedding` 将 JSON 数据向量化并存入 ChromaDB。 |
| `rag/retriever_v2_mix_Reranking.py` | **检索核心引擎**：集成了混合召回（向量+BM25）与重排序逻辑。 |
| `rag/test_batch_mixreranking.py` | **测试基准**：评估检索质量、相关度及召回/重排耗时指标。 |
| `models/` | **权重仓库**：存放 `Qwen3-Embedding` 及 `BGE-Reranker` 模型权重。 |

## 2. 环境依赖

请在项目根目录下安装必要的环境依赖：

```bash
pip install -r requirements.txt
```

*注意：依赖包含 `pandas`, `openpyxl`, `chromadb`, `sentence-transformers` 等。*

## 3. 快速部署流程 (Data Pipeline)

如果数据发生了变动，请按照以下顺序完成从数据到索引的闭环：

### Step 1: 清洗数据
将原始 Excel 转为系统要求的标准 JSON 格式：
```bash
python -m rag.excel_to_json --input data/raw/data.xlsx --output data/processed/standard.json
```

### Step 2: 构建索引
将 JSON 内容向量化并入库（会自动生成 `data/chroma_db_qwen3` 文件夹）：
```bash
python -m rag.ingest_from_json
```

### Step 3: 运行验证
执行单次查询或批量测试：
```bash
# 单次查询测试
python -m rag.retriever_v2_mix_Reranking -q "给我讲一下永栖所"

# 批量性能测试
python -m rag.test_batch_mixreranking
```

## 4. 核心接口说明 (MuseumRetriever)

在集成开发时，直接通过 `retriever_v2_mix_Reranking.py` 中的 `MuseumRetriever` 类调用：

- **`hybrid_query(query, top_k=3)`**：完整链路检索，包含向量召回、BM25 召回、重排序。返回 `(results, metrics)`。
- **`vector_search(query, collection_name, n_results=5)`**：针对特定集合的原子向量召回。
- **`rerank(query, candidates)`**：使用重排序模型进行深度语义精排。
- **`get_stats()`**：返回当前知识库规模、引擎类型等系统状态。
- **`rebuild_index()`**：刷新内存中的 BM25 倒排索引。

## 5. Python 集成示例 (Integration Example)

若需在其他 Python 服务（如大模型 Agent）中调用本检索系统，请参考以下范例：

```python
from rag.retriever_v2_mix_Reranking import MuseumRetriever

# 1. 实例化检索器 (建议设为全局单例)
retriever = MuseumRetriever()

# 2. 封装检索函数
def get_rag_context(user_query: str) -> str:
    """给大模型用的函数：输入用户问题，返回知识库上下文"""
    # 执行混合检索 (top_k=2 表示取最相关的2条结果)
    results, metrics = retriever.hybrid_query(user_query, top_k=2)
    
    # 融合并返回大模型可直接阅读的文本
    return retriever.fuse_knowledge(results)

# 3. 使用示例
user_input = "给我讲一下永栖所"
context = get_rag_context(user_input)

# 将 context 拼接到你的 LLM Prompt 中即可
print(f"=== 准备发送给大模型的上下文 ===\n{context}")
```

---
*注：本系统采用本地模型预加载模式。若 `models/` 文件夹缺失，系统启动时会自动尝试从 HuggingFace 自动下载对应模型，请确保网络通畅。*
