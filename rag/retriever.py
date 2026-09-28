"""
主链路兼容入口。

当前 Web / ROS 主链路仍然从 `rag.retriever` 导入 `MuseumRetriever`，
这里直接转发到最新的混合检索 + 重排序实现，避免上层服务继续依赖旧版检索器。
"""

from rag.retriever_v2_mix_Reranking import MuseumRetriever, Retriever

__all__ = ["MuseumRetriever", "Retriever"]
