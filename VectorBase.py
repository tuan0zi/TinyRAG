import json
import os

import numpy as np

from Embeddings import BaseEmbeddings, OpenAIEmbedding
from utils import ReadFiles


class VectorStore:
    """
    一个最简单的向量数据库。

    主要负责：

    1. 保存文档 Chunk
    2. 把每个 Chunk 转换成 Embedding
    3. 保存向量到磁盘
    4. 从磁盘加载向量
    5. 根据用户 Query 检索最相关的 Chunk
    """

    def __init__(
        self,
        document: list[str] | None = None,
    ) -> None:

        # 文档 Chunk
        if document is None:
            document = []

        self.document = document

        # 每个 Chunk 对应的 Embedding 向量
        self.vectors: list[list[float]] = []

    def get_vector(
        self,
        EmbeddingModel: BaseEmbeddings,
    ) -> list[list[float]]:
        """
        把所有文档 Chunk 转换成 Embedding 向量。
        """

        # 每次重新生成向量前先清空
        self.vectors = []

        # 逐个处理每一个 Chunk
        for i, doc in enumerate(self.document):

            print(
                f"正在生成第 {i + 1}/"
                f"{len(self.document)} 个向量..."
            )

            # Chunk -> Embedding
            vector = EmbeddingModel.get_embedding(
                doc
            )

            # 保存这个 Chunk 对应的向量
            self.vectors.append(
                vector
            )

        return self.vectors

    @classmethod
    def get_similarity(
        cls,
        vector1: list[float],
        vector2: list[float],
    ) -> float:
        """
        计算两个向量的余弦相似度。
        """

        return BaseEmbeddings.cosine_similarity(
            vector1,
            vector2,
        )

    def persist(
        self,
        path: str = "storage",
    ) -> None:
        """
        把文档 Chunk 和对应向量保存到磁盘。
        """


        # 如果 storage 文件夹不存在，
        # 就自动创建
        os.makedirs(
            path,
            exist_ok=True,
        )

        # 文档保存路径
        document_path = os.path.join(
            path,
            "document.json",
        )

        # 向量保存路径
        vector_path = os.path.join(
            path,
            "vectors.json",
        )

        # 保存文档 Chunk
        with open(
            document_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.document,
                f,
                ensure_ascii=False,
                indent=2,
            )

        # 保存向量
        with open(
            vector_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.vectors,
                f,
            )

        print(
            f"向量数据库已经保存到: {path}"
        )

    def load_vector(
        self,
        path: str = "storage",
    ) -> None:
        """
        从磁盘加载文档 Chunk 和向量。
        """

        document_path = os.path.join(
            path,
            "document.json",
        )

        vector_path = os.path.join(
            path,
            "vectors.json",
        )

        # 加载文档
        with open(
            document_path,
            "r",
            encoding="utf-8",
        ) as f:

            self.document = json.load(f)

        # 加载向量
        with open(
            vector_path,
            "r",
            encoding="utf-8",
        ) as f:

            self.vectors = json.load(f)

        print(
            f"已经从 {path} 加载向量数据库"
        )

        print(
            f"文档 Chunk 数量: "
            f"{len(self.document)}"
        )

        print(
            f"向量数量: "
            f"{len(self.vectors)}"
        )

    def query(
        self,
        query: str,
        EmbeddingModel: BaseEmbeddings,
        k: int = 1,
    ) -> list[str]:
        """
        根据用户问题，检索最相关的 k 个文档 Chunk。
        """

        # ----------------------------
        # 第 1 步
        # 用户问题 -> Query Embedding
        # ----------------------------

        query_vector = (
            EmbeddingModel.get_embedding(
                query
            )
        )

        # ----------------------------
        # 第 2 步
        # Query 向量和所有文档向量
        # 分别计算余弦相似度
        # ----------------------------

        result = np.array(
            [
                self.get_similarity(
                    query_vector,
                    vector,
                )
                for vector in self.vectors
            ]
        )

        # ----------------------------
        # 第 3 步
        # 找出相似度最高的 Top-K
        # ----------------------------

        top_k_index = (
            result
            .argsort()[-k:][::-1]  #argsort（）返回按照数值从小到大排序的索引
        )

        # ----------------------------
        # 第 4 步
        # 根据索引找到原始 Chunk
        # ----------------------------

        top_k_documents = (
            np.array(
                self.document
            )[top_k_index]
            .tolist()
        )

        return top_k_documents


# --------------------------------
# 测试代码
# --------------------------------
if __name__ == "__main__":

    # 1. 读取 data 目录中的文档
    docs = ReadFiles(
        "./data"
    ).get_content(
        max_token_len=50,
        cover_content=10,
    )

    print(
        f"\n一共得到 {len(docs)} 个 Chunk"
    )

    # 2. 创建向量数据库对象
    vector_store = VectorStore(
        docs
    )

    # 3. 创建 Embedding 模型
    embedding_model = OpenAIEmbedding()

    # 4. 所有 Chunk -> Embedding
    vector_store.get_vector(
        EmbeddingModel=embedding_model
    )

    print(
        f"\n一共生成 "
        f"{len(vector_store.vectors)} 个向量"
    )

    # 5. 保存向量数据库
    vector_store.persist(
        path="./storage"
    )

    # 6. 测试检索
    question = "RAG 的原理是什么？"

    result = vector_store.query(
        query=question,
        EmbeddingModel=embedding_model,
        k=2,
    )

    print(
        "\n用户问题："
    )
    print(question)

    print(
        "\n检索到的相关 Chunk："
    )

    for i, chunk in enumerate(result):
        print(
            f"\n========== Top {i + 1} =========="
        )
        print(chunk)