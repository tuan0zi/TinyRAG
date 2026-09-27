import os

import numpy as np
from openai import OpenAI


class BaseEmbeddings:
    """
    Embedding 模型的基类。

    主要定义两件事：

    1. get_embedding()
       文本 -> 向量

    2. cosine_similarity()
       计算两个向量之间的余弦相似度
    """

    def __init__(
        self,
        path: str = "",
        is_api: bool = True,
    ) -> None:
        # 模型路径
        # 如果以后使用本地 Embedding 模型，可以使用这个变量
        self.path = path

        # 是否通过 API 调用 Embedding 模型
        self.is_api = is_api

    def get_embedding(
        self,
        text: str,
        model: str = "",
    ) -> list[float]:
        """
        输入文本，返回 Embedding 向量。

        BaseEmbeddings 只是定义接口，
        真正的实现交给子类。
        """
        raise NotImplementedError(
            "子类必须实现 get_embedding() 方法"
        )

    @classmethod
    def cosine_similarity(
        cls,
        vector1: list[float],
        vector2: list[float],
    ) -> float:
        """
        计算两个向量的余弦相似度。
        """

        # Python list -> numpy array
        v1 = np.array(
            vector1,
            dtype=np.float32,
        )

        v2 = np.array(
            vector2,
            dtype=np.float32,
        )

        # 检查向量中是否存在 NaN / inf
        if (
            not np.all(np.isfinite(v1))
            or not np.all(np.isfinite(v2))
        ):
            return 0.0

        # 点积
        dot_product = np.dot(
            v1,
            v2,
        )

        # 分别计算两个向量的模长
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        # 两个模长相乘
        magnitude = norm_v1 * norm_v2

        # 防止除以 0
        if magnitude == 0:
            return 0.0

        # 余弦相似度
        return float(
            dot_product / magnitude
        )


class OpenAIEmbedding(BaseEmbeddings):
    """
    使用 OpenAI 兼容 API 获取 Embedding。

    注意：
    这里使用 openai Python SDK，
    不代表一定调用 OpenAI 官方模型。

    只要某个平台兼容 OpenAI API 格式，
    就可以通过 base_url 调用。
    """

    def __init__(
        self,
        path: str = "",
        is_api: bool = True,
        model: str = "BAAI/bge-m3",
    ):
        # 调用父类初始化方法
        super().__init__(
            path=path,
            is_api=is_api,
        )

        # 保存默认使用的 Embedding 模型名字
        self.model = model

        if self.is_api:

            # 从环境变量读取 API Key
            api_key = os.getenv(
                "OPENAI_API_KEY"
            )

            # 从环境变量读取 API 地址
            base_url = os.getenv(
                "OPENAI_BASE_URL"
            )

            # 如果没有设置 API Key，直接报错
            if not api_key:
                raise ValueError(
                    "没有找到环境变量 OPENAI_API_KEY"
                )

            # 如果没有设置 Base URL，给出明确提示
            if not base_url:
                raise ValueError(
                    "没有找到环境变量 OPENAI_BASE_URL"
                )

            # 创建 OpenAI 客户端
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )

    def get_embedding(
        self,
        text: str,
        model: str = "",
    ) -> list[float]:
        """
        文本 -> Embedding 向量
        """

        if not self.is_api:
            raise NotImplementedError(
                "当前只实现了 API Embedding"
            )

        # 如果调用函数时没有单独指定模型，
        # 就使用 __init__ 中保存的默认模型
        if not model:
            model = self.model

        # 把换行符替换为空格
        text = text.replace(
            "\n",
            " ",
        )

        # 调用 Embedding API
        response = (
            self.client.embeddings.create(
                input=[text],
                model=model,
            )
        )

        # API 返回的数据中取出第一个文本的向量
        embedding = (
            response
            .data[0]
            .embedding
        )

        return embedding


# ----------------------------
# 测试代码
# ----------------------------
if __name__ == "__main__":

    embedding_model = OpenAIEmbedding()

    text1 = "RAG 是一种检索增强生成技术。"

    text2 = "RAG 会先从知识库中检索相关内容。"

    text3 = "今天天气很好，我准备出去跑步。"

    print("正在获取 Embedding...")

    vector1 = embedding_model.get_embedding(
        text1
    )

    vector2 = embedding_model.get_embedding(
        text2
    )

    vector3 = embedding_model.get_embedding(
        text3
    )

    print("\nvector1 类型：")
    print(type(vector1))

    print("\nEmbedding 维度：")
    print(len(vector1))

    print("\nvector1 前 10 个数字：")
    print(vector1[:10])

    similarity_12 = (
        BaseEmbeddings.cosine_similarity(
            vector1,
            vector2,
        )
    )

    similarity_13 = (
        BaseEmbeddings.cosine_similarity(
            vector1,
            vector3,
        )
    )

    print("\n文本 1 和文本 2 的相似度：")
    print(similarity_12)

    print("\n文本 1 和文本 3 的相似度：")
    print(similarity_13)