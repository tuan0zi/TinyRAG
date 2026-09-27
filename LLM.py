import os

from openai import OpenAI


# --------------------------------
# RAG Prompt 模板
# --------------------------------

RAG_PROMPT_TEMPLATE = """
使用以下上下文来回答用户的问题。

如果你不知道答案，就说你不知道。
总是使用中文回答。

问题：
{question}

可参考的上下文：
--------------------
{context}
--------------------

如果给定的上下文无法让你做出回答，
请回答：数据库中没有这个内容，我不知道。

有用的回答：
"""


class BaseModel:
    """
    LLM 基类。

    主要定义：

    1. chat()
       输入问题 + 历史对话 + 检索上下文
       返回模型回答

    2. load_model()
       预留给以后本地加载模型使用
    """

    def __init__(
        self,
        path: str = "",
    ):
        # 如果以后使用本地模型，
        # 可以保存模型路径
        self.path = path

    def chat(
        self,
        prompt: str,
        history: list[dict],
        content: str,
    ) -> str:
        """
        调用大模型生成回答。
        """

        raise NotImplementedError(
            "子类必须实现 chat() 方法"
        )

    def load_model(self):
        """
        预留：
        如果以后加载本地 LLM，
        可以在这里实现。
        """

        raise NotImplementedError(
            "当前没有实现本地模型加载"
        )


class OpenAIChat(BaseModel):
    """
    使用 OpenAI-Compatible API 调用大语言模型。
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-32B-Instruct",
    ):
        # 调用父类初始化
        super().__init__()

        # 保存模型名称
        self.model = model

        # 读取 API Key
        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        # 读取 API 地址
        base_url = os.getenv(
            "OPENAI_BASE_URL"
        )

        # 检查 API Key
        if not api_key:
            raise ValueError(
                "没有找到环境变量 OPENAI_API_KEY"
            )

        # 检查 API 地址
        if not base_url:
            raise ValueError(
                "没有找到环境变量 OPENAI_BASE_URL"
            )

        # 创建 API 客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def chat(
        self,
        prompt: str,
        history: list[dict],
        content: str,
    ) -> str:
        """
        prompt:
            用户问题

        history:
            历史对话消息

        content:
            VectorStore 检索出来的相关知识
        """

        # --------------------------------
        # 第 1 步
        # 构造 RAG Prompt
        # --------------------------------

        rag_prompt = (
            RAG_PROMPT_TEMPLATE.format(
                question=prompt,
                context=content,
            )
        )

        # --------------------------------
        # 第 2 步
        # 把本轮用户问题加入历史消息
        # --------------------------------

        history.append(
            {
                "role": "user",
                "content": rag_prompt,
            }
        )

        # --------------------------------
        # 第 3 步
        # 调用大模型 API
        # --------------------------------

        response = (
            self.client
            .chat
            .completions
            .create(
                model=self.model,
                messages=history,
                max_tokens=2048,
                temperature=0.1,
            )
        )

        # --------------------------------
        # 第 4 步
        # 取出模型生成的文本
        # --------------------------------

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        return answer


# --------------------------------
# 测试代码
# --------------------------------

if __name__ == "__main__":

    # 创建 LLM 对象
    chat_model = OpenAIChat()

    # 模拟用户问题
    question = "RAG 的原理是什么？"

    # 模拟 VectorStore 检索出来的内容
    context = """
RAG 的全称是 Retrieval-Augmented Generation。

RAG 会先根据用户问题从外部知识库中检索相关内容，
然后将检索结果和用户问题一起交给大语言模型生成最终回答。
"""

    # 暂时没有历史对话
    history = []

    # 调用大模型
    answer = chat_model.chat(
        prompt=question,
        history=history,
        content=context,
    )

    print("\n用户问题：")
    print(question)

    print("\n检索上下文：")
    print(context)

    print("\n模型回答：")
    print(answer)