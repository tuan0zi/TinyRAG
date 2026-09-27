from utils import ReadFiles
from Embeddings import OpenAIEmbedding
from VectorBase import VectorStore
from LLM import OpenAIChat


def build_vector_database():
    """
    第一次运行时：

    1. 读取文档
    2. 切分 Chunk
    3. 生成 Embedding
    4. 保存向量数据库

    返回：
        VectorStore 对象
    """

    print("========== 开始构建向量数据库 ==========")

    # --------------------------------
    # 第 1 步
    # 读取 data 目录并切分 Chunk
    # --------------------------------

    docs = ReadFiles(
        "./data"
    ).get_content(
        max_token_len=600,
        cover_content=150,  #除了第一个块之外 其他每个都加上前一个的最后cover_content长度个token
    )

    print(
        f"\n一共得到 {len(docs)} 个 Chunk"
    )

    # --------------------------------
    # 第 2 步
    # 创建向量数据库
    # --------------------------------

    vector_store = VectorStore(
        docs
    )

    # --------------------------------
    # 第 3 步
    # 创建 Embedding 模型
    # --------------------------------

    embedding_model = OpenAIEmbedding()

    # --------------------------------
    # 第 4 步
    # 所有 Chunk -> Embedding
    # --------------------------------

    vector_store.get_vector(
        EmbeddingModel=embedding_model
    )

    # --------------------------------
    # 第 5 步
    # 保存到 storage
    # --------------------------------

    vector_store.persist(
        path="./storage"
    )

    print(
        "\n========== 向量数据库构建完成 =========="
    )

    return vector_store


def load_vector_database():
    """
    如果已经提前生成过向量数据库，
    就直接从 storage 加载。

    不需要重新计算所有文档的 Embedding。
    """

    print(
        "========== 正在加载向量数据库 =========="
    )

    vector_store = VectorStore()

    vector_store.load_vector(
        path="./storage"
    )

    print(
        "========== 向量数据库加载完成 =========="
    )

    return vector_store


def main():
    """
    TinyRAG 主流程：

    用户问题
        ↓
    Query Embedding
        ↓
    VectorStore 检索
        ↓
    Top-K Chunk
        ↓
    拼接 Context
        ↓
    LLM
        ↓
    最终回答
    """

    # --------------------------------
    # 是否重新构建向量数据库
    # --------------------------------

    rebuild_vector_database = True

    # --------------------------------
    # 第 1 步
    # 创建 / 加载向量数据库
    # --------------------------------

    if rebuild_vector_database:

        vector_store = (
            build_vector_database()
        )

    else:

        vector_store = (
            load_vector_database()
        )

    # --------------------------------
    # 第 2 步
    # 创建 Embedding 模型
    #
    # 注意：
    # Query 使用的 Embedding 模型
    # 必须和文档建库时兼容
    # --------------------------------

    embedding_model = OpenAIEmbedding()

    # --------------------------------
    # 第 3 步
    # 创建 LLM
    # --------------------------------

    chat_model = OpenAIChat()

    # --------------------------------
    # 第 4 步
    # 获取用户问题
    # --------------------------------

    question = input(
        "\n请输入你的问题："
    )

    # --------------------------------
    # 第 5 步
    # 向量检索
    # --------------------------------

    retrieved_chunks = (
        vector_store.query(
            query=question,
            EmbeddingModel=embedding_model,
            k=2,
        )
    )

    # --------------------------------
    # 第 6 步
    # 打印检索结果
    # --------------------------------

    print(
        "\n========== 检索到的 Chunk =========="
    )

    for i, chunk in enumerate(
        retrieved_chunks
    ):

        print(
            f"\n---------- Top {i + 1} ----------"
        )

        print(chunk)

    # --------------------------------
    # 第 7 步
    # 把多个 Chunk 拼成一个 Context
    # --------------------------------

    context = "\n\n".join(
        retrieved_chunks      #由于content 接收字符串 所以要把字符串列表转换成字符串
    )

    # --------------------------------
    # 第 8 步
    # 调用 LLM
    # --------------------------------

    answer = chat_model.chat(
        prompt=question,
        history=[],
        content=context,
    )

    # --------------------------------
    # 第 9 步
    # 输出最终回答
    # --------------------------------

    print(
        "\n========== RAG 最终回答 =========="
    )

    print(answer)


if __name__ == "__main__":
    main()