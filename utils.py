import os

import tiktoken
from pypdf import PdfReader


# 用于计算文本的 token 数量
# 书中的代码直接使用了 enc.encode() / enc.decode()
enc = tiktoken.get_encoding("cl100k_base")


class ReadFiles:
    """
    用于：
    1. 读取目录中的 txt / md / pdf 文件
    2. 将文档切分成多个 Chunk
    """

    def __init__(self, path: str):
        # 保存数据目录路径
        self.path = path

    @classmethod
    def read_pdf(cls, file_path: str) -> str:
        """
        读取 PDF 文件
        """
        reader = PdfReader(file_path)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    @classmethod
    def read_markdown(cls, file_path: str) -> str:
        """
        读取 Markdown 文件
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @classmethod
    def read_text(cls, file_path: str) -> str:
        """
        读取 txt 文件
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @classmethod
    def read_file_content(cls, file_path: str) -> str:
        """
        根据文件扩展名选择不同的读取方式
        """
        if file_path.endswith(".pdf"):
            return cls.read_pdf(file_path)

        elif file_path.endswith(".md"):
            return cls.read_markdown(file_path)

        elif file_path.endswith(".txt"):
            return cls.read_text(file_path)

        else:
            raise ValueError(
                f"Unsupported file type: {file_path}"
            )

    @classmethod
    def get_chunk(
        cls,
        text: str,
        max_token_len: int = 600,
        cover_content: int = 150,
    ):
        """
        将一篇长文本切成多个 Chunk

        max_token_len:
            每个 Chunk 最大 token 长度

        cover_content:
            相邻 Chunk 保留的重叠内容大小
        """

        # 最终保存所有 Chunk
        chunk_text = []

        # 当前 Chunk 已经有多少 token
        curr_len = 0

        # 当前正在构建的 Chunk
        curr_chunk = ""

        # 为 overlap 预留空间
        token_len = max_token_len - cover_content

        # 按换行符粗切分
        lines = text.splitlines()

        for line in lines:

            # 去掉行首和行尾空格
            line = line.strip()

            # 空行直接跳过
            if not line:
                continue

            # 当前这一行有多少 token
            line_len = len(enc.encode(line))

            # ----------------------------
            # 情况 1：
            # 单独一行就超过最大长度
            # ----------------------------
            if line_len > max_token_len:

                # 先保存之前正在构建的 Chunk
                if curr_chunk:
                    chunk_text.append(curr_chunk)

                    curr_chunk = ""
                    curr_len = 0

                # 把整行变成 token
                line_tokens = enc.encode(line)

                # 计算需要分成多少块
                num_chunks = (
                    len(line_tokens)
                    + token_len
                    - 1
                ) // token_len

                for i in range(num_chunks):

                    start_token = i * token_len

                    end_token = min(
                        start_token + token_len,
                        len(line_tokens),
                    )

                    # 当前 token 片段
                    chunk_tokens = line_tokens[
                        start_token:end_token
                    ]

                    # token → 文本
                    chunk_part = enc.decode(
                        chunk_tokens
                    )

                    # 除了第一块之外，
                    # 给当前块增加上一块的重叠内容
                    if i > 0 and chunk_text:

                        prev_chunk = chunk_text[-1]

                        if len(prev_chunk) > cover_content:
                            cover_part = prev_chunk[
                                -cover_content:
                            ]
                        else:
                            cover_part = prev_chunk

                        chunk_part = (
                            cover_part
                            + chunk_part
                        )

                    chunk_text.append(
                        chunk_part
                    )

                curr_chunk = ""
                curr_len = 0

            # ----------------------------
            # 情况 2：
            # 当前行还能塞进当前 Chunk
            # ----------------------------
            elif (
                curr_len
                + line_len
                + 1
                <= token_len
            ):

                if curr_chunk:
                    curr_chunk += "\n"

                    # 粗略把换行记为一个长度
                    curr_len += 1

                curr_chunk += line

                curr_len += line_len

            # ----------------------------
            # 情况 3：
            # 当前 Chunk 已经装不下这一行
            # ----------------------------
            else:

                # 保存旧 Chunk
                if curr_chunk:
                    chunk_text.append(
                        curr_chunk
                    )

                # 新 Chunk 加入上一块的覆盖内容
                if chunk_text:

                    prev_chunk = chunk_text[-1]

                    if len(prev_chunk) > cover_content:
                        cover_part = prev_chunk[
                            -cover_content:
                        ]
                    else:
                        cover_part = prev_chunk

                    curr_chunk = (
                        cover_part
                        + "\n"
                        + line
                    )

                    curr_len = (
                        len(
                            enc.encode(
                                cover_part
                            )
                        )
                        + 1
                        + line_len
                    )

                else:
                    curr_chunk = line
                    curr_len = line_len  #以token为长度单位

        # 最后一块别忘记保存   因为没装满的chunk没有保存 
        if curr_chunk:
            chunk_text.append(
                curr_chunk
            )

        return chunk_text

    def get_content(
        self,
        max_token_len: int = 600,
        cover_content: int = 150,
    ):
        """
        读取 self.path 目录下所有支持的文件，
        并把它们全部切成 Chunk。
        """

        all_chunks = []

        # 遍历目录及其子目录
        for root, _, files in os.walk(
            self.path
        ):
            for file_name in sorted(files):

                # 只处理支持的文件
                if not file_name.endswith(
                    (".txt", ".md", ".pdf")
                ):
                    continue

                file_path = os.path.join(
                    root,
                    file_name,
                )

                print(
                    f"正在读取: {file_path}"
                )

                # 文件 → 长字符串
                text = self.read_file_content(
                    file_path
                )

                # 长字符串 → 多个 Chunk
                chunks = self.get_chunk(
                    text,
                    max_token_len=max_token_len,
                    cover_content=cover_content,
                )

                all_chunks.extend(chunks)

        return all_chunks