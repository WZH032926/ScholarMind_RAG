import os
import chromadb
from openai import OpenAI


class MultiDocRAGBot:
    def __init__(self, api_key, base_url, db_path="./chroma_db"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # 1. 初始化 ChromaDB 持久化客户端
        print(f"📦 连接向量数据库: {db_path}...")
        self.chroma_client = chromadb.PersistentClient(path=db_path)

        # 2. 获取或创建 Collection (类似于关系型数据库里的 Table)
        # 默认使用 L2 (欧式距离) 或 Cosine (余弦相似度)
        self.collection = self.chroma_client.get_or_create_collection(
            name="scholar_papers",
            metadata={"hnsw:space": "cosine"}  # 指定使用余弦相似度
        )
        print(f"✅ 数据库连接成功，当前共有 {self.collection.count()} 个数据块。")

    def get_embedding(self, text):
        """调用云端大模型获取向量"""
        response = self.client.embeddings.create(
            input=text.replace("\n", " "),
            model="embedding-2"  # 确保和你的模型名称一致
        )
        return response.data[0].embedding

    def add_document(self, filename, chunks):
        """
        【全新核心逻辑】将切分好的文本块注入向量数据库
        """
        print(f"📥 正在将 {filename} 写入向量数据库...")

        # 准备 ChromaDB 需要的数据结构
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            # 给每个块生成一个全球唯一的 ID (文件名 + 序号)
            chunk_id = f"{filename}_chunk_{i}"

            # 查重：如果库里已经有这个文件了，就不重复算了 (节省 Token 费！)
            existing = self.collection.get(ids=[chunk_id])
            if existing and len(existing['ids']) > 0:
                continue

                # 只有不存在的数据，才去算向量
            vector = self.get_embedding(chunk)

            ids.append(chunk_id)
            documents.append(chunk)
            embeddings.append(vector)
            metadatas.append({"source": filename})  # 存入元数据

            # 批量写入数据库
            if ids:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                print(f"✅ 成功写入 {len(ids)} 个新数据块！")
            else:
                print(f"⏭️ 文档 {filename} 已存在，跳过写入。")

    def retrieve(self, query, top_k=3):
        """
        【算法重构】使用 ChromaDB 自带的 HNSW 索引进行极速检索
        """
        query_vector = self.get_embedding(query)

        # 直接调用底层的 query 方法
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )

        # ChromaDB 返回的格式很嵌套，我们把它解析成前端能看懂的列表格式
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "text": results['documents'][0][i],
                    "source": results['metadatas'][0][i]['source'],
                    # ChromaDB 余弦距离越小越相似，为了统一，我们转换为相似度得分
                    "score": 1.0 - results['distances'][0][i]
                })
        return formatted_results

    def _translate_query(self, query):
        response = self.client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "system",
                 "content": "你是一个精准的翻译官。请将用户的提问翻译成英文。直接输出翻译后的句子，不要包含任何解释。"},
                {"role": "user", "content": query}
            ],
            temperature=0
        )
        return response.choices[0].message.content

    def chat(self, query):
        """对外主入口"""
        print(f"\n 用户提问: {query}")

        # 0. 跨语言检索
        search_query = self._translate_query(query)
        print(f"内部检索词已转换: {search_query}")

        # 1.检索
        print(" 正在检索...")
        context = self.retrieve(search_query)
        # 检索到内容再回答或者相关性低，节省token
        if not context or context[0]['score'] < 0.4:
            return {"answer": "抱歉，知识库中没有找到相关内容。",
                    "source": []
                    }

        # 2.构建提示词 之前是封装
        context_str = ""
        for i, chunk in enumerate(context):
            context_str += f"<doc id='{i + 1}'>{chunk['text']}</doc>\n"

        prompt = f"""
    你是一个双语科研助手。请基于以下提供的【英文参考文档】回答用户的【中文问题】。

    【参考文档】：
    {context_str}

    【用户问题】：
    {query}

    【回答要求】：
    1. 必须用流利专业的**中文**回答。
    2. 你的回答必须完全基于【参考文档】的内容，不要编造。
    3. 每一个关键论点后面，必须用 [doc id] 的格式标注来源，例如：[doc 1]。
    """

        # 5. 调用大模型生成回答
        print("✍️ 正在生成最终回答...")
        response = self.client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "system", "content": "你是一个严谨的科研助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        # 6. 【核心】返回字典，把答案和参考的文档块一起交出去
        return {
            "answer": response.choices[0].message.content,
            "sources": context
        }