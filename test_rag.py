from dotenv import load_dotenv
load_dotenv()

import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 本地 embedding 模型：免费、离线、中文友好
# 首次运行会自动下载模型文件（约 400MB），之后走缓存
embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
client = OpenAI()

# ====== 准备文档 ======
documents = [
    """RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索和文本生成的 AI 技术。它的核心思想是：在 LLM 生成回答之前，先从外部知识库中检索相关文档，然后将检索到的文档作为上下文提供给 LLM，从而减少幻觉、提升回答的准确性。""",
    """FastAPI 是一个现代化的 Python Web 框架，用于构建 RESTful API。它的核心特性包括：自动生成 OpenAPI 文档、基于 Pydantic 的数据校验、依赖注入系统、异步支持。""",
    """LangChain 是一个用于构建 LLM 应用的框架。它提供了 Chains、Agents、Memory、Retrieval 等核心模块。""",
]

# ====== ① 文档切分 ======
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# ====== ② 向量化 ======
def get_embedding(text):
    """用本地 BGE 模型做向量化，不调 API"""
    return embedding_model.encode(text)

# ====== ③ 构建向量索引 ======
chunks = []
for doc in documents:
    chunks.extend(split_text(doc))
embeddings = [get_embedding(chunk) for chunk in chunks]
print(f"共 {len(chunks)} 个 chunk，每个向量维度：{len(embeddings[0])}")

# ====== ④ 检索 ======
def search(query, k=3):
    query_emb = get_embedding(query)
    similarities = [
        np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
        for doc_emb in embeddings
    ]
    top_k = np.argsort(similarities)[-k:][::-1]
    return [(chunks[i], similarities[i]) for i in top_k]  # 返回 chunk + 相似度分数

# ====== ⑤ 生成 ======
def generate(query):
    retrieved = search(query)
    print(f"\n检索到 {len(retrieved)} 条相关文档：")
    for i, (chunk, score) in enumerate(retrieved):
        print(f"  [{i+1}] 相似度={score:.3f} | {chunk[:80]}...")

    context = "\n---\n".join([c for c, _ in retrieved])
    prompt = f"""根据以下参考资料回答问题。如果资料中没有相关信息，请如实说"未找到相关信息"。

    参考资料：
    {context}
    
    问题：{query}
    回答："""
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="high",
        extra_body={
            "thinking": {"type": "enabled"},  # DeepSeek 特有参数，必须走 extra_body
        },
    )
    return response.choices[0].message.content

# ====== 测试 ======
if __name__ == "__main__":
    queries = [
        "什么是RAG？",
        "FastAPI 有哪些特点？",
        "LangChain 和 FastAPI 有什么关系？",  # 这个问题在文档里没有直接答案
    ]
    for q in queries:
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        print(f"{'='*60}")
        answer = generate(q)
        print(f"\n回答：{answer}")
