"""RAG 聊天路由 —— 把 test_rag.py 的逻辑搬进 FastAPI"""
from fastapi import APIRouter
from backend.models.response import APIResponse

router = APIRouter()


# ⚠️ 先保持简单：把 test_rag.py 里证明能用的函数搬过来
# 后续 Project 1 会替换为 LangChain 版本的检索链路
# 暂时用内存数据，验证接口通畅即可


@router.post("/chat", response_model=APIResponse[dict])
async def rag_chat(req: dict):
    """RAG 问答接口（v0.1 占位版）"""
    query = req.get("query", "")

    # TODO: 把 test_rag.py 里的 search() + generate() 搬过来
    # 目前先返回占位响应，确认骨架 + 新路由能正常工作

    return APIResponse(
        message="RAG 接口已就绪",
        data={"query": query, "answer": f"[占位回答] 你问的是：{query}"},
    )