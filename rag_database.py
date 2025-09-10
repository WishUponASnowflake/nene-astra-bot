import lancedb
import asyncio  # 确保导入 asyncio
from PIL import Image
from pathlib import Path
from astrbot.api import logger

from pydantic import BaseModel
from lancedb.pydantic import LanceModel, Vector

from .dashscope_provider import DashscopeProvider

class RAGDatabase:
    def __init__(self, db_path: Path, embedding_provider: DashscopeProvider | None):
        self.db_path = db_path
        self.db = lancedb.connect(self.db_path)
        self.table = None
        self.schema = None
        self.provider = embedding_provider

    def init_table(self):
        if not self.provider:
            logger.error("没有可用的 Dashscope 服务提供商，无法初始化数据库表。")
            return

        vector_dim = self.provider.embedding_dimension
        class ChatRecord(LanceModel):
            vector: Vector(vector_dim)
            text: str
            image_path: str
            group_id: str
            sender_id: str
            sender_name: str
            timestamp: int
        self.schema = ChatRecord
        
        table_names = self.db.table_names()
        if "chat_history" not in table_names:
            self.table = self.db.create_table("chat_history", schema=self.schema, mode="overwrite")
            logger.info(f"RAG 数据库表 'chat_history' 创建成功，使用模型 '{self.provider.model_name}'，维度: {vector_dim}。")
        else:
            self.table = self.db.open_table("chat_history")
            logger.info(f"已连接到现有的 RAG 数据库表 'chat_history'。")

    async def add_text(self, group_id: str, sender_id: str, sender_name: str, text_content: str, timestamp: int):
        if not self.provider or self.table is None: return
        try:
            # --- 主要修改点 ---
            # 使用 asyncio.to_thread 来异步执行同步的 SDK 调用
            vector = await asyncio.to_thread(self.provider.get_text_embedding, text_content)
            # ------------------
            data = {"vector": vector, "text": text_content, "image_path": "", "group_id": group_id, "sender_id": sender_id, "sender_name": sender_name, "timestamp": timestamp}
            self.table.add([data])
        except Exception as e:
            logger.error(f"调用 Dashscope 进行文本 embedding 或写入数据库失败: {e}")

    async def add_image(self, group_id: str, sender_id: str, sender_name: str, image_path: Path, timestamp: int):
        if not self.provider or self.table is None: return
        try:
            # --- 主要修改点 ---
            # 使用 asyncio.to_thread
            vector = await asyncio.to_thread(self.provider.get_image_embedding, image_path)
            # ------------------
            data = {"vector": vector, "text": "[图片消息]", "image_path": str(image_path), "group_id": group_id, "sender_id": sender_id, "sender_name": sender_name, "timestamp": timestamp}
            self.table.add([data])
        except Exception as e:
            logger.error(f"调用 Dashscope 进行图片 embedding 或写入数据库失败: {e}")
            
    async def query(self, query_text: str, group_id: str, top_k: int = 5, top_j: int = 3) -> list:
        if not self.provider or self.table is None: return []
        try:
            # 使用 asyncio.to_thread
            query_vector = await asyncio.to_thread(self.provider.get_text_embedding, query_text)
            
            # --- 核心修改点 ---
            # 查询文本记录，移除了 group_id 的限制
            text_results = self.table.search(query_vector)\
                .where(f"group_id = '{group_id}' AND image_path = ''")\
                .limit(top_k)\
                .to_list()

            # 查询图片记录，移除了 group_id 的限制
            image_results = self.table.search(query_vector)\
                .where(f"group_id = '{group_id}' AND text = '[图片消息]'")\
                .limit(top_j)\
                .to_list()
            # ------------------
            
            # 合并并排序
            results = text_results + image_results
            results.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return results
        except Exception as e:
            logger.error(f"查询 RAG 数据库失败: {e}")
            return []