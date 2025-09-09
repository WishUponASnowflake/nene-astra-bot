import lancedb
import asyncio  # 確保導入 asyncio
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
            logger.error("沒有可用的 Dashscope 服務提供商，無法初始化數據庫表。")
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
            logger.info(f"RAG 數據庫表 'chat_history' 創建成功，使用模型 '{self.provider.model_name}'，維度: {vector_dim}。")
        else:
            self.table = self.db.open_table("chat_history")
            logger.info(f"已連接到現有的 RAG 數據庫表 'chat_history'。")

    async def add_text(self, group_id: str, sender_id: str, sender_name: str, text_content: str, timestamp: int):
        if not self.provider or self.table is None: return
        try:
            # --- 主要修改點 ---
            # 使用 asyncio.to_thread 來異步執行同步的 SDK 調用
            vector = await asyncio.to_thread(self.provider.get_text_embedding, text_content)
            # ------------------
            data = {"vector": vector, "text": text_content, "image_path": "", "group_id": group_id, "sender_id": sender_id, "sender_name": sender_name, "timestamp": timestamp}
            self.table.add([data])
        except Exception as e:
            logger.error(f"調用 Dashscope 進行文本 embedding 或寫入數據庫失敗: {e}")

    async def add_image(self, group_id: str, sender_id: str, sender_name: str, image_path: Path, timestamp: int):
        if not self.provider or self.table is None: return
        try:
            # --- 主要修改點 ---
            # 使用 asyncio.to_thread
            vector = await asyncio.to_thread(self.provider.get_image_embedding, image_path)
            # ------------------
            data = {"vector": vector, "text": "[圖片消息]", "image_path": str(image_path), "group_id": group_id, "sender_id": sender_id, "sender_name": sender_name, "timestamp": timestamp}
            self.table.add([data])
        except Exception as e:
            logger.error(f"調用 Dashscope 進行圖片 embedding 或寫入數據庫失敗: {e}")
            
    async def query(self, query_text: str, group_id: str, top_k: int = 5, top_j: int = 3) -> list:
        if not self.provider or self.table is None: return []
        try:
            # --- 主要修改點 ---
            # 使用 asyncio.to_thread
            query_vector = await asyncio.to_thread(self.provider.get_text_embedding, query_text)
            # ------------------
            
            # 查詢文本記錄
            text_results = self.table.search(query_vector)\
                .where(f"group_id = '{group_id}' AND image_path = ''")\
                .limit(top_k)\
                .to_list()

            # 查詢圖片記錄
            image_results = self.table.search(query_vector)\
                .where(f"group_id = '{group_id}' AND text = '[圖片消息]'")\
                .limit(top_j)\
                .to_list()
            
            # 合併並排序
            results = text_results + image_results
            results.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return results
        except Exception as e:
            logger.error(f"查詢 RAG 數據庫失敗: {e}")
            return []