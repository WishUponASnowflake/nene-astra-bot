import aiohttp
import uuid
import asyncio
import os
from pathlib import Path
from datetime import datetime

import astrbot.api.message_components as Comp
from astrbot.api.event import AstrMessageEvent
from astrbot.api import logger, AstrBotConfig
from astrbot.api.star import Context

from .rag_database import RAGDatabase

class EventHandler:
    def __init__(self, context: Context, config: AstrBotConfig, rag_db: RAGDatabase, image_save_path: Path):
        self.context = context
        self.config = config
        self.rag_db = rag_db
        self.image_save_path = image_save_path
        self.image_save_path.mkdir(parents=True, exist_ok=True)
        # 從配置中讀取圖片數量上限
        self.max_local_images = self.config.get("max_local_images", 50)

    def _cleanup_old_images(self):
        """
        [功能] 清理舊的圖片文件，維持本地緩存數量不超過上限。
        """
        try:
            # 獲取目錄下所有的圖片文件
            files = [f for f in self.image_save_path.iterdir() if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.gif')]
            
            # 如果文件數量超過上限
            if len(files) > self.max_local_images:
                # 按修改時間排序，最舊的在前
                files.sort(key=lambda x: x.stat().st_mtime)
                
                # 計算需要刪除的文件數量
                files_to_delete_count = len(files) - self.max_local_images
                files_to_delete = files[:files_to_delete_count]
                
                logger.info(f"本地圖片數量 {len(files)} 已超過上限 {self.max_local_images}，正在清理 {len(files_to_delete)} 張最舊的圖片...")
                for f in files_to_delete:
                    try:
                        f.unlink() # 刪除文件
                    except Exception as e:
                        logger.error(f"刪除舊圖片 {f} 失敗: {e}")
        except Exception as e:
            logger.error(f"執行圖片清理任務時發生錯誤: {e}")

    async def _download_image(self, url: str) -> Path | None:
        """
        異步從 URL 下載圖片並保存到本地。
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        file_name = f"{uuid.uuid4()}.jpg"
                        save_path = self.image_save_path / file_name
                        with open(save_path, 'wb') as f:
                            f.write(await resp.read())
                        return save_path
        except Exception as e:
            logger.error(f"下載圖片時發生異常: {e}")
            return None

    async def handle_group_message_logging(self, event: AstrMessageEvent):
        """
        [業務邏輯] 處理群聊消息，提取文本和圖片，調用數據庫層進行存儲，並觸發圖片清理。
        """
        group_id = event.get_group_id()
        sender_id = event.get_sender_id()
        sender_name = event.get_sender_name()
        timestamp = event.message_obj.timestamp
        
        for component in event.message_obj.message:
            if isinstance(component, Comp.Plain):
                text_content = component.text.strip()
                if text_content:
                    await self.rag_db.add_text(group_id, sender_id, sender_name, text_content, timestamp)
            elif isinstance(component, Comp.Image):
                if image_url := component.url:
                    if saved_path := await self._download_image(image_url):
                        await self.rag_db.add_image(group_id, sender_id, sender_name, saved_path, timestamp)
                        # 每次成功保存新圖片後，觸發一次清理檢查
                        self._cleanup_old_images()

    async def process_rag_search_command(self, event: AstrMessageEvent, query: str):
        """
        [業務邏輯] 處理 RAG 搜索指令，並將結果格式化後發送（文本直接發，圖片發送圖片文件）。
        """
        group_id = event.get_group_id()
        if not group_id:
            yield event.plain_result("抱歉，此命令只能在群聊中使用。")
            return

        top_k = self.config.get("top_k_results", 5)
        #yield event.plain_result(f"正在為你在群聊 {group_id} 中搜索“{query}”的相關記錄...")
        
        results = await self.rag_db.query(query, group_id, top_k)
        
        if not results:
            yield event.plain_result("沒有找到相關的聊天記錄。")
            return
            
        #yield event.plain_result(f"找到關於“{query}”的最相關的 {len(results)} 條記錄：")

        # 遍歷結果並逐條發送
        for res in results:
            ts = datetime.fromtimestamp(res['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            sender_info = f"[{ts}] {res['sender_name']}:"
            #logger.info(res['text'])
            # 檢查是否是圖片記錄
            image_path_str = res.get('image_path')
            if res['text'] == "[圖片消息]":
                image_path = Path(image_path_str)
                #logger.info(image_path_str)
                # 檢查本地圖片文件是否還存在
                if image_path.exists():
                    # 如果存在，發送“發送者信息” + 圖片
                    yield event.chain_result([
                        Comp.Plain(sender_info),
                        Comp.Image.fromFileSystem(image_path_str) # 從本地文件系統發送圖片
                    ])
                else:
                    # 如果圖片因被清理而不再存在
                    yield event.plain_result(f"{sender_info}\n[一張已過期的圖片] (ID: {image_path.stem})")
            else:
                # 如果是文本記錄，直接發送
                text_content = res['text']
                #yield event.plain_result(f"{sender_info} {text_content}")
            
            # 每條消息之間稍微停頓一下，避免刷屏
            await asyncio.sleep(0.5)