import asyncio
from pathlib import Path

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, AstrBotConfig

from .rag_database import RAGDatabase
from .event_handler import EventHandler
from .dashscope_provider import DashscopeProvider

@register(
    "rag_collector",
    "您的名字",
    "一個使用 Dashscope 的 RAG 數據庫插件 (API Key from Env)",
    "4.2.0", # 版本號更新
    "https://github.com/your-repo/astrbot_plugin_rag_collector"
)
class RAGCollectorPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        
        provider = None
        try:
            # 從插件配置中讀取 dashscope 的配置 (現在主要為 model_name)
            ds_config = self.config.get("dashscope_config", {})
            
            # 創建 DashscopeProvider 實例。
            # 這個類的 __init__ 方法會自動檢查環境變量 DASHSCOPE_API_KEY。
            # 如果未設置，它將會拋出 ValueError，並被下面的 except 捕獲。
            provider = DashscopeProvider(config=ds_config)
            
            logger.info(f"插件 'rag_collector' 成功創建 Dashscope 提供商，模型: {provider.model_name}。")

        except Exception as e:
            # 捕獲因環境變量未設置或其他原因導致的初始化失敗
            logger.error(f"插件 'rag_collector' 創建 Dashscope 提供商失敗: {e}")
            logger.error("請確保您已在環境變量中正確設置了 DASHSCOPE_API_KEY。")

        # 設置文件和數據庫路徑
        plugin_data_path = Path("data") / "rag_collector_plugin"
        db_path = plugin_data_path / "lancedb"
        image_path = plugin_data_path / "images"
        
        # 初始化數據庫實例，並傳入我們創建的 provider
        # 即使 provider 為 None，程序也不會崩潰，只是後續功能無法使用
        self.db = RAGDatabase(db_path=db_path, embedding_provider=provider)
        
        # 初始化事件處理器
        self.handler = EventHandler(
            context=context, 
            config=self.config,
            rag_db=self.db, 
            image_save_path=image_path
        )
        
        # 進行數據庫表的初始化
        self.db.init_table()
        
        logger.info("RAG Collector 插件已加載。")

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message_listener(self, event: AstrMessageEvent):
        """
        [註冊] 被動監聽所有群消息，並創建後台任務交由 handler 處理。
        """
        asyncio.create_task(self.handler.handle_group_message_logging(event))

    @filter.command("rag_search", alias={'搜索記錄', 'rag'})
    async def search_rag_records(self, event: AstrMessageEvent, *, query: str):
        """
        [註冊] /rag_search 指令，將請求交由 handler 處理並返回結果。
        """
        async for result in self.handler.process_rag_search_command(event, query):
            yield result

    async def terminate(self):
        """
        插件卸載/停用時調用。
        """
        logger.info("RAG Collector 插件已卸載。")