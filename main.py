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
    "sdy_zjx", # 作者名
    "一个使用 Dashscope 的 RAG 数据库插件 (API Key from Env)",
    "4.2.0", 
    "https://github.com/your-repo/astrbot_plugin_rag_collector"
)
class RAGCollectorPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        
        provider = None
        try:
            # 从插件配置中读取 dashscope 的配置 (现在主要为 model_name)
            ds_config = self.config.get("dashscope_config", {})
            
            # 创建 DashscopeProvider 实例。
            provider = DashscopeProvider(config=ds_config)
            
            logger.info(f"插件 'rag_collector' 成功创建 Dashscope 提供商，模型: {provider.model_name}。")

        except Exception as e:
            # 捕获因环境变量未设置或其他原因导致的初始化失败
            logger.error(f"插件 'rag_collector' 创建 Dashscope 提供商失败: {e}")
            logger.error("请确保您已在环境变量中正确设置了 DASHSCOPE_API_KEY。")

        # 设置文件和数据库路径
        plugin_data_path = Path("data") / "rag_collector_plugin"
        db_path = plugin_data_path / "lancedb"
        image_path = plugin_data_path / "images"
        
        # 初始化数据库实例
        self.db = RAGDatabase(db_path=db_path, embedding_provider=provider)
        
        # 初始化事件处理器
        self.handler = EventHandler(
            context=context, 
            config=self.config,
            rag_db=self.db, 
            image_save_path=image_path
        )
        
        # 进行数据库表的初始化
        self.db.init_table()
        
        logger.info("RAG Collector 插件已加载。")

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message_listener(self, event: AstrMessageEvent):
        """
        [注册] 被动监听所有群消息，并交由 handler 的新流程处理。
        """
        async for result in self.handler.handle_new_message(event):
            yield result

    @filter.command("rag_search", alias={'搜索记录', 'rag'})
    async def search_rag_records(self, event: AstrMessageEvent, *, query: str):
        """
        [注册] /rag_search 指令，将请求交由 handler 处理并返回结果。
        """
        async for result in self.handler.process_rag_search_command(event, query):
            yield result

    async def terminate(self):
        """
        插件卸载/停用时调用。
        """
        logger.info("RAG Collector 插件已卸载。")