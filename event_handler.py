import aiohttp
import uuid
import asyncio
import os
import random
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
        self.max_local_images = self.config.get("max_local_images", 50)
        
        # 读取回复意愿判断模型的配置
        self.judgement_provider_id = self.config.get("judgement_model", {}).get("provider_id")
        self.reply_possibility = self.config.get("reply_possibility", 0.1)
        
        # 读取最终回复大模型和人格的配置
        main_llm_config = self.config.get("main_llm_provider", {})
        self.main_llm_provider_id = main_llm_config.get("provider_id")
        self.persona_prompt = main_llm_config.get("persona_prompt", "你是一个友好、乐于助人的群聊助手。")


    def _cleanup_old_images(self):
        """
        [功能] 清理旧的图片文件，维持本地缓存数量不超过上限。
        """
        try:
            files = [f for f in self.image_save_path.iterdir() if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.gif')]
            if len(files) > self.max_local_images:
                files.sort(key=lambda x: x.stat().st_mtime)
                files_to_delete_count = len(files) - self.max_local_images
                files_to_delete = files[:files_to_delete_count]
                logger.info(f"本地图片数量 {len(files)} 已超过上限 {self.max_local_images}，正在清理 {len(files_to_delete)} 张最旧的图片...")
                for f in files_to_delete:
                    try:
                        f.unlink()
                    except Exception as e:
                        logger.error(f"删除旧图片 {f} 失败: {e}")
        except Exception as e:
            logger.error(f"执行图片清理任务时发生错误: {e}")

    async def _download_image(self, url: str) -> Path | None:
        """
        异步从 URL 下载图片并保存到本地。
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
            logger.error(f"下载图片时发生异常: {e}")
            return None

    async def handle_group_message_logging(self, event: AstrMessageEvent):
        """
        [业务逻辑] 处理群聊消息，提取文本和图片，调用数据库层进行存储，并触发图片清理。
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
                        self._cleanup_old_images()

    async def process_rag_search_command(self, event: AstrMessageEvent, query: str):
        """
        [业务逻辑] 处理 RAG 搜索指令，并将结果格式化后发送（文本直接发，图片发送图片文件）。
        """
        group_id = event.get_group_id()
        if not group_id:
            yield event.plain_result("抱歉，此命令只能在群聊中使用。")
            return

        top_k = self.config.get("top_k_text_results", 5)
        top_j = self.config.get("top_j_image_results", 3)
        
        results = await self.rag_db.query(query, group_id, top_k, top_j)
        
        if not results:
            yield event.plain_result("没有找到相关的聊天记录。")
            return
            
        for res in results:
            ts = datetime.fromtimestamp(res['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            sender_info = f"[{ts}] {res['sender_name']}:"
            image_path_str = res.get('image_path')
            if res['text'] == "[图片消息]" and image_path_str:
                image_path = Path(image_path_str)
                if image_path.exists():
                    yield event.chain_result([
                        Comp.Plain(sender_info),
                        Comp.Image.fromFileSystem(image_path_str)
                    ])
                else:
                    yield event.plain_result(f"{sender_info}\n[一张已过期的图片] (ID: {image_path.stem})")
            else:
                text_content = res['text']
                yield event.plain_result(f"{sender_info} {text_content}")
            
            await asyncio.sleep(0.5)

    async def _should_reply(self, event: AstrMessageEvent) -> bool:
        """
        使用小模型通过 Prompt 判断是否应该回复。
        """
        is_mentioned = any(
            isinstance(c, Comp.At) and str(c.qq) == str(event.message_obj.self_id)
            for c in event.message_obj.message
        )
        if is_mentioned:
            logger.info("机器人被提及，跳过判断，直接进入回复流程。")
            return True

        if random.random() > self.reply_possibility:
            logger.info(f"随机触发失败 ({self.reply_possibility * 100}%)，不回复。")
            return False

        if not self.judgement_provider_id:
            logger.warning("未在插件设定中指定用于判断意愿的小模型，将跳过意愿判断并默认为不回复。")
            return False
            
        provider = self.context.get_provider_by_id(self.judgement_provider_id)
        if not provider:
            logger.error(f"找不到 ID 为 '{self.judgement_provider_id}' 的服务提供商，请检查设定。")
            return False
        
        message_text = event.message_str.strip()
        if not message_text:
            message_text = "[用户发送了非文本消息]"

        prompt = f"""
        你是一个群聊中 AI 助手的回复决策模型。你的任务是判断 AI 助手是否应该对以下这条新消息进行回复。
        回复标准：
        1.  群友们在轻松闲聊，而不是在讨论严肃、专业或工作的议题时，可以回复。
        2.  话题直接与 AI、机器人、模型或代码相关时，应该回复。
        3.  用户在寻求帮助、提问时，应该回复。
        你的回答必须非常简洁，只能是 "YES" 或 "NO"。
        ---
        群聊新消息: "{message_text}"
        ---
        根据以上标准，AI 助手是否应该回复？
        """
        
        try:
            logger.info(f"正在调用小模型 ({self.judgement_provider_id}) 判断回复意愿...")
            llm_resp = await provider.text_chat(prompt=prompt)
            decision = llm_resp.completion_text.strip().upper()
            logger.info(f"小模型的回复是: '{decision}'")
            
            return "YES" in decision
        except Exception as e:
            logger.error(f"调用小模型判断意愿时发生错误: {e}")
            return False

    def _format_rag_results_for_prompt(self, results: list) -> tuple[str, list[str]]:
        """
        [修改后] 将 RAG 搜索结果格式化为 Prompt 的一部分，并提取图片路径。
        返回一个元组：(格式化后的文本, 图片路径列表)
        """
        if not results:
            return "", []
        
        image_paths = []
        prompt_text = "--- 以下是从历史聊天记录中检索到的相关消息，请参考 ---\n\n"
        for res in sorted(results, key=lambda x: x['timestamp'], reverse=True)[:5]:
            ts = datetime.fromtimestamp(res['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            sender_info = f"[{ts}] {res['sender_name']}:"
            
            image_path_str = res.get('image_path')
            if image_path_str and Path(image_path_str).exists():
                prompt_text += f"{sender_info} [用户发送了一张图片，图片内容已附上，请结合图片进行理解]\n"
                image_paths.append(image_path_str)
            else:
                prompt_text += f"{sender_info} {res['text']}\n"
        
        prompt_text += "\n--- 历史消息结束 ---\n"
        return prompt_text, image_paths

    async def handle_new_message(self, event: AstrMessageEvent):
        asyncio.create_task(self.handle_group_message_logging(event))

        current_message_text = event.message_str.strip()
        if not current_message_text:
            current_message_text = "[用户发送了非文本消息]"

        group_id = event.get_group_id()
        if not group_id:
            return

        logger.info(f"对新消息进行 RAG 搜索: '{current_message_text}'")
        top_k = self.config.get("top_k_text_results", 5)
        top_j = self.config.get("top_j_image_results", 3)
        rag_results = await self.rag_db.query(current_message_text, group_id, top_k, top_j)

        should_reply = await self._should_reply(event)

        if should_reply:
            if not self.main_llm_provider_id:
                logger.error("未在插件设定中指定用于最终回复的大语言模型，无法生成回复。")
                yield event.plain_result("抱歉，管理员还未设置我的回复大脑，我暂时无法回答。")
                return

            provider = self.context.get_provider_by_id(self.main_llm_provider_id)
            if not provider:
                logger.error(f"找不到 ID 为 '{self.main_llm_provider_id}' 的服务提供商，请检查设定。")
                yield event.plain_result("抱歉，管理员指定的回复大脑好像不见了，我暂时无法回答。")
                return
            
            # --- 核心修改 ---
            # 1. 接收格式化后的文本和图片URL列表
            rag_context_str, image_urls = self._format_rag_results_for_prompt(rag_results)
            
            prompt = (
                f"{rag_context_str}\n"
                f"现在，请基于以上可能相关的历史消息（包括文字和图片），以前后文一致、符合群聊氛围的口语化方式，自然地回复以下这条新消息。\n"
                f"你的回复应当简洁、直接，不要重复历史消息的内容，也不要提及你参考了历史消息。\n\n"
                f"新消息来自「{event.get_sender_name()}」: \"{current_message_text}\"\n\n"
                f"你的回复："
            )

            logger.info(f"组装完成，准备调用大模型 ({self.main_llm_provider_id}) 进行处理。")
            if image_urls:
                logger.info(f"本次请求将附带 {len(image_urls)} 张图片: {image_urls}")

            try:
                # 2. 将 image_urls 传递给 text_chat 方法
                llm_resp = await provider.text_chat(
                    prompt=prompt,
                    system_prompt=self.persona_prompt,
                    image_urls=image_urls
                )
                reply_text = llm_resp.completion_text.strip()

                if reply_text:
                    logger.info(f"LLM 返回结果: {reply_text}")
                    yield event.plain_result(reply_text)
                else:
                    logger.warning("LLM 返回了空内容，本次不回复。")
            except Exception as e:
                logger.error(f"调用大模型时发生错误: {e}")
                yield event.plain_result("抱歉，我的大脑出了一点小问题，稍后再试吧~")