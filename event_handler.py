import aiohttp
import uuid
import asyncio
import os
import random
import json
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

        # 更新判断模型的 Prompt，要求返回包含概率和回复方式的 JSON
        judgement_model_config = self.config.get("judgement_model", {})
        self.judgement_provider_id = judgement_model_config.get("provider_id")
        self.judgement_prompt = judgement_model_config.get("judgement_prompt", """你是一个群聊中 AI 助手的回复决策模型。你的任务是判断 AI 助手对以下新消息进行回复的“必要性”和“趣味性”，并决定回复方式。

请遵循以下标准：1.  **必要性判断**：给出一个 0.0 到 1.0 之间的概率分数。你的名字是宁宁。当有人叫你的名字的时候你的回复概率为1：1. 话题轻松有趣、适合闲聊时，概率更高。2. 话题直接与 AI、机器人、模型或代码相关时，概率应该较高（0.3-0.6）。3. 用户在明确寻求帮助、提问时，概率应该为 0.4-0.7。4. 如果是严肃、专业或工作相关的话题，概率应该为0。5. 无意义的闲聊或表情符号，概率应该接近 0.0。6. 用户在和其他用户对话（如at其他用户 ，出现形如[At:xxxxxx]；聊天内容中出现人名/网名；话题明显与你无关）时你不应该插嘴。你的回复概率应该为0。你应该学会读气氛。 7. **你的名字是宁宁。当有人叫你的名字的时候你的回复概率为1. **  
2.  **回复方式判断**：
    -   如果消息是对话题的延续、补充或者是一个开放性的陈述，适合用 'direct' (直接回复)。
    -   如果消息是一个明确的问题，或者上下文不清晰，为了让对话更清晰，适合用 'quote' (引用回复)。
    -   默认情况下，倾向于使用 'direct'，让对话更自然。
你的回答必须是一个严格的 JSON 对象，格式如下：
{{"probability": <概率值>, "reply_style": "<direct 或 quote>"}}

例如：
{{"probability": 0.85, "reply_style": "direct"}}

群聊新消息："{message_text}"
根据以上标准，你的决策是？""")
        self.reply_possibility = self.config.get("reply_possibility", 0.1)

        # 读取最终回复大模型和人格的配置
        main_llm_config = self.config.get("main_llm_provider", {})
        self.main_llm_provider_id = main_llm_config.get("provider_id")
        self.persona_prompt = main_llm_config.get("persona_prompt", "你是一个友好、乐于助人的群聊助手。")

        # 读取 @ 消息等待窗口的配置
        self.at_message_window = self.config.get("at_message_window", 15)
        self.at_message_prompt = self.config.get("at_message_prompt", "我在，请讲。在 {timeout} 秒内我会一并收听您的后续消息...")
        self.user_conversations = {}  # 用于存储用户会话状态

        # 初始化静音列表
        self.mute_list_path = self.image_save_path.parent / "mute_list.json"
        self.muted_groups = self._load_mute_list()

    def _load_mute_list(self) -> set:
        if self.mute_list_path.exists():
            try:
                with open(self.mute_list_path, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"加载静音列表失败: {e}")
                return set()
        return set()

    def _save_mute_list(self):
        try:
            with open(self.mute_list_path, 'w', encoding='utf-8') as f:
                json.dump(list(self.muted_groups), f, indent=4)
        except IOError as e:
            logger.error(f"保存静音列表失败: {e}")

    def mute(self, scope: str):
        """静音指定范围"""
        self.muted_groups.add(scope)
        self._save_mute_list()

    def unmute(self, scope: str):
        """解除静音指定范围"""
        if scope == 'all':
            self.muted_groups.clear()
        else:
            self.muted_groups.discard(scope)
        self._save_mute_list()


    async def _parse_message_components(self, event: AstrMessageEvent) -> tuple[list[str], list[str]]:
        """
        解析消息事件，提取文本内容和图片路径。
        """
        texts = []
        image_paths = []

        for component in event.message_obj.message:
            if isinstance(component, Comp.Plain):
                cleaned_text = component.text.strip()
                if isinstance(component, Comp.At) and str(component.qq) == str(event.message_obj.self_id):
                    continue
                if cleaned_text:
                    texts.append(cleaned_text)
            elif isinstance(component, Comp.Image):
                if image_url := component.url:
                    if saved_path := await self._download_image(image_url):
                        image_paths.append(str(saved_path))
                        texts.append("[图片]")

        return texts, image_paths


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
        
        texts, image_paths = await self._parse_message_components(event)

        text_content = " ".join(texts)
        if text_content:
            await self.rag_db.add_text(group_id, sender_id, sender_name, text_content, timestamp)
        
        for image_path in image_paths:
            await self.rag_db.add_image(group_id, sender_id, sender_name, Path(image_path), timestamp)

        if image_paths:
             self._cleanup_old_images()

    async def process_rag_search_command(self, event: AstrMessageEvent, query: str):
        """
        [业务逻辑] 处理 RAG 搜索指令，并将结果格式化后发送（直接发送）。
        """
        group_id = event.get_group_id()
        if not group_id:
            yield event.chain_result([Comp.Plain("抱歉，此命令只能在群聊中使用。")])
            return

        top_k = self.config.get("top_k_text_results", 5)
        top_j = self.config.get("top_j_image_results", 3)
        
        results = await self.rag_db.query(query, group_id, top_k, top_j)
        
        if not results:
            yield event.chain_result([Comp.Plain("没有找到相关的聊天记录。")])
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
                    yield event.chain_result([Comp.Plain(f"{sender_info}\n[一张已过期的图片] (ID: {image_path.stem})")])
            else:
                text_content = res['text']
                yield event.chain_result([Comp.Plain(f"{sender_info} {text_content}")])
            
            await asyncio.sleep(0.5)

    async def _should_reply(self, event: AstrMessageEvent, is_mentioned: bool) -> tuple[bool, str]:
        """
        [新逻辑] 使用小模型判断回复概率和方式，决定是否及如何回复。
        返回一个元组 (should_reply: bool, reply_style: str)。
        """
        group_id = event.get_group_id()
        if 'all' in self.muted_groups or (group_id and group_id in self.muted_groups):
            return False, 'quote'

        if is_mentioned:
            logger.info("机器人被提及，强制引用回复。")
            return True, 'quote'

        if not self.judgement_provider_id:
            logger.info(f"未配置判断模型，使用备用随机概率 ({self.reply_possibility * 100}%)。")
            return random.random() < self.reply_possibility, 'direct'
            
        provider = self.context.get_provider_by_id(self.judgement_provider_id)
        if not provider:
            logger.error(f"找不到 ID 为 '{self.judgement_provider_id}' 的服务提供商，本次不回复。")
            return False, 'quote'
        
        message_text = event.message_str.strip()
        if not message_text:
            message_text = "[用户发送了非文本消息]"

        prompt = self.judgement_prompt.format(message_text=message_text)
        
        try:
            logger.info(f"正在调用小模型 ({self.judgement_provider_id}) 判断回复概率和方式...")
            llm_resp = await provider.text_chat(prompt=prompt)
            decision_text = llm_resp.completion_text.strip()
            logger.info(prompt)

            decision_data = json.loads(decision_text)
            probability = float(decision_data.get("probability", 0))
            reply_style = decision_data.get("reply_style", "direct")
            
            if reply_style not in ['direct', 'quote']:
                reply_style = 'direct'

            logger.info(f"小模型返回的回复概率是: {probability:.2f}, 回复方式: {reply_style}")

            should = random.random() < probability
            logger.info(f"判断结果: {'回复' if should else '不回复'}。")
            return should, reply_style

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"小模型返回的内容 '{decision_text}' 不是有效的 JSON 或概率值，本次不回复。错误: {e}")
            return False, 'quote'
        except Exception as e:
            logger.error(f"调用小模型判断概率时发生错误: {e}")
            return False, 'quote'


    def _format_rag_results_for_prompt(self, results: list) -> tuple[str, list[str]]:
        """
        将 RAG 搜索结果格式化为 Prompt 的一部分，并提取图片路径。
        """
        if not results:
            return "", []
        
        image_paths = []
        prompt_text = "--- 以下是从历史聊天记录中检索到的相关消息，请参考 ---\n\n"
        # 仅取最近5条记录用于上下文
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

    async def _process_user_conversation(self, conversation_key: tuple, event: AstrMessageEvent, reply_style: str = 'quote'):
        """
        处理收集到的用户消息，并根据 reply_style 决定回复方式。
        """
        try:
            if conversation_key not in self.user_conversations:
                return

            conversation_data = self.user_conversations[conversation_key]
            
            combined_text = " ".join(conversation_data['texts'])
            collected_image_paths = conversation_data['image_paths']
            sender_name = conversation_data['sender_name']
            group_id = conversation_key[0]

            logger.info(f"开始处理来自 {sender_name} 的消息: '{combined_text}'")

            if not combined_text and not collected_image_paths:
                logger.info("收集到的消息为空，不处理。")
                return
                
            top_k = self.config.get("top_k_text_results", 5)
            top_j = self.config.get("top_j_image_results", 3)
            rag_results = await self.rag_db.query(combined_text, group_id, top_k, top_j)

            if not self.main_llm_provider_id:
                logger.error("未在插件设定中指定用于最终回复的大语言模型，无法生成回复。")
                yield event.plain_result("抱歉，管理员还未设置我的回复大脑，我暂时无法回答。")
                return

            provider = self.context.get_provider_by_id(self.main_llm_provider_id)
            if not provider:
                logger.error(f"找不到 ID 为 '{self.main_llm_provider_id}' 的服务提供商，请检查设定。")
                yield event.plain_result("抱歉，管理员指定的回复大脑好像不见了，我暂时无法回答。")
                return
                
            rag_context_str, rag_image_paths = self._format_rag_results_for_prompt(rag_results)
            all_image_urls = rag_image_paths + collected_image_paths
                
            prompt = (
                f"{rag_context_str}\n"
                f"现在，请基于以上可能相关的历史消息（包括文字和图片），以前后文一致、符合群聊氛围的口语化方式，自然地回复以下这条新消息。\n"
                f"你的回复应当简洁、直接，不要重复历史消息的内容，也不要提及你参考了历史消息。\n\n"
                f"新消息来自「{sender_name}」: \"{combined_text}\"\n\n"
                f"你的回复："
            )

            logger.info(f"组装完成，准备调用大模型 ({self.main_llm_provider_id}) 进行处理。")
            if all_image_urls:
                logger.info(f"本次请求将附带 {len(all_image_urls)} 张图片: {all_image_urls}")

            try:
                llm_resp = await provider.text_chat(
                    prompt=prompt,
                    system_prompt=self.persona_prompt,
                    image_urls=all_image_urls
                )
                reply_text = llm_resp.completion_text.strip()

                if reply_text:
                    logger.info(f"LLM 返回结果: {reply_text}")
                    if reply_style == 'direct':
                        logger.info("执行直接回复 (chain_result)。")
                        await event.send(event.plain_result(reply_text))
                    else:
                        logger.info("执行引用回复 (plain_result)。")
                        yield event.plain_result(reply_text)
                else:
                    logger.warning("LLM 返回了空内容，本次不回复。")
            except Exception as e:
                logger.error(f"调用大模型时发生错误: {e}")
                yield event.plain_result("抱歉，我的大脑出了一点小问题，稍后再试吧~")
        finally:
            if conversation_key in self.user_conversations:
                del self.user_conversations[conversation_key]
                logger.info(f"已清理会话: {conversation_key}")


    async def handle_new_message(self, event: AstrMessageEvent):
        asyncio.create_task(self.handle_group_message_logging(event))

        group_id = event.get_group_id()
        sender_id = event.get_sender_id()
        if not group_id or not sender_id:
            return

        conversation_key = (group_id, sender_id)

        is_mentioned = any(
            isinstance(c, Comp.At) and str(c.qq) == str(event.message_obj.self_id)
            for c in event.message_obj.message
        )

        if conversation_key in self.user_conversations:
            logger.info(f"在固定窗口期内，收到来自 {event.get_sender_name()} 的后续消息。")
            texts, image_paths = await self._parse_message_components(event)
            self.user_conversations[conversation_key]['texts'].extend(texts)
            self.user_conversations[conversation_key]['image_paths'].extend(image_paths)
            return

        if is_mentioned and self.at_message_window > 0:
            logger.info(f"机器人被 {event.get_sender_name()} 提及，启动一个固定的消息等待窗口 ({self.at_message_window}秒)。")
            
            # prompt_text = self.at_message_prompt.format(timeout=self.at_message_window)
            # yield event.plain_result(prompt_text)

            texts, image_paths = await self._parse_message_components(event)
            self.user_conversations[conversation_key] = {
                "texts": texts,
                "image_paths": image_paths,
                "sender_name": event.get_sender_name()
            }
            
            await asyncio.sleep(self.at_message_window)
            
            async for result in self._process_user_conversation(conversation_key, event, reply_style='quote'):
                yield result
            return
            
        should_reply, reply_style = await self._should_reply(event, is_mentioned)
        if should_reply:
            single_reply_key = (group_id, f"single_reply_{sender_id}_{uuid.uuid4()}")
            texts, image_paths = await self._parse_message_components(event)
            self.user_conversations[single_reply_key] = {
                "texts": texts,
                "image_paths": image_paths,
                "sender_name": event.get_sender_name()
            }
            async for result in self._process_user_conversation(single_reply_key, event, reply_style=reply_style):
                 yield result