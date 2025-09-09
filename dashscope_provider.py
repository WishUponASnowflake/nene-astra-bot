import dashscope
import base64
import os
from pathlib import Path
from http import HTTPStatus
from astrbot.api import logger

class DashscopeProvider:
    """
    使用阿里雲 Dashscope SDK 進行 Embedding 的服務提供商。
    API Key 會自動從環境變量 DASHSCOPE_API_KEY 讀取。
    """
    def __init__(self, config: dict):
        # 檢查環境變量是否存在，提供清晰的啟動提示
        if 'DASHSCOPE_API_KEY' not in os.environ:
            raise ValueError("環境變量 DASHSCOPE_API_KEY 未設置，Dashscope 提供商無法初始化。")

        self.model_name = config.get("model_name", "multimodal-embedding-v1")
        self.id = f"dashscope_embedder_{self.model_name}"
        
        # multimodal-embedding-v1 的維度是固定的
        self.dimension = 1024
        # 注意：不再需要 dashscope.api_key = ... 這一行，SDK 會自動讀取環境變量

    @property
    def embedding_dimension(self) -> int:
        """返回模型的固定向量維度。"""
        return self.dimension

    # ... get_text_embedding 和 get_image_embedding 方法無需任何修改 ...
    def get_text_embedding(self, text: str) -> list[float]:
        """
        同步方法：調用 Dashscope SDK 獲取文本的 embedding。
        """
        input_data = [{'text': text}]
        resp = dashscope.MultiModalEmbedding.call(
            model=self.model_name,
            input=input_data
        )

        if resp.status_code == HTTPStatus.OK:
            embedding = resp.output['embeddings'][0]['embedding']
            return embedding
        else:
            logger.error(f"調用 Dashscope 文本 Embedding API 失敗: Code: {resp.code}, Message: {resp.message}")
            raise ConnectionError(f"Dashscope API Error: {resp.message}")

    def get_image_embedding(self, image_path: str | Path) -> list[float]:
        """
        同步方法：調用 Dashscope SDK 獲取圖片的 embedding。
        """
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            image_format = Path(image_path).suffix.lstrip('.')
            image_data = f"data:image/{image_format};base64,{base64_image}"
            input_data = [{'image': image_data}]

            resp = dashscope.MultiModalEmbedding.call(
                model=self.model_name,
                input=input_data
            )

            if resp.status_code == HTTPStatus.OK:
                embedding = resp.output['embeddings'][0]['embedding']
                return embedding
                
            else:
                logger.error(f"調用 Dashscope 圖片 Embedding API 失敗: Code: {resp.code}, Message: {resp.message}")
                raise ConnectionError(f"Dashscope API Error: {resp.message}")
        except FileNotFoundError:
            logger.error(f"找不到圖片文件: {image_path}")
            raise
        except Exception as e:
            logger.error(f"處理圖片 Embedding 時發生未知錯誤: {e}")
            raise