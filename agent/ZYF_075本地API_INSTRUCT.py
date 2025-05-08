import requests
import base64
import logging
from typing import Dict, Any
from dataclasses import dataclass, field
from copy import deepcopy

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

DEFAULT_HOST = "10.110.237.138"
DEFAULT_PORT = "3000"
API_ENDPOINT = "/api/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
BEARER_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImQ5MWQ3YjRiLTk3OWQtNGU3Yi1hN2MyLWNjZmY5ODRjNTIzNyJ9.8VtmHdivvEqNEqIH7iswtnfy81yHtIzBM-p2dqQBt5k"
DEFAULT_HEADERS = {"Authorization": f"Bearer {BEARER_TOKEN}", "Content-Type": "application/json"}
REQUEST_TIMEOUT = 60
IMAGE_CONTENT_TYPE = "image/jpeg"

@dataclass
class ApiConfig:
    host: str = DEFAULT_HOST
    port: str = DEFAULT_PORT
    model: str = DEFAULT_MODEL
    headers: Dict[str, str] = field(default_factory=lambda: deepcopy(DEFAULT_HEADERS))
    temperature: float = 0.0
    stream: bool = False

class ImageProcessor:
    @staticmethod
    def encode_to_base64(image_path: str) -> str:
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except FileNotFoundError:
            logger.error("没这个文件")
            raise

class RequestBuilder:
    @staticmethod
    def build_request_body(config: ApiConfig, base64_image: str) -> Dict[str, Any]:
        return {
            "model": config.model,
            "options": {"temperature": config.temperature},
            "stream": config.stream,
            "messages": [{
                "role": "user",
                "content": [
                    #{"type": "text", "text": ""},
                    {"type": "image_url", "image_url": {"url": f"data:{IMAGE_CONTENT_TYPE};base64,{base64_image}"}
                }]
            }]
        }

class ApiClient:
    def __init__(self, config: ApiConfig):
        self.config = config
        self.base_url = f"http://{config.host}:{config.port}{API_ENDPOINT}"

    def send_request(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = requests.post(self.base_url, json=request_body, headers=self.config.headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return {}

def main_workflow():
    config = ApiConfig()
    try:
        image_path = input("图像文件路径: ").strip()#在这里用"image_path"传路径，绝对相对路径均可，我这里为了测试方便用了input，可以改成其他函数，最后给到"image_path"就行
        if not image_path:
            raise ValueError("无效路径")
        
        base64_image = ImageProcessor.encode_to_base64(image_path)
        request_body = RequestBuilder.build_request_body(config, base64_image)
        response = ApiClient(config).send_request(request_body)
        
        if response and (content := response.get('choices', [{}])[0].get('message', {}).get('content')):
            print(content)
        else:
            print("API炸缸了")
    except Exception:
        print("本次处理失败，另外注意文件体积需要小于5MB")
if __name__ == "__main__":
    try:
        main_workflow()
    except Exception as e:
        logger.critical(f"程序意外终止: {str(e)}", exc_info=True)
