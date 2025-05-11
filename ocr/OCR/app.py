"""
基于Flask的OCR及多模型协同处理系统
功能包含：图片上传、OCR识别、视觉模型分析、文本校验模型验证
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # 处理跨域请求
from werkzeug.utils import secure_filename  # 安全文件名处理
import os
import logging
from dgocr.dgocr import DGOCR  # OCR核心库
import base64
import requests
from typing import Dict, Any
from dataclasses import dataclass, field

# 初始化Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求
app.logger.setLevel(logging.DEBUG)

# ======================= 配置区 ======================= #
# 注意：以下配置需根据实际部署环境修改
UPLOAD_FOLDER = 'uploads'    # 文件上传保存目录
RESULT_FOLDER = 'results'    # 处理结果保存目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# OCR模型配置（示例路径，实际需根据模型存放位置修改）
OCR_CONFIG = {
    "rec_path": "./models/recognition_model",      # 识别模型路径
    "det_path": "./models/detection_model.onnx",   # 检测模型路径
    "img_size": 1024,          # 处理图像尺寸
    "model_type": "seglink",   # 模型类型
    "cpu_thread_num": 12       # CPU线程数
}

# ===================== 服务初始化 ===================== #
try:
    # OCR引擎初始化
    ocr = DGOCR(
        OCR_CONFIG["rec_path"],
        OCR_CONFIG["det_path"],
        img_size=OCR_CONFIG["img_size"],
        model_type=OCR_CONFIG["model_type"],
        cpu_thread_num=OCR_CONFIG["cpu_thread_num"]
    )
    app.logger.info("OCR引擎初始化成功")
except Exception as e:
    app.logger.error(f"OCR初始化失败: {str(e)}")
    raise

# ==================== API配置类 ===================== #
@dataclass
class ApiConfig:
    """远程API基础配置类"""
    host: str = "api.server.com"      # API服务地址（示例）
    port: str = "3000"                # API服务端口
    model: str = "default-model"      # 默认使用模型
    headers: Dict[str, str] = field(default_factory=lambda: {
        "Authorization": "Bearer your_api_key_here",  # 替换为实际API密钥
        "Content-Type": "application/json"
    })
    temperature: float = 0.0          # 模型温度参数
    stream: bool = False              # 是否流式传输
    api_endpoint: str = "/api/chat/completions"  # API端点
    request_timeout: int = 60         # 请求超时时间（秒）
    image_content_type: str = "image/jpeg"  # 图片编码格式

@dataclass
class ValidationModelConfig(ApiConfig):
    """文本校验模型专用配置"""
    model: str = "text-validation-model"  # 校验模型名称

# ==================== 功能模块 ====================== #
class ImageProcessor:
    """图片处理工具类"""
    
    @staticmethod
    def encode_to_base64(image_path: str) -> str:
        """将图片编码为Base64字符串"""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except FileNotFoundError:
            app.logger.error("图片文件不存在")
            raise

class RemoteModelClient:
    """远程模型API客户端"""
    
    def __init__(self, config: ApiConfig):
        self.config = config
        self.base_url = f"http://{self.config.host}:{self.config.port}{self.config.api_endpoint}"

    def build_request_body(self, base64_image: str) -> Dict[str, Any]:
        """构建API请求体"""
        return {
            "model": self.config.model,
            "options": {"temperature": self.config.temperature},
            "stream": self.config.stream,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{self.config.image_content_type};base64,{base64_image}"
                    }
                }]
            }]
        }

    def send_request(self, image_path: str) -> Dict[str, Any]:
        """发送API请求"""
        try:
            base64_image = ImageProcessor.encode_to_base64(image_path)
            request_body = self.build_request_body(base64_image)
            response = requests.post(
                self.base_url,
                json=request_body,
                headers=self.config.headers,
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            app.logger.error(f"API请求失败: {str(e)}")
            return {"error": str(e)}

# ===================== 路由处理 ===================== #
@app.route('/ocr', methods=['POST'])
def process_ocr():
    """主处理路由：接收图片，协调OCR、视觉模型、校验模型工作"""
    try:
        # 校验文件上传
        if 'file' not in request.files:
            return jsonify({'error': '未检测到文件上传'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400

        # 保存上传文件
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)
        app.logger.debug(f"文件保存成功：{upload_path}")

        # 并行处理流程
        try:
            # OCR处理
            ocr_result = ocr.run(upload_path)
            # 视觉模型处理
            vision_client = RemoteModelClient(ApiConfig())
            vision_result = vision_client.send_request(upload_path)
        except Exception as e:
            app.logger.error(f"处理过程中发生错误: {str(e)}")
            return jsonify({'error': '模型处理失败'}), 500

        # 生成可视化结果
        result_filename = f"result_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        ocr.draw(upload_path, ocr_result, result_path)

        # 格式化结果
        formatted_ocr = [{
            "id": i,
            "text": item[1][0],
            "confidence": float(item[1][1]),
            "position": item[0]
        } for i, item in enumerate(ocr_result)]

        formatted_vision = vision_result.get('choices', [{}])[0].get('message', {}).get('content', "") \
            if not vision_result.get("error") else "视觉模型处理失败"

        # 文本校验处理
        combined_text = f"{' '.join([item['text'] for item in formatted_ocr]} || {formatted_vision}"
        try:
            validation_client = RemoteModelClient(ValidationModelConfig())
            validation_response = requests.post(
                validation_client.base_url,
                json={
                    "model": validation_client.config.model,
                    "messages": [{"role": "user", "content": combined_text}]
                },
                headers=validation_client.config.headers,
                timeout=validation_client.config.request_timeout
            )
            validation_result = validation_response.json().get('choices', [{}])[0].get('message', {}).get('content', "")
        except Exception as e:
            validation_result = f"校验失败: {str(e)}"

        return jsonify({
            'ocr_result': formatted_ocr,
            'vision_result': formatted_vision,
            'validation_result': validation_result,
            'result_image': f"{request.host_url}results/{result_filename}"
        }), 200

    except Exception as e:
        app.logger.error(f"全局异常: {str(e)}")
        return jsonify({'error': '服务器内部错误'}), 500

@app.route('/results/<filename>')
def serve_result(filename):
    """提供结果图片访问"""
    return send_from_directory(RESULT_FOLDER, filename)

# ==================== 启动入口 ====================== #
if __name__ == '__main__':
    # 启动参数配置（生产环境应使用WSGI服务器）
    app.run(
        host='0.0.0.0',   # 绑定所有网络接口
        port=5000,        # 服务端口
        debug=True        # 调试模式（生产环境应设为False）
    )
