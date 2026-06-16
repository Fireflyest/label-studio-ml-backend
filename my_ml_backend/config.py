# SAM3 自动标注 - 配置文件
#
# 敏感信息(如 HF_TOKEN)建议优先通过环境变量或 .env 文件设置,
# 此处的默认值仅作为兜底,生产环境建议在 .env 中覆盖。

import os

# my_ml_backend 目录的绝对路径(config.py 所在目录)
# 用于将相对路径(如模型权重路径)正确解析,避免因启动时工作目录不同导致找不到文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv
    # 显式指定 .env 路径为 BASE_DIR 下,避免因启动命令的工作目录不同而读取不到
    load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass  # 未安装 python-dotenv 时,仅依赖系统环境变量 / 默认值


# ========== SAM3 模型相关 ==========
# HuggingFace Token,用于下载/校验模型权限
SAM3_HF_TOKEN = os.environ.get("SAM3_HF_TOKEN", "")

# 本地模型权重路径
# 默认值基于 BASE_DIR(即 my_ml_backend 目录)拼接,对应 my_ml_backend/model/sam3/sam3.pt
# 如需自定义,在 .env 中设置 SAM3_CHECKPOINT_PATH 为绝对路径即可覆盖
SAM3_CHECKPOINT_PATH = os.environ.get(
    "SAM3_CHECKPOINT_PATH",
    os.path.join(BASE_DIR, "model", "sam3", "sam3.pt"),
)

# 检测置信度阈值
SAM3_CONFIDENCE_THRESHOLD = float(os.environ.get("SAM3_CONFIDENCE_THRESHOLD", "0.3"))

# 文本提示词
SAM3_PROMPT_PERSON = os.environ.get("SAM3_PROMPT_PERSON", "person")
SAM3_PROMPT_PHONE = os.environ.get("SAM3_PROMPT_PHONE", "smartphone")


# ========== 业务逻辑相关 ==========
# 类别ID -> Label Studio 标签名 映射
# 注意: 这里的标签名需要与 Label Studio 标注界面配置(label_config)中的
# RectangleLabels 标签名完全一致
CLASS_NAMES = {
    0: "play",
    1: "call",
}

# 手机中心点相对于人物框高度的位置阈值,小于该值判定为 "call"(打电话),
# 否则判定为 "play"(玩手机)
RELATIVE_Y_THRESHOLD = float(os.environ.get("RELATIVE_Y_THRESHOLD", "0.4"))


# ========== Label Studio 相关 ==========
# 如需在 model.py 中通过 API 下载图片资源,需要配置以下两项
# (label_studio_ml 框架通常也会自动读取同名环境变量)
LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
LABEL_STUDIO_API_KEY = os.environ.get("LABEL_STUDIO_API_KEY", "")