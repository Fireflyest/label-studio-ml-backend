from typing import List, Dict, Optional

from PIL import Image

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

import config


def _get_detector_class():
    """根据 config.DETECTOR_BACKEND 动态选择检测器类"""
    backend = config.DETECTOR_BACKEND.strip().lower()
    if backend == "gemma":
        from gemma_playcall_detector import GemmaDetector
        return GemmaDetector
    elif backend == "sam3":
        from sam3_playcall_detector import SAM3Detector
        return SAM3Detector
    else:
        raise ValueError(
            f"未知的 DETECTOR_BACKEND='{backend}', 可选值: 'gemma' / 'sam3'\n"
            f"请在 .env 或环境变量中设置 DETECTOR_BACKEND"
        )


class NewModel(LabelStudioMLBase):
    """自动标注后端

    支持两种检测器后端(通过 config.DETECTOR_BACKEND / .env 切换):
      - gemma : 使用本地 Gemma 视觉语言模型 (默认)
      - sam3  : 使用 SAM3 检测器

    两种后端均输出相同格式的结果, 此处代码无需区分。
    """

    def setup(self):
        """Configure any parameters of your model here"""
        backend = config.DETECTOR_BACKEND.strip().lower()
        self.set("model_version", f"{backend}")
        # 懒加载: 首次 predict 时再初始化,避免后端启动卡顿
        self._detector = None
        print(f"[NewModel] 检测器后端: {backend}")

    def _get_detector(self):
        if self._detector is None:
            detector_cls = _get_detector_class()
            self._detector = detector_cls.get_instance()
        return self._detector

    def _get_image_control(self):
        """从 label_config 中解析 RectangleLabels 控件的 from_name / to_name / 图片字段名"""
        for control_name, control in self.parsed_label_config.items():
            if control["type"] == "RectangleLabels":
                from_name = control_name
                to_name = control["to_name"][0]
                value_key = control["inputs"][0]["value"]
                return from_name, to_name, value_key
        raise ValueError("未在 label_config 中找到 RectangleLabels 控件,请检查标注模板配置")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        from_name, to_name, value_key = self._get_image_control()
        detector = self._get_detector()

        predictions = []
        for task in tasks:
            image_url = task["data"][value_key]

            # 需要设置环境变量 LABEL_STUDIO_URL / LABEL_STUDIO_API_KEY
            # (可在 config.py / .env 中配置,框架会自动读取同名环境变量)
            image_path = self.get_local_path(image_url, task_id=task.get("id"))

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"图片读取失败 {image_url}: {e}")
                predictions.append({"result": [], "score": 0.0, "model_version": self.get("model_version")})
                continue

            img_w, img_h = image.size

            try:
                detections = detector.detect(image)
            except Exception as e:
                print(f"检测失败 {image_url}: {e}")
                predictions.append({"result": [], "score": 0.0, "model_version": self.get("model_version")})
                continue

            results = []
            scores = []
            for det in detections:
                class_id = det["class_id"]
                label = config.CLASS_NAMES.get(class_id)
                if label is None:
                    continue

                x1, y1, x2, y2 = det["box"]
                avg_score = (det["body_score"] + det["phone_score"]) / 2
                scores.append(avg_score)

                results.append({
                    "from_name": from_name,
                    "to_name": to_name,
                    "type": "rectanglelabels",
                    "value": {
                        # Label Studio 使用百分比坐标(相对图片宽高)
                        "x": float(x1) / img_w * 100,
                        "y": float(y1) / img_h * 100,
                        "width": float(x2 - x1) / img_w * 100,
                        "height": float(y2 - y1) / img_h * 100,
                        "rectanglelabels": [label],
                    },
                    "score": float(avg_score),
                    "original_width": img_w,
                    "original_height": img_h,
                })

            predictions.append({
                "result": results,
                "score": float(max(scores)) if scores else 0.0,
                "model_version": self.get("model_version"),
            })

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        当前自动标注后端不需要在线训练,保留空实现以兼容框架接口
        """
        pass