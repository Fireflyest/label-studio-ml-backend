"""
SAM3 检测与匹配逻辑封装

从原自动标注脚本中提取出与"业务无关的IO/可视化"部分,
只保留: 模型加载 + 检测 + person/phone 匹配 + call/play 分类
供 Label Studio ML Backend (model.py) 调用
"""

import os
import torch

import config


class SAM3Detector:
    """单例封装: 避免每次 predict 都重新加载模型"""

    _instance = None

    def __init__(self):
        self.processor = self._load_sam3()

    @classmethod
    def get_instance(cls) -> "SAM3Detector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ---------------- 模型加载 ----------------

    def _load_sam3(self):
        import sam3
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        # API Key 从配置文件读取(支持通过 .env 覆盖)
        os.environ["HF_TOKEN"] = config.SAM3_HF_TOKEN

        model = build_sam3_image_model(
            bpe_path=f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz",
            checkpoint_path=config.SAM3_CHECKPOINT_PATH,
            load_from_HF=False,
        )
        processor = Sam3Processor(model, confidence_threshold=config.SAM3_CONFIDENCE_THRESHOLD)
        return processor

    # ---------------- 工具函数 ----------------

    @staticmethod
    def _boxes_intersect(b1, b2):
        return (b1[0] < b2[2] and b1[2] > b2[0] and
                b1[1] < b2[3] and b1[3] > b2[1])

    @staticmethod
    def _merge_box(b1, b2):
        return (min(b1[0], b2[0]), min(b1[1], b2[1]),
                max(b1[2], b2[2]), max(b1[3], b2[3]))

    # ---------------- 检测主逻辑 ----------------

    def detect(self, image):
        """
        对单张图片进行检测、匹配、分类

        :param image: PIL.Image (RGB)
        :return: List[Dict],每个元素为一个检测结果:
            {
                "box": (x1, y1, x2, y2)  像素坐标,
                "class_id": 0 或 1,       0=play, 1=call (映射见 config.CLASS_NAMES)
                "body_score": float,
                "phone_score": float,
            }
        """
        inference_state = self.processor.set_image(image)

        upper_bodies, phones = [], []

        self.processor.reset_all_prompts(inference_state)
        inference_state = self.processor.set_text_prompt(state=inference_state, prompt=config.SAM3_PROMPT_PERSON)
        for box, score in zip(inference_state.get("boxes", []), inference_state.get("scores", [])):
            upper_bodies.append({"box": box.cpu().numpy(), "score": float(score.cpu())})

        self.processor.reset_all_prompts(inference_state)
        inference_state = self.processor.set_text_prompt(state=inference_state, prompt=config.SAM3_PROMPT_PHONE)
        for box, score in zip(inference_state.get("boxes", []), inference_state.get("scores", [])):
            phones.append({"box": box.cpu().numpy(), "score": float(score.cpu())})

        # ---- 找出 person / phone 的最佳重叠匹配 ----
        candidates = []
        for i, b in enumerate(upper_bodies):
            for j, p in enumerate(phones):
                if self._boxes_intersect(b["box"], p["box"]):
                    x1 = max(b["box"][0], p["box"][0])
                    y1 = max(b["box"][1], p["box"][1])
                    x2 = min(b["box"][2], p["box"][2])
                    y2 = min(b["box"][3], p["box"][3])
                    overlap = max(0, x2 - x1) * max(0, y2 - y1)
                    candidates.append((i, j, overlap))

        candidates.sort(key=lambda x: x[2], reverse=True)

        used_b, used_p = set(), set()
        matches = []
        for bi, pi, _ in candidates:
            if bi in used_b or pi in used_p:
                continue
            used_b.add(bi)
            used_p.add(pi)
            matches.append((bi, pi))

        # ---- 合并框 + call/play 分类 ----
        results = []
        for bi, pi in matches:
            body = upper_bodies[bi]
            phone = phones[pi]

            merged = self._merge_box(body["box"], phone["box"])

            body_h = body["box"][3] - body["box"][1]
            phone_cy = (phone["box"][1] + phone["box"][3]) * 0.5
            relative_y = (phone_cy - body["box"][1]) / body_h if body_h > 0 else 0.5
            class_id = 1 if relative_y < config.RELATIVE_Y_THRESHOLD else 0

            results.append({
                "box": merged,
                "class_id": class_id,
                "body_score": body["score"],
                "phone_score": phone["score"],
            })

        return results