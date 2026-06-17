import os
import re
import json
import torch
from PIL import Image

import config


class GemmaDetector:
    _instance = None

    def __init__(self):
        self.processor, self.model = self._load_gemma()

    @classmethod
    def get_instance(cls) -> "GemmaDetector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ---------------- 模型加载 ----------------

    def _load_gemma(self):
        from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig

        model_path = config.GEMMA_MODEL_PATH
        print(f"加载 Gemma 模型: {model_path}")

        # int4 量化: 权重约 13GB, 可整体装入 16GB 显存
        # 不再需要 CPU offloading, 消除所有设备冲突
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        processor = AutoProcessor.from_pretrained(model_path)

        # device_map="auto": 全部层都在 GPU (因为 int4 装得下)
        print("加载模型 (int4, 全部放 GPU)...")
        model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()

        # 验证加载结果
        device_counts = {}
        for name, dev in model.hf_device_map.items():
            device_counts[str(dev)] = device_counts.get(str(dev), 0) + 1
        print(f"设备分布: {device_counts}")

        vram_used = torch.cuda.memory_allocated() / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"显存占用: {vram_used:.1f} / {vram_total:.1f} GB")

        return processor, model

    # ---------------- Prompt 构造 ----------------
    def _build_prompt(self) -> str:
        task_text = (
            "Analyze this image and detect all people who are using a smartphone. "
            "For each person using a smartphone, output a bounding box that tightly encloses "
            "both the person's body and the smartphone together.\n\n"
            "Classify each detection as:\n"
            '  - "call": the person is holding the phone near their ear/face (making a call)\n'
            '  - "play": the person is looking at the phone screen (browsing, typing, etc.)\n\n'
            "Return ONLY a JSON object with no extra text, markdown, or explanation:\n"
            "{\n"
            '  "detections": [\n'
            "    {\n"
            '      "x1": <float 0-1>,\n'
            '      "y1": <float 0-1>,\n'
            '      "x2": <float 0-1>,\n'
            '      "y2": <float 0-1>,\n'
            '      "class": "call" or "play",\n'
            '      "body_score": <float 0-1>,\n'
            '      "phone_score": <float 0-1>\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            'If no person is using a smartphone, return: {"detections": []}'
        )

        # <|image|> 是 Gemma4 视觉输入的占位符,
        # processor 看到它才会把视觉编码器的输出 token 插入到这个位置
        return (
            "<start_of_turn>user\n"
            f"<|image|>\n{task_text}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

    # ---------------- 推理 ----------------

    def _infer(self, image: Image.Image) -> str:
        prompt_text = self._build_prompt()

        inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        # 全部层都在 cuda:0, 直接搬到 GPU 即可
        inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v
                  for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.GEMMA_MAX_NEW_TOKENS,
                do_sample=False,
            )

        response = self.processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )
        return response.strip()


    # ---------------- JSON 解析 (与原版相同) ----------------

    @staticmethod
    def _parse_response(response: str, img_w: int, img_h: int) -> list:
        print(response)
        json_str = response.strip()

        if not json_str.startswith("{"):
            match = re.search(r"\{.*\}", json_str, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                print(f"  [警告] 无法解析 JSON:\n{response[:200]}")
                return []

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"  [警告] JSON 解析失败: {e}\n原始: {response[:200]}")
            return []

        detections = data.get("detections", [])
        results = []

        for det in detections:
            try:
                x1 = int(float(det["x1"]) * img_w)
                y1 = int(float(det["y1"]) * img_h)
                x2 = int(float(det["x2"]) * img_w)
                y2 = int(float(det["y2"]) * img_h)
                cls_name = str(det.get("class", "play")).strip().lower()
                body_score = float(det.get("body_score", 0.8))
                phone_score = float(det.get("phone_score", 0.8))
            except (KeyError, ValueError, TypeError) as e:
                print(f"  [警告] 跳过: {det}, 错误: {e}")
                continue

            x1, x2 = max(0, x1), min(img_w, x2)
            y1, y2 = max(0, y1), min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            class_id = config.CLASS_NAME_TO_ID.get(cls_name, 0)
            results.append({
                "box": (x1, y1, x2, y2),
                "class_id": class_id,
                "body_score": body_score,
                "phone_score": phone_score,
            })

        return results

    # ---------------- 对外接口 ----------------

    def detect(self, image: Image.Image) -> list:
        img_w, img_h = image.size
        response = self._infer(image)              # 直接传 PIL Image
        return self._parse_response(response, img_w, img_h)
