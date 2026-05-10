# -*- coding: utf-8 -*-
"""
MSF-YOLO：多源异构数据协同下的目标跟踪完整代码

严格对应论文流程：
1. 正常状态：YOLOv8 检测 person + DeepSORT 实时跟踪
2. 目标连续 N 帧未匹配：判定 LOST
3. LOST 时记录锚点：目标框、裁剪图、深度、原 track_id、速度
4. 新 person 出现：深度信息初筛 + Res2Net 声纹确认 + qwen3-vl-plus 语义验证
5. 多模态加权融合：Score = w_depth*C_depth + w_audio*C_audio + w_vlm*C_vlm
6. Score >= 0.65：确认目标身份，恢复 people_1 跟踪

安装依赖：
pip install ultralytics deep-sort-realtime opencv-python numpy openai modelscope soundfile
如使用 RealSense 深度相机，还需要：
pip install pyrealsense2

使用前设置 Qwen API Key：
Windows:
set DASHSCOPE_API_KEY=你的API_KEY

Linux/Mac:
export DASHSCOPE_API_KEY=你的API_KEY
"""

import os
import re
import cv2
import json
import time
import base64
import argparse
import traceback
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

import numpy as np


# ============================================================
# 1. 第三方库导入
# ============================================================

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception:
    DeepSort = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
except Exception:
    pipeline = None
    Tasks = None

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


# ============================================================
# 2. 参数配置
# ============================================================

@dataclass
class MSFConfig:
    # YOLOv8
    yolo_weight: str = "yolov8n.pt"
    yolo_conf: float = 0.35
    yolo_iou: float = 0.50
    person_class_id: int = 0

    # DeepSORT
    deepsort_max_age: int = 30
    deepsort_n_init: int = 3
    deepsort_max_cosine_distance: float = 0.30
    deepsort_nn_budget: int = 100

    # 目标丢失判定
    lost_threshold: int = 5
    max_lost_seconds: float = 30.0

    # 时空筛选半径
    spatial_search_radius: float = 150.0
    search_radius_growth_per_frame: float = 0.35
    search_radius_max: float = 320.0
    enable_global_fallback: bool = True
    global_fallback_after_frames: int = 10
    full_frame_search_after_frames: int = 45
    stop_search_center_at_frame_edge: bool = True

    # 深度验证阈值
    depth_tau_near: float = 0.30
    depth_tau_far: float = 0.50
    depth_near_boundary: float = 3.0
    hard_depth_reject: bool = True

    # 声纹验证
    speaker_model_id: str = "iic/speech_res2net_sv_zh-cn_3dspeaker_16k"
    audio_threshold: float = 0.65
    voice_recent_seconds: float = 5.0

    # VLM 验证
    use_vlm: bool = True
    vlm_model: str = "qwen3-vl-plus"
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    vlm_timeout: int = 60
    vlm_min_interval_frames: int = 10

    # 融合权重
    w_depth: float = 0.25
    w_audio: float = 0.25
    w_vlm: float = 0.50
    fusion_threshold: float = 0.65

    # 显示名称
    target_name: str = "people_1"
    show_window: bool = True


# ============================================================
# 3. 数据结构
# ============================================================

@dataclass
class Detection:
    bbox_xyxy: np.ndarray
    conf: float
    cls_id: int


@dataclass
class TrackInfo:
    track_id: int
    bbox_xyxy: np.ndarray


@dataclass
class TargetAnchor:
    target_name: str
    old_track_id: int
    bbox_xyxy: np.ndarray
    crop_image: np.ndarray
    depth: Optional[float]
    frame_idx: int
    timestamp: float
    velocity_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))


# ============================================================
# 4. 通用工具函数
# ============================================================

def xyxy_to_ltwh(bbox: np.ndarray) -> List[float]:
    x1, y1, x2, y2 = bbox.astype(float).tolist()
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(float)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a.astype(float)
    bx1, by1, bx2, by2 = b.astype(float)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter

    return float(inter / max(union, 1e-6))


def clamp_bbox(bbox: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(float)

    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))

    if x2 <= x1:
        x2 = min(float(w - 1), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(h - 1), y1 + 1.0)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def crop_person(frame: np.ndarray, bbox: np.ndarray, expand_ratio: float = 0.20) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(float)

    bw = x2 - x1
    bh = y2 - y1

    x1 = x1 - bw * expand_ratio
    x2 = x2 + bw * expand_ratio
    y1 = y1 - bh * expand_ratio
    y2 = y2 + bh * expand_ratio

    box = clamp_bbox(np.array([x1, y1, x2, y2], dtype=np.float32), w, h).astype(int)
    x1, y1, x2, y2 = box.tolist()

    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        crop = np.zeros((224, 224, 3), dtype=np.uint8)

    return crop


def preprocess_vlm_image(img_bgr: np.ndarray, size: int = 448) -> np.ndarray:
    if img_bgr is None or img_bgr.size == 0:
        return np.full((size, size, 3), 128, dtype=np.uint8)

    h, w = img_bgr.shape[:2]
    scale = min(size / max(w, 1), size / max(h, 1))
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    resized = cv2.resize(img_bgr, (nw, nh))
    canvas = np.full((size, size, 3), 128, dtype=np.uint8)

    x0 = (size - nw) // 2
    y0 = (size - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def image_to_data_uri(img_bgr: np.ndarray) -> str:
    img = preprocess_vlm_image(img_bgr, size=448)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("VLM 图像编码失败")

    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return "data:image/jpeg;base64," + b64


def parse_vlm_json(text: str) -> Dict[str, Any]:
    if text is None:
        return {"answer": "否", "confidence": 0.0}

    text = text.strip()

    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    nums = re.findall(r"(?:0\.\d+|1\.0|1|0)", text)
    confidence = 0.5
    if nums:
        try:
            confidence = float(nums[-1])
            confidence = max(0.0, min(1.0, confidence))
        except Exception:
            confidence = 0.5

    if any(word in text for word in ["不是", "否", "不同", "不一致"]):
        answer = "否"
    elif any(word in text for word in ["是", "同一个", "相同", "一致"]):
        answer = "是"
    else:
        answer = "否"

    return {"answer": answer, "confidence": confidence}


def latest_audio_file(audio_dir: Optional[str], recent_seconds: float) -> Optional[str]:
    if not audio_dir or not os.path.isdir(audio_dir):
        return None

    now = time.time()
    candidates = []

    for name in os.listdir(audio_dir):
        if not name.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
            continue

        path = os.path.join(audio_dir, name)
        mtime = os.path.getmtime(path)
        if now - mtime <= recent_seconds:
            candidates.append((mtime, path))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


# ============================================================
# 5. 深度图读取与目标深度提取
# ============================================================

class DepthProvider:
    """
    支持读取 depth_dir 下的深度文件：
    000001.npy / 000001.png / 1.npy / 1.png

    npy 默认单位：米
    png 默认单位：毫米，通过 depth_scale 转换为米
    """

    def __init__(self, depth_dir: Optional[str] = None, depth_scale: float = 1000.0):
        self.depth_dir = depth_dir
        self.depth_scale = depth_scale

    def load(self, frame_idx: int) -> Optional[np.ndarray]:
        if not self.depth_dir:
            return None

        names = [
            f"{frame_idx:06d}.npy",
            f"{frame_idx:05d}.npy",
            f"{frame_idx}.npy",
            f"{frame_idx:06d}.png",
            f"{frame_idx:05d}.png",
            f"{frame_idx}.png",
        ]

        for name in names:
            path = os.path.join(self.depth_dir, name)
            if not os.path.exists(path):
                continue

            if path.lower().endswith(".npy"):
                return np.load(path).astype(np.float32)

            depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                return None

            return depth.astype(np.float32) / float(self.depth_scale)

        return None

    @staticmethod
    def get_depth(depth_map: Optional[np.ndarray], bbox: np.ndarray) -> Optional[float]:
        if depth_map is None:
            return None

        h, w = depth_map.shape[:2]
        box = clamp_bbox(bbox, w, h).astype(int)
        x1, y1, x2, y2 = box.tolist()

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        center_depth = float(depth_map[cy, cx])
        if np.isfinite(center_depth) and center_depth > 0:
            return center_depth

        bw = x2 - x1
        bh = y2 - y1

        rx1 = int(x1 + 0.30 * bw)
        rx2 = int(x2 - 0.30 * bw)
        ry1 = int(y1 + 0.30 * bh)
        ry2 = int(y2 - 0.30 * bh)

        patch = depth_map[max(0, ry1):max(0, ry2), max(0, rx1):max(0, rx2)]
        valid = patch[np.isfinite(patch) & (patch > 0)]

        if valid.size == 0:
            return None

        return float(np.median(valid))


class RealSenseProvider:
    """
    RealSense 实时 RGB-D 数据读取模块。

    输出：
    1. color_bgr：OpenCV 使用的 BGR 彩色图
    2. depth_map_m：与彩色图对齐后的深度图，单位为米

    注意：必须将 depth 对齐到 color，否则 YOLO 检测框坐标不能直接用于深度图取值。
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        if rs is None:
            raise RuntimeError("未安装 pyrealsense2。请执行：pip install pyrealsense2")

        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self.profile = self.pipeline.start(self.config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # 将深度帧对齐到彩色帧
        self.align = rs.align(rs.stream.color)

        print(
            f"[INFO] RealSense 深度相机启用："
            f"width={width}, height={height}, fps={fps}, depth_scale={self.depth_scale}"
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return False, None, None

        color_bgr = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)

        # RealSense 原始深度为 uint16，需要乘 depth_scale 转换为米
        depth_map_m = depth_raw * float(self.depth_scale)

        return True, color_bgr, depth_map_m

    def release(self):
        self.pipeline.stop()


# ============================================================
# 6. YOLOv8 person 检测
# ============================================================

class YOLOPersonDetector:
    def __init__(self, cfg: MSFConfig):
        if YOLO is None:
            raise RuntimeError("未安装 ultralytics。请执行：pip install ultralytics")

        self.cfg = cfg
        self.model = YOLO(cfg.yolo_weight)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.cfg.yolo_conf,
            iou=self.cfg.yolo_iou,
            verbose=False
        )

        detections = []
        if len(results) == 0 or results[0].boxes is None:
            return detections

        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if cls_id != self.cfg.person_class_id:
                continue

            xyxy = box.xyxy[0].detach().cpu().numpy().astype(np.float32)
            detections.append(Detection(bbox_xyxy=xyxy, conf=conf, cls_id=cls_id))

        return detections


# ============================================================
# 7. DeepSORT 跟踪
# ============================================================

class DeepSORTTracker:
    def __init__(self, cfg: MSFConfig):
        if DeepSort is None:
            raise RuntimeError("未安装 deep-sort-realtime。请执行：pip install deep-sort-realtime")

        self.tracker = DeepSort(
            max_age=cfg.deepsort_max_age,
            n_init=cfg.deepsort_n_init,
            max_cosine_distance=cfg.deepsort_max_cosine_distance,
            nn_budget=cfg.deepsort_nn_budget,
            nms_max_overlap=1.0
        )

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[TrackInfo]:
        ds_inputs = []

        for det in detections:
            ltwh = xyxy_to_ltwh(det.bbox_xyxy)
            ds_inputs.append((ltwh, det.conf, "person"))

        tracks = self.tracker.update_tracks(ds_inputs, frame=frame)

        output = []
        for trk in tracks:
            if not trk.is_confirmed():
                continue

            if trk.time_since_update > 0:
                continue

            bbox = np.array(trk.to_ltrb(), dtype=np.float32)
            output.append(TrackInfo(track_id=int(trk.track_id), bbox_xyxy=bbox))

        return output


# ============================================================
# 8. 深度验证
# ============================================================

class DepthVerifier:
    def __init__(self, cfg: MSFConfig):
        self.cfg = cfg

    def verify(self, lost_depth: Optional[float], curr_depth: Optional[float]) -> Optional[Dict[str, Any]]:
        if lost_depth is None or curr_depth is None:
            return None

        delta = abs(float(lost_depth) - float(curr_depth))
        base_depth = min(float(lost_depth), float(curr_depth))

        if base_depth < self.cfg.depth_near_boundary:
            tau = self.cfg.depth_tau_near
        else:
            tau = self.cfg.depth_tau_far

        confidence = max(0.0, 1.0 - delta / max(tau, 1e-6))
        passed = delta <= tau

        return {
            "available": True,
            "passed": bool(passed),
            "confidence": float(confidence),
            "delta": float(delta),
            "tau": float(tau),
            "lost_depth": float(lost_depth),
            "curr_depth": float(curr_depth)
        }


# ============================================================
# 9. Res2Net 声纹验证
# ============================================================

class VoiceprintVerifier:
    def __init__(self, cfg: MSFConfig, target_voice: Optional[str]):
        self.cfg = cfg
        self.target_voice = target_voice
        self.pipe = None
        self.enabled = False

        if not target_voice or not os.path.exists(target_voice):
            print("[INFO] 未提供目标者注册语音，声纹模块关闭。")
            return

        if pipeline is None or Tasks is None:
            print("[INFO] ModelScope 不可用，声纹模块关闭。")
            return

        try:
            print("[INFO] 正在加载声纹模型：", cfg.speaker_model_id)
            self.pipe = pipeline(
                task=Tasks.speaker_verification,
                model=cfg.speaker_model_id
            )
            self.enabled = True
            print("[INFO] 声纹模型加载完成。")
        except Exception as e:
            print("[WARN] 声纹模型加载失败，声纹模块关闭：", e)
            self.enabled = False

    def verify(self, current_voice: Optional[str]) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        if not current_voice or not os.path.exists(current_voice):
            return None

        try:
            result = self.pipe([self.target_voice, current_voice])
            score = self._parse_score(result)
            passed = score >= self.cfg.audio_threshold

            return {
                "available": True,
                "passed": bool(passed),
                "confidence": float(score),
                "score": float(score),
                "threshold": float(self.cfg.audio_threshold),
                "raw": str(result)
            }

        except Exception as e:
            print("[WARN] 声纹验证失败：", e)
            return None

    @staticmethod
    def _parse_score(result: Any) -> float:
        if result is None:
            return 0.0

        if isinstance(result, dict):
            for key in ["score", "similarity", "cosine_score", "cos_score"]:
                if key in result:
                    return float(result[key])

            if "scores" in result and isinstance(result["scores"], list) and result["scores"]:
                return float(result["scores"][0])

            if "output" in result:
                return VoiceprintVerifier._parse_score(result["output"])

        if isinstance(result, list) and result:
            return VoiceprintVerifier._parse_score(result[0])

        nums = re.findall(r"(?:0\.\d+|1\.0|1|0)", str(result))
        if nums:
            return float(nums[-1])

        return 0.0


# ============================================================
# 10. qwen3-vl-plus 语义验证
# ============================================================

class QwenVLMVerifier:
    def __init__(self, cfg: MSFConfig):
        self.cfg = cfg
        self.enabled = False
        self.client = None

        if not cfg.use_vlm:
            print("[INFO] VLM 模块关闭。")
            return

        if OpenAI is None:
            print("[INFO] openai SDK 不可用，VLM 模块关闭。")
            return

        # 方法一：可直接把下面引号里的内容替换为真实 DASHSCOPE_API_KEY。
        # 方法二：也可以不改代码，使用系统环境变量 DASHSCOPE_API_KEY。
        api_key = "你的API_KEY".strip()
        if api_key == "你的API_KEY":
            api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()

        if not api_key:
            print("[INFO] 未检测到 DASHSCOPE_API_KEY，VLM 模块关闭。")
            return

        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url=cfg.dashscope_base_url,
                timeout=cfg.vlm_timeout
            )
            self.enabled = True
            print("[INFO] VLM 模块启用：", cfg.vlm_model)
        except Exception as e:
            print("[WARN] VLM 初始化失败：", e)
            self.enabled = False

    def verify(self, lost_crop: np.ndarray, curr_crop: np.ndarray) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        try:
            lost_uri = image_to_data_uri(lost_crop)
            curr_uri = image_to_data_uri(curr_crop)

            prompt = (
                "你是机器人目标重识别模块。"
                "现在给你两张人物图像：第一张是目标丢失前的人物图像，第二张是当前重新出现的候选人物图像。"
                "请判断两张图中的人物是否为同一个人。"
                "请重点比较衣服颜色、衣服款式、体型、姿态、随身物品、头发轮廓和整体外观。"
                "只输出严格 JSON，不要输出解释。"
                "格式：{\"answer\":\"是或否\",\"confidence\":0到1之间的小数}"
            )

            completion = self.client.chat.completions.create(
                model=self.cfg.vlm_model,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": lost_uri}},
                            {"type": "image_url", "image_url": {"url": curr_uri}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )

            content = completion.choices[0].message.content
            parsed = parse_vlm_json(content)

            answer = str(parsed.get("answer", "否"))
            raw_confidence = float(parsed.get("confidence", 0.0))
            raw_confidence = max(0.0, min(1.0, raw_confidence))

            same_person = ("是" in answer) and ("否" not in answer)

            if same_person:
                c_vlm = raw_confidence
            else:
                c_vlm = 1.0 - raw_confidence

            return {
                "available": True,
                "passed": bool(same_person),
                "confidence": float(c_vlm),
                "raw_answer": answer,
                "raw_confidence": float(raw_confidence),
                "raw_text": content
            }

        except Exception as e:
            print("[WARN] VLM 验证失败：", e)
            return None


# ============================================================
# 11. 多模态加权融合
# ============================================================

class FusionDecision:
    def __init__(self, cfg: MSFConfig):
        self.cfg = cfg

    def decide(
        self,
        depth_result: Optional[Dict[str, Any]],
        audio_result: Optional[Dict[str, Any]],
        vlm_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:

        confs = {}
        weights = {}

        if depth_result is not None and depth_result.get("available", False):
            confs["depth"] = float(depth_result["confidence"])
            weights["depth"] = self.cfg.w_depth

        if audio_result is not None and audio_result.get("available", False):
            confs["audio"] = float(audio_result["confidence"])
            weights["audio"] = self.cfg.w_audio

        if vlm_result is not None and vlm_result.get("available", False):
            confs["vlm"] = float(vlm_result["confidence"])
            weights["vlm"] = self.cfg.w_vlm

        if not confs:
            return {
                "accepted": False,
                "score": 0.0,
                "confs": confs,
                "weights": weights,
                "reason": "no_modal_available"
            }

        total_weight = sum(weights.values())
        norm_weights = {k: v / total_weight for k, v in weights.items()}

        score = 0.0
        for k in confs:
            score += norm_weights[k] * confs[k]

        accepted = score >= self.cfg.fusion_threshold

        return {
            "accepted": bool(accepted),
            "score": float(score),
            "confs": confs,
            "weights": norm_weights,
            "reason": "score_ok" if accepted else "score_low"
        }


# ============================================================
# 12. MSF-YOLO 主跟踪器
# ============================================================

class MSFYOLOTracker:
    def __init__(
        self,
        cfg: MSFConfig,
        target_voice: Optional[str] = None,
        target_track_id: Optional[int] = None,
        auto_select_first: bool = True
    ):
        self.cfg = cfg

        self.detector = YOLOPersonDetector(cfg)
        self.tracker = DeepSORTTracker(cfg)

        self.depth_verifier = DepthVerifier(cfg)
        self.voice_verifier = VoiceprintVerifier(cfg, target_voice)
        self.vlm_verifier = QwenVLMVerifier(cfg)
        self.fusion = FusionDecision(cfg)

        self.target_track_id = target_track_id
        self.auto_select_first = auto_select_first

        if self.target_track_id is None:
            self.state = "INIT"
        else:
            self.state = "TRACKING"

        self.lost_count = 0
        self.anchor: Optional[TargetAnchor] = None

        self.last_bbox: Optional[np.ndarray] = None
        self.last_crop: Optional[np.ndarray] = None
        self.last_depth: Optional[float] = None
        self.last_seen_frame: Optional[int] = None

        self.bbox_history: List[Tuple[int, np.ndarray]] = []
        self.last_vlm_frame = -10 ** 9

        # 当前画面尺寸，用于 LOST 状态下把预测搜索框/搜索圆限制在画面边缘。
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None

    def process_frame(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
        frame_idx: int,
        current_voice: Optional[str]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:

        self.frame_height, self.frame_width = frame.shape[:2]

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame)

        debug = {
            "frame_idx": frame_idx,
            "state": self.state,
            "target_track_id": self.target_track_id,
            "detections": len(detections),
            "tracks": len(tracks),
            "event": None
        }

        if self.state == "INIT":
            self._auto_select_target(frame, depth_map, frame_idx, tracks, debug)

        elif self.state == "TRACKING":
            self._tracking_state(frame, depth_map, frame_idx, tracks, debug)

        elif self.state == "LOST":
            self._lost_state(frame, depth_map, frame_idx, detections, tracks, current_voice, debug)

        vis = self._draw(frame.copy(), tracks, debug, depth_map)
        return vis, debug

    def _auto_select_target(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
        frame_idx: int,
        tracks: List[TrackInfo],
        debug: Dict[str, Any]
    ):
        if not self.auto_select_first:
            debug["event"] = "wait_target_id"
            return

        if not tracks:
            debug["event"] = "wait_person"
            return

        # 默认选择画面中面积最大的 person 作为 people_1
        tracks_sorted = sorted(
            tracks,
            key=lambda t: (t.bbox_xyxy[2] - t.bbox_xyxy[0]) * (t.bbox_xyxy[3] - t.bbox_xyxy[1]),
            reverse=True
        )

        target = tracks_sorted[0]
        self.target_track_id = target.track_id
        self.state = "TRACKING"
        self._save_observation(frame, depth_map, target.bbox_xyxy, frame_idx)

        debug["event"] = "auto_select_target"
        depth_text = "None" if self.last_depth is None else f"{self.last_depth:.3f} m"
        print(
            f"[INIT] 自动选择目标：{self.cfg.target_name}, "
            f"DeepSORT ID={self.target_track_id}, depth={depth_text}"
        )

    def _tracking_state(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
        frame_idx: int,
        tracks: List[TrackInfo],
        debug: Dict[str, Any]
    ):
        target_track = None
        for trk in tracks:
            if trk.track_id == self.target_track_id:
                target_track = trk
                break

        if target_track is not None:
            self.lost_count = 0
            self._save_observation(frame, depth_map, target_track.bbox_xyxy, frame_idx)
            debug["event"] = "tracking"
            return

        self.lost_count += 1
        debug["event"] = f"target_missing_{self.lost_count}"

        if self.lost_count >= self.cfg.lost_threshold:
            self._enter_lost(frame_idx)
            debug["event"] = "enter_lost"

    def _enter_lost(self, frame_idx: int):
        if self.last_bbox is None or self.last_crop is None:
            self.state = "INIT"
            return

        self.anchor = TargetAnchor(
            target_name=self.cfg.target_name,
            old_track_id=int(self.target_track_id) if self.target_track_id is not None else -1,
            bbox_xyxy=self.last_bbox.copy(),
            crop_image=self.last_crop.copy(),
            depth=self.last_depth,
            frame_idx=frame_idx,
            timestamp=time.time(),
            velocity_xy=self._estimate_velocity()
        )

        self.state = "LOST"

        print(
            f"[LOST] {self.cfg.target_name} 丢失："
            f"old_track_id={self.anchor.old_track_id}, "
            f"depth={self.anchor.depth}, "
            f"velocity={self.anchor.velocity_xy.tolist()}"
        )

    def _lost_state(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
        frame_idx: int,
        detections: List[Detection],
        tracks: List[TrackInfo],
        current_voice: Optional[str],
        debug: Dict[str, Any]
    ):
        if self.anchor is None:
            self.state = "INIT"
            debug["event"] = "lost_without_anchor"
            return

        if time.time() - self.anchor.timestamp > self.cfg.max_lost_seconds:
            print(f"[TIMEOUT] {self.cfg.target_name} 长时间未恢复，清理目标。")
            self._reset_target()
            debug["event"] = "lost_timeout"
            return

        candidates = self._filter_candidates_by_spatial(detections, frame_idx)
        debug["candidate_count"] = len(candidates)

        if not candidates:
            debug["event"] = "lost_wait_candidate"
            return

        best_result = None

        for det in candidates:
            curr_bbox = det.bbox_xyxy
            curr_crop = crop_person(frame, curr_bbox)
            curr_depth = DepthProvider.get_depth(depth_map, curr_bbox)

            # 1. 深度初筛
            depth_result = self.depth_verifier.verify(self.anchor.depth, curr_depth)

            if (
                self.cfg.hard_depth_reject
                and depth_result is not None
                and not depth_result.get("passed", False)
            ):
                fusion_result = {
                    "accepted": False,
                    "score": 0.0,
                    "confs": {"depth": depth_result["confidence"]},
                    "weights": {"depth": 1.0},
                    "reason": "hard_depth_reject"
                }

                result_pack = (det, depth_result, None, None, fusion_result)
                best_result = self._select_better_result(best_result, result_pack)
                continue

            # 2. 声纹确认
            audio_result = self.voice_verifier.verify(current_voice)

            # 3. VLM 语义验证
            vlm_result = None
            if (
                self.vlm_verifier.enabled
                and frame_idx - self.last_vlm_frame >= self.cfg.vlm_min_interval_frames
            ):
                self.last_vlm_frame = frame_idx
                vlm_result = self.vlm_verifier.verify(self.anchor.crop_image, curr_crop)

            # 4. 多模态融合
            fusion_result = self.fusion.decide(depth_result, audio_result, vlm_result)

            result_pack = (det, depth_result, audio_result, vlm_result, fusion_result)
            best_result = self._select_better_result(best_result, result_pack)

            if fusion_result["accepted"]:
                self._recover_target(frame, depth_map, frame_idx, curr_bbox, tracks)

                debug["event"] = "recover_success"
                debug["depth"] = depth_result
                debug["audio"] = audio_result
                debug["vlm"] = vlm_result
                debug["fusion"] = fusion_result

                print(
                    f"[RECOVER] {self.cfg.target_name} 恢复成功："
                    f"score={fusion_result['score']:.3f}, "
                    f"confs={fusion_result['confs']}, "
                    f"weights={fusion_result['weights']}"
                )
                return

        if best_result is not None:
            _, depth_result, audio_result, vlm_result, fusion_result = best_result
            debug["event"] = "recover_failed"
            debug["depth"] = depth_result
            debug["audio"] = audio_result
            debug["vlm"] = vlm_result
            debug["fusion"] = fusion_result

    @staticmethod
    def _select_better_result(old_result, new_result):
        if old_result is None:
            return new_result

        old_score = old_result[4].get("score", 0.0)
        new_score = new_result[4].get("score", 0.0)

        if new_score > old_score:
            return new_result

        return old_result

    def _recover_target(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
        frame_idx: int,
        candidate_bbox: np.ndarray,
        tracks: List[TrackInfo]
    ):
        # DeepSORT 内部 ID 不强行改写；程序层面继承 people_1 身份。
        # 如果候选框已经被 DeepSORT 分配了新 ID，则把该新 ID 作为后续视觉跟踪 ID。
        best_track_id = None
        best_iou = 0.0

        for trk in tracks:
            iou = bbox_iou(candidate_bbox, trk.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_track_id = trk.track_id

        if best_track_id is not None and best_iou > 0.30:
            self.target_track_id = best_track_id

        self.state = "TRACKING"
        self.lost_count = 0
        self.anchor = None
        self._save_observation(frame, depth_map, candidate_bbox, frame_idx)

    def _filter_candidates_by_spatial(self, detections: List[Detection], frame_idx: int) -> List[Detection]:
        """
        目标丢失后的候选筛选。
        说明：
        1. 黄色搜索圆/预测框会停在画面边缘，不会继续跑到画面外。
        2. 圆内候选优先进入深度/声纹/VLM 验证。
        3. 如果圆内没有候选，且已经丢失超过 global_fallback_after_frames，
           则启用全局候选回退，避免目标从圆外返回时完全无法识别。
        4. 如果丢失时间超过 full_frame_search_after_frames，直接启用整帧候选搜索，
           不再只依赖黄色圆区域。
        """
        if self.anchor is None:
            return detections

        dt = max(0, frame_idx - self.anchor.frame_idx)

        # 时间较长后直接全画面找。这样目标从画面外绕路返回时，不会被黄色圆限制住。
        if self.cfg.enable_global_fallback and dt >= self.cfg.full_frame_search_after_frames:
            return detections

        pred_center = self._predicted_search_center(frame_idx)
        radius = self._current_search_radius(frame_idx)

        inside_candidates = []

        for det in detections:
            center = bbox_center(det.bbox_xyxy)
            dist = float(np.linalg.norm(center - pred_center))
            if dist <= radius:
                inside_candidates.append(det)

        if inside_candidates:
            return inside_candidates

        # 工程回退：目标可能从预测圆外重新进入画面，不能因为不在黄色圆里就完全不验证。
        if self.cfg.enable_global_fallback and dt >= self.cfg.global_fallback_after_frames:
            return detections

        return inside_candidates

    def _raw_predicted_center(self, frame_idx: int) -> np.ndarray:
        if self.anchor is None:
            return np.zeros(2, dtype=np.float32)

        dt = max(0, frame_idx - self.anchor.frame_idx)
        anchor_center = bbox_center(self.anchor.bbox_xyxy)
        return (anchor_center + self.anchor.velocity_xy * dt).astype(np.float32)

    def _predicted_search_center(self, frame_idx: int) -> np.ndarray:
        """
        LOST 状态下用于显示和筛选的预测中心。
        如果 stop_search_center_at_frame_edge=True，中心点预测到画面外后会被钳制到画面边缘，
        这样黄色框/黄色圆停在边上，而不是继续飞出画面。
        """
        center = self._raw_predicted_center(frame_idx)

        if not self.cfg.stop_search_center_at_frame_edge:
            return center

        if self.frame_width is None or self.frame_height is None:
            return center

        center[0] = np.clip(center[0], 0, max(0, self.frame_width - 1))
        center[1] = np.clip(center[1], 0, max(0, self.frame_height - 1))
        return center.astype(np.float32)

    def _prediction_outside_frame(self, frame_idx: int) -> bool:
        if self.frame_width is None or self.frame_height is None or self.anchor is None:
            return False

        c = self._raw_predicted_center(frame_idx)
        return bool(c[0] < 0 or c[0] > self.frame_width - 1 or c[1] < 0 or c[1] > self.frame_height - 1)

    def _current_search_radius(self, frame_idx: int) -> float:
        if self.anchor is None:
            return float(self.cfg.spatial_search_radius)

        dt = max(0, frame_idx - self.anchor.frame_idx)
        radius = self.cfg.spatial_search_radius + self.cfg.search_radius_growth_per_frame * dt
        radius = min(float(self.cfg.search_radius_max), float(radius))
        return float(radius)

    def _current_lost_box(self, frame_idx: int) -> Optional[np.ndarray]:
        """根据丢失前 bbox 尺寸，在当前预测中心处生成 LOST 预测框，并限制在画面内。"""
        if self.anchor is None or self.frame_width is None or self.frame_height is None:
            return None

        x1, y1, x2, y2 = self.anchor.bbox_xyxy.astype(float)
        bw = max(2.0, x2 - x1)
        bh = max(2.0, y2 - y1)
        cx, cy = self._predicted_search_center(frame_idx).astype(float)

        box = np.array([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], dtype=np.float32)
        return clamp_bbox(box, self.frame_width, self.frame_height)

    def _save_observation(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
        bbox: np.ndarray,
        frame_idx: int
    ):
        self.last_bbox = bbox.copy()
        self.last_crop = crop_person(frame, bbox)
        self.last_depth = DepthProvider.get_depth(depth_map, bbox)
        self.last_seen_frame = frame_idx

        self.bbox_history.append((frame_idx, bbox.copy()))
        if len(self.bbox_history) > 8:
            self.bbox_history.pop(0)

    def _estimate_velocity(self) -> np.ndarray:
        if len(self.bbox_history) < 2:
            return np.zeros(2, dtype=np.float32)

        f1, b1 = self.bbox_history[-2]
        f2, b2 = self.bbox_history[-1]

        dt = max(1, f2 - f1)
        c1 = bbox_center(b1)
        c2 = bbox_center(b2)

        return ((c2 - c1) / float(dt)).astype(np.float32)

    def _reset_target(self):
        self.state = "INIT"
        self.target_track_id = None
        self.lost_count = 0
        self.anchor = None
        self.last_bbox = None
        self.last_crop = None
        self.last_depth = None
        self.last_seen_frame = None
        self.bbox_history.clear()

    def _draw(self, frame: np.ndarray, tracks: List[TrackInfo], debug: Dict[str, Any], depth_map: Optional[np.ndarray]) -> np.ndarray:
        h, w = frame.shape[:2]

        for trk in tracks:
            box = clamp_bbox(trk.bbox_xyxy, w, h).astype(int)
            x1, y1, x2, y2 = box.tolist()

            depth_value = DepthProvider.get_depth(depth_map, trk.bbox_xyxy)
            depth_text = "depth=None" if depth_value is None else f"depth={depth_value:.2f}m"

            if self.state == "TRACKING" and trk.track_id == self.target_track_id:
                color = (0, 255, 0)
                label = f"{self.cfg.target_name} | DS_ID={trk.track_id} | {depth_text}"
            else:
                color = (160, 160, 160)
                label = f"person | DS_ID={trk.track_id} | {depth_text}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

        if self.state == "LOST" and self.anchor is not None:
            cur_frame_idx = int(debug.get("frame_idx", self.anchor.frame_idx))
            dt = max(0, cur_frame_idx - self.anchor.frame_idx)
            pred_center = self._predicted_search_center(cur_frame_idx)
            radius = int(self._current_search_radius(cur_frame_idx))

            # 画黄色搜索圆；预测中心会停在画面边缘。
            cv2.circle(frame, tuple(pred_center.astype(int).tolist()), radius, (0, 165, 255), 2)

            # 画黄色 LOST 预测框，避免只看到一个快速飘走的圆。
            lost_box = self._current_lost_box(cur_frame_idx)
            if lost_box is not None:
                lx1, ly1, lx2, ly2 = lost_box.astype(int).tolist()
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 165, 255), 2)
                cv2.putText(
                    frame,
                    "LOST predicted box",
                    (lx1, max(20, ly1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 165, 255),
                    2
                )

            lost_depth_text = "None" if self.anchor.depth is None else f"{self.anchor.depth:.2f}m"
            cv2.putText(
                frame,
                f"LOST: {self.cfg.target_name} | lost_depth={lost_depth_text} | search_radius={radius}px",
                (20, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 165, 255),
                2
            )
            if self._prediction_outside_frame(cur_frame_idx):
                cv2.putText(
                    frame,
                    "Prediction reached frame edge",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 165, 255),
                    2
                )

            if self.cfg.enable_global_fallback and dt >= self.cfg.full_frame_search_after_frames:
                cv2.putText(
                    frame,
                    "Full-frame search enabled",
                    (20, 178),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 165, 255),
                    2
                )
            elif self.cfg.enable_global_fallback and dt >= self.cfg.global_fallback_after_frames:
                cv2.putText(
                    frame,
                    "Global fallback enabled if circle has no candidates",
                    (20, 178),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 165, 255),
                    2
                )

        status = (
            f"STATE={self.state} | target_track_id={self.target_track_id} | "
            f"event={debug.get('event')}"
        )

        cv2.rectangle(frame, (10, 10), (min(w - 10, 1100), 50), (0, 0, 0), -1)
        cv2.putText(
            frame,
            status,
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2
        )

        fusion = debug.get("fusion")
        if fusion is not None:
            text = f"Fusion score={fusion.get('score', 0.0):.3f}, reason={fusion.get('reason')}"
            cv2.putText(
                frame,
                text,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 0),
                2
            )

        return frame


# ============================================================
# 13. 主程序
# ============================================================

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MSF-YOLO 多源异构目标跟踪")

    parser.add_argument("--video", type=str, default="0", help="视频路径或摄像头编号，例如 0")
    parser.add_argument("--output", type=str, default="", help="输出视频路径，可为空")
    parser.add_argument("--yolo", type=str, default="yolov8n.pt", help="YOLOv8 权重路径")

    parser.add_argument("--target_track_id", type=int, default=-1, help="指定初始 DeepSORT ID；-1 表示自动选择第一人")
    parser.add_argument("--no_auto_select", action="store_true", help="关闭自动选择第一人")

    parser.add_argument("--depth_dir", type=str, default="", help="深度图文件夹，可为空")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="png 深度图缩放系数，默认毫米转米")

    parser.add_argument("--use_realsense", action="store_true", help="使用 RealSense 深度相机实时采集 RGB-D")
    parser.add_argument("--rs_width", type=int, default=640, help="RealSense 彩色/深度宽度")
    parser.add_argument("--rs_height", type=int, default=480, help="RealSense 彩色/深度高度")
    parser.add_argument("--rs_fps", type=int, default=30, help="RealSense 帧率")

    parser.add_argument("--search_radius", type=float, default=150.0, help="目标丢失后的黄色搜索圆初始半径，单位像素")
    parser.add_argument("--search_radius_growth", type=float, default=0.35, help="黄色搜索圆每帧扩大速度，单位像素/帧")
    parser.add_argument("--search_radius_max", type=float, default=320.0, help="黄色搜索圆最大半径，单位像素")
    parser.add_argument("--global_fallback_after", type=int, default=10, help="丢失多少帧后：如果圆内没有候选，则允许圆外候选进入验证")
    parser.add_argument("--full_frame_search_after", type=int, default=45, help="丢失多少帧后直接整帧搜索所有候选目标")
    parser.add_argument("--no_edge_stop", action="store_true", help="关闭 LOST 预测框/搜索圆停在画面边缘的逻辑")
    parser.add_argument("--max_lost_seconds", type=float, default=120.0, help="目标最多等待恢复时间，单位秒")

    parser.add_argument("--target_voice", type=str, default="", help="目标者注册语音文件路径")
    parser.add_argument("--current_voice", type=str, default="", help="当前候选人语音文件路径，测试用")
    parser.add_argument("--voice_dir", type=str, default="", help="实时语音目录，自动读取最近的音频文件")

    parser.add_argument("--no_vlm", action="store_true", help="关闭 qwen3-vl-plus 验证")
    parser.add_argument("--vlm_model", type=str, default="qwen3-vl-plus", help="VLM 模型名")
    parser.add_argument(
        "--dashscope_base_url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="DashScope OpenAI 兼容接口地址"
    )

    parser.add_argument("--no_show", action="store_true", help="不显示窗口")

    return parser


def main():
    args = build_argparser().parse_args()

    cfg = MSFConfig(
        yolo_weight=args.yolo,
        use_vlm=not args.no_vlm,
        vlm_model=args.vlm_model,
        dashscope_base_url=args.dashscope_base_url,
        show_window=not args.no_show,
        spatial_search_radius=args.search_radius,
        search_radius_growth_per_frame=args.search_radius_growth,
        search_radius_max=args.search_radius_max,
        global_fallback_after_frames=args.global_fallback_after,
        full_frame_search_after_frames=args.full_frame_search_after,
        stop_search_center_at_frame_edge=not args.no_edge_stop,
        max_lost_seconds=args.max_lost_seconds
    )

    target_track_id = None if args.target_track_id < 0 else args.target_track_id
    auto_select_first = (not args.no_auto_select) and (target_track_id is None)

    target_voice = args.target_voice if args.target_voice else None

    msf_tracker = MSFYOLOTracker(
        cfg=cfg,
        target_voice=target_voice,
        target_track_id=target_track_id,
        auto_select_first=auto_select_first
    )

    depth_provider = DepthProvider(
        depth_dir=args.depth_dir if args.depth_dir else None,
        depth_scale=args.depth_scale
    )

    rgbd_provider = None
    cap = None

    if args.use_realsense:
        # 实时 RGB-D 模式：不再从 depth_dir 读取深度图，而是由 RealSense 直接输出 depth_map。
        rgbd_provider = RealSenseProvider(
            width=args.rs_width,
            height=args.rs_height,
            fps=args.rs_fps
        )
        fps = float(args.rs_fps)
        width = int(args.rs_width)
        height = int(args.rs_height)
        print("[INFO] 深度模块启用：RealSense 实时深度输入。")
    else:
        # 离线视频模式：RGB 来自视频；深度来自 depth_dir 中预先保存好的深度图。
        if args.depth_dir:
            print(f"[INFO] 深度模块启用：depth_dir={args.depth_dir}, depth_scale={args.depth_scale}")
        else:
            print("[INFO] 未提供深度图目录，深度模块关闭。")

        if args.video.isdigit():
            video_source = int(args.video)
        else:
            video_source = args.video

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源：{args.video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print("[INFO] MSF-YOLO 开始运行。")
    print("[INFO] 按 q 退出。")

    frame_idx = 0

    while True:
        frame_idx += 1

        if args.use_realsense:
            ret, frame, depth_map = rgbd_provider.read()
            if not ret:
                print("[WARN] RealSense 当前帧读取失败。")
                continue
        else:
            ret, frame = cap.read()
            if not ret:
                break

            depth_map = depth_provider.load(frame_idx)

        if args.current_voice:
            current_voice = args.current_voice
        elif args.voice_dir:
            current_voice = latest_audio_file(args.voice_dir, cfg.voice_recent_seconds)
        else:
            current_voice = None

        try:
            vis, debug = msf_tracker.process_frame(
                frame=frame,
                depth_map=depth_map,
                frame_idx=frame_idx,
                current_voice=current_voice
            )
        except Exception as e:
            print("[ERROR] 当前帧处理失败：", e)
            traceback.print_exc()
            vis = frame

        if writer is not None:
            writer.write(vis)

        if cfg.show_window:
            cv2.imshow("MSF-YOLO", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    if cap is not None:
        cap.release()

    if rgbd_provider is not None:
        rgbd_provider.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()
    print("[INFO] MSF-YOLO 运行结束。")


if __name__ == "__main__":
    main()
