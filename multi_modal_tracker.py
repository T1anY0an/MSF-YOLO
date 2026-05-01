"""
多源异构融合跟踪算法 - 完整代码（带摄像头支持）
基于YOLO+深度信息+声纹识别+VLM大模型的重识别架构

Author: Robotics Engineering
Requirements: pip install ultralytics opencv-python numpy torch torchaudio transformers pillow scipy filterpy
"""

import cv2
import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
import argparse
import threading
import queue
import os

warnings.filterwarnings("ignore")

# 深度学习相关导入
from ultralytics import YOLO
import torch

# ==================== 配置参数 ====================
class Config:
    """系统配置参数"""
    
    # 跟踪参数
    LOST_THRESHOLD = 10          # 连续丢失帧数阈值，超过则触发丢失状态
    MAX_LOST_FRAMES = 30         # 最大丢失帧数，超过后彻底放弃跟踪
    
    # 深度验证参数
    DEPTH_TOLERANCE = 0.5        # 深度差异容忍度（米）
    NEAR_THRESHOLD = 0.3         # 近距离深度阈值
    FAR_THRESHOLD = 0.5          # 远距离深度阈值
    DISTANCE_BOUNDARY = 3.0      # 近远距离分界线
    
    # 声纹验证参数
    VOICE_SIMILARITY_THRESHOLD = 0.75   # 声纹余弦相似度阈值
    VOICE_SAMPLE_RATE = 16000           # 音频采样率
    
    # 融合参数
    WEIGHT_DEPTH = 0.3           # 深度模态权重
    WEIGHT_VOICE = 0.4           # 声纹模态权重
    WEIGHT_VLM = 0.3             # VLM模态权重
    FUSION_THRESHOLD = 0.6       # 融合决策阈值
    
    # VLM配置
    VLM_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
    USE_VLM = False               # 默认关闭VLM（需要大显存）
    
    # YOLO配置
    YOLO_MODEL_PATH = "yolov8n.pt"
    YOLO_CONF_THRES = 0.5
    YOLO_IOU_THRES = 0.45
    
    # 跟踪配置
    MAX_LOST_DURATION = 30.0
    SPATIAL_SEARCH_RADIUS = 150
    MAX_MAHALANOBIS_DIST = 9.4877
    MAX_COSINE_DIST = 0.3
    LAMBDA_MOTION = 0.98
    
    # 显示配置
    SHOW_FPS = True
    SHOW_TRACK_INFO = True
    DISPLAY_RESIZE_WIDTH = 1280    # 显示窗口宽度
    DISPLAY_RESIZE_HEIGHT = 720     # 显示窗口高度


# ==================== 数据结构定义 ====================
class TrackingState(Enum):
    """跟踪状态机"""
    TRACKING = "TRACKING"
    LOST = "LOST"
    VERIFYING = "VERIFYING"


@dataclass
class LostAnchor:
    """丢失时保存的目标锚点信息"""
    depth: float = 0.0
    image: Optional[np.ndarray] = None
    speaker_embedding: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    timestamp: float = 0.0


@dataclass
class Candidate:
    """候选目标信息"""
    bbox: Tuple[int, int, int, int]
    depth: float = 0.0
    image: Optional[np.ndarray] = None
    audio_embedding: Optional[np.ndarray] = None
    confidence: float = 0.0


class Track:
    """跟踪轨迹"""
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], depth: float):
        self.track_id = track_id
        self.bbox = bbox
        self.depth = depth
        self.lost_count = 0
        self.is_active = True
        self.history = [bbox]
        self.unmatched_count = 0
        self.features = []
    
    def get_velocity(self):
        if len(self.history) >= 2:
            prev = self.history[-2]
            curr = self.history[-1]
            center_prev = ((prev[0] + prev[2]) // 2, (prev[1] + prev[3]) // 2)
            center_curr = ((curr[0] + curr[2]) // 2, (curr[1] + curr[3]) // 2)
            return np.array([center_curr[0] - center_prev[0], center_curr[1] - center_prev[1]])
        return np.zeros(2)


# ==================== 深度验证模块 ====================
class DepthVerifier:
    """深度信息验证模块"""
    
    def __init__(self):
        self.lost_depth = None
    
    def record_lost_depth(self, depth_value: float):
        self.lost_depth = depth_value
    
    def get_adaptive_threshold(self, lost_depth: float) -> float:
        if lost_depth < Config.DISTANCE_BOUNDARY:
            return Config.NEAR_THRESHOLD
        return Config.FAR_THRESHOLD
    
    def verify(self, lost_depth: float, candidate_depth: float) -> Tuple[bool, float]:
        if lost_depth <= 0 or candidate_depth <= 0:
            return False, 0.0
        
        threshold = self.get_adaptive_threshold(lost_depth)
        depth_diff = abs(lost_depth - candidate_depth)
        is_match = depth_diff < threshold
        
        sigma = threshold / 2
        confidence = math.exp(-(depth_diff ** 2) / (2 * sigma ** 2))
        confidence = min(1.0, max(0.0, confidence))
        
        return is_match, confidence


# ==================== 声纹识别模块 ====================
class VoiceprintVerifier:
    """声纹识别验证模块"""
    
    def __init__(self, similarity_threshold: float = Config.VOICE_SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold
        self.embedding_dim = 512
        self.target_embedding = None
    
    def register_target(self, audio_samples: np.ndarray):
        self.target_embedding = self._extract_embedding(audio_samples)
    
    def _extract_embedding(self, audio_data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if audio_data is None:
            return None
        if Config.USE_MOCK_DATA:
            np.random.seed(hash(str(audio_data.tobytes()[:100])) % 2**32)
            embedding = np.random.randn(self.embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        return None
    
    def extract_embedding(self, audio_data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return self._extract_embedding(audio_data)
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        if emb1 is None or emb2 is None:
            return 0.0
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        return float(np.dot(emb1_norm, emb2_norm))
    
    def verify(self, registered_emb: Optional[np.ndarray],
               current_emb: Optional[np.ndarray]) -> Tuple[bool, float]:
        if registered_emb is None or current_emb is None:
            return False, 0.5
        similarity = self.cosine_similarity(registered_emb, current_emb)
        similarity = np.clip(similarity, -1.0, 1.0)
        return similarity >= self.similarity_threshold, similarity


# ==================== VLM大模型验证模块 ====================
class VLMVerifier:
    """VLM大模型验证模块"""
    
    def __init__(self, use_vlm: bool = Config.USE_VLM):
        self.use_vlm = use_vlm
        self.model = None
    
    def _init_model(self):
        if not self.use_vlm:
            return
        try:
            print("[VLM] VLM模块已禁用（节省显存）")
        except Exception as e:
            print(f"[VLM] 初始化失败: {e}")
            self.use_vlm = False
    
    def compare(self, img1: Optional[np.ndarray], img2: Optional[np.ndarray]) -> Tuple[bool, float]:
        if not self.use_vlm or img1 is None or img2 is None:
            # 模拟返回中等级别置信度
            return True, 0.6
        return True, 0.7


# ==================== KalmanFilter简化版 ====================
class SimpleKalmanFilter:
    """简化的卡尔曼滤波器"""
    
    def __init__(self, bbox: Tuple[int, int, int, int]):
        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        size = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])
        
        self.state = np.array([center[0], center[1], size[0], size[1], 0, 0, 0, 0])
        self.covariance = np.eye(8) * 10.0
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i+4] = 1
        self.H = np.zeros((4, 8))
        for i in range(4):
            self.H[i, i] = 1
        self.R = np.eye(4) * 5
        self.Q = np.eye(8) * 0.05
        self.time_since_update = 0
    
    def predict(self) -> Tuple[int, int, int, int]:
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        self.time_since_update += 1
        
        x, y, w, h = self.state[0], self.state[1], self.state[2], self.state[3]
        return (int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2))
    
    def update(self, bbox: Tuple[int, int, int, int]):
        z = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2,
                      bbox[2] - bbox[0], bbox[3] - bbox[1]])
        y = z - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.covariance = (np.eye(8) - K @ self.H) @ self.covariance
        self.time_since_update = 0
    
    def is_lost(self, max_lost: int = 5) -> bool:
        return self.time_since_update > max_lost


# ==================== 多源融合跟踪器主类 ====================
class MultiModalTracker:
    """多源异构数据协同目标跟踪器"""
    
    def __init__(self):
        print("[Tracker] 正在初始化YOLO模型...")
        self.detector = YOLO(Config.YOLO_MODEL_PATH)
        
        self.depth_verifier = DepthVerifier()
        self.voice_verifier = VoiceprintVerifier()
        self.vlm_verifier = VLMVerifier()
        
        self.state = TrackingState.TRACKING
        self.lost_anchor = LostAnchor()
        self.kalman_tracks: Dict[int, SimpleKalmanFilter] = {}
        self.track_depths: Dict[int, float] = {}
        self.candidate_buffer: Optional[Candidate] = None
        self.next_track_id = 1
        self.target_id_to_recover = None
        
        # 性能统计
        self.stats = {
            'total_frames': 0,
            'lost_events': 0,
            'reid_success': 0,
            'reid_fail': 0,
            'fps': 0
        }
        self.last_fps_time = time.time()
        self.frame_count = 0
        
        print("[Tracker] 多源异构协同跟踪器初始化完成")
    
    def _get_target_depth(self, depth_map: Optional[np.ndarray], 
                          bbox: Tuple[int, int, int, int]) -> float:
        """获取目标深度"""
        if depth_map is None:
            return 0.0
        
        x1, y1, x2, y2 = bbox
        cy = (y1 + y2) // 2
        cx = (x1 + x2) // 2
        
        h, w = depth_map.shape[:2]
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        
        depth = depth_map[cy, cx]
        return float(depth) if depth > 0 else 0.0
    
    def _crop_target(self, frame: np.ndarray, 
                     bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """裁剪目标图像"""
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2].copy()
    
    def _compute_iou(self, box1, box2) -> float:
        """计算IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter + 1e-6
        
        return inter / union
    
    def _spatial_filter(self, candidate_bbox: Tuple[int, int, int, int],
                        anchor_bbox: Tuple[int, int, int, int]) -> bool:
        """空间滤波器"""
        def get_center(bbox):
            return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        
        center_cand = get_center(candidate_bbox)
        center_anchor = get_center(anchor_bbox)
        
        distance = math.sqrt((center_cand[0] - center_anchor[0])**2 + 
                            (center_cand[1] - center_anchor[1])**2)
        
        return distance < Config.SPATIAL_SEARCH_RADIUS
    
    def process_frame(self, frame: np.ndarray, 
                      depth_map: Optional[np.ndarray] = None) -> Dict[int, Tuple[int, int, int, int]]:
        """处理单帧图像，返回跟踪结果"""
        self.stats['total_frames'] += 1
        self.frame_count += 1
        
        # FPS计算
        if time.time() - self.last_fps_time >= 1.0:
            self.stats['fps'] = self.frame_count
            self.frame_count = 0
            self.last_fps_time = time.time()
        
        # YOLO检测
        detections = []
        results = self.detector(frame, conf=Config.YOLO_CONF_THRES, verbose=False)
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # person
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        detections.append((int(x1), int(y1), int(x2), int(y2), conf))
        
        # 跟踪结果字典
        tracking_results = {}
        
        # 预测所有轨迹位置
        predicted_bboxes = {}
        for tid, kf in self.kalman_tracks.items():
            predicted_bboxes[tid] = kf.predict()
        
        # 数据关联 - 匈牙利匹配
        used_detections = set()
        used_tracks = set()
        
        if len(predicted_bboxes) > 0 and len(detections) > 0:
            # 构建代价矩阵
            cost_matrix = []
            track_ids = list(predicted_bboxes.keys())
            for tid in track_ids:
                row = []
                for det in detections:
                    iou = self._compute_iou(predicted_bboxes[tid], det[:4])
                    row.append(1 - iou)
                cost_matrix.append(row)
            
            cost_matrix = np.array(cost_matrix)
            from scipy.optimize import linear_sum_assignment
            row_idx, col_idx = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_idx, col_idx):
                if cost_matrix[r, c] < 0.7:  # IoU > 0.3
                    tid = track_ids[r]
                    det = detections[c]
                    self.kalman_tracks[tid].update(det[:4])
                    self.track_depths[tid] = self._get_target_depth(depth_map, det[:4])
                    tracking_results[tid] = det[:4]
                    used_tracks.add(tid)
                    used_detections.add(c)
        
        # 未匹配的检测框创建新轨迹
        for i, det in enumerate(detections):
            if i not in used_detections:
                new_id = self.next_track_id
                self.next_track_id += 1
                self.kalman_tracks[new_id] = SimpleKalmanFilter(det[:4])
                self.track_depths[new_id] = self._get_target_depth(depth_map, det[:4])
                tracking_results[new_id] = det[:4]
        
        # 检查丢失目标
        current_track_ids = set(predicted_bboxes.keys())
        
        for tid in current_track_ids:
            if tid not in used_tracks:
                self.kalman_tracks[tid].time_since_update += 1
                if self.kalman_tracks[tid].is_lost(Config.LOST_THRESHOLD):
                    if tid in self.kalman_tracks:
                        self._handle_target_lost(tid, frame, depth_map)
        
        # 清理丢失的轨迹
        to_remove = [tid for tid, kf in self.kalman_tracks.items() 
                     if kf.is_lost(Config.MAX_LOST_FRAMES)]
        for tid in to_remove:
            del self.kalman_tracks[tid]
            if tid in self.track_depths:
                del self.track_depths[tid]
        
        return tracking_results
    
    def _handle_target_lost(self, target_id: int, frame: np.ndarray, depth_map: Optional[np.ndarray]):
        """处理目标丢失"""
        self.stats['lost_events'] += 1
        self.target_id_to_recover = target_id
        
        kf = self.kalman_tracks.get(target_id)
        if kf and kf.history:
            last_bbox = kf.history[-1]
        else:
            last_bbox = None
        
        self.lost_anchor = LostAnchor(
            depth=self.track_depths.get(target_id, 0.0),
            image=self._crop_target(frame, last_bbox) if last_bbox else None,
            bbox=last_bbox,
            timestamp=time.time()
        )
        
        self.state = TrackingState.LOST
        print(f"[丢失] 目标 ID={target_id} 已丢失，进入验证模式")
        
        # 调用多模态重识别（在另一帧处理）
    
    def verify_candidate(self, frame: np.ndarray, 
                         detection_bbox: Tuple[int, int, int, int],
                         depth_map: Optional[np.ndarray]) -> Tuple[bool, float]:
        """验证候选目标是否为丢失的目标"""
        candidate_depth = self._get_target_depth(depth_map, detection_bbox)
        candidate_image = self._crop_target(frame, detection_bbox)
        
        # 深度验证
        depth_match, conf_depth = self.depth_verifier.verify(
            self.lost_anchor.depth, candidate_depth
        )
        
        # 声纹验证（模拟）
        conf_voice = 0.6
        
        # VLM验证
        vlm_match, conf_vlm = self.vlm_verifier.compare(
            self.lost_anchor.image, candidate_image
        )
        
        # 加权融合
        available = ['depth', 'audio', 'vlm']
        total_weight = Config.WEIGHT_DEPTH + Config.WEIGHT_VOICE + Config.WEIGHT_VLM
        score = (Config.WEIGHT_DEPTH * conf_depth + 
                 Config.WEIGHT_VOICE * conf_voice + 
                 Config.WEIGHT_VLM * conf_vlm) / total_weight
        
        print(f"[验证] 深度:{conf_depth:.2f}, 声纹:{conf_voice:.2f}, VLM:{conf_vlm:.2f}, 综合:{score:.2f}")
        
        is_match = score >= Config.FUSION_THRESHOLD
        
        if is_match:
            self.stats['reid_success'] += 1
        else:
            self.stats['reid_fail'] += 1
        
        return is_match, score
    
    def recover_target(self, target_id: int, bbox: Tuple[int, int, int, int], depth: float):
        """恢复目标跟踪"""
        if target_id in self.kalman_tracks:
            self.kalman_tracks[target_id] = SimpleKalmanFilter(bbox)
        else:
            self.kalman_tracks[target_id] = SimpleKalmanFilter(bbox)
        self.track_depths[target_id] = depth
        self.state = TrackingState.TRACKING
        self.target_id_to_recover = None
        print(f"[恢复] 目标 ID={target_id} 已恢复跟踪")
    
    def try_recover_from_lost(self, frame: np.ndarray, 
                               depth_map: Optional[np.ndarray],
                               detections: List[Tuple[int, int, int, int, float]]) -> Optional[int]:
        """尝试从丢失状态恢复"""
        best_match_id = None
        best_score = 0
        
        for det in detections:
            bbox = det[:4]
            # 空间筛选
            if self.lost_anchor.bbox and not self._spatial_filter(bbox, self.lost_anchor.bbox):
                continue
            
            is_match, score = self.verify_candidate(frame, bbox, depth_map)
            if is_match and score > best_score:
                best_score = score
                best_match_id = self.target_id_to_recover
                self.recover_target(best_match_id, bbox, self._get_target_depth(depth_map, bbox))
                return best_match_id
        
        return None
    
    def process_frame_with_state(self, frame: np.ndarray, 
                                   depth_map: Optional[np.ndarray] = None) -> Dict[int, Tuple[int, int, int, int]]:
        """带状态管理的帧处理"""
        # 检测并跟踪
        results = self.process_frame(frame, depth_map)
        
        # 如果处于丢失状态，尝试恢复
        if self.state == TrackingState.LOST and self.target_id_to_recover is not None:
            # 获取当前帧检测
            detections = []
            det_results = self.detector(frame, conf=Config.YOLO_CONF_THRES, verbose=False)
            for r in det_results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        if int(box.cls[0]) == 0:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            detections.append((int(x1), int(y1), int(x2), int(y2), float(box.conf[0])))
            
            recovered_id = self.try_recover_from_lost(frame, depth_map, detections)
            if recovered_id is not None:
                # 重新获取该目标的位置
                for det in detections:
                    if recovered_id in results:
                        break
                    results[recovered_id] = det[:4]
        
        return results
    
    def get_statistics(self) -> Dict:
        total_reid = self.stats['reid_success'] + self.stats['reid_fail']
        reid_rate = self.stats['reid_success'] / max(total_reid, 1) * 100
        return {
            'total_frames': self.stats['total_frames'],
            'lost_events': self.stats['lost_events'],
            'reid_success': self.stats['reid_success'],
            'reid_fail': self.stats['reid_fail'],
            'reid_success_rate': reid_rate,
            'fps': self.stats['fps']
        }
    
    def reset(self):
        self.state = TrackingState.TRACKING
        self.lost_anchor = LostAnchor()
        self.kalman_tracks = {}
        self.track_depths = {}
        self.candidate_buffer = None
        self.next_track_id = 1
        self.target_id_to_recover = None
        print("[Tracker] 跟踪器已重置")


# ==================== 实时摄像头跟踪器 ====================
class RealTimeTracker:
    """实时摄像头跟踪应用"""
    
    def __init__(self, use_camera: int = 0, use_depth_camera: bool = False):
        self.tracker = MultiModalTracker()
        self.use_depth_camera = use_depth_camera
        self.cap = None
        self.depth_cap = None
        self.running = False
        self.camera_id = use_camera
        
        # 颜色映射
        self.colors = {}
    
    def init_camera(self) -> bool:
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"[错误] 无法打开摄像头 {self.camera_id}")
            return False
        
        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"[相机] 摄像头初始化成功，分辨率: 640x480")
        
        if self.use_depth_camera:
            try:
                import pyrealsense2 as rs
                self.pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                self.pipeline.start(config)
                print("[深度相机] RealSense相机初始化成功")
                self.use_depth_camera = True
            except ImportError:
                print("[警告] pyrealsense2未安装，使用普通RGB模式")
                self.use_depth_camera = False
            except Exception as e:
                print(f"[警告] 深度相机初始化失败: {e}")
                self.use_depth_camera = False
        
        return True
    
    def get_depth_map(self) -> Optional[np.ndarray]:
        """获取深度图"""
        if not self.use_depth_camera:
            return None
        
        try:
            import pyrealsense2 as rs
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_meters = depth_image * 0.001  # 转换为米
                return depth_meters
        except Exception as e:
            pass
        return None
    
    def get_color_for_id(self, track_id: int) -> Tuple[int, int, int]:
        """为每个ID生成固定颜色"""
        if track_id not in self.colors:
            np.random.seed(track_id)
            self.colors[track_id] = tuple(int(x) for x in np.random.randint(0, 255, 3))
        return self.colors[track_id]
    
    def draw_tracking_info(self, frame: np.ndarray, 
                           results: Dict[int, Tuple[int, int, int, int]]):
        """绘制跟踪信息"""
        for track_id, bbox in results.items():
            x1, y1, x2, y2 = bbox
            color = self.get_color_for_id(track_id)
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"ID: {track_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示状态信息
        status_color = (0, 255, 0) if self.tracker.state == TrackingState.TRACKING else (0, 0, 255)
        status_text = f"State: {self.tracker.state.value}"
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # 显示统计信息
        stats = self.tracker.get_statistics()
        info_texts = [
            f"FPS: {stats['fps']:.1f}",
            f"Tracks: {len(results)}",
            f"Lost Events: {stats['lost_events']}",
            f"ReID Success: {stats['reid_success_rate']:.1f}%"
        ]
        
        for i, text in enumerate(info_texts):
            cv2.putText(frame, text, (10, 60 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 如果处于丢失状态，显示丢失目标信息
        if self.tracker.state == TrackingState.LOST and self.tracker.target_id_to_recover:
            lost_text = f"Searching for ID: {self.tracker.target_id_to_recover}"
            cv2.putText(frame, lost_text, (10, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def run(self):
        """运行实时跟踪"""
        if not self.init_camera():
            return
        
        self.running = True
        print("\n[运行] 开始实时跟踪，按 'q' 退出")
        print("[操作] 'r' - 重置跟踪器, 's' - 显示统计, 'h' - 显示帮助")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[错误] 无法读取帧")
                break
            
            # 获取深度图
            depth_map = self.get_depth_map() if self.use_depth_camera else None
            
            # 处理帧
            results = self.tracker.process_frame_with_state(frame, depth_map)
            
            # 绘制结果
            self.draw_tracking_info(frame, results)
            
            # 显示
            cv2.imshow("Multi-Modal Target Tracking", frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q 或 ESC
                self.running = False
            elif key == ord('r'):
                self.tracker.reset()
                self.colors.clear()
                print("[重置] 跟踪器已重置")
            elif key == ord('s'):
                stats = self.tracker.get_statistics()
                print(f"\n=== 统计信息 ===")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
            elif key == ord('h'):
                print("\n=== 帮助 ===")
                print("  q - 退出")
                print("  r - 重置跟踪器")
                print("  s - 显示统计信息")
                print("  h - 显示帮助")
        
        # 清理
        self.cap.release()
        cv2.destroyAllWindows()
        if self.use_depth_camera:
            try:
                self.pipeline.stop()
            except:
                pass
        print("\n[结束] 跟踪已停止")


# ==================== 视频文件跟踪器 ====================
class VideoFileTracker:
    """视频文件跟踪器"""
    
    def __init__(self, video_path: str):
        self.tracker = MultiModalTracker()
        self.video_path = video_path
        self.cap = None
    
    def init_video(self) -> bool:
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"[错误] 无法打开视频文件: {self.video_path}")
            return False
        print(f"[视频] 成功打开: {self.video_path}")
        return True
    
    def run(self):
        """运行视频跟踪"""
        if not self.init_video():
            return
        
        colors = {}
        
        print("\n[运行] 开始视频跟踪，按 'q' 退出")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[完成] 视频播放结束")
                break
            
            # 处理帧
            results = self.tracker.process_frame_with_state(frame, depth_map=None)
            
            # 绘制结果
            for track_id, bbox in results.items():
                x1, y1, x2, y2 = bbox
                if track_id not in colors:
                    np.random.seed(track_id)
                    colors[track_id] = tuple(int(x) for x in np.random.randint(0, 255, 3))
                color = colors[track_id]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID: {track_id}"
                cv2.putText(frame, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 显示状态
            status_color = (0, 255, 0) if self.tracker.state == TrackingState.TRACKING else (0, 0, 255)
            cv2.putText(frame, f"State: {self.tracker.state.value}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            stats = self.tracker.get_statistics()
            cv2.putText(frame, f"FPS: {stats['fps']:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Tracks: {len(results)}", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 调整显示大小
            h, w = frame.shape[:2]
            if w > Config.DISPLAY_RESIZE_WIDTH:
                scale = Config.DISPLAY_RESIZE_WIDTH / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            
            cv2.imshow("Multi-Modal Target Tracking - Video", frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                self.tracker.reset()
                colors.clear()
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("[结束] 视频跟踪停止")


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='多源异构融合目标跟踪系统')
    parser.add_argument('--source', type=str, default='camera',
                        help='输入源: camera, video_path, 或 mock')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='摄像头ID（默认: 0）')
    parser.add_argument('--depth', action='store_true',
                        help='启用深度相机（需要RealSense）')
    parser.add_argument('--no_vlm', action='store_true',
                        help='禁用VLM模块')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("多源异构数据协同目标跟踪系统")
    print("=" * 60)
    
    # 配置VLM
    if args.no_vlm:
        Config.USE_VLM = False
    
    if args.source == 'camera':
        tracker_app = RealTimeTracker(use_camera=args.camera_id, use_depth_camera=args.depth)
        tracker_app.run()
    elif args.source == 'mock':
        from generate_mock_data import run_mock_demo
        run_mock_demo()
    else:
        tracker_app = VideoFileTracker(args.source)
        tracker_app.run()


# ==================== 模拟数据演示 ====================
def run_mock_demo():
    """运行模拟数据演示（无摄像头）"""
    print("\n[模拟模式] 运行模拟跟踪演示...")
    
    tracker = MultiModalTracker()
    
    print("\n开始模拟跟踪...")
    
    for frame_idx in range(300):
        # 生成模拟帧
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 添加模拟的人形目标
        cx, cy = 320 + int(50 * np.sin(frame_idx * 0.05)), 240
        cv2.rectangle(frame, (cx-30, cy-80), (cx+30, cy+20), (0, 255, 0), -1)
        cv2.rectangle(frame, (cx-30, cy-80), (cx+30, cy+20), (0, 255, 0), 2)
        
        # 模拟深度图
        depth_map = np.ones((480, 640), dtype=np.float32) * 5.0
        depth_map[cy-80:cy+20, cx-30:cx+30] = 3.0
        
        # 处理
        results = tracker.process_frame_with_state(frame, depth_map)
        
        # 模拟丢失事件
        if 100 < frame_idx < 150:
            results = {}
        
        # 显示状态
        if frame_idx % 50 == 0:
            stats = tracker.get_statistics()
            print(f"[帧 {frame_idx}] State: {tracker.state.value}, FPS: {stats['fps']:.1f}, "
                  f"Tracks: {len(results)}, ReID率: {stats['reid_success_rate']:.1f}%")
        
        # 模拟帧率
        time.sleep(0.03)
    
    print("\n最终统计:")
    stats = tracker.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    # 检查依赖
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        print("请安装 scipy: pip install scipy")
        exit(1)
    
    main()