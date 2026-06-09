import sys
import os
import time
import math
import queue
import threading
import collections
import numpy as np
import cv2
import psutil
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

# Ép buộc OpenCV sử dụng giao thức TCP cho luồng RTSP (tránh mất gói UDP và tăng độ ổn định)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
# GStreamer: do NOT force-disable — required for NVDEC hardware decode.
# Remove the line below if it exists; it was preventing GStreamer from opening RTSP.
# os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QSlider, QCheckBox,
    QTextEdit, QGroupBox, QFormLayout, QGridLayout, QListWidget,
    QListWidgetItem, QSplitter, QStyle, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont, QPainter

# Triton gRPC client
import json
import tritonclient.grpc as grpcclient
from concurrent.futures import ThreadPoolExecutor, wait

from pipeline_utils import (
    ENSEMBLE_MODEL,
    GRPC_TIMEOUT_SEC,
    INFER_MAX_EDGE,
    get_thread_grpc_client,
    infer_frame,
    normalize_rtsp_url,
    is_single_model,
    clear_ensemble_cache,
)
from video_source import VideoSourceReader, DEFAULT_DECODER, opencv_gstreamer_enabled

# ==============================================================================
# HÀM BỎ CUỘN CHUỘT CHO COMBOBOX (CHỐNG LĂN CHUỘT NHẦM)
# ==============================================================================
class NonScrollComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, e):
        e.ignore()

# ==============================================================================
# TRACKER ĐƠN GIẢN — IoU MATCHING + EMA BBOX SMOOTHING
# ==============================================================================
class SimpleTracker:
    """
    Tracker nhẹ không cần thư viện ngoài.
    - Greedy IoU matching để giữ ID ổn định qua các frame.
    - EMA smoothing để bbox trượt mượt giữa các lần inference round-robin.
    - max_age giữ track sống khi camera chưa được infer lại.
    """
    def __init__(self, smooth_alpha=0.45, max_age=20, iou_thresh=0.25):
        self.alpha      = smooth_alpha  # 0=frozen, 1=raw detection (không smooth)
        self.max_age    = max_age       # frame tối đa giữ track khi mất detection
        self.iou_thresh = iou_thresh    # ngưỡng IoU tối thiểu để match
        self._tracks    = {}            # tid -> track dict
        self._next_id   = 0

    @staticmethod
    def _iou(a, b):
        ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
        ix1=max(ax1,bx1); iy1=max(ay1,by1)
        ix2=min(ax2,bx2); iy2=min(ay2,by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0: return 0.0
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

    def update(self, dets_by_model: dict):
        """Nhận detection mới {model: [det...]}, cập nhật EMA tracks."""
        # Flatten tất cả detections kèm model tag
        new_dets = []
        for model_name, dets in dets_by_model.items():
            for d in dets:
                new_dets.append({**d, "_model": model_name})

        # Tăng tuổi tất cả tracks, xóa track quá cũ
        for tid in list(self._tracks):
            self._tracks[tid]["age"] += 1
            if self._tracks[tid]["age"] > self.max_age:
                del self._tracks[tid]

        if not new_dets:
            return

        matched_tids = set()
        matched_dis  = set()

        # Greedy IoU matching — ưu tiên IoU cao nhất cùng class và model
        for di, det in enumerate(new_dets):
            best_iou, best_tid = self.iou_thresh, None
            for tid, t in self._tracks.items():
                if tid in matched_tids: continue
                if t["class_id"] != det["class_id"]: continue
                if t["model"] != det["_model"]: continue
                iou = self._iou(det["bbox"], t["bbox_raw"])
                if iou > best_iou:
                    best_iou, best_tid = iou, tid

            if best_tid is not None:
                t = self._tracks[best_tid]
                a = self.alpha
                # EMA: bbox mượt dần về vị trí mới
                t["bbox"]     = [a*det["bbox"][i] + (1-a)*t["bbox"][i] for i in range(4)]
                t["bbox_raw"] = list(det["bbox"])
                t["confidence"] = det["confidence"]
                t["age"] = 0
                matched_tids.add(best_tid)
                matched_dis.add(di)

        # Tạo track mới cho detection chưa được match
        for di, det in enumerate(new_dets):
            if di not in matched_dis:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "bbox":       list(det["bbox"]),
                    "bbox_raw":   list(det["bbox"]),
                    "class_id":   det["class_id"],
                    "confidence": det["confidence"],
                    "model":      det["_model"],
                    "age":        0,
                    "id":         tid,
                }

    def get_draw_dets(self) -> dict:
        """Chỉ render track có age <= 1 (vừa được detection match).
        Tránh ghost box khi đối tượng đã di chuyển khỏi vị trí cũ."""
        result = {}
        for t in self._tracks.values():
            if t["age"] > 1:   # Ẩn ngay nếu miss 2 cycle liên tiếp
                continue
            m = t["model"]
            if m not in result:
                result[m] = []
            result[m].append({
                "bbox":       [round(v, 1) for v in t["bbox"]],
                "class_id":   t["class_id"],
                "confidence": t["confidence"],
                "track_id":   t["id"],
            })
        return result

# ==============================================================================
# VẼ ĐỐI TƯỢNG (DÙNG MÀU ĐEN - TRẮNG TỐI GIẢN)
# ==============================================================================
def draw(frame: np.ndarray, all_detections: dict, class_names: dict, bbox_style: str = "White Outline", custom_colors: dict = None):
    out = frame.copy()
    
    # Mặc định ban đầu
    main_color = (255, 255, 255)
    text_color = (0, 0, 0)
    bg_color = (255, 255, 255)

    for model_name in all_detections.keys():
        dets = all_detections.get(model_name) or []
        names = class_names.get(model_name)

        for det in dets:
            x1, y1, x2, y2 = (int(v) for v in det["bbox"])
            conf = det["confidence"]
            cid = det["class_id"]

            lbl_name = names[cid] if names and cid < len(names) else f"{model_name}_{cid}"
            
            # Định nghĩa màu vẽ dựa trên cấu hình trắng-đen hoặc có màu phân biệt đối tượng
            if bbox_style == "White Outline":
                main_color = (255, 255, 255)
                text_color = (0, 0, 0)
                bg_color = (255, 255, 255)
            elif bbox_style == "Light Gray":
                main_color = (200, 200, 200)
                text_color = (0, 0, 0)
                bg_color = (200, 200, 200)
            elif bbox_style == "Medium Gray":
                main_color = (128, 128, 128)
                text_color = (255, 255, 255)
                bg_color = (128, 128, 128)
            elif bbox_style == "Dark Gray":
                main_color = (64, 64, 64)
                text_color = (255, 255, 255)
                bg_color = (64, 64, 64)
            elif bbox_style == "Gray Scale":
                main_color = (160, 160, 160)
                text_color = (255, 255, 255)
                bg_color = (80, 80, 80)
            elif bbox_style == "Dark Outline":
                main_color = (0, 0, 0)
                text_color = (255, 255, 255)
                bg_color = (0, 0, 0)
            elif bbox_style == "Custom Colors" and custom_colors:
                name_lower = lbl_name.lower()
                main_color = None
                for k, color in custom_colors.items():
                    if k.lower() in name_lower or name_lower in k.lower():
                        main_color = color
                        break
                if main_color is None:
                    # Fallback to Color-Coded
                    if "fire" in name_lower or "smoke" in name_lower:
                        main_color = (0, 69, 255)
                    elif "person" in name_lower:
                        main_color = (255, 255, 255)
                    elif "fall" in name_lower:
                        main_color = (0, 200, 255)
                    elif "vehicle" in name_lower or "car" in name_lower:
                        main_color = (200, 200, 200)
                    else:
                        h = hash(name_lower)
                        main_color = (max(100, (h & 0xFF)), max(100, ((h >> 8) & 0xFF)), max(100, ((h >> 16) & 0xFF)))
                text_color = (0, 0, 0) if (main_color[0] + main_color[1] + main_color[2]) > 380 else (255, 255, 255)
                bg_color = main_color
            else: # "Color-Coded"
                name_lower = lbl_name.lower()
                if "fire" in name_lower or "smoke" in name_lower:
                    main_color = (0, 69, 255)   # Cam đỏ
                elif "person" in name_lower:
                    main_color = (255, 255, 255) # Trắng
                elif "fall" in name_lower:
                    main_color = (0, 200, 255)  # Vàng
                elif "vehicle" in name_lower or "car" in name_lower:
                    main_color = (200, 200, 200) # Xám nhạt
                else:
                    h = hash(name_lower)
                    main_color = (max(100, (h & 0xFF)), max(100, ((h >> 8) & 0xFF)), max(100, ((h >> 16) & 0xFF)))
                text_color = (0, 0, 0)
                bg_color = main_color

            tid_str = f"#{det.get('track_id', '')} " if "track_id" in det else ""
            label = f"{tid_str}{lbl_name} {conf:.2f}"
            _font_scale = 0.65
            _font_thick = 2
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, _font_scale, _font_thick)
            lx, ly = x1, max(y1 - th - baseline - 6, 0)

            # Vẽ nền nhãn
            cv2.rectangle(out, (lx, ly), (lx + tw + 6, ly + th + baseline + 4), bg_color, -1)
            # Vẽ văn bản
            cv2.putText(out, label, (lx + 3, ly + th + 3), cv2.FONT_HERSHEY_SIMPLEX, _font_scale, text_color, _font_thick, cv2.LINE_AA)
            
            # Vẽ viền
            if bbox_style == "White Outline":
                cv2.rectangle(out, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), 1)
                cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)
            elif bbox_style in ["Gray Scale", "Light Gray", "Medium Gray", "Dark Gray"]:
                cv2.rectangle(out, (x1, y1), (x2, y2), main_color, 2)
            elif bbox_style == "Dark Outline":
                cv2.rectangle(out, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 255, 255), 1)
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), 1)
            else: # Color-Coded hoặc Custom Colors
                cv2.rectangle(out, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), 1)
                cv2.rectangle(out, (x1, y1), (x2, y2), main_color, 2)

    return out

# ==============================================================================
# HÀM GHÉP LƯỚI ĐA LUỒNG TỰ ĐỘNG GIỮ TỶ LỆ KHUNG HÌNH (NO-DISTORTION GRID)
# ==============================================================================
def create_grid(frames: list, cell_width=640, cell_height=480):
    N = len(frames)
    
    # Tính aspect ratio thực tế từ khung hình hợp lệ đầu tiên để tránh méo hình
    ref_w, ref_h = cell_width, cell_height
    for f in frames:
        if f is not None:
            fh, fw = f.shape[:2]
            if fh > 0 and fw > 0:
                ref_w = cell_width
                ref_h = int(cell_width * fh / fw)
                break

    if N == 0:
        return np.zeros((ref_h, ref_w, 3), dtype=np.uint8)

    cols = math.ceil(math.sqrt(N))
    rows = math.ceil(N / cols)

    resized_frames = []
    for frame in frames:
        if frame is None:
            resized_frames.append(np.zeros((ref_h, ref_w, 3), dtype=np.uint8))
        else:
            if frame.shape[0] > 0 and frame.shape[1] > 0:
                resized_frames.append(cv2.resize(frame, (ref_w, ref_h)))
            else:
                resized_frames.append(np.zeros((ref_h, ref_w, 3), dtype=np.uint8))

    total_cells = rows * cols
    while len(resized_frames) < total_cells:
        resized_frames.append(np.zeros((ref_h, ref_w, 3), dtype=np.uint8))

    grid_rows = []
    for r in range(rows):
        row_frames = resized_frames[r * cols : (r + 1) * cols]
        grid_rows.append(np.hstack(row_frames))

    grid = np.vstack(grid_rows)
    return grid

# ==============================================================================
# QTHREAD ĐIỀU PHỐI SUY LUẬN AI BẤT ĐỒNG BỘ
# ==============================================================================
class TritonInferenceThread(QThread):
    # Phát tín hiệu: frame, lat_infer, lat_post, fps, api_info, lat_e2e, lat_grpc_ping
    frame_ready = pyqtSignal(np.ndarray, float, float, float, str, float, float, str, float, float, float)
    log_emitted = pyqtSignal(str)
    # Fractional playback position (0.0–1.0) for video-file sources; drives the seek slider in the UI
    video_position_updated = pyqtSignal(float)

    def __init__(self, host, sources, models, ensemble_model, use_ensemble, conf, iou, stride, labels_map, bbox_style, custom_colors, decoder="Auto", grid_size=(640, 480)):
        super().__init__()
        self.host = host
        self.sources = [normalize_rtsp_url(s) for s in sources]
        self.models_list = models
        self.ensemble_model = ensemble_model
        self.use_ensemble = use_ensemble
        self.conf = conf
        self.iou = iou
        self.stride = stride
        self.labels_map = labels_map
        self.bbox_style = bbox_style
        self.custom_colors = custom_colors
        self.decoder = decoder
        self.grid_size = grid_size
        self.running = True
        self._playback_speed = 1          # 1 = normal, 2/3/4 = fast forward
        self._speed_lock = threading.Lock()
        # Thread-safe seek request: GUI sets this via seek_video(); run() drains it each cycle
        self._seek_pct  = None
        self._seek_lock = threading.Lock()

    def set_speed(self, multiplier: int):
        """Thread-safe: set playback speed multiplier (1/2/3/4). Only affects video files."""
        with self._speed_lock:
            self._playback_speed = max(1, int(multiplier))

    def run(self):
        self.log_emitted.emit(f"[CLIENT] Đang gửi yêu cầu gRPC kết nối tới Triton Server: {self.host}...")
        try:
            client = grpcclient.InferenceServerClient(url=self.host, verbose=False)
            if not client.is_server_ready(client_timeout=GRPC_TIMEOUT_SEC):
                self.log_emitted.emit("[SERVER] LỖI: Server Triton phản hồi chưa sẵn sàng!")
                return
            else:
                self.log_emitted.emit("[SERVER] Xác nhận: Server Triton hoạt động bình thường, trạng thái READY.")
        except Exception as e:
            self.log_emitted.emit(f"[SERVER] LỖI: Không thể thiết lập phiên gRPC tới Server: {e}")
            return

        # Đo gRPC network latency ban đầu
        self._grpc_ping_ms = 0.0
        self._ping_counter = 0
        try:
            t_ping = time.perf_counter()
            client.is_server_ready(client_timeout=GRPC_TIMEOUT_SEC)
            self._grpc_ping_ms = (time.perf_counter() - t_ping) * 1000.0
            self.log_emitted.emit(f"[CLIENT->SERVER] Độ trễ đường truyền mạng (gRPC Ping): {self._grpc_ping_ms:.1f} ms")
        except Exception:
            pass

        self.log_emitted.emit("[CLIENT] Đang tạo các luồng độc lập để giải mã camera...")
        readers = []
        for idx, src_path in enumerate(self.sources):
            r = VideoSourceReader(src_path, name=f"src_{idx}", decoder=self.decoder)
            if r.is_opened():
                readers.append(r)
                self.log_emitted.emit(f"[CLIENT] Khởi chạy camera nguồn #{idx} THÀNH CÔNG.")
                self.log_emitted.emit(f"[DECODER RAW LOG] Nguồn #{idx} -> {r.raw_pipeline_info}")
            else:
                self.log_emitted.emit(f"[CLIENT] LỖI THẤT BẠI: Không thể mở camera nguồn #{idx}: {src_path}")
                r.release()

        if not readers:
            self.log_emitted.emit("LỖI: Không có nguồn video nào hoạt động!")
            return

        self.log_emitted.emit("Bắt đầu xử lý luồng suy luận AI...")
        frame_idx = 0
        cached_dets = {}
        last_time = time.perf_counter()
        _fps_ema         = 0.0   # EMA display-loop FPS (nội bộ, không emit)
        _infer_fps_ema   = 0.0   # EMA inference-output FPS (tốc độ model thực tế)
        _last_infer_time = 0.0   # timestamp lần inference gần nhất
        _last_lat_infer  = 0.0   # lat_infer persisted từ stride gần nhất
        _last_lat_post   = 0.0   # lat_post persisted từ stride gần nhất
        # Separate pools: nesting model jobs on the camera pool deadlocks when max_workers=1.
        _cam_executor = ThreadPoolExecutor(
            max_workers=min(32, max(1, len(readers))))

        # ── Pre-compute stable inference params ONCE ────────────────────────────────
        # These values don't change during a session. Computing them here avoids
        # repeating list comprehensions and attribute lookups on every stride cycle.
        _host          = self.host
        _ensembles     = [m for m in self.models_list if not is_single_model(get_thread_grpc_client(_host), m)]
        _use_ens       = self.use_ensemble or len(_ensembles) > 0
        _ens_name      = _ensembles[0] if _ensembles else ENSEMBLE_MODEL
        _conf          = self.conf
        _iou           = self.iou
        _labels_map    = self.labels_map
        _bbox_style    = self.bbox_style
        _custom_colors = self.custom_colors

        # ── Define parallel worker functions ONCE before the loop ───────────────────
        # Python closures are cheap but redefining them inside a hot loop is wasteful
        # and confusing. Defining them here makes the loop body cleaner and avoids
        # recreating the closure object on every stride-th frame.

        def _read_cam(item):
            """Read one decoded frame from a VideoSourceReader.
            Parallelising reads overlaps I/O wait and GPU-decode time across cameras."""
            cam_idx, reader = item
            ret, f = reader.read()
            return cam_idx, ret, f

        def _infer_cam(item):
            """Send one camera frame to Triton via gRPC and return detections.
            Each call obtains its own thread-local gRPC stub so multiple cameras
            can fire concurrent RPCs without channel contention."""
            cam_idx, cam_frame = item
            dets, li, lp, err = infer_frame(
                get_thread_grpc_client(_host),
                cam_frame,
                self.models_list,
                _conf,
                _iou,
                use_ensemble=_use_ens,
                ensemble_name=_ens_name,
                max_edge=INFER_MAX_EDGE,
            )
            return cam_idx, dets, li, lp, err

        def _annotate_cam(item):
            """Draw bounding boxes and camera overlay on one frame (CPU-bound).
            Parallelising this step saves ≈N × draw_time when N cameras are active."""
            cam_idx, raw_frame = item
            src_dets  = cached_dets.get(cam_idx, {})
            annotated = draw(raw_frame, src_dets, _labels_map, _bbox_style, _custom_colors)
            _, cam_fps, cam_drop = readers[cam_idx].get_stats()
            cam_label = f"CAM #{cam_idx} | FPS: {cam_fps:.1f} | Drop: {cam_drop:.1f}%"
            (clw, clh), _ = cv2.getTextSize(cam_label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
            cv2.rectangle(annotated, (8, 8), (clw + 20, clh + 20), (0, 0, 0), -1)
            cv2.putText(annotated, cam_label, (14, clh + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
            return cam_idx, annotated

        while self.running:
            t_loop_start = time.perf_counter()

            # ── Apply pending seek (video files only) ───────────────────────────────
            # seek_video() writes _seek_pct from the GUI thread; we drain it here
            # under the lock so reads are always consistent.
            with self._seek_lock:
                seek_pct = self._seek_pct
                self._seek_pct = None
            if seek_pct is not None:
                for r in readers:
                    try:
                        r.seek(seek_pct)
                    except Exception:
                        pass  # never touch cap directly — races with reader thread

            # ── PARALLEL frame read: all cameras decoded concurrently ───────────────
            # Sequential reads block on each camera in turn; parallel reads let slow
            # RTSP or GPU-decode waits overlap, reducing wall-clock time proportionally.
            read_futs    = [_cam_executor.submit(_read_cam, (idx, r))
                            for idx, r in enumerate(readers)]
            raw_frames   = [None] * len(readers)
            valid_frames = {}
            for fut in read_futs:
                idx, ret, f = fut.result()
                if ret and f is not None and f.ndim == 3 and f.shape[0] > 0 and f.shape[1] > 0 and f.shape[2] == 3:
                    f = np.ascontiguousarray(f)
                    raw_frames[idx] = f
                    valid_frames[idx] = f

            # ── Playback speed: push speed multiplier to all video readers ──────────
            with self._speed_lock:
                _speed = self._playback_speed
            for r in readers:
                try:
                    r.set_speed(_speed)
                except Exception:
                    pass

            if all(f is None for f in raw_frames):
                time.sleep(0.01)
                continue

            # ── Emit video playback position (0.0–1.0) for the seek slider ──────────
            # Probe the first reader's underlying VideoCapture for frame position.
            # This is a best-effort operation: silently skipped for RTSP sources
            # (FRAME_COUNT == 0) and for any VideoSourceReader that hides the cap.
            if seek_pct is None:
                try:
                    pos_pct = readers[0].get_position()  # thread-safe method on VideoSourceReader
                    if pos_pct is not None:
                        self.video_position_updated.emit(float(pos_pct))
                except Exception:
                    pass  # get_position() not implemented or RTSP source — slider stays frozen

            frame_idx += 1
            # Dùng lại latency thực từ lần inference gần nhất (không reset về 0)
            lat_infer = _last_lat_infer
            lat_post  = _last_lat_post

            # Infer song song tất cả camera cùng lúc — mỗi camera 1 thread
            if frame_idx % self.stride == 0:
                t0 = time.perf_counter()

                futures = {
                    _cam_executor.submit(_infer_cam, item): item[0]
                    for item in valid_frames.items()
                }
                lat_infer, lat_post = 0.0, 0.0
                n_done = 0
                done, pending = wait(
                    futures, timeout=GRPC_TIMEOUT_SEC + 5.0)
                for fut in done:
                    cam_idx, cam_result, li, lp, err = fut.result()
                    if err:
                        self.log_emitted.emit(f"LỖI inference cam #{cam_idx}: {err}")
                    else:
                        cached_dets[cam_idx] = cam_result
                        lat_infer += li
                        lat_post += lp
                        n_done += 1
                for fut in pending:
                    cam_idx = futures[fut]
                    self.log_emitted.emit(
                        f"LỖI inference cam #{cam_idx}: timeout sau {GRPC_TIMEOUT_SEC:.0f}s")
                if n_done:
                    lat_infer /= n_done
                    lat_post  /= n_done
                # Lưu lại để dùng trên các frame không phải stride
                _last_lat_infer = lat_infer
                _last_lat_post  = lat_post
                # Đo inference-output FPS thực tế (tốc độ model xử lý)
                _t_infer_now = time.perf_counter()
                if _last_infer_time > 0.0:
                    _dt_infer = _t_infer_now - _last_infer_time
                    _raw_infer_fps = 1.0 / _dt_infer if _dt_infer > 1e-6 else 0.0
                    _infer_fps_ema = 0.2 * _raw_infer_fps + 0.8 * _infer_fps_ema if _infer_fps_ema > 0.0 else _raw_infer_fps
                _last_infer_time = _t_infer_now

            # ── PARALLEL frame annotation: draw bboxes on all cameras concurrently ──
            # draw() is pure CPU work; running N cameras in parallel reduces wall-clock
            # annotation time from N × draw_time to ≈ max(draw_time_i).
            ann_items = [(idx, f) for idx, f in enumerate(raw_frames) if f is not None]
            if ann_items:
                ann_futs      = [_cam_executor.submit(_annotate_cam, item) for item in ann_items]
                annotated_map = {}
                for fut in ann_futs:
                    cam_idx, ann = fut.result()
                    annotated_map[cam_idx] = ann
                annotated_frames = [annotated_map.get(i) for i in range(len(raw_frames))]
            else:
                annotated_frames = [None] * len(raw_frames)

            grid_img = create_grid(annotated_frames, cell_width=self.grid_size[0], cell_height=self.grid_size[1])

            # Đo display-loop FPS (nội bộ, không emit ra UI)
            t_now    = time.perf_counter()
            dt       = t_now - last_time
            raw_fps  = 1.0 / dt if dt > 1e-6 else 0.0
            _fps_ema = 0.15 * raw_fps + 0.85 * _fps_ema if _fps_ema > 0.0 else raw_fps
            # fps emit = tốc độ inference thực tế, không phải tốc độ vòng lặp hiển thị
            fps      = _infer_fps_ema
            last_time = t_now

            # ── OUTPUT FPS overlay — scales with grid width, always readable ──
            fps_text = f"OUT {fps:.1f} FPS"
            _fscale = max(0.8, min(1.8, grid_img.shape[1] / 1000.0))
            _thickness = 2
            (ftw, fth), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, _fscale, _thickness)
            _gx = grid_img.shape[1] - ftw - 18
            _gy = fth + 18
            cv2.rectangle(grid_img, (_gx - 8, 4), (_gx + ftw + 8, _gy + 8), (0, 0, 0), -1)
            _fps_color = (0, 80, 255) if fps < 8 else (0, 200, 255) if fps < 15 else (0, 255, 128)
            cv2.putText(grid_img, fps_text, (_gx, _gy), cv2.FONT_HERSHEY_SIMPLEX, _fscale, _fps_color, _thickness, cv2.LINE_AA)

            # E2E Latency: toàn bộ pipeline từ đọc frame đến sẵn sàng hiển thị
            lat_e2e = (time.perf_counter() - t_loop_start) * 1000.0

            # Đo gRPC ping mỗi 100 frame để không ảnh hưởng hiệu năng
            self._ping_counter += 1
            if self._ping_counter % 100 == 0:
                try:
                    t_ping = time.perf_counter()
                    client.is_server_ready(client_timeout=GRPC_TIMEOUT_SEC)
                    self._grpc_ping_ms = (time.perf_counter() - t_ping) * 1000.0
                except Exception:
                    pass

            # Chi tiết hoạt động API Triton gRPC đang sử dụng
            if _use_ens:
                api_info = f"gRPC ensemble/{_ens_name} (1 RPC, client preprocess)"
            else:
                api_info = f"gRPC ModelInfer (client preprocess + {','.join(self.models_list)})"

            # Lấy thông tin giải mã từ các nguồn
            decoder_info = ", ".join(sorted(list(set(r.decoder_backend for r in readers))))

            # Thu thập decoder performance stats từ tất cả readers
            all_decode_lats, all_src_fps, all_drop_rates = [], [], []
            for r in readers:
                dl, sf, dr = r.get_stats()
                all_decode_lats.append(dl)
                all_src_fps.append(sf)
                all_drop_rates.append(dr)
            avg_decode_lat = float(np.mean(all_decode_lats)) if all_decode_lats else 0.0
            avg_src_fps   = float(np.mean(all_src_fps))    if all_src_fps    else 0.0
            avg_drop_rate = float(np.mean(all_drop_rates)) if all_drop_rates else 0.0

            self.frame_ready.emit(grid_img, lat_infer, lat_post, fps, api_info, lat_e2e, self._grpc_ping_ms, decoder_info, avg_decode_lat, avg_src_fps, avg_drop_rate)

            # Giới hạn FPS hiển thị
            elapsed = time.perf_counter() - t_loop_start
            sleep_time = max(0.008, 0.050 - elapsed)   # Cap ~20 FPS display; reduce CPU spin
            time.sleep(sleep_time)

        # Giải phóng khi tắt luồng
        _cam_executor.shutdown(wait=False, cancel_futures=True)
        for r in readers:
            r.release()
        self.log_emitted.emit("Bộ xử lý luồng AI đã dừng giải phóng tài nguyên.")

    def seek_video(self, pct: float):
        """Request a seek to fractional position *pct* (0.0–1.0) across all video-file readers.
        Thread-safe — the seek is applied at the top of the next run() cycle.
        No-op for live RTSP streams where CAP_PROP_FRAME_COUNT == 0."""
        with self._seek_lock:
            self._seek_pct = max(0.0, min(1.0, pct))

    def stop(self):
        """Non-blocking: do not call wait() here — that freezes the Qt GUI thread."""
        self.running = False

# ==============================================================================
# LỚP HIỂN THỊ HÌNH ẢNH VIDEO TỰ CO GIÃN HỢP LỆ (VIDEO VIEWPORT)
# ==============================================================================
class VideoLabel(QLabel):
    fullscreen_requested = pyqtSignal()
    escape_requested = pyqtSignal()

    def __init__(self, text="TRITON MONOCHROME WORKBENCH", parent=None):
        super().__init__(text, parent)
        self.pix = None
        self.setMinimumSize(160, 120)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_frame(self, q_img):
        self.pix = QPixmap.fromImage(q_img)
        self.update()

    def paintEvent(self, event):
        if self.pix and not self.pix.isNull():
            painter = QPainter(self)
            # Điền màu nền đen
            painter.fillRect(self.rect(), Qt.GlobalColor.black)
            # Tính toán tỷ lệ co giãn tối ưu giữ nguyên tỷ lệ khung hình
            size = self.size()
            scaled_pix = self.pix.scaled(size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
            # Căn giữa pixmap
            x = (size.width() - scaled_pix.width()) // 2
            y = (size.height() - scaled_pix.height()) // 2
            painter.drawPixmap(x, y, scaled_pix)
        else:
            super().paintEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F11 or event.key() == Qt.Key.Key_Escape:
            self.escape_requested.emit()
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.fullscreen_requested.emit()

# ==============================================================================
# GIAO DIỆN CHÍNH TRẮNG ĐEN (MINIMALISTIC BLACK & WHITE GUI)
# ==============================================================================
class TritonMinimalistDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TRITON MONOCHROME DASHBOARD")
        self.resize(1280, 800)
        self.infer_thread = None
        self.proc = psutil.Process(os.getpid())
        self.proc.cpu_percent() # Gọi trước một lần để thiết lập mốc thời gian ban đầu

        self._session_samples   = []    # [(decode_lat, src_fps, drop_rate, fps, e2e), ...]
        self._seeking = False           # True while user holds/releases slider; blocks position snap-back
        # EMA smoothing cho resource stats (tránh nhảy số quá nhanh)
        self._res_cpu_ema  = 0.0
        self._res_ram_ema  = 0.0
        self._res_gpu_ema  = 0.0
        self._res_vram_ema = 0.0

        # Lưu trữ màu sắc tự định nghĩa cho nhãn hiển thị (BGR)
        self.custom_colors = {
            "person": (255, 255, 255),
            "fire": (0, 69, 255),
            "smoke": (0, 69, 255),
            "fall": (0, 200, 255)
        }

        # Thiết lập timer tự động cập nhật danh sách mô hình từ Triton Server mỗi 5 giây
        self.resource_timer = QTimer(self)
        self.resource_timer.setInterval(2000)  # 2s: đủ để đọc, không nhấp nháy
        self.resource_timer.timeout.connect(self.update_resource_stats)
        self.resource_timer.start()

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(5000)
        self.refresh_timer.timeout.connect(self.poll_triton_models)
        self.refresh_timer.start()

        # Cấu hình StyleSheet Đen Trắng phẳng
        self.setStyleSheet("""
            QWidget {
                background-color: #0A0A0A;
                color: #E0E0E0;
                font-family: 'JetBrains Mono', 'Courier New', monospace;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #222222;
                border-radius: 0px;
                margin-top: 15px;
                padding-top: 15px;
                font-weight: bold;
                color: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                background-color: #0A0A0A;
            }
            QPushButton {
                background-color: #1E1E1E;
                border: 1px solid #3A3A3A;
                color: #EAEAEA;
                padding: 6px 12px;
                border-radius: 2px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2D2D2D;
                color: #FFFFFF;
                border-color: #888888;
            }
            QPushButton:pressed {
                background-color: #404040;
                color: #FFFFFF;
                border-color: #AAAAAA;
            }
            QPushButton:disabled {
                background-color: #0F0F0F;
                color: #555555;
                border-color: #222222;
            }
            QLineEdit, QComboBox, QTextEdit, QListWidget {
                background-color: #111111;
                border: 1px solid #222222;
                color: #FFFFFF;
                padding: 5px;
                border-radius: 0px;
            }
            QLineEdit:focus, QComboBox:focus, QTextEdit:focus, QListWidget:focus {
                border: 1px solid #FFFFFF;
            }
            QSlider::groove:horizontal {
                border: 1px solid #222222;
                height: 4px;
                background: #111111;
            }
            QSlider::handle:horizontal {
                background: #FFFFFF;
                border: 1px solid #444444;
                width: 12px;
                height: 12px;
                margin: -4px 0;
            }
            QLabel {
                color: #CCCCCC;
            }
            QScrollBar:vertical {
                border: none;
                background: #0A0A0A;
                width: 8px;
            }
            QScrollBar::handle:vertical {
                background: #333333;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #FFFFFF;
            }
        """)

        self.init_ui()

    def init_ui(self):
        # Widget trung tâm
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Bộ chia màn hình chia đôi Controls và Video
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ── PHẦN ĐIỀU KHIỂN (LEFT PANEL TRONG SCROLL AREA) ──
        from PyQt6.QtWidgets import QSizePolicy # Đảm bảo import cho chính sách co giãn
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        # Thay thế setFixedWidth bằng setMinimumWidth giúp thực đơn không bị bóp nghẹt chữ mà vẫn giãn được theo chiều rộng khi bung to màn hình
        scroll_area.setMinimumWidth(380) 
        scroll_area.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #0A0A0A; }")

        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")
        left_panel.setStyleSheet("QWidget#leftPanel { background-color: #0A0A0A; }")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 10, 0)
        left_layout.setSpacing(10)

        # 1. Kết nối Triton
        conn_group = QGroupBox("1. KẾT NỐI TRITON API")
        conn_form = QFormLayout(conn_group)
        self.host_input = QLineEdit("localhost:8001")
        self.connect_btn = QPushButton("KẾT NỐI")
        self.connect_btn.clicked.connect(self.connect_triton)
        self.status_lbl = QLabel("TRẠNG THÁI: CHƯA KẾT NỐI")
        self.status_lbl.setStyleSheet("color: #888888; font-weight: bold;")
        
        conn_form.addRow("Triton Host:", self.host_input)
        conn_form.addRow("", self.connect_btn)
        conn_form.addRow("", self.status_lbl)
        left_layout.addWidget(conn_group)

        # 2. Lựa chọn Model
        model_group = QGroupBox("2. CHỌN MÔ HÌNH (TRITON ACTIVE)")
        model_form = QVBoxLayout(model_group)
        self.model_list_widget = QListWidget()
        # Nới rộng không gian hiển thị danh sách từ 140px đến tối đa 220px để dễ cuộn
        self.model_list_widget.setMinimumHeight(140)
        self.model_list_widget.setMaximumHeight(220)
        # Đảm bảo cơ chế thanh cuộn hoạt động mượt mà tự động khi danh sách quá dài
        self.model_list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        model_form.addWidget(QLabel("Tích chọn các mô hình muốn kích hoạt:"))
        model_form.addWidget(self.model_list_widget)
        left_layout.addWidget(model_group)

        # 3. Nguồn hình ảnh (Sources)
        src_group = QGroupBox("3. CẤU HÌNH CAMERA NGUỒN")
        src_form = QFormLayout(src_group)
        self.src_combo = NonScrollComboBox()
        self.src_combo.addItems([
            "Webcam cục bộ (0)",
            "Nhiều nguồn (Webcam + Video)",
            "Đường dẫn tùy chỉnh (phân cách bằng dấu phẩy)"
        ])
        self.src_combo.currentIndexChanged.connect(self.toggle_source_view)
        self.src_input = QLineEdit("/home/lelam/Documents/ecosystem-ai-server/client/video.mp4")

        self.decoder_combo = NonScrollComboBox()
        self.decoder_combo.addItems([
            "Auto",
            "GStreamer NVDEC",
            "GStreamer CPU",
            "FFmpeg",
            "OpenCV",
        ])
        self.decoder_combo.setCurrentText("FFmpeg")
        self.decoder_combo.setToolTip(
            "Auto: thử GStreamer NVDEC → CPU → FFmpeg\n"
            "GStreamer NVDEC: GPU decode (cần nvidia)\n"
            "GStreamer CPU: avdec_h264 software\n"
            "FFmpeg: CAP_FFMPEG trực tiếp\n"
            "OpenCV: mặc định OpenCV"
        )

        # Cấu hình Preset chất lượng / độ phân giải hiển thị ô lưới (Grid Cell)
        self.quality_combo = NonScrollComboBox()
        self.quality_combo.addItems([
            "Low (480x360)",
            "Medium (640x480)",
            "High (1280x720)",
            "Full HD (1920x1080)"
        ])
        self.quality_combo.setCurrentText("High (1280x720)") # Mặc định để High cho sắc nét nét căng

        src_form.addRow("Bộ chọn:", self.src_combo)
        src_form.addRow("Nguồn:", self.src_input)
        src_form.addRow("Decoder:", self.decoder_combo)
        src_form.addRow("Chất lượng hiển thị:", self.quality_combo)
        left_layout.addWidget(src_group)

        # 4. Tham số suy luận
        param_group = QGroupBox("4. THAM SỐ PHÂN TÍCH")
        param_form = QFormLayout(param_group)
        
        # Confidence
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(10, 100)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.update_param_labels)
        self.conf_lbl = QLabel("0.50")
        param_form.addRow("Confidence Threshold:", self.conf_lbl)
        param_form.addRow("", self.conf_slider)

        # IOU
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setRange(10, 100)
        self.iou_slider.setValue(45)
        self.iou_slider.valueChanged.connect(self.update_param_labels)
        self.iou_lbl = QLabel("0.45")
        param_form.addRow("IOU NMS Threshold:", self.iou_lbl)
        param_form.addRow("", self.iou_slider)

        # Stride
        self.stride_slider = QSlider(Qt.Orientation.Horizontal)
        self.stride_slider.setRange(1, 10)
        self.stride_slider.setValue(2)
        self.stride_slider.valueChanged.connect(self.update_param_labels)
        self.stride_lbl = QLabel("2")
        param_form.addRow("Frame Stride:", self.stride_lbl)
        param_form.addRow("", self.stride_slider)

        # Bbox Style
        self.style_combo = NonScrollComboBox()
        self.style_combo.addItems([
            "White Outline",
            "Light Gray",
            "Medium Gray",
            "Dark Gray",
            "Gray Scale",
            "Dark Outline",
            "Color-Coded",
            "Custom Colors"
        ])
        self.style_combo.setCurrentText("Color-Coded") # Chọn Color-Coded làm mặc định
        param_form.addRow("Bounding Box Style:", self.style_combo)
        
        left_layout.addWidget(param_group)

        # 5. Nhãn tùy chỉnh (Labels Customization)
        label_group = QGroupBox("5. TÙY CHỈNH NHÃN HIỂN THỊ")
        label_layout = QVBoxLayout(label_group)
        self.labels_edit = QTextEdit("person=person\nweapon=weapon\nfall=fall\nfire=fire,smoke\nhelmet=helmet")
        self.labels_edit.setFixedHeight(70)
        label_layout.addWidget(self.labels_edit)
        
        # Nút chọn màu sắc tự định nghĩa cho nhãn
        self.pick_color_btn = QPushButton("CHỌN MÀU SẮC NHÃN...")
        self.pick_color_btn.clicked.connect(self.choose_label_colors)
        label_layout.addWidget(self.pick_color_btn)

        left_layout.addWidget(label_group)

        # Nút khởi chạy
        self.action_btn = QPushButton("BẮT ĐẦU LUỒNG PHÂN TÍCH")
        self.action_btn.setFixedHeight(40)
        self.action_btn.clicked.connect(self.toggle_pipeline)
        self.action_btn.setEnabled(False)
        left_layout.addWidget(self.action_btn)

        scroll_area.setWidget(left_panel)
        splitter.addWidget(scroll_area)

        # ── PHẦN VIEWPORT & LOGS (RIGHT PANEL) ──
        right_panel = QWidget()
        self.right_layout = QVBoxLayout(right_panel)
        right_layout = self.right_layout
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Viewport chính tự co giãn khi fullscreen
        self.viewport = VideoLabel("TRITON MONOCHROME WORKBENCH")
        self.viewport.fullscreen_requested.connect(self._toggle_fullscreen)
        self.viewport.escape_requested.connect(self._toggle_fullscreen)
        self.viewport.setStyleSheet("""
            background-color: #020202;
            border: 1px solid #1A1A1A;
            font-size: 14px;
            color: #444444;
            font-weight: bold;
        """)
        right_layout.addWidget(self.viewport, stretch=5)

        # ── VIDEO SEEK SLIDER ────────────────────────────────────────────────────────
        # Active for file-based sources (MP4, AVI, …); silently ignored for live RTSP
        # streams where CAP_PROP_FRAME_COUNT == 0.  The inference thread emits
        # video_position_updated to keep the handle in sync during normal playback.
        seek_row = QHBoxLayout()
        seek_row.setSpacing(8)
        self.lbl_video_time = QLabel("--:-- / --:--")
        self.lbl_video_time.setFixedWidth(90)
        self.lbl_video_time.setStyleSheet(
            "color: #555555; font-size: 11px; font-family: monospace;")
        self.video_seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_seek_slider.setRange(0, 1000)
        self.video_seek_slider.setValue(0)
        self.video_seek_slider.setToolTip(
            "Seek video file — drag to scrub.  No-op for live RTSP streams.")
        self.video_seek_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 3px; background: #1C1C1C; border-radius: 1px;
            }
            QSlider::sub-page:horizontal {
                background: #3A3A3A; border-radius: 1px;
            }
            QSlider::handle:horizontal {
                background: #888888; border: 1px solid #555555;
                width: 12px; height: 12px; margin: -5px 0; border-radius: 6px;
            }
            QSlider::handle:horizontal:hover { background: #FFFFFF; }
        """)
        self.video_seek_slider.sliderPressed.connect(self._on_seek_slider_pressed)
        self.video_seek_slider.sliderReleased.connect(self._on_seek_slider_released)
        seek_row.addWidget(self.lbl_video_time)
        seek_row.addWidget(self.video_seek_slider, stretch=1)

        # Speed buttons (x1 / x2 / x3 / x4)
        _speed_btn_style = """
            QPushButton {{
                background: #111111; color: #666666;
                border: 1px solid #2A2A2A; padding: 2px 8px;
                font-size: 11px; font-weight: bold; border-radius: 3px;
            }}
            QPushButton:checked {{
                background: #2A2A2A; color: #FFFFFF; border: 1px solid #555555;
            }}
            QPushButton:hover {{ border-color: #888888; }}
        """
        self._speed_btns = {}
        for spd in [1, 2, 3, 4]:
            btn = QPushButton(f"x{spd}")
            btn.setCheckable(True)
            btn.setFixedWidth(34)
            btn.setFixedHeight(22)
            btn.setStyleSheet(_speed_btn_style)
            btn.clicked.connect(lambda checked, s=spd: self._set_playback_speed(s))
            seek_row.addWidget(btn)
            self._speed_btns[spd] = btn
        self._speed_btns[1].setChecked(True)  # default x1

        # Fullscreen button
        self._is_fullscreen = False
        self.btn_fullscreen = QPushButton("⛶")
        self.btn_fullscreen.setFixedWidth(28)
        self.btn_fullscreen.setFixedHeight(22)
        self.btn_fullscreen.setToolTip("Toggle fullscreen (F11)")
        self.btn_fullscreen.setStyleSheet("""
            QPushButton {
                background: #111111; color: #888888;
                border: 1px solid #2A2A2A; font-size: 13px; border-radius: 3px;
            }
            QPushButton:hover { color: #FFFFFF; border-color: #888888; }
        """)
        self.btn_fullscreen.clicked.connect(self._toggle_fullscreen)
        seek_row.addWidget(self.btn_fullscreen)

        right_layout.addLayout(seek_row)

        # Thông số hiệu năng (Metrics) - 2 dòng
        metrics_bar_1 = QHBoxLayout()
        self.lbl_metric_infer = QLabel("INFERENCE: - ms")
        self.lbl_metric_post = QLabel("POST: - ms")
        self.lbl_metric_fps = QLabel("FPS: -")
        self.lbl_metric_total = QLabel("TOTAL: - ms")
        self.lbl_metric_ping = QLabel("gRPC PING: - ms")
        self.lbl_metric_decoder = QLabel("DECODER: -")
        
        _metric_min_widths = [110, 100, 170, 130, 140, 120]
        for lbl, mw in zip(
            [self.lbl_metric_infer, self.lbl_metric_post, self.lbl_metric_fps,
             self.lbl_metric_total, self.lbl_metric_ping, self.lbl_metric_decoder],
            _metric_min_widths
        ):
            lbl.setStyleSheet("background-color: #111111; padding: 6px; border: 1px solid #222222; font-weight: bold; font-size: 11px;")
            lbl.setMinimumWidth(mw)
            metrics_bar_1.addWidget(lbl)
        
        right_layout.addLayout(metrics_bar_1)

        metrics_bar_2 = QHBoxLayout()
        self.lbl_api_details = QLabel("TRITON API: -")
        self.lbl_api_details.setStyleSheet("background-color: #111111; padding: 6px; border: 1px solid #222222; font-weight: bold; font-size: 11px;")
        metrics_bar_2.addWidget(self.lbl_api_details)
        right_layout.addLayout(metrics_bar_2)

        # Decoder performance comparison row
        metrics_bar_3 = QHBoxLayout()
        self.lbl_decode_lat = QLabel("DECODE LAT: - ms")
        self.lbl_src_fps    = QLabel("SRC FPS: -")
        self.lbl_drop_rate  = QLabel("DROP RATE: -%")
        self.lbl_vs_last    = QLabel("AVG: —")
        _dec_style = "background-color: #0A0A0A; padding: 6px; border: 1px solid #1A1A1A; font-size: 11px; color: #666666;"
        for lbl in [self.lbl_decode_lat, self.lbl_src_fps, self.lbl_drop_rate, self.lbl_vs_last]:
            lbl.setStyleSheet(_dec_style)
            metrics_bar_3.addWidget(lbl)
        right_layout.addLayout(metrics_bar_3)

        metrics_bar_4 = QHBoxLayout()
        self.lbl_cpu = QLabel("CPU: -%")
        self.lbl_ram = QLabel("RAM: - MB")
        self.lbl_gpu_util = QLabel("GPU: -%")
        self.lbl_gpu_mem = QLabel("VRAM: - MB")
        _res_style = "background-color: #0A0A0A; padding: 6px; border: 1px solid #1A1A1A; font-size: 11px; color: #888888;"
        for lbl in [self.lbl_cpu, self.lbl_ram, self.lbl_gpu_util, self.lbl_gpu_mem]:
            lbl.setStyleSheet(_res_style)
            metrics_bar_4.addWidget(lbl)
        right_layout.addLayout(metrics_bar_4)

        # Khung chứa logs console bên dưới
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFixedHeight(120)
        self.log_console.setStyleSheet("""
            background-color: #030303;
            border: 1px solid #111111;
            color: #A0A0A0;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
        """)
        right_layout.addWidget(self.log_console, stretch=1)

        splitter.addWidget(right_panel)
        
        # Thiết lập tỷ lệ co giãn (Stretch Factor): 
        # Chỉ số 0 (scroll_area/menu) gán 0 để giữ kích thước vừa đủ không bị chiếm không gian vô ích.
        # Chỉ số 1 (right_panel/video) gán 1 để chiếm trọn phần diện tích màn hình còn lại khi phóng to.
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        
        # Đồng bộ hiển thị ban đầu
        self.src_combo.setCurrentIndex(2) # Chọn Đường dẫn tùy chỉnh (Index 2 sau khi xóa file demo)
        self.toggle_source_view()
        self.log("Bảng điều khiển Desktop Monochrome khởi chạy thành công.")

    # ── ĐIỀU KHIỂN HÀNH VI GIAO DIỆN ──
    def log(self, text):
        t_str = time.strftime("%H:%M:%S")
        self.log_console.append(f"[{t_str}] {text}")

    def toggle_source_view(self):
        idx = self.src_combo.currentIndex()
        if idx == 0:
            self.src_input.setText("0")
            self.src_input.setReadOnly(True)
        elif idx == 1:
            self.src_input.setText("0, rtsp://...")
            self.src_input.setReadOnly(False)
        else:
            self.src_input.setReadOnly(False)
            self.src_input.setPlaceholderText("Nhập camera, ví dụ: rtsp://192.168.1.100:554/stream1, 0")

    def choose_label_colors(self):
        # Tách danh sách nhãn hiện tại từ labels_edit
        labels = []
        import re
        raw_text = self.labels_edit.toPlainText().strip()
        normalized_text = re.sub(r'\s+(\w+)\s*=', r'\n\1=', raw_text)
        normalized_text = normalized_text.replace(";", "\n")
        
        for line in normalized_text.split("\n"):
            line = line.strip()
            if "=" in line:
                parts = line.split("=", 1)
                if len(parts) == 2:
                    m_name, m_lbls = parts
                    labels.append(m_name.strip())
                    for l in m_lbls.split(","):
                        if l.strip():
                            labels.append(l.strip())

        labels = sorted(list(set(labels)))
        if not labels:
            self.log("Không tìm thấy nhãn hợp lệ trong khung cấu hình nhãn!")
            return

        from PyQt6.QtWidgets import QInputDialog, QColorDialog
        from PyQt6.QtGui import QColor
        item, ok = QInputDialog.getItem(self, "Cấu hình màu sắc nhãn", "Chọn nhãn muốn thay đổi màu:", labels, 0, False)
        if ok and item:
            curr_bgr = self.custom_colors.get(item, (255, 255, 255))
            initial_color = QColor(curr_bgr[2], curr_bgr[1], curr_bgr[0]) # RGB
            color = QColorDialog.getColor(initial_color, self, f"Chọn màu sắc nhãn: {item}")
            if color.isValid():
                # Lưu dưới dạng BGR
                self.custom_colors[item] = (color.blue(), color.green(), color.red())
                self.log(f"Cập nhật màu sắc: nhãn '{item}' -> BGR {self.custom_colors[item]}")

    def update_param_labels(self):
        self.conf_lbl.setText(f"{self.conf_slider.value() / 100:.2f}")
        self.iou_lbl.setText(f"{self.iou_slider.value() / 100:.2f}")
        self.stride_lbl.setText(f"{self.stride_slider.value()}")

    def _on_seek_slider_pressed(self):
        """Block position snap-back while the user holds the slider."""
        self._seeking = True

    def _on_seek_slider_released(self):
        """Send the seek, then unblock position updates after the seek has time to land."""
        self._on_seek_slider_moved(self.video_seek_slider.value())
        QTimer.singleShot(600, self._clear_seeking)

    def _clear_seeking(self):
        self._seeking = False

    def _on_seek_slider_moved(self, value: int):
        """Called when the user drags the seek slider.
        Converts the 0–1000 integer to a 0.0–1.0 fraction and forwards the
        request to the running inference thread.  Safe to call while the thread
        is live; the thread applies the seek at the top of its next cycle."""
        if self.infer_thread and self.infer_thread.isRunning():
            self.infer_thread.seek_video(value / 1000.0)

    def _update_video_position(self, pct: float):
        if self._seeking:
            return  # don't snap the handle back while user is dragging or seek is in-flight
        self.video_seek_slider.blockSignals(True)
        self.video_seek_slider.setValue(int(pct * 1000))
        self.video_seek_slider.blockSignals(False)
        self.lbl_video_time.setText(f"{pct * 100:.1f}%")

    def _set_playback_speed(self, speed: int):
        """Update speed buttons highlight and push speed to running thread."""
        for s, btn in self._speed_btns.items():
            btn.setChecked(s == speed)
        if self.infer_thread and self.infer_thread.isRunning():
            self.infer_thread.set_speed(speed)

    def _toggle_fullscreen(self):
        if self._is_fullscreen:
            # Restore to layout
            self.viewport.setWindowFlags(Qt.WindowType.Widget)
            self.viewport.showNormal()
            self.right_layout.insertWidget(0, self.viewport, stretch=5)
            self._is_fullscreen = False
            self.btn_fullscreen.setText("⛶")
            self.btn_fullscreen.setToolTip("Toggle fullscreen (F11)")
            self.setFocus()
        else:
            # Detach and make fullscreen
            self.right_layout.removeWidget(self.viewport)
            self.viewport.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
            self.viewport.showFullScreen()
            self._is_fullscreen = True
            self.btn_fullscreen.setText("⧉")
            self.btn_fullscreen.setToolTip("Exit fullscreen (F11 / Esc)")
            self.viewport.setFocus()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F11:
            self._toggle_fullscreen()
        elif event.key() == Qt.Key.Key_Escape and self._is_fullscreen:
            self._toggle_fullscreen()
        else:
            super().keyPressEvent(event)

    # ── KẾT NỐI TRITON SERVER QUA API gRPC ──
    def connect_triton(self):
        host = self.host_input.text().strip()
        self.log(f"[CLIENT] Đang kích hoạt gRPC gửi tín hiệu bắt tay tới Triton {host}...")
        self.connect_btn.setEnabled(False)
        self.status_lbl.setText("ĐANG KẾT NỐI...")
        self.status_lbl.setStyleSheet("color: #FFFFFF;")

        # Chạy trong luồng riêng để tránh khóa GUI khi connect timeout
        def _bg_connect():
            try:
                client = grpcclient.InferenceServerClient(url=host)
                is_ready = client.is_server_ready(client_timeout=GRPC_TIMEOUT_SEC)
                if is_ready:
                    index = client.get_model_repository_index()
                    models = [m.name for m in index.models if m.state == "READY"]
                    return True, models, ""
                return False, [], "Server is not ready"
            except Exception as e:
                return False, [], str(e)

        def _callback(res):
            success, models, err_msg = res
            self.connect_btn.setEnabled(True)
            if success:
                self.log(f"[SERVER] Kết nối thành công. Danh sách Repository mô hình khả dụng: {models}")
                self.status_lbl.setText("TRẠNG THÁI: ĐANG HOẠT ĐỘNG")
                self.status_lbl.setStyleSheet("color: #FFFFFF; background-color: #222222; border: 1px solid #FFFFFF; padding: 2px;")
                
                # Đưa danh sách mô hình lên giao diện (Để trống, không tự chọn mô hình nào)
                self.model_list_widget.clear()
                for m in models:
                    item = QListWidgetItem(m)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Unchecked)
                    self.model_list_widget.addItem(item)
                
                self.action_btn.setEnabled(True)
            else:
                self.log(f"[SERVER] LỖI phản hồi kết nối từ Triton: {err_msg}")
                self.status_lbl.setText("TRẠNG THÁI: LỖI KẾT NỐI")
                self.status_lbl.setStyleSheet("color: #FF0000; font-weight: bold;")
                self.action_btn.setEnabled(False)

        # Worker đơn giản hỗ trợ Thread
        class ConnectWorker(QThread):
            done = pyqtSignal(tuple)
            def run(self):
                res = _bg_connect()
                self.done.emit(res)

        self.conn_worker = ConnectWorker(self)
        self.conn_worker.done.connect(_callback)
        self.conn_worker.start()

    def _on_infer_thread_finished(self):
        """Reset UI when QThread exits (never block GUI in stop())."""
        self.infer_thread = None
        self.action_btn.setText("BẮT ĐẦU LUỒNG PHÂN TÍCH")
        self.action_btn.setEnabled(True)
        self.viewport.clear()
        self.viewport.setText("TRITON MONOCHROME WORKBENCH")
        self.lbl_metric_infer.setText("INFERENCE: - ms")
        self.lbl_metric_post.setText("POST: - ms")
        self.lbl_metric_fps.setText("FPS: -")
        self.lbl_metric_total.setText("TOTAL: - ms")
        self.lbl_metric_decoder.setText("DECODER: -")
        self.lbl_api_details.setText("TRITON API: -")
        self.lbl_decode_lat.setText("DECODE LAT: - ms")
        self.lbl_src_fps.setText("SRC FPS: -")
        self.lbl_drop_rate.setText("DROP RATE: -%")
        self.lbl_vs_last.setText("AVG: —")
        # Reset seek slider so it doesn't show stale position on next session
        self.video_seek_slider.blockSignals(True)
        self.video_seek_slider.setValue(0)
        self.video_seek_slider.blockSignals(False)
        self.lbl_video_time.setText("--:-- / --:--")
        # Reset speed buttons to x1
        self._set_playback_speed(1)

    # ── ĐIỀU KHIỂN BẮT ĐẦU / DỪNG SUY LUẬN AI ──
    def toggle_pipeline(self):
        if self.infer_thread and self.infer_thread.isRunning():
            # Dừng — lưu thống kê phiên trước khi dừng thread
            self.log("Yêu cầu dừng luồng AI...")
            self.action_btn.setEnabled(False)
            self.action_btn.setText("ĐANG DỪNG...")

            self._session_samples.clear()

            self.infer_thread.stop()
        else:
            # Bắt đầu
            host = self.host_input.text().strip()
            sources = [s.strip() for s in self.src_input.text().split(",") if s.strip()]
            
            # Chọn models
            selected_models = []
            for i in range(self.model_list_widget.count()):
                item = self.model_list_widget.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    selected_models.append(item.text())

            use_ensemble = False
            ensemble_name = ""

            if not selected_models:
                self.log("CẢNH BÁO: Phải chọn ít nhất 1 mô hình hoạt động!")
                return

            conf = self.conf_slider.value() / 100.0
            iou = self.iou_slider.value() / 100.0
            stride = self.stride_slider.value()
            bbox_style = self.style_combo.currentText()

            # Lấy thông số độ phân giải từ quality_combo
            q_text = self.quality_combo.currentText()
            grid_w, grid_h = 640, 480
            if "480x360" in q_text: grid_w, grid_h = 480, 360
            elif "1280x720" in q_text: grid_w, grid_h = 1280, 720
            elif "1920x1080" in q_text: grid_w, grid_h = 1920, 1080

            # Phân tích nhãn tự định nghĩa (Đã sửa lỗi parse dính chữ bằng Regex)
            labels_map = {}
            raw_text = self.labels_edit.toPlainText().strip()
            
            import re
            normalized_text = re.sub(r'\s+(\w+)\s*=', r'\n\1=', raw_text)
            normalized_text = normalized_text.replace(";", "\n")
            
            for line in normalized_text.split("\n"):
                line = line.strip()
                if not line or "=" not in line:
                    continue
                
                parts = line.split("=", 1)
                if len(parts) == 2:
                    model_name = parts[0].strip()
                    model_labels = [lbl.strip() for lbl in parts[1].split(",") if lbl.strip()]
                    
                    if model_name and model_labels:
                        labels_map[model_name] = model_labels

            self._session_samples.clear()
            clear_ensemble_cache()   # ← force re-query ensemble config từ Triton

            self.log(f"Khởi chạy AI: Models={selected_models} | Ensemble={ensemble_name} | Sources={sources} | Decoder={self.decoder_combo.currentText()}")
            self.action_btn.setText("DỪNG LUỒNG PHÂN TÍCH")

            # Tạo luồng xử lý
            self.infer_thread = TritonInferenceThread(
                host=host,
                sources=sources,
                models=selected_models,
                ensemble_model=ensemble_name,
                use_ensemble=use_ensemble,
                conf=conf,
                iou=iou,
                stride=stride,
                labels_map=labels_map,
                bbox_style=bbox_style,
                custom_colors=self.custom_colors,
                decoder=self.decoder_combo.currentText(),
                grid_size=(grid_w, grid_h)
            )
            self.infer_thread.frame_ready.connect(self.on_frame_ready)
            self.infer_thread.log_emitted.connect(self.log)
            self.infer_thread.finished.connect(self._on_infer_thread_finished)
            self.infer_thread.video_position_updated.connect(self._update_video_position)
            self.infer_thread.start()

    # ── HIỂN THỊ KHUNG HÌNH NHẬN ĐƯỢC LÊN GIAO DIỆN ──
    def on_frame_ready(self, frame, lat_infer, lat_post, fps, api_info, lat_e2e, lat_ping, decoder_info, decode_lat, src_fps, drop_rate):
        # Chuyển đổi BGR sang RGB
        if frame is None or frame.ndim != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
            return
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        # Chuyển đổi sang QImage với bản sao độc lập dữ liệu tránh crash
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

        # Nạp hình ảnh vào viewport tự co giãn
        self.viewport.set_frame(q_img)

        # ── Tính tổng độ trễ pipeline đo được ──────────────────────────────────
        # TOTAL = decode (camera→RAM) + infer (gRPC round-trip) + postprocess (NMS)
        # E2E   = thời gian toàn bộ vòng lặp (bao gồm annotate + queue + sleep)
        lat_total = decode_lat + lat_infer + lat_post

        # ── Metrics row 1: inference details ───────────────────────────────────
        self.lbl_metric_infer.setText(f"INFER: {lat_infer:.1f} ms")
        self.lbl_metric_post.setText(f"POST: {lat_post:.1f} ms")

        # SRC FPS: tốc độ camera nguồn thực tế
        # OUT FPS: tốc độ inference thực tế (số frame/s model xử lý, không phải display-loop)
        _fps_color = "#FF4444" if src_fps < 8 else "#FFAA00" if src_fps < 15 else "#44FF88"
        fps_str = f"{fps:5.1f}" if fps > 0.0 else " --.-"
        self.lbl_metric_fps.setStyleSheet(
            f"background-color: #0A0A0A; padding: 6px; border: 1px solid #1A1A1A;"
            f"font-size: 13px; font-weight: bold; color: {_fps_color};")
        self.lbl_metric_fps.setText(f"SRC {src_fps:5.1f} / OUT {fps_str} FPS")

        

        _total_color = "#FF4444" if lat_total > 200 else "#FFAA00" if lat_total > 80 else "#44FF88"
        self.lbl_metric_total.setStyleSheet(
            f"background-color: #0A0A0A; padding: 6px; border: 1px solid #1A1A1A;"
            f"font-size: 11px; font-weight: bold; color: {_total_color};")
        self.lbl_metric_total.setText(f"TOTAL: {lat_total:.1f} ms")
        self.lbl_metric_ping.setText(f"gRPC PING: {lat_ping:.1f} ms")
        self.lbl_metric_decoder.setText(f"DECODER: {decoder_info}")
        self.lbl_api_details.setText(f"{api_info} | Host: {self.host_input.text().strip()}")

        # ── Decoder perf row ────────────────────────────────────────────────────
        self.lbl_decode_lat.setText(f"DECODE: {decode_lat:.1f} ms")
        self.lbl_src_fps.setText(f"SRC FPS: {src_fps:.1f}")
        drop_color = "#FF4444" if drop_rate > 5.0 else "#FFAA00" if drop_rate > 1.0 else "#666666"
        self.lbl_drop_rate.setStyleSheet(
            f"background-color: #0A0A0A; padding: 6px; border: 1px solid #1A1A1A; font-size: 11px; color: {drop_color};")
        self.lbl_drop_rate.setText(f"DROP: {drop_rate:.2f}%")

        # Tích lũy mẫu phiên hiện tại (thêm lat_infer, lat_post vào tuple)
        self._session_samples.append((decode_lat, src_fps, drop_rate, fps, lat_e2e, lat_infer, lat_post))

        # ── Hiển thị trung bình cuộn 60 mẫu gần nhất (ổn định sau ~3 giây) ──────
        # Thay thế so sánh delta khó hiểu → hiển thị số trung bình thực tế rõ ràng
        N_AVG = 60
        if len(self._session_samples) >= 10:
            recent      = self._session_samples[-N_AVG:]
            n           = len(recent)
            avg_decode  = float(np.mean([s[0] for s in recent]))
            avg_src_fps = float(np.mean([s[1] for s in recent]))
            avg_infer   = float(np.mean([s[5] for s in recent]))
            avg_post    = float(np.mean([s[6] for s in recent]))
            avg_total   = avg_decode + avg_infer + avg_post
            avg_drop    = float(np.mean([s[2] for s in recent]))

            # Màu tổng độ trễ: xanh tốt / vàng trung bình / đỏ chậm
            avg_color = "#44FF88" if avg_total < 60 else "#FFAA00" if avg_total < 150 else "#FF4444"
            avg_text = (
                f"AVG({n}):  "
                f"decode {avg_decode:.1f}ms  "
                f"infer {avg_infer:.1f}ms  "
                f"post {avg_post:.1f}ms  │  "
                f"total {avg_total:.1f}ms  "
                f"fps {avg_src_fps:.1f}  "
                f"drop {avg_drop:.2f}%"
            )
            self.lbl_vs_last.setStyleSheet(
                f"background-color: #0A0A0A; padding: 6px; border: 1px solid #1A1A1A;"
                f"font-size: 11px; color: {avg_color};")
            self.lbl_vs_last.setText(avg_text)

    def update_resource_stats(self):
        # 1. Cập nhật CPU và RAM của Client máy cục bộ
        _a = 0.3  # EMA alpha: 0.3 = mượt, phản ứng đủ nhanh
        cpu    = self.proc.cpu_percent()
        ram_mb = self.proc.memory_info().rss / (1024 * 1024)
        # EMA smoothing để tránh nhảy số
        self._res_cpu_ema  = _a * cpu    + (1 - _a) * self._res_cpu_ema  if self._res_cpu_ema  > 0 else cpu
        self._res_ram_ema  = _a * ram_mb + (1 - _a) * self._res_ram_ema  if self._res_ram_ema  > 0 else ram_mb
        cpu    = self._res_cpu_ema
        ram_mb = self._res_ram_ema
        
        cpu_color = "#FF4444" if cpu > 80 else "#FFAA00" if cpu > 50 else "#888888"
        ram_color = "#FF4444" if ram_mb > 4000 else "#888888"
        
        self.lbl_cpu.setStyleSheet(f"background-color: #0A0A0A; padding: 6px; border: 1px solid #1A1A1A; font-size: 11px; color: {cpu_color};")
        self.lbl_cpu.setText(f"CPU: {cpu:5.1f}%")
        self.lbl_ram.setStyleSheet(f"background-color: #0A0A0A; padding: 6px; border: 1px solid #1A1A1A; font-size: 11px; color: {ram_color};")
        self.lbl_ram.setText(f"RAM: {ram_mb:6.0f} MB")
        
        # 2. Cập nhật GPU và VRAM của máy Client (RTX 3070 Ti)
        if _NVML_AVAILABLE:
            try:
                # Khởi tạo lại NVML nếu trước đó bị lỗi ngắt quãng
                try:
                    pynvml.nvmlInit()
                except Exception:
                    pass
                
                # Lấy handle card đồ họa đầu tiên của máy client
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Lấy thông số tải (%) và bộ nhớ VRAM
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_pct    = util.gpu
                vram_used  = mem.used  // (1024 * 1024)
                vram_total = mem.total // (1024 * 1024)
                # EMA smoothing GPU (dùng _a từ block CPU bên trên)
                self._res_gpu_ema  = _a * gpu_pct   + (1 - _a) * self._res_gpu_ema  if self._res_gpu_ema  > 0 else float(gpu_pct)
                self._res_vram_ema = _a * vram_used + (1 - _a) * self._res_vram_ema if self._res_vram_ema > 0 else float(vram_used)
                gpu_pct   = self._res_gpu_ema
                vram_used = int(self._res_vram_ema)
                
                gpu_color = "#FF4444" if gpu_pct > 90 else "#FFAA00" if gpu_pct > 60 else "#888888"
                
                self.lbl_gpu_util.setStyleSheet(f"background-color: #0A0A0A; padding: 6px; border: 1px solid #1A1A1A; font-size: 11px; color: {gpu_color};")
                self.lbl_gpu_util.setText(f"GPU: {gpu_pct:5.1f}%")
                self.lbl_gpu_mem.setText(f"VRAM: {vram_used:5d}/{vram_total} MB")
                
            except Exception as e:
                # Nếu crash giữa chừng (ví dụ driver bận), hiển thị thông báo lỗi ngắn gọn
                self.lbl_gpu_util.setText("GPU: ERR")
                self.lbl_gpu_mem.setText("VRAM: ERR")
        else:
            # Trường hợp thư viện pynvml không thể init khi chạy app
            self.lbl_gpu_util.setText("GPU: N/A")
            self.lbl_gpu_mem.setText("VRAM: N/A")

    def poll_triton_models(self):
        # Tự động cập nhật danh sách mô hình chạy ngầm khi không ở luồng suy luận
        if self.status_lbl.text() == "TRẠNG THÁI: ĐANG HOẠT ĐỘNG" and (not self.infer_thread or not self.infer_thread.isRunning()):
            host = self.host_input.text().strip()
            
            class PollWorker(QThread):
                done = pyqtSignal(bool, list)
                def __init__(self, host):
                    super().__init__()
                    self.host = host
                def run(self):
                    try:
                        client = grpcclient.InferenceServerClient(url=self.host)
                        if client.is_server_ready(client_timeout=GRPC_TIMEOUT_SEC):
                            index = client.get_model_repository_index()
                            models = [m.name for m in index.models if m.state == "READY"]
                            self.done.emit(True, models)
                            return
                        self.done.emit(False, [])
                    except Exception:
                        self.done.emit(False, [])

            self.poll_worker = PollWorker(host)
            def _on_poll_done(success, models):
                if success:
                    # So sánh danh sách hiện tại để tránh vẽ lại không cần thiết
                    current_models = [self.model_list_widget.item(i).text() for i in range(self.model_list_widget.count())]
                    if set(models) != set(current_models):
                        # Lưu trạng thái check cũ
                        checked_models = []
                        for i in range(self.model_list_widget.count()):
                            item = self.model_list_widget.item(i)
                            if item.checkState() == Qt.CheckState.Checked:
                                checked_models.append(item.text())
                        
                        self.model_list_widget.clear()
                        for m in models:
                            item = QListWidgetItem(m)
                            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                            if m in checked_models:
                                item.setCheckState(Qt.CheckState.Checked)
                            else:
                                item.setCheckState(Qt.CheckState.Unchecked)
                            self.model_list_widget.addItem(item)
                        self.log(f"gRPC: /inference.GRPCInferenceService/ModelRepositoryIndex (Tự động đồng bộ hóa danh sách mô hình: {models})")

            self.poll_worker.done.connect(_on_poll_done)
            self.poll_worker.start()

    def closeEvent(self, event):
        # Đóng luồng tính toán khi tắt ứng dụng
        if self.infer_thread and self.infer_thread.isRunning():
            self.infer_thread.stop()
        event.accept()

# ==============================================================================
# HÀM KHỞI CHẠY CHÍNH
# ==============================================================================
def main():
    app = QApplication(sys.argv)
    
    # Thiết lập font Monospace thanh lịch
    font = QFont("JetBrains Mono", 10)
    app.setFont(font)
    
    dashboard = TritonMinimalistDashboard()
    dashboard.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()