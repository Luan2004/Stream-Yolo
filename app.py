import os
import uuid
import shutil
import cv2
import subprocess
import traceback
import time
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from typing import Optional


# ============================================================
# Config
# ============================================================

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Đổi đúng đường dẫn model của bạn tại đây
PERSON_MODEL_PATH = "yolov8n.pt"       # Model COCO, có class person
FALL_MODEL_PATH = "best_fall.pt"         # Tạm test. Khi có model té ngã thì đổi thành best_fall.pt
FIRE_MODEL_PATH = "best_fire.pt"       # Model cháy/khói của bạn

# Nếu chưa có model fire, tạm đổi thành "yolov8n.pt" để test API chạy được.
# FIRE_MODEL_PATH = "yolov8n.pt"

MODEL_STATUS = {
    "person": {"path": PERSON_MODEL_PATH, "loaded": False, "error": None},
    "fall": {"path": FALL_MODEL_PATH, "loaded": False, "error": None},
    "fire": {"path": FIRE_MODEL_PATH, "loaded": False, "error": None},
}

person_model = None
fall_model = None
fire_model = None



# ============================================================
# GStreamer + Motion Detection Config
# ============================================================

# Lưu trạng thái cho endpoint nhận frame từ camera điện thoại.
# Lưu ý: bản demo dùng global state, đủ để test 1 camera.
# Khi chạy nhiều camera/user, nên tách theo camera_id/session_id.
LAST_CAMERA_GRAY = None
LAST_CAMERA_ANNOTATED = None
LAST_CAMERA_DETECTIONS = []
LAST_CAMERA_TS = 0.0

MOTION_DEFAULT_THRESHOLD = 3.0      # % pixel thay đổi; càng cao càng bỏ qua nhiều frame
MOTION_DEFAULT_MIN_INTERVAL = 1.0   # giây; ép detect lại sau mỗi N giây dù ít chuyển động
MOTION_RESIZE_WIDTH = 320           # resize nhỏ để tính motion nhanh hơn

# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(title="3 YOLO Models Video Demo API - H264 Output")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cho phép mở output qua:
# http://127.0.0.1:8000/outputs/ten_file.mp4
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# ============================================================
# Model loading
# ============================================================

def load_model_safe(model_name: str, model_path: str):
    """
    Load model an toàn.
    Nếu model bị thiếu/lỗi thì API vẫn chạy, /health sẽ báo lỗi model đó.
    """
    try:
        # Các model như yolov8n.pt có thể tự download nên không cần os.path.exists
        if not os.path.exists(model_path) and not model_path.startswith("yolo"):
            raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

        model = YOLO(model_path)
        MODEL_STATUS[model_name]["loaded"] = True
        MODEL_STATUS[model_name]["error"] = None
        return model

    except Exception as e:
        MODEL_STATUS[model_name]["loaded"] = False
        MODEL_STATUS[model_name]["error"] = str(e)
        print(f"[MODEL ERROR] {model_name}: {e}")
        return None


@app.on_event("startup")
def startup_load_models():
    global person_model, fall_model, fire_model

    print("Đang load YOLO models...")

    person_model = load_model_safe("person", PERSON_MODEL_PATH)
    fall_model = load_model_safe("fall", FALL_MODEL_PATH)
    fire_model = load_model_safe("fire", FIRE_MODEL_PATH)

    print("Model status:", MODEL_STATUS)


# ============================================================
# Helper functions
# ============================================================

def safe_filename(filename: str) -> str:
    filename = os.path.basename(filename or "video.mp4")
    filename = filename.replace(" ", "_")
    return filename


def check_at_least_one_model_loaded():
    return any(item["loaded"] for item in MODEL_STATUS.values())


def get_video_writer(output_path: str, fps: float, width: int, height: int):
    """
    Ghi video raw bằng OpenCV.
    File này sau đó sẽ được convert sang H.264 để trình duyệt xem được.
    """
    if fps <= 0:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return writer


def get_ffmpeg_executable():
    """
    Ưu tiên dùng ffmpeg trong hệ thống.
    Nếu không có, thử dùng imageio-ffmpeg.
    Cài fallback bằng:
        pip install imageio-ffmpeg
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def convert_to_h264(input_path: str, output_path: str):
    """
    Convert video sang H.264 + yuv420p để Chrome/Edge/HTML5 video xem được.

    Cần có ffmpeg:
        - Windows: cài ffmpeg và thêm vào PATH
        - Hoặc: pip install imageio-ffmpeg
    """
    ffmpeg = get_ffmpeg_executable()

    if ffmpeg is None:
        raise RuntimeError(
            "Không tìm thấy ffmpeg. Hãy cài: pip install imageio-ffmpeg "
            "hoặc cài ffmpeg và thêm vào PATH."
        )

    cmd = [
        ffmpeg,
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "FFmpeg convert H.264 thất bại.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDERR:\n{completed.stderr}"
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Convert xong nhưng file H.264 output bị rỗng.")

    return output_path


def draw_yolo_results(frame, results, model_name: str, color, allowed_classes=None):
    detections = []

    for r in results:
        if r.boxes is None:
            continue

        names = r.names

        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if isinstance(names, dict):
                class_name = names.get(cls_id, str(cls_id))
            else:
                class_name = str(cls_id)

            if allowed_classes is not None and class_name not in allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{model_name}: {class_name} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            detections.append({
                "model": model_name,
                "class_name": class_name,
                "confidence": round(conf, 4),
                "box": [x1, y1, x2, y2]
            })

    return frame, detections


def predict_one_model(model, frame, model_name, color, conf, allowed_classes=None):
    if model is None:
        return frame, []

    results = model.predict(
        source=frame,
        conf=conf,
        imgsz=640,
        verbose=False
    )

    return draw_yolo_results(
        frame,
        results,
        model_name=model_name,
        color=color,
        allowed_classes=allowed_classes
    )


def run_3_models_on_frame(frame, conf: float = 0.4):
    """
    Chạy 3 model trên frame:
    - person: chỉ lấy class person
    - fall: lấy tất cả class của model fall
    - fire: lấy tất cả class của model fire
    """
    annotated = frame.copy()
    all_detections = []

    annotated, dets = predict_one_model(
        person_model,
        annotated,
        model_name="person",
        color=(0, 255, 0),
        conf=conf,
        allowed_classes=["person"]
    )
    all_detections.extend(dets)

    annotated, dets = predict_one_model(
        fall_model,
        annotated,
        model_name="fall",
        color=(255, 0, 0),
        conf=conf,
        allowed_classes=None
    )
    all_detections.extend(dets)

    annotated, dets = predict_one_model(
        fire_model,
        annotated,
        model_name="fire",
        color=(0, 0, 255),
        conf=conf,
        allowed_classes=None
    )
    all_detections.extend(dets)

    return annotated, all_detections




# ============================================================
# GStreamer + Motion Detection Helpers
# ============================================================

def build_gstreamer_pipeline(
    source: str,
    source_type: str = "rtsp",
    latency: int = 100,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None
):
    """
    Tạo pipeline GStreamer cho OpenCV.

    source_type:
      - rtsp: source là rtsp://...
      - file: source là đường dẫn video local
      - webcam: source là index camera, ví dụ "0" trên Linux

    Lưu ý:
      OpenCV phải được build có GStreamer thì cv2.CAP_GSTREAMER mới dùng được.
    """
    caps_parts = []
    if width and height:
        caps_parts.append(f"width={int(width)},height={int(height)}")
    if fps:
        caps_parts.append(f"framerate={int(fps)}/1")

    caps = ""
    if caps_parts:
        caps = " ! video/x-raw," + ",".join(caps_parts)

    if source_type == "rtsp":
        return (
            f"rtspsrc location={source} latency={int(latency)} drop-on-latency=true ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! videoscale"
            f"{caps} ! "
            "appsink drop=true sync=false max-buffers=1"
        )

    if source_type == "file":
        return (
            f"filesrc location={source} ! "
            "decodebin ! videoconvert ! videoscale"
            f"{caps} ! "
            "appsink drop=true sync=false max-buffers=1"
        )

    if source_type == "webcam":
        # Linux webcam. Trên Windows thường không dùng v4l2src.
        return (
            f"v4l2src device=/dev/video{source} ! "
            "videoconvert ! videoscale"
            f"{caps} ! "
            "appsink drop=true sync=false max-buffers=1"
        )

    raise ValueError("source_type phải là: rtsp, file hoặc webcam")


def open_capture_gstreamer_or_default(pipeline_or_path: str, use_gstreamer: bool = True):
    """
    Mở video/stream bằng GStreamer nếu có.
    Nếu fail thì fallback về cv2.VideoCapture thường.
    """
    if use_gstreamer:
        cap = cv2.VideoCapture(pipeline_or_path, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap, "gstreamer"

    cap = cv2.VideoCapture(pipeline_or_path)
    return cap, "opencv"


def prepare_motion_gray(frame, resize_width: int = MOTION_RESIZE_WIDTH):
    """
    Chuyển frame về ảnh xám nhỏ để so sánh nhanh.
    """
    h, w = frame.shape[:2]
    if w > resize_width:
        scale = resize_width / float(w)
        frame = cv2.resize(frame, (resize_width, int(h * scale)))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def motion_changed(prev_gray, curr_frame, threshold_percent: float = MOTION_DEFAULT_THRESHOLD):
    """
    Trả về:
      changed: True nếu frame có thay đổi đủ lớn
      motion_score: % pixel thay đổi sau threshold ảnh khác biệt

    threshold_percent nên thử:
      - 1.0 - 2.0: nhạy, phù hợp phát hiện người đi qua
      - 3.0 - 5.0: ít gọi model hơn
      - > 8.0: chỉ detect khi chuyển động lớn
    """
    curr_gray = prepare_motion_gray(curr_frame)

    if prev_gray is None:
        return True, 100.0, curr_gray

    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    changed_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    motion_score = (changed_pixels / max(total_pixels, 1)) * 100.0

    return motion_score >= float(threshold_percent), float(motion_score), curr_gray


def encode_jpg_response(frame, detections, skipped: bool, motion_score: float):
    """
    Encode frame đã vẽ box thành FileResponse.
    """
    ok, encoded_img = cv2.imencode(".jpg", frame)

    if not ok:
        raise RuntimeError("Không encode được ảnh kết quả.")

    output_name = f"{uuid.uuid4()}_camera_motion.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    with open(output_path, "wb") as f:
        f.write(encoded_img.tobytes())

    return FileResponse(
        output_path,
        media_type="image/jpeg",
        filename="camera_result_motion.jpg",
        headers={
            "X-Detections-Count": str(len(detections or [])),
            "X-Skipped-By-Motion": str(skipped).lower(),
            "X-Motion-Score": f"{motion_score:.4f}"
        }
    )


# ============================================================
# Routes
# ============================================================

@app.get("/")
def home():
    return {
        "message": "3-model video detection API - H264 Output",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "snapshot": "POST /detect/snapshot",
            "frames": "POST /detect/frames",
            "video": "POST /detect/video",
            "video_info": "POST /detect/video-info",
            "outputs": "GET /outputs/{filename}"
        }
    }


@app.get("/health")
def health():
    ffmpeg = get_ffmpeg_executable()

    return {
        "ok": check_at_least_one_model_loaded(),
        "ffmpeg_found": ffmpeg is not None,
        "ffmpeg_path": ffmpeg,
        "upload_dir": os.path.abspath(UPLOAD_DIR),
        "output_dir": os.path.abspath(OUTPUT_DIR),
        "models": MODEL_STATUS
    }


@app.post("/detect/snapshot")
async def detect_snapshot(
    file: UploadFile = File(...),
    frame_index: int = Form(0),
    conf: float = Form(0.4)
):
    """
    Upload video .mp4.
    Lấy 1 frame, chạy 3 model, trả về ảnh JPG.
    """
    try:
        if not check_at_least_one_model_loaded():
            return JSONResponse(
                status_code=500,
                content={"error": "Chưa load được model nào.", "models": MODEL_STATUS}
            )

        video_id = str(uuid.uuid4())
        input_name = safe_filename(file.filename)

        input_path = os.path.join(UPLOAD_DIR, f"{video_id}_{input_name}")
        output_path = os.path.join(OUTPUT_DIR, f"{video_id}_snapshot.jpg")

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            return JSONResponse(
                status_code=400,
                content={"error": "Không mở được video. Kiểm tra file .mp4."}
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_index < 0:
            frame_index = 0

        if total_frames > 0 and frame_index >= total_frames:
            frame_index = total_frames - 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return JSONResponse(
                status_code=400,
                content={"error": "Không đọc được frame từ video."}
            )

        annotated, detections = run_3_models_on_frame(frame, conf=conf)
        cv2.imwrite(output_path, annotated)

        return FileResponse(
            output_path,
            media_type="image/jpeg",
            filename="labeled_snapshot.jpg"
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc(), "models": MODEL_STATUS}
        )


@app.post("/detect/frames")
async def detect_frames(
    file: UploadFile = File(...),
    every_n_frames: int = Form(30),
    max_images: int = Form(5),
    conf: float = Form(0.4)
):
    """
    Upload video .mp4.
    Lấy nhiều frame, chạy 3 model, trả về danh sách ảnh.
    """
    try:
        if not check_at_least_one_model_loaded():
            return JSONResponse(
                status_code=500,
                content={"error": "Chưa load được model nào.", "models": MODEL_STATUS}
            )

        if every_n_frames <= 0:
            every_n_frames = 30

        if max_images <= 0:
            max_images = 5

        video_id = str(uuid.uuid4())
        input_name = safe_filename(file.filename)
        input_path = os.path.join(UPLOAD_DIR, f"{video_id}_{input_name}")

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            return JSONResponse(
                status_code=400,
                content={"error": "Không mở được video. Kiểm tra file .mp4."}
            )

        frame_id = 0
        saved_count = 0
        results = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_id % every_n_frames == 0:
                annotated, detections = run_3_models_on_frame(frame, conf=conf)

                image_name = f"{video_id}_frame_{frame_id}.jpg"
                image_path = os.path.join(OUTPUT_DIR, image_name)
                image_url = f"/outputs/{image_name}"

                cv2.imwrite(image_path, annotated)

                results.append({
                    "frame_index": frame_id,
                    "image_path": image_path,
                    "image_url": image_url,
                    "detections": detections
                })

                saved_count += 1

                if saved_count >= max_images:
                    break

            frame_id += 1

        cap.release()

        return {
            "message": "Đã xử lý video thành các ảnh",
            "video_file": file.filename,
            "saved_images": saved_count,
            "results": results
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc(), "models": MODEL_STATUS}
        )


@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    conf: float = Form(0.4),
    every_n_frames: int = Form(1)
):
    """
    Upload video .mp4.
    Chạy 3 model trên video, vẽ box + label,
    sau đó convert output sang H.264 để xem được trên trình duyệt.
    """
    try:
        if not check_at_least_one_model_loaded():
            return JSONResponse(
                status_code=500,
                content={"error": "Chưa load được model nào.", "models": MODEL_STATUS}
            )

        if every_n_frames <= 0:
            every_n_frames = 1

        video_id = str(uuid.uuid4())
        input_name = safe_filename(file.filename)

        input_path = os.path.join(UPLOAD_DIR, f"{video_id}_{input_name}")

        raw_output_name = f"{video_id}_raw.mp4"
        raw_output_path = os.path.join(OUTPUT_DIR, raw_output_name)

        final_output_name = f"{video_id}_labeled_h264.mp4"
        final_output_path = os.path.join(OUTPUT_DIR, final_output_name)

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            return JSONResponse(
                status_code=400,
                content={"error": "Không mở được video. Kiểm tra file .mp4."}
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 25

        if width <= 0 or height <= 0:
            cap.release()
            return JSONResponse(
                status_code=400,
                content={"error": "Không lấy được kích thước video."}
            )

        writer = get_video_writer(raw_output_path, fps, width, height)

        if not writer.isOpened():
            cap.release()
            return JSONResponse(
                status_code=500,
                content={"error": "Không tạo được file video raw bằng OpenCV."}
            )

        frame_id = 0
        processed_frames = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_id % every_n_frames == 0:
                annotated, detections = run_3_models_on_frame(frame, conf=conf)
                processed_frames += 1
            else:
                annotated = frame

            writer.write(annotated)
            frame_id += 1

        cap.release()
        writer.release()

        if frame_id == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Video không có frame nào để xử lý."}
            )

        if not os.path.exists(raw_output_path) or os.path.getsize(raw_output_path) == 0:
            return JSONResponse(
                status_code=500,
                content={"error": "File raw output bị rỗng, không tạo được video."}
            )

        # ====================================================
        # Convert sang H.264 để thẻ <video> của trình duyệt xem được
        # ====================================================
        convert_to_h264(raw_output_path, final_output_path)

        return FileResponse(
            final_output_path,
            media_type="video/mp4",
            filename="labeled_video_h264.mp4",
            headers={
                "X-Video-Output-Url": f"/outputs/{final_output_name}",
                "X-Raw-Video-Output-Url": f"/outputs/{raw_output_name}",
                "X-Total-Frames": str(total_frames),
                "X-Processed-Frames": str(processed_frames),
                "X-Codec": "h264"
            }
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc(),
                "models": MODEL_STATUS,
                "ffmpeg_found": get_ffmpeg_executable() is not None
            }
        )


@app.post("/detect/video-info")
async def detect_video_info(
    file: UploadFile = File(...),
    conf: float = Form(0.4),
    every_n_frames: int = Form(5)
):
    """
    Giống /detect/video nhưng trả về JSON có video_url thay vì trả trực tiếp file.
    Hữu ích nếu frontend muốn nhận đường dẫn /outputs/xxx.mp4.
    """
    try:
        if not check_at_least_one_model_loaded():
            return JSONResponse(
                status_code=500,
                content={"error": "Chưa load được model nào.", "models": MODEL_STATUS}
            )

        if every_n_frames <= 0:
            every_n_frames = 5

        video_id = str(uuid.uuid4())
        input_name = safe_filename(file.filename)

        input_path = os.path.join(UPLOAD_DIR, f"{video_id}_{input_name}")

        raw_output_name = f"{video_id}_raw.mp4"
        raw_output_path = os.path.join(OUTPUT_DIR, raw_output_name)

        final_output_name = f"{video_id}_labeled_h264.mp4"
        final_output_path = os.path.join(OUTPUT_DIR, final_output_name)

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            return JSONResponse(
                status_code=400,
                content={"error": "Không mở được video. Kiểm tra file .mp4."}
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 25

        writer = get_video_writer(raw_output_path, fps, width, height)

        if not writer.isOpened():
            cap.release()
            return JSONResponse(
                status_code=500,
                content={"error": "Không tạo được file video raw bằng OpenCV."}
            )

        frame_id = 0
        processed_frames = 0
        detection_frames = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_id % every_n_frames == 0:
                annotated, detections = run_3_models_on_frame(frame, conf=conf)
                processed_frames += 1

                if detections:
                    detection_frames.append({
                        "frame_index": frame_id,
                        "detections": detections
                    })
            else:
                annotated = frame

            writer.write(annotated)
            frame_id += 1

        cap.release()
        writer.release()

        if frame_id == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Video không có frame nào để xử lý."}
            )

        convert_to_h264(raw_output_path, final_output_path)

        return {
            "message": "Đã xử lý video và convert sang H.264",
            "input_file": file.filename,
            "output_file": final_output_name,
            "video_url": f"/outputs/{final_output_name}",
            "full_video_url": f"http://127.0.0.1:8000/outputs/{final_output_name}",
            "raw_video_url": f"/outputs/{raw_output_name}",
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "every_n_frames": every_n_frames,
            "codec": "h264",
            "detections": detection_frames,
            "models": MODEL_STATUS,
            "ffmpeg_found": get_ffmpeg_executable() is not None
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc(),
                "models": MODEL_STATUS,
                "ffmpeg_found": get_ffmpeg_executable() is not None
            }
        )


@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(0.4)
):
    """
    Nhận 1 ảnh/frame từ camera điện thoại,
    chạy 3 model YOLO,
    trả về ảnh JPG đã vẽ box + label.
    """
    try:
        if not check_at_least_one_model_loaded():
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Chưa load được model nào.",
                    "models": MODEL_STATUS
                }
            )

        image_bytes = await file.read()

        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Không đọc được ảnh từ camera."}
            )

        annotated, detections = run_3_models_on_frame(frame, conf=conf)

        ok, encoded_img = cv2.imencode(".jpg", annotated)

        if not ok:
            return JSONResponse(
                status_code=500,
                content={"error": "Không encode được ảnh kết quả."}
            )

        output_name = f"{uuid.uuid4()}_camera.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        with open(output_path, "wb") as f:
            f.write(encoded_img.tobytes())

        return FileResponse(
            output_path,
            media_type="image/jpeg",
            filename="camera_result.jpg",
            headers={
                "X-Detections-Count": str(len(detections))
            }
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc(),
                "models": MODEL_STATUS
            }
        )



@app.post("/detect/image-motion")
async def detect_image_motion(
    file: UploadFile = File(...),
    conf: float = Form(0.4),
    motion_threshold: float = Form(MOTION_DEFAULT_THRESHOLD),
    force_interval: float = Form(MOTION_DEFAULT_MIN_INTERVAL)
):
    """
    Nhận 1 ảnh/frame từ frontend.
    Nếu frame gần như không đổi so với frame trước thì KHÔNG chạy YOLO lại,
    mà trả về kết quả detect gần nhất để tiết kiệm thời gian.

    Header trả về:
      X-Skipped-By-Motion: true/false
      X-Motion-Score: % pixel thay đổi
    """
    global LAST_CAMERA_GRAY, LAST_CAMERA_ANNOTATED, LAST_CAMERA_DETECTIONS, LAST_CAMERA_TS

    try:
        if not check_at_least_one_model_loaded():
            return JSONResponse(
                status_code=500,
                content={"error": "Chưa load được model nào.", "models": MODEL_STATUS}
            )

        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Không đọc được ảnh từ camera."}
            )

        changed, motion_score, curr_gray = motion_changed(
            LAST_CAMERA_GRAY,
            frame,
            threshold_percent=motion_threshold
        )

        now = time.time()
        force_detect = (now - LAST_CAMERA_TS) >= float(force_interval)

        if (not changed) and (not force_detect) and LAST_CAMERA_ANNOTATED is not None:
            # Không chạy model lại, trả ảnh kết quả cũ.
            return encode_jpg_response(
                LAST_CAMERA_ANNOTATED.copy(),
                LAST_CAMERA_DETECTIONS,
                skipped=True,
                motion_score=motion_score
            )

        annotated, detections = run_3_models_on_frame(frame, conf=conf)

        LAST_CAMERA_GRAY = curr_gray
        LAST_CAMERA_ANNOTATED = annotated.copy()
        LAST_CAMERA_DETECTIONS = detections
        LAST_CAMERA_TS = now

        return encode_jpg_response(
            annotated,
            detections,
            skipped=False,
            motion_score=motion_score
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc(), "models": MODEL_STATUS}
        )


@app.get("/detect/gstreamer-info")
def detect_gstreamer_info(
    source: str,
    source_type: str = "rtsp",
    latency: int = 100,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None
):
    """
    Test tạo pipeline GStreamer và kiểm tra OpenCV có mở được không.
    Ví dụ:
      /detect/gstreamer-info?source=rtsp://user:pass@192.168.1.10:554/stream1&source_type=rtsp
    """
    try:
        pipeline = build_gstreamer_pipeline(
            source=source,
            source_type=source_type,
            latency=latency,
            width=width,
            height=height,
            fps=fps
        )
        cap, backend = open_capture_gstreamer_or_default(pipeline, use_gstreamer=True)
        opened = cap.isOpened()
        cap.release()

        return {
            "opened": opened,
            "backend": backend,
            "pipeline": pipeline,
            "note": "Nếu opened=false, OpenCV của bạn có thể chưa build GStreamer hoặc thiếu plugin GStreamer."
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()}
        )


@app.get("/detect/gstreamer-stream")
def detect_gstreamer_stream(
    source: str,
    source_type: str = "rtsp",
    conf: float = 0.4,
    seconds: int = 10,
    every_n_frames: int = 1,
    motion_threshold: float = MOTION_DEFAULT_THRESHOLD,
    force_interval: float = MOTION_DEFAULT_MIN_INTERVAL,
    latency: int = 100,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None
):
    """
    Đọc RTSP/video bằng GStreamer, chạy YOLO khi có chuyển động,
    ghi output ra mp4 H.264 và trả về file video.

    Ví dụ RTSP:
      /detect/gstreamer-stream?source=rtsp://user:pass@192.168.1.10:554/stream1&source_type=rtsp&seconds=10

    Ví dụ file local:
      /detect/gstreamer-stream?source=D:/video/test.mp4&source_type=file&seconds=10
    """
    try:
        if not check_at_least_one_model_loaded():
            return JSONResponse(
                status_code=500,
                content={"error": "Chưa load được model nào.", "models": MODEL_STATUS}
            )

        if every_n_frames <= 0:
            every_n_frames = 1

        pipeline = build_gstreamer_pipeline(
            source=source,
            source_type=source_type,
            latency=latency,
            width=width,
            height=height,
            fps=fps
        )

        cap, backend = open_capture_gstreamer_or_default(pipeline, use_gstreamer=True)

        if not cap.isOpened():
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Không mở được stream/video bằng GStreamer hoặc OpenCV.",
                    "pipeline": pipeline
                }
            )

        out_fps = float(fps or cap.get(cv2.CAP_PROP_FPS) or 25)
        if out_fps <= 0:
            out_fps = 25

        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            return JSONResponse(
                status_code=400,
                content={"error": "Mở được stream nhưng không đọc được frame đầu tiên.", "pipeline": pipeline}
            )

        h, w = first_frame.shape[:2]

        video_id = str(uuid.uuid4())
        raw_output_name = f"{video_id}_gst_raw.mp4"
        raw_output_path = os.path.join(OUTPUT_DIR, raw_output_name)
        final_output_name = f"{video_id}_gst_motion_h264.mp4"
        final_output_path = os.path.join(OUTPUT_DIR, final_output_name)

        writer = get_video_writer(raw_output_path, out_fps, w, h)

        if not writer.isOpened():
            cap.release()
            return JSONResponse(
                status_code=500,
                content={"error": "Không tạo được file output video."}
            )

        prev_gray = None
        last_annotated = None
        last_detections = []
        last_detect_ts = 0.0

        frame_id = 0
        processed_frames = 0
        skipped_frames = 0
        total_frames = 0

        start = time.time()
        frame = first_frame

        while True:
            if total_frames > 0:
                ret, frame = cap.read()
                if not ret:
                    break

            total_frames += 1

            if time.time() - start >= int(seconds):
                break

            should_consider = (frame_id % every_n_frames == 0)

            if should_consider:
                changed, motion_score, curr_gray = motion_changed(
                    prev_gray,
                    frame,
                    threshold_percent=motion_threshold
                )

                force_detect = (time.time() - last_detect_ts) >= float(force_interval)

                if changed or force_detect or last_annotated is None:
                    annotated, detections = run_3_models_on_frame(frame, conf=conf)
                    last_annotated = annotated.copy()
                    last_detections = detections
                    last_detect_ts = time.time()
                    processed_frames += 1
                else:
                    annotated = last_annotated.copy()
                    skipped_frames += 1

                prev_gray = curr_gray
            else:
                annotated = last_annotated.copy() if last_annotated is not None else frame
                skipped_frames += 1

            cv2.putText(
                annotated,
                f"backend={backend} processed={processed_frames} skipped={skipped_frames}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            writer.write(annotated)
            frame_id += 1

        cap.release()
        writer.release()

        if total_frames == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Không đọc được frame nào từ stream/video."}
            )

        convert_to_h264(raw_output_path, final_output_path)

        return FileResponse(
            final_output_path,
            media_type="video/mp4",
            filename="gst_motion_result_h264.mp4",
            headers={
                "X-Video-Output-Url": f"/outputs/{final_output_name}",
                "X-Total-Frames": str(total_frames),
                "X-Processed-Frames": str(processed_frames),
                "X-Skipped-Frames": str(skipped_frames),
                "X-Backend": backend
            }
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc(),
                "models": MODEL_STATUS,
                "ffmpeg_found": get_ffmpeg_executable() is not None
            }
        )


# ============================================================
# Run command:
# uvicorn app_h264:app --reload --host 0.0.0.0 --port 8000
# ============================================================
