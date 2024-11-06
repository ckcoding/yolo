# ============== 系统和基础库导入 ==============
import os
import time
import base64
from threading import Thread, Event
from datetime import datetime
from queue import Queue
import gc
from concurrent.futures import ThreadPoolExecutor

# ============== 第三方库导入 ==============
import cv2
import numpy as np
import torch
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from PIL import Image, ImageDraw, ImageFont
from flask_cors import CORS

# ============== YOLOv7相关导入 ==============
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

# ============== 全局配置类 ==============
class Config:
    SAVE_DIR = 'saved_frames'
    FONT_PATH = "./Alimama_ShuHeiTi_Bold.ttf"
    MODEL_PATH = 'yolov7.pt'
    HELMET_MODEL_PATH = 'yolov7.pt'  # 安全帽模型路径
    VEST_MODEL_PATH = 'best.pt'      
    
    STREAM_URL = "https://v9.cdn88.cn:7018/hls/18057/1/0/1.m3u8"
    FRAME_SKIP = 5
    RESIZE_DIMS = (320, 240)
    
    IMG_SIZE = 320
    CONF_THRES = 0.5
    IOU_THRES = 0.5
    
    DEVICE = 'cpu'
    BATCH_SIZE = 1
    
    FRAME_BUFFER_SIZE = 5  # 增加队列缓冲区
    WARMUP_ITERATIONS = 1
    MAX_RETRIES = 3
    
    CHINESE_LABELS = {
        'person': '人',
        'car': '汽车',
        'truck': '卡车',
        'bus': '公交车',
    }
    
    LABEL_COLORS = {
        'car': (255, 0, 0),
        'person': (0, 255, 0),
    }
    
    SHOW_LABELS = ['person', 'car']

if not os.path.exists(Config.SAVE_DIR):
    os.makedirs(Config.SAVE_DIR)

# ============== Flask应用初始化 ==============
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# print("正在初始化设备...")
# device = select_device(Config.DEVICE)
# print(f"使用设备: {device}")

# print(f"正在加载模型: {Config.MODEL_PATH}")
# model = attempt_load(Config.MODEL_PATH, map_location=device)
# model.eval()


# 修改模型加载部分
print("正在初始化设备...")
device = select_device(Config.DEVICE)
print(f"使用设备: {device}")

print(f"正在加载安全帽检测模型: {Config.HELMET_MODEL_PATH}")
helmet_model = attempt_load(Config.HELMET_MODEL_PATH, map_location=device)
helmet_model.eval()

print(f"正在加载安全衣检测模型: {Config.VEST_MODEL_PATH}")
vest_model = attempt_load(Config.VEST_MODEL_PATH, map_location=device)
vest_model.eval()

# ============== 流状态管理类 ==============
class StreamState:
    def __init__(self):
        self.active = True
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.stop_event = Event()

stream_state = StreamState()

def calculate_fps():
    current_time = time.time()
    fps = 1.0 / (current_time - stream_state.last_fps_time)
    stream_state.last_fps_time = current_time
    stream_state.fps = fps
    return fps

def draw_label(img, xyxy, label_text, color):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(Config.FONT_PATH, 8)
        x1, y1 = int(xyxy[0]), int(xyxy[1])
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        bg_coords = [x1, y1 - text_height - 4, x1 + text_width + 8, y1]
        draw.rectangle(bg_coords, fill=color[::-1])
        draw.text((x1 + 4, y1 - text_height - 2), label_text, font=font, fill=(255, 255, 255))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"绘制标签时出错: {str(e)}")
        return img

# 修改处理帧的函数
def process_frame(frame):
    try:
        # 准备输入图像
        img = cv2.resize(frame, (Config.IMG_SIZE, Config.IMG_SIZE))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        im0 = frame.copy()
        
        # 先进行安全帽检测
        with torch.no_grad():
            pred_helmet = helmet_model(img)[0]
            pred_helmet = non_max_suppression(pred_helmet, Config.CONF_THRES, Config.IOU_THRES)
        
        if len(pred_helmet[0]):
            det_helmet = pred_helmet[0]
            det_helmet[:, :4] = scale_coords(img.shape[2:], det_helmet[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det_helmet):
                cls_name = helmet_model.names[int(cls)]
                if cls_name in Config.HELMET_LABELS:
                    color = Config.LABEL_COLORS.get(cls_name, (0, 255, 0))
                    chinese_label = Config.CHINESE_LABELS.get(cls_name, cls_name)
                    plot_one_box(xyxy, im0, label=None, color=color, line_thickness=1)
                    label_text = f"{chinese_label} {conf:.2f}"
                    im0 = draw_label(im0, xyxy, label_text, color)
        
        # 再进行安全衣检测
        with torch.no_grad():
            pred_vest = vest_model(img)[0]
            pred_vest = non_max_suppression(pred_vest, Config.CONF_THRES, Config.IOU_THRES)
        
        if len(pred_vest[0]):
            det_vest = pred_vest[0]
            det_vest[:, :4] = scale_coords(img.shape[2:], det_vest[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det_vest):
                cls_name = vest_model.names[int(cls)]
                if cls_name in Config.VEST_LABELS:
                    color = Config.LABEL_COLORS.get(cls_name, (0, 255, 0))
                    chinese_label = Config.CHINESE_LABELS.get(cls_name, cls_name)
                    plot_one_box(xyxy, im0, label=None, color=color, line_thickness=1)
                    label_text = f"{chinese_label} {conf:.2f}"
                    im0 = draw_label(im0, xyxy, label_text, color)
        
        fps = calculate_fps()
        cv2.putText(im0, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return im0
    except Exception as e:
        print(f"处理帧时出错: {str(e)}")
        return frame
    finally:
        if 'img' in locals():
            del img
        if 'pred_helmet' in locals():
            del pred_helmet
        if 'pred_vest' in locals():
            del pred_vest
        gc.collect()
        
def read_frames(frame_queue, stream_state):
    cap = None
    retry_count = 0
    
    while not stream_state.stop_event.is_set() and retry_count < Config.MAX_RETRIES:
        try:
            if cap is None:
                cap = cv2.VideoCapture(Config.STREAM_URL)
                if not cap.isOpened():
                    raise Exception("无法打开视频流")
            
            while not stream_state.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    raise Exception("无法读取帧")
                
                stream_state.frame_count += 1
                if stream_state.frame_count % Config.FRAME_SKIP == 0:
                    frame = cv2.resize(frame, Config.RESIZE_DIMS)
                    if not frame_queue.full():
                        frame_queue.put(frame)
                
                time.sleep(0.1)
                
        except Exception as e:
            print(f"读取帧时出错 (尝试 {retry_count + 1}/{Config.MAX_RETRIES}): {str(e)}")
            retry_count += 1
            if cap is not None:
                cap.release()
                cap = None
            time.sleep(2)
    
    if cap is not None:
        cap.release()

def process_frames_in_thread(frame_queue, stream_state):
    with ThreadPoolExecutor(max_workers=2) as executor:
        while not stream_state.stop_event.is_set():
            if frame_queue.empty():
                time.sleep(0.1)
                continue
            
            frame = frame_queue.get()
            future = executor.submit(process_frame, frame)
            
            processed_frame = future.result()
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('video_frame', {'image': frame_base64})
            
            time.sleep(0.2)

def video_stream():
    frame_queue = Queue(maxsize=Config.FRAME_BUFFER_SIZE)
    read_thread = Thread(target=read_frames, args=(frame_queue, stream_state), daemon=True)
    read_thread.start()
    
    process_frames_in_thread(frame_queue, stream_state)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('客户端已连接')

@socketio.on('disconnect')
def handle_disconnect():
    print('客户端已断开连接')

def cleanup():
    stream_state.stop_event.set()
    print("正在关闭视频流...")
    time.sleep(1)

if __name__ == '__main__':
    try:
        print("启动视频流处理线程...")
        Thread(target=video_stream, daemon=True).start()
        print(f"启动Flask服务器在端口 5002...")
        socketio.run(app, debug=False, host='0.0.0.0', port=5002)
    except KeyboardInterrupt:
        cleanup()
    finally:
        cleanup()
