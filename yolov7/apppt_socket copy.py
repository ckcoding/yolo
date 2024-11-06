# ============== 系统和基础库导入 ==============
import os                  # 操作系统接口，用于文件和目录操作
import time               # 时间相关函数，用于延时和计时
import base64             # Base64编码解码，用于图像数据传输
from threading import Thread, Event  # 线程相关，用于并发处理
from datetime import datetime        # 日期时间处理
from queue import Queue             # 队列，用于线程间数据传递
import gc                 # 垃圾回收，用于内存管理

# ============== 第三方库导入 ==============
import cv2                # OpenCV库，用于图像处理和视频流处理
import numpy as np        # NumPy库，用于数值计算和数组操作
import torch             # PyTorch深度学习框架
from flask import Flask, render_template  # Flask Web框架
from flask_socketio import SocketIO, emit # Flask-SocketIO，用于WebSocket通信
from PIL import Image, ImageDraw, ImageFont  # PIL库，用于图像处理和中文文字绘制

# ============== YOLOv7相关导入 ==============
from models.experimental import attempt_load  # YOLOv7模型加载函数
from utils.general import non_max_suppression, scale_coords  # YOLOv7工具函数
from utils.plots import plot_one_box  # 边界框绘制函数
from utils.torch_utils import select_device  # 设备选择函数

# ============== 全局配置类 ==============
class Config:
    """
    全局配置类，存储所有配置参数
    """
    # 目录配置
    SAVE_DIR = 'saved_frames'  # 保存帧的目录
    FONT_PATH = "./Alimama_ShuHeiTi_Bold.ttf"  # 中文字体路径
    MODEL_PATH = 'merged_model.pt'  # YOLOv7模型路径
    
    # 视频流配置
    STREAM_URL = "https://v9.cdn88.cn:7018/hls/18057/1/0/1.m3u8"  # 视频流URL
    FRAME_SKIP = 5    # 跳帧数，每N帧处理一帧
    RESIZE_DIMS = (320, 240)  # 处理分辨率，降低以提高性能
    
    # 模型配置
    IMG_SIZE = 320    # 模型输入图像大小
    CONF_THRES = 0.5  # 置信度阈值，高于此值的检测框才会显示
    IOU_THRES = 0.5   # IOU阈值，用于非极大值抑制
    
    # 设备配置
    DEVICE = ''       # 计算设备，空字符串表示自动选择
    BATCH_SIZE = 1    # 批处理大小，CPU模式建议使用1
    
    # 性能优化参数
    FRAME_BUFFER_SIZE = 2  # 帧缓冲区大小
    WARMUP_ITERATIONS = 1  # 模型预热迭代次数
    MAX_RETRIES = 3       # 最大重试次数
    
    # 标签配置：英文标签到中文的映射
    CHINESE_LABELS = {
        'person': '人',
        'car': '汽车',
        'truck': '卡车',
        'bus': '公交车',
    }
    
    # 标签颜色配置：不同类别使用不同颜色（BGR格式）
    LABEL_COLORS = {
        'car': (255, 0, 0),     # 红色
        'person': (0, 255, 0),   # 绿色
    }
    
    # 需要显示的标签列表
    SHOW_LABELS = ['person', 'car']

# ============== 创建保存目录 ==============
if not os.path.exists(Config.SAVE_DIR):
    os.makedirs(Config.SAVE_DIR)

# ============== Flask应用初始化 ==============
app = Flask(__name__)  # 创建Flask应用实例
socketio = SocketIO(app)  # 创建SocketIO实例，用于WebSocket通信

# ============== 初始化模型和设备 ==============
print("正在初始化设备...")
device = select_device(Config.DEVICE)  # 选择计算设备（CPU/GPU）
print(f"使用设备: {device}")

print(f"正在加载模型: {Config.MODEL_PATH}")
model = attempt_load(Config.MODEL_PATH, map_location=device)  # 加载YOLOv7模型
model.eval()  # 设置模型为评估模式

# ============== 流状态管理类 ==============
class StreamState:
    """
    用于管理视频流状态的类
    """
    def __init__(self):
        self.active = True  # 流是否活动
        self.fps = 0        # 当前FPS
        self.frame_count = 0  # 帧计数器
        self.last_fps_time = time.time()  # 上次FPS计算时间
        self.stop_event = Event()  # 停止事件，用于优雅关闭

# 创建全局流状态实例
stream_state = StreamState()

def calculate_fps():
    """
    计算并更新FPS（每秒帧数）
    Returns:
        float: 当前的FPS值
    """
    current_time = time.time()  # 获取当前时间
    fps = 1.0 / (current_time - stream_state.last_fps_time)  # 计算FPS
    stream_state.last_fps_time = current_time  # 更新上次计算时间
    stream_state.fps = fps  # 更新FPS值
    return fps

def draw_label(img, xyxy, label_text, color):
    """
    在图像上绘制标签和边框
    
    Args:
        img (numpy.ndarray): 输入图像
        xyxy (list): 边界框坐标 [x1, y1, x2, y2]
        label_text (str): 要显示的标签文本
        color (tuple): BGR颜色值
    
    Returns:
        numpy.ndarray: 处理后的图像
    """
    try:
        # 将OpenCV格式(BGR)转换为PIL格式(RGB)以支持中文
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)  # 创建绘图对象
        
        # 设置字体和大小
        font = ImageFont.truetype(Config.FONT_PATH, 12)
        
        # 获取文本框的位置和大小
        x1, y1 = int(xyxy[0]), int(xyxy[1])  # 边界框左上角坐标
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]  # 文本宽度
        text_height = text_bbox[3] - text_bbox[1]  # 文本高度
        
        # 绘制文本背景框
        bg_coords = [x1, y1 - text_height - 4, x1 + text_width + 8, y1]
        draw.rectangle(bg_coords, fill=color[::-1])  # 填充背景色
        
        # 绘制文本
        draw.text((x1 + 4, y1 - text_height - 2), label_text, font=font, fill=(255, 255, 255))
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"绘制标签时出错: {str(e)}")
        return img  # 出错时返回原图

def process_frame(frame):
    """
    处理单帧图像，进行目标检测和标注
    
    Args:
        frame (numpy.ndarray): 输入帧
    
    Returns:
        numpy.ndarray: 处理后的帧
    """
    try:
        # ===== 图像预处理 =====
        # 调整图像大小到模型输入尺寸
        img = cv2.resize(frame, (Config.IMG_SIZE, Config.IMG_SIZE))
        # BGR转RGB，并调整维度顺序（HWC到CHW）
        img = img[:, :, ::-1].transpose(2, 0, 1)
        # 确保数组内存连续，提高运行效率
        img = np.ascontiguousarray(img)
        # 转换为PyTorch张量并移到指定设备
        img = torch.from_numpy(img).to(device)
        # 归一化到0-1范围
        img = img.float() / 255.0
        
        # 添加批处理维度
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # ===== 模型推理 =====
        with torch.no_grad():  # 不计算梯度，节省内存
            pred = model(img)[0]  # 前向传播
            # 非极大值抑制，过滤重叠的检测框
            pred = non_max_suppression(
                pred, 
                Config.CONF_THRES,  # 置信度阈值
                Config.IOU_THRES,   # IOU阈值
                classes=None,       # 不限制类别
                agnostic=False      # 不进行类别无关的NMS
            )
        
        # ===== 处理预测结果 =====
        det = pred[0]  # 获取第一张图片的检测结果
        im0 = frame.copy()  # 复制原图用于绘制
        
        if len(det):  # 如果有检测到的目标
            # 将检测框坐标从模型尺寸缩放到原图尺寸
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            
            # 处理每个检测框
            for *xyxy, conf, cls in reversed(det):
                cls_name = model.names[int(cls)]  # 获取类别名称
                if cls_name in Config.SHOW_LABELS:  # 如果是需要显示的类别
                    # 获取该类别的颜色
                    color = Config.LABEL_COLORS.get(cls_name, (0, 255, 0))
                    # 获取中文标签
                    chinese_label = Config.CHINESE_LABELS.get(cls_name, cls_name)
                    
                    # 绘制边界框
                    plot_one_box(xyxy, im0, label=None, color=color, line_thickness=1)
                    
                    # 绘制标签
                    label_text = f"{chinese_label} {conf:.2f}"
                    im0 = draw_label(im0, xyxy, label_text, color)
        
        # 添加FPS显示
        fps = calculate_fps()
        cv2.putText(im0, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return im0
    except Exception as e:
        print(f"处理帧时出错: {str(e)}")
        return frame
    finally:
        # 清理临时变量，释放内存
        if 'img' in locals():
            del img
        if 'pred' in locals():
            del pred
        if 'det' in locals():
            del det
        gc.collect()  # 触发垃圾回收

def read_frames(frame_queue, stream_state):
    """读取视频流帧的线程函数，持续捕获视频帧并放入队列"""
    cap = None  # 视频捕获对象初始化为空
    retry_count = 0  # 重试计数器
    
    # 当停止事件未触发且重试次数在允许范围内时继续循环
    while not stream_state.stop_event.is_set() and retry_count < Config.MAX_RETRIES:
        try:
            # 如果视频捕获对象未初始化，则创建新的捕获对象
            if cap is None:
                cap = cv2.VideoCapture(Config.STREAM_URL)  # 使用配置的URL创建视频捕获对象
                if not cap.isOpened():
                    raise Exception("无法打开视频流")  # 如果打开失败则抛出异常
            
            # 持续读取视频帧的内循环
            while not stream_state.stop_event.is_set():
                ret, frame = cap.read()  # 读取一帧视频
                if not ret:
                    raise Exception("无法读取帧")  # 读取失败时抛出异常
                
                stream_state.frame_count += 1  # 帧计数器递增
                # 根据跳帧设置决定是否处理当前帧
                if stream_state.frame_count % Config.FRAME_SKIP == 0:
                    frame = cv2.resize(frame, Config.RESIZE_DIMS)  # 调整帧的大小
                    if not frame_queue.full():  # 确保队列未满
                        frame_queue.put(frame)  # 将处理后的帧放入队列
                
                time.sleep(0.1)  # 降低CPU使用率的延时
                
        except Exception as e:
            # 发生异常时的错误处理
            print(f"读取帧时出错 (尝试 {retry_count + 1}/{Config.MAX_RETRIES}): {str(e)}")
            retry_count += 1  # 增加重试计数
            if cap is not None:
                cap.release()  # 释放当前的视频捕获对象
                cap = None
            time.sleep(2)  # 重试前的等待时间
    
    # 退出时确保释放视频捕获对象
    if cap is not None:
        cap.release()

def video_stream():
    """视频流处理的主函数，负责帧的处理和推送"""
    # 创建用于存储视频帧的队列，设置最大容量
    frame_queue = Queue(maxsize=Config.FRAME_BUFFER_SIZE)
    
    # 创建并启动读取帧的后台线程
    read_thread = Thread(target=read_frames, args=(frame_queue, stream_state), daemon=True)
    read_thread.start()
    
    # 主循环：处理队列中的帧并推送到客户端
    while not stream_state.stop_event.is_set():
        try:
            # 检查队列是否为空
            if frame_queue.empty():
                time.sleep(0.1)  # 队列为空时等待
                continue
            
            frame = frame_queue.get()  # 从队列获取一帧
            processed_frame = process_frame(frame)  # 处理帧（应用特效或转换）
            
            # 将处理后的帧编码为JPEG格式并转换为base64字符串
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            # 通过WebSocket发送到客户端
            socketio.emit('video_frame', {'image': frame_base64})
            
            time.sleep(0.2)  # 控制帧处理速率的延时
            
        except Exception as e:
            print(f"视频流处理错误: {str(e)}")
            time.sleep(1)  # 发生错误时的等待时间

# Flask路由和事件处理
@app.route('/')
def index():
    """处理根路由请求，返回主页面"""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """处理WebSocket客户端连接事件"""
    print('客户端已连接')

@socketio.on('disconnect')
def handle_disconnect():
    """处理WebSocket客户端断开连接事件"""
    print('客户端已断开连接')

def cleanup():
    """清理资源的函数，确保程序正确关闭"""
    stream_state.stop_event.set()  # 设置停止事件
    print("正在关闭视频流...")
    time.sleep(1)  # 等待所有线程完成清理

if __name__ == '__main__':
    try:
        print("启动视频流处理线程...")
        # 启动视频流处理线程
        Thread(target=video_stream, daemon=True).start()
        print(f"启动Flask服务器在端口 5002...")
        # 启动Flask服务器
        socketio.run(app, debug=False, host='0.0.0.0', port=5002)
    except KeyboardInterrupt:
        cleanup()  # 处理键盘中断
    finally:
        cleanup()  # 确保在任何情况下都执行清理