# ============== 系统和基础库导入 ==============
import os
import gc
import time
from concurrent.futures import ThreadPoolExecutor
# ============== 第三方库导入 ==============
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ============== YOLOv7相关导入 ==============
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

# ============== 全局配置类 ==============
class Config:
    # 模型配置
    MODELS = {
        'default': {
            'path': 'yolov7.pt',
            'color': (0, 255, 0)  # 绿色
        },
        'crack': {
            'path': 'crack.pt',
            'color': (255, 0, 0) # 红色
        },
        'smoke': {
            'path': 'smoke.pt',
            'color': (0, 0, 255) # 蓝色
        }
    }
    FONT_PATH = "./Alimama_ShuHeiTi_Bold.ttf"
    # 标签中英文映射
    LABEL_MAP = {
        'person': '人',
        'car': '汽车',
        'bike': '自行车',
        'motorcycle': '摩托车',
        'bus': '公交车',
        'truck': '卡车',
        'traffic_light': '交通灯',
        'traffic_sign': '交通标志',
        'smoke': '香烟',
        'crack': '裂缝',
        'fire': '火灾',
        "toothbrush": "牙刷",
        # 可以添加更多映射
    }

    # 标签颜色映射 (BGR格式)
    LABEL_COLORS = {
        'person': (0, 255, 0),    # 绿色
        'car': (0, 0, 255),       # 红色
        'bike': (255, 255, 0),    # 青色
        'motorcycle': (255, 0, 255), # 紫色
        'bus': (0, 255, 255),     # 黄色
        'truck': (255, 255, 255),  # 白色
        'traffic_light': (255, 0, 0), # 蓝色
        'traffic_sign': (0, 255, 255), # 青色
        'default': (255, 255, 0),  # 默认青色
        'smoke': (125, 255, 0),    # 青色
        'crack': (25, 255, 0),    # 青色
        'fire': (255, 55, 0),    # 青色
        
    }
    
    IMG_SIZE = 416
    IOU_THRES = 0.45
    DEVICE = '0' if torch.cuda.is_available() else 'cpu'  # 使用GPU加速
    
    INPUT_DIR = './images'
    OUTPUT_DIR = './results'
    
    # 提高置信度阈值
    CONF_THRESHOLDS = {
        'smoke': 0.7,    # 略微降低阈值
        'person': 0.6,   
        'default': 0.6   
    }
    
    # 修改设备配置，优先使用GPU
    BATCH_SIZE = 16 if torch.cuda.is_available() else 4  # 增加GPU批量
    
    # 增加并行处理参数
    NUM_WORKERS = min(4, os.cpu_count())  # 减少worker数量避免overhead
    
    # 优化图像处理参数
    IMG_SIZE = 416  # 降低分辨率以提高速度
    HALF = True if torch.cuda.is_available() else False  # GPU时启用半精度
    
    # 缓存字体对象
    FONT = None  # 将在初始化时加载

# 初始化时加载字体
try:
    Config.FONT = ImageFont.truetype(Config.FONT_PATH, 20)
except:
    print(f"无法加载字体文件 {Config.FONT_PATH}，使用默认字体")
    Config.FONT = ImageFont.load_default()

# 初始化模型
print("正在初始化设备...")
device = select_device(Config.DEVICE)
print(f"使用设备: {device}")

# 加载所有模型
models = {}
for model_name, model_config in Config.MODELS.items():
    print(f"正在加载{model_name}模型: {model_config['path']}")
    model = attempt_load(model_config['path'], map_location=device)
    if Config.HALF and device.type != 'cpu':
        model.half()  # 将模型转换为半精度
    model.eval()
    models[model_name] = model

# 绘制中文标签的边界框
def plot_chinese_box(img, xyxy, label, color, line_thickness):
    """优化的边界框绘制函数"""
    # 转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 绘制边界框
    x1, y1, x2, y2 = map(int, xyxy)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_thickness)
    
    # 使用缓存的字体对
    bbox = draw.textbbox((x1, y1), label, font=Config.FONT)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # 计算标签的可能位置（顺序：左上、左下、右上、右下）
    label_positions = [
        (x1, y1 - text_h if y1 - text_h >= 0 else y1),  # 左上
        (x1, y2),                                        # 左下
        (x2 - text_w, y1 - text_h if y1 - text_h >= 0 else y1),  # 右上
        (x2 - text_w, y2)                               # 右下
    ]
    
    # 获取图像尺寸
    img_h, img_w = img.shape[:2]
    
    # 选择最合适的位置
    for label_x, label_y in label_positions:
        # 确保标签在图像范围内
        if (0 <= label_x <= img_w - text_w and 
            0 <= label_y <= img_h - text_h):
            # 绘制标签背景
            draw.rectangle(
                [label_x, label_y, label_x + text_w, label_y + text_h],
                fill=color
            )
            # 绘制标签文字
            draw.text((label_x, label_y), label, fill=(255, 255, 255), font=Config.FONT)
            break
    
    # 转换回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 处理图片
def process_batch(image_paths):
    """批量处理图像"""
    try:
        batch_imgs = []
        original_imgs = []
        valid_paths = []  # 添加这行来跟踪有效的图片路径
        
        # 直接串行处理，避免多线程开销
        for path in image_paths:
            frame = cv2.imread(path)
            if frame is None:
                continue
                
            # 预处理
            img = cv2.resize(frame, (Config.IMG_SIZE, Config.IMG_SIZE))
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.float() / 255.0
            
            batch_imgs.append(img)
            original_imgs.append(frame)
            valid_paths.append(path)  # 记录有效的图片路径
        
        if not batch_imgs:
            return [], []  # 返回空列表和对应的路径
            
        # 合并批次
        batch = torch.stack(batch_imgs)
        
        # 使用半精度计算
        if Config.HALF and device.type != 'cpu':
            batch = batch.half()
            
        # 初始化results列表，确保每个图片都有对应的空列表
        results = [[] for _ in range(len(batch_imgs))]
        
        # 批量推理
        for model_name, model in models.items():
            with torch.no_grad():
                pred = model(batch)[0]
                conf_thres = Config.CONF_THRESHOLDS.get(model_name, Config.CONF_THRESHOLDS['default'])
                pred = non_max_suppression(pred, conf_thres, Config.IOU_THRES)
                
            # 处理每张图片的预测结果
            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_coords(batch.shape[2:], det[:, :4], original_imgs[i].shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        eng_label = model.names[int(cls)]
                        cn_label = Config.LABEL_MAP.get(eng_label, eng_label)
                        color = Config.LABEL_COLORS.get(eng_label, Config.LABEL_COLORS['default'])
                        
                        # 不需要检查results长度，因为已经初始化好了
                        results[i].append({
                            'xyxy': xyxy,
                            'conf': conf,
                            'label': f'{cn_label} {conf:.2f}',
                            'color': color
                        })
        
        # 处理结果
        processed_images = []
        for i, detections in enumerate(results):
            if detections:
                # 按置信度排序
                detections.sort(key=lambda x: x['conf'], reverse=True)
                
                # 一次性绘制
                result_img = original_imgs[i].copy()
                for det in detections:
                    result_img = plot_chinese_box(
                        result_img,
                        det['xyxy'],
                        det['label'],
                        det['color'],
                        1
                    )
                processed_images.append(result_img)
            else:
                processed_images.append(None)  # 如果没有检测到物体，添加None
                
        return processed_images, valid_paths  # 返回处理后的图片和对应的原始路径
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # 清理内存
        if 'batch' in locals():
            del batch
        torch.cuda.empty_cache()
        gc.collect()

# 主函数
def main():
    # 删除results目录下所有文件和子目录
    if os.path.exists(Config.OUTPUT_DIR):
        for file in os.listdir(Config.OUTPUT_DIR):
            file_path = os.path.join(Config.OUTPUT_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"删除文件/目录时出错: {file_path}, 错误: {e}")
    
    # 创建输出目录
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)
    
    # 获取所有图片文件并按名称排序
    image_files = [os.path.join(Config.INPUT_DIR, f) 
                  for f in sorted(os.listdir(Config.INPUT_DIR))
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"找到 {len(image_files)} 个图片文件")
    print("处理顺序：")
    for f in image_files:
        print(f"  - {os.path.basename(f)}")
    
    # 将图片分成多个批次
    batches = [image_files[i:i + Config.BATCH_SIZE] 
              for i in range(0, len(image_files), Config.BATCH_SIZE)]
    
    # 处理每个批次
    for i, batch in enumerate(batches):
        print(f"正在处理批次 {i+1}/{len(batches)}")
        results, valid_paths = process_batch(batch)  # 获取结果和对应的路径
        
        # 保存结果
        for result_img, input_path in zip(results, valid_paths):
            if result_img is not None:
                # 修改输出文件命名方式
                filename = os.path.basename(input_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(
                    Config.OUTPUT_DIR, 
                    f"{name}_result{ext}"
                )
                cv2.imwrite(output_path, result_img)
                print(f"已保存结果到: {output_path}")

if __name__ == '__main__':
    #打印执行时间
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"执行时间: {end_time - start_time:.2f} 秒")