# ============== 系统和基础库导入 ==============
import os
import gc
import time
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
    
    IMG_SIZE = 320
    CONF_THRES = 0.7
    IOU_THRES = 0.45
    DEVICE = 'cpu'
    
    INPUT_DIR = './images'
    OUTPUT_DIR = './results'
    
    # 提高置信度阈值
    CONF_THRESHOLDS = {
        'smoke': 0.8,    # 香烟检测要求更高的置信度
        'person': 0.7,   # 人物检测
        'default': 0.7   # 默认阈值
    }

# 初始化模型
print("正在初始化设备...")
device = select_device(Config.DEVICE)
print(f"使用设备: {device}")

# 加载所有模型
models = {}
for model_name, model_config in Config.MODELS.items():
    print(f"正在加载{model_name}模型: {model_config['path']}")
    model = attempt_load(model_config['path'], map_location=device)
    model.eval()
    models[model_name] = model

# 绘制中文标签的边界框
def plot_chinese_box(img, xyxy, label, color, line_thickness):
    """使用PIL绘制中文标签的边界框"""
    # 转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 设置字体（使用配置中指定的字体文件）
    try:
        font = ImageFont.truetype(Config.FONT_PATH, 20)
    except:
        print(f"无法加载字体文件 {Config.FONT_PATH}，使用默认字体")
        font = ImageFont.load_default()
    
    # 绘制边界框
    x1, y1, x2, y2 = map(int, xyxy)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_thickness)
    
    # 使用 textbbox 获取文本边界框
    bbox = draw.textbbox((x1, y1), label, font=font)
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
            draw.text((label_x, label_y), label, fill=(255, 255, 255), font=font)
            break
    
    # 转换回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 处理图片
def process_image(image_path):
    try:
        # 读取图片
        frame = cv2.imread(image_path)
        if frame is None:
            raise Exception("无法读取图片")

        # 准备输入图像
        img = cv2.resize(frame, (Config.IMG_SIZE, Config.IMG_SIZE))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        im0 = frame.copy()
        
        # 创建一个列表存储所有检测结果
        all_detections = []
        
        # 对每个模型进行检测
        for model_name, model in models.items():
            with torch.no_grad():
                pred = model(img)[0]
                
                # 使用对应类别的置信度阈值
                conf_thres = Config.CONF_THRESHOLDS.get(model_name, Config.CONF_THRESHOLDS['default'])
                pred = non_max_suppression(pred, conf_thres, Config.IOU_THRES)
            
            if len(pred[0]):
                det = pred[0]
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    eng_label = model.names[int(cls)]
                    cn_label = Config.LABEL_MAP.get(eng_label, eng_label)
                    color = Config.LABEL_COLORS.get(eng_label, Config.LABEL_COLORS['default'])
                    
                    # 将检测结果添加到列表中
                    all_detections.append({
                        'xyxy': xyxy,
                        'conf': conf,
                        'label': f'{cn_label} {conf:.2f}',
                        'color': color
                    })
        
        # 按置信度排序
        all_detections.sort(key=lambda x: x['conf'], reverse=True)
        
        print(all_detections)
        # 绘制边框，优先绘制置信度高的
        for det in all_detections:
            im0 = plot_chinese_box(
                im0,
                det['xyxy'],
                label=det['label'],
                color=det['color'],
                line_thickness=1
            )
        
        return im0

    except Exception as e:
        print(f"处理图片时出错: {str(e)}")
        return None
    finally:
        if 'img' in locals():
            del img
        torch.cuda.empty_cache()
        gc.collect()

# 主函数
def main():
    # 删除results目录下所有文件
    for file in os.listdir(Config.OUTPUT_DIR):
        os.remove(os.path.join(Config.OUTPUT_DIR, file))
    
    # 创建输出目录
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(Config.INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 处理每张图片
    for image_file in image_files:
        print(f"正在处理: {image_file}")
        input_path = os.path.join(Config.INPUT_DIR, image_file)
        output_path = os.path.join(Config.OUTPUT_DIR, f"detected_{image_file}")
        
        # 处理图片
        result = process_image(input_path)
        
        if result is not None:
            # 保存结果
            cv2.imwrite(output_path, result)
            print(f"已保存结果到: {output_path}")
        else:
            print(f"处理失败: {image_file}")

if __name__ == '__main__':
    #打印执行时间
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"执行时间: {end_time - start_time:.2f} 秒")