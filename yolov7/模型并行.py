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
        },
        'crack': {
            'path': 'crack.pt',
        },
        'smoke': {
            'path': 'smoke.pt',
        }
    }

    IMG_SIZE = 320
    CONF_THRES = 0.1
    IOU_THRES = 0.1
    DEVICE = 'cpu'
    
    INPUT_DIR = './测试'
    OUTPUT_DIR = './结果'

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
def plot_chinese_box(img, xyxy, label, line_thickness):
    """使用PIL绘制中文标签的边界框"""
    # 转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 设置绿色
    green_color = (0, 255, 0)
    
    # 绘制边界框
    x1, y1, x2, y2 = map(int, xyxy)
    draw.rectangle([x1, y1, x2, y2], outline=green_color, width=line_thickness)
    
    # 使用 textbbox 获取文本边界框
    bbox = draw.textbbox((x1, y1), label)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # 绘制标签背景
    draw.rectangle([x1, y1, x1 + text_w, y1 + text_h], fill=green_color)
    
    # 绘制标签文字
    draw.text((x1, y1), label, fill=(255, 255, 255))
    
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
        
        # 对每个模型进行检测
        for model_name, model in models.items():
            with torch.no_grad():
                pred = model(img)[0]
                pred = non_max_suppression(pred, Config.CONF_THRES, Config.IOU_THRES)
            
            if len(pred[0]):
                det = pred[0]
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    # 获取英文标签
                    eng_label = model.names[int(cls)]
                    # 获取该标签的颜色（如果没有配置则使用默认颜色）
                    label = f'{eng_label} {conf:.2f}'
                    im0 = plot_chinese_box(
                        im0,
                        xyxy,
                        label=eng_label,
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
    # 创建输出目录
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(Config.INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 处理每张图片
    for image_file in image_files:
        time.sleep(5)
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
    main()