import os
from pathlib import Path
import sys
import os

# 修改系统路径添加方式
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将当前目录添加到环境变量

# 确保YOLOv7的根目录在系统路径中
YOLOV7_ROOT = os.path.join(ROOT, 'yolov7')  # 假设yolov7文件夹在当前目录下
if os.path.exists(YOLOV7_ROOT) and YOLOV7_ROOT not in sys.path:
    sys.path.append(YOLOV7_ROOT)

from utils.dataloaders import LoadImages

import torch
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, Profile, check_img_size, colorstr, cv2,
                         non_max_suppression, scale_boxes, xyxy2xywh)

from utils.torch_utils import DetectMultiBackend  # 从 utils.torch_utils 导入而不是 models.common
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from models.experimental import attempt_load  # 替代方案

def detect(weights_hat, weights_safety, source, save_dir, half, line_thickness):
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 创建多级目录
    if not os.path.exists(save_dir + 'labels/'):
        os.makedirs(save_dir + 'labels/')

    # 初始化设备
    device = select_device('0')

    # 加载模型
    model_hat = attempt_load(weights_hat, device=device)
    model_safety = attempt_load(weights_safety, device=device)

    # 设置推理模式
    model_hat.eval()
    model_safety.eval()

    # 如果使用半精度
    if half:
        model_hat.half()
        model_safety.half()

    # 获取步长和类别名称
    stride_hat = int(model_hat.stride.max())
    names_hat = model_hat.module.names if hasattr(model_hat, 'module') else model_hat.names
    names_safety = model_safety.module.names if hasattr(model_safety, 'module') else model_safety.names

    # 检查图像尺寸
    imgsz = check_img_size(imgsz=640, s=stride_hat)
    
    # 加载数据
    dataset = LoadImages(source, img_size=imgsz, stride=stride_hat, auto=True)

    # 初始化计数器和计时器
    seen, dt = 0, Profile()

    # 处理每一张图片
    for path, im, im0s, vid_cap, s in dataset:
        # 预处理图像
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        # 推理
        with dt:
            pred_hat = model_hat(im, augment=False, visualize=False)[0]
            pred_safety = model_safety(im, augment=False, visualize=False)[0]

        # NMS
        pred_hat = non_max_suppression(pred_hat, 0.25, 0.45, None, False, max_det=1000)
        pred_safety = non_max_suppression(pred_safety, 0.25, 0.45, None, False, max_det=1000)

        # 处理安全帽检测结果
        for i, det in enumerate(pred_hat):
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)
            save_path = str(save_dir + p.name)
            txt_path = str(save_dir + 'labels/' + p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=line_thickness, example=str(names_hat))

            if len(det):
                # 将边界框从img_size缩放到im0的尺寸
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 写入结果
                for *xyxy, conf, cls in reversed(det):
                    # 归一化xywh
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh)  # 标签格式
                    with open(f'{txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 添加边界框到图像
                    c = int(cls)
                    label = f'{names_hat[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # 保存结果
            im0 = annotator.result()
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)

        # 处理安全衣检测结果
        for i, det in enumerate(pred_safety):
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)
            save_path = str(save_dir + p.name.replace('.', '_safety.'))
            txt_path = str(save_dir + 'labels/' + p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            annotator = Annotator(im0, line_width=line_thickness, example=str(names_safety))

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls + len(names_hat), *xywh)  # 安全衣类别号接续安全帽类别号
                    with open(f'{txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    c = int(cls)
                    label = f'{names_safety[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c + len(names_hat), True))

            im0 = annotator.result()
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)

        LOGGER.info(f'{s}Done. ({dt.dt:.3f}s)')
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

if __name__ == "__main__":
    with torch.no_grad():
        half = False  # 是否使用半精度
        line_thickness = 3  # 边界框厚度
        weights_hat = r'./yolov7.pt'  # 安全帽模型路径
        weights_safety = r'./best.pt'  # 安全衣模型路径
        source = r'./images'  # 输入图片文件夹
        save_dir = r'./detect/'  # 结果保存路径
        detect(weights_hat, weights_safety, source, save_dir, half, line_thickness)