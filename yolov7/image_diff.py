from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import cv2
import numpy as np

# 解析输入参数
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="first input image")
ap.add_argument("-s", "--second", required=True, help="second input image")
args = vars(ap.parse_args())

# 加载输入图像
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])

# 检查图像是否加载正确
if imageA is None or imageB is None:
    print("Error loading images!")
    exit(1)

# 打印图像尺寸
print("Image A size: ", imageA.shape)
print("Image B size: ", imageB.shape)

# 转换为灰度图像
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# 调整图像尺寸使 imageB 与 imageA 匹配
imageB_resized = cv2.resize(imageB, (grayA.shape[1], grayA.shape[0]))

# 计算结构相似性指数（SSIM）
(score, diff) = compare_ssim(grayA, cv2.cvtColor(imageB_resized, cv2.COLOR_BGR2GRAY), full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# 对差异图像进行阈值处理前，先进行高斯模糊以减少噪声
diff = cv2.GaussianBlur(diff, (3, 3), 0)
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# 进行形态学操作来去除小噪点
kernel = np.ones((3,3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# 在寻找轮廓时添加面积过滤
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# 只绘制面积大于特定阈值的轮廓
min_area = 100  # 可以根据实际需求调整这个值
for c in cnts:
    if cv2.contourArea(c) > min_area:  # 添加面积过滤
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB_resized, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 确保所有图像尺寸一致
imageB_resized = cv2.resize(imageB_resized, (imageA.shape[1], imageA.shape[0]))
diff = cv2.resize(diff, (imageA.shape[1], imageA.shape[0]))
thresh = cv2.resize(thresh, (imageA.shape[1], imageA.shape[0]))

# 将差异图像转换为彩色伪彩色显示
diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
# 将差异图像转换为 4 通道 (BGRA) 格式
diff_colored_bgra = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2BGRA)

# 将阈值图转换为 4 通道 (BGRA) 格式
thresh_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGRA)

# 定义半透明黄色
yellow_color = [0, 255, 255, 128]  # 128 表示半透明

# 将非零区域设为半透明黄色
thresh_colored[np.where(thresh != 0)] = yellow_color

# 保存透明的阈值图像
output_thresh_path = "images/thresh_transparent.png"
cv2.imwrite(output_thresh_path, thresh_colored)
print(f"透明的黄色阈值图已保存至 {output_thresh_path}")

# 如果希望生成黑色背景的可视化图像以便在不支持透明通道的查看器中查看，可以生成一个带黑色背景的示例图
thresh_black_bg = thresh_colored.copy()
thresh_black_bg[np.where(thresh == 0)] = [0, 0, 0, 255]  # 设置背景为黑色
output_black_bg_path = "images/thresh_black_bg.png"
cv2.imwrite(output_black_bg_path, thresh_black_bg)
print(f"黑色背景的黄色阈值图已保存至 {output_black_bg_path}")

# 创建阈值图重叠到修改后图像上的效果
overlay_image = imageB_resized.copy()
# 转换为BGRA以支持透明度
overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2BGRA)
# 在修改后的图像上叠加黄色半透明遮罩
mask = thresh != 0  # 创建布尔掩码
overlay_image[mask] = cv2.addWeighted(
    overlay_image[mask], 
    0.7,  # 原图权重
    np.full(overlay_image[mask].shape, yellow_color, dtype=np.uint8),
    0.3,  # 黄色遮罩权重
    0
)
# 保存重叠效果图像
output_overlay_path = "images/overlay_result.png"
cv2.imwrite(output_overlay_path, overlay_image)
print(f"重叠效果图像已保存至 {output_overlay_path}")

# 将 top_row 转换为 4 通道
top_row_bgra = cv2.cvtColor(np.hstack([imageA, imageB_resized]), cv2.COLOR_BGR2BGRA)
bottom_row = np.hstack([diff_colored_bgra, thresh_colored])
combined_image = np.vstack([top_row_bgra, bottom_row])

# 保存最终合成图像
output_path = "images/comparison_result.png"
cv2.imwrite(output_path, combined_image)
print(f"合成图像已保存至 {output_path}")
