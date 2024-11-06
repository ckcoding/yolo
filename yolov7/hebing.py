import torch

def load_model_weights(weight_path):
    """
    加载模型权重
    """
    try:
        return torch.load(weight_path, map_location='cpu')
    except Exception as e:
        print(f"加载权重文件 {weight_path} 时出错：{str(e)}")
        return None

def merge_models(model1_weights, model2_weights, weight1=0.7, weight2=0.3):
    """
    合并两个模型的权重，支持加权合并
    """
    merged_weights = {}
    
    model1_keys = set(model1_weights.keys())
    model2_keys = set(model2_weights.keys())
    
    print("正在合并以下键：")
    common_keys = model1_keys.intersection(model2_keys)
    
    for key in common_keys:
        if isinstance(model1_weights[key], torch.Tensor):
            if model1_weights[key].shape != model2_weights[key].shape:
                print(f"警告：键 {key} 的张量形状不匹配")
                print(f"model1: {model1_weights[key].shape}, model2: {model2_weights[key].shape}")
                continue
            # 加权合并
            merged_weights[key] = weight1 * model1_weights[key] + weight2 * model2_weights[key]
        else:
            print(f"信息：键 {key} 不是张量，使用 model1 的值")
            # 其他合并逻辑保持不变
            merged_weights[key] = model1_weights[key] if model1_weights[key] is not None else model2_weights[key]

        print(f"合并键 {key} 的值：")
        print(f"model1: {model1_weights[key]}")
        print(f"model2: {model2_weights[key]}")
        print(f"merged: {merged_weights[key]}")
    
    return merged_weights

def save_merged_weights(merged_weights, output_path):
    """
    保存合并后的权重
    """
    try:
        torch.save(merged_weights, output_path)
        print(f"合并后的权重已保存到：{output_path}")
    except Exception as e:
        print(f"保存权重文件时出错：{str(e)}")

def main():
    weight_path1 = 'best1.pt'  # 第一个权重文件路径
    weight_path2 = 'best2.pt'  # 第二个权重文件路径
    output_path = 'merged_model.pt'  # 输出文件路径
    
    print("正在加载第一个模型的权重...")
    weights1 = load_model_weights(weight_path1)
    if weights1 is None:
        return
        
    print("正在加载第二个模型的权重...")
    weights2 = load_model_weights(weight_path2)
    if weights2 is None:
        return
    
    print("\n模型权重信息：")
    print("Model1 权重键：", weights1.keys())
    print("Model2 权重键：", weights2.keys())
    
    print("\n检查权重形状：")
    for key in weights1.keys():
        if key in weights2:
            if isinstance(weights1[key], torch.Tensor) and isinstance(weights2[key], torch.Tensor):
                if weights1[key].shape != weights2[key].shape:
                    print(f"警告：层 {key} 的形状不匹配：")
                    print(f"Model1: {weights1[key].shape}")
                    print(f"Model2: {weights2[key].shape}")
            else:
                print(f"信息：键 {key} 不是张量")
    
    print("\n正在合并模型权重...")
    merged_weights = merge_models(weights1, weights2)
    
    print("\n正在保存合并后的权重...")
    save_merged_weights(merged_weights, output_path)

if __name__ == '__main__':
    main()
