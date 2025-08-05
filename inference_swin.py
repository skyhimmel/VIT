import torch
import torchvision.transforms as transforms
from swin import SwinTransformerMNIST
import matplotlib.pyplot as plt
from dataset import MNIST

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载模型
model = SwinTransformerMNIST(
    img_size=28,
    patch_size=4,
    in_chans=1,
    num_classes=10,
    embed_dim=64,
    depths=[2, 2, 2],
    num_heads=[2, 4, 8],
    window_size=1
).to(DEVICE)

# 尝试加载最佳模型，如果不存在则加载普通模型
try:
    model.load_state_dict(torch.load('swin_model_best.pth', map_location=DEVICE))
    print("Loaded best model")
except:
    try:
        model.load_state_dict(torch.load('swin_model.pth', map_location=DEVICE))
        print("Loaded regular model")
    except:
        print("No trained model found, using random weights")

model.eval()

# 加载测试数据集
test_dataset = MNIST(is_train=False)

# 随机选择几张图片进行测试
import random

print("随机测试几张图片：")
for i in range(5):
    idx = random.randint(0, len(test_dataset) - 1)
    img, label = test_dataset[idx]
    
    # 添加batch维度并移动到设备
    img_batch = img.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(img_batch)
        pred = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred].item()
    
    # 显示图片和预测结果
    plt.figure(figsize=(3, 3))
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f'真实标签: {label}\n预测: {pred} ({confidence:.2%})')
    plt.axis('off')
    plt.show()
    
    print(f"图片 {i+1}: 真实标签={label}, 预测={pred}, 置信度={confidence:.2%}")

# 计算整体测试准确率
def evaluate_accuracy():
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"\n整体测试集准确率: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    evaluate_accuracy()