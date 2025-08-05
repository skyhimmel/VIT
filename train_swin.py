import torch
from dataset import MNIST
from swin import SwinTransformerMNIST
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据集
dataset = MNIST()

# Swin Transformer模型
model = SwinTransformerMNIST(
    img_size=28,
    patch_size=4,
    in_chans=1,
    num_classes=10,
    embed_dim=64,
    depths=[2, 2, 2],
    num_heads=[2, 4, 8],
    window_size=1,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    drop_path_rate=0.1
).to(DEVICE)

try:    # 加载模型
    model.load_state_dict(torch.load('swin_model.pth'))
    print("Loaded existing model")
except:
    print("No existing model found, starting from scratch")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

EPOCH = 50
BATCH_SIZE = 64

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                       num_workers=10, persistent_workers=True)

iter_count = 0
best_acc = 0.0

for epoch in range(EPOCH):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if iter_count % 1000 == 0:
            acc = 100. * correct / total
            print(f'Epoch: {epoch}, Iter: {iter_count}, Loss: {loss:.4f}, Acc: {acc:.2f}%')
            
            # 保存临时模型
            torch.save(model.state_dict(), '.swin_model_temp.pth')
            os.replace('.swin_model_temp.pth', 'swin_model.pth')
            
        iter_count += 1
    
    # 每个epoch结束后的统计
    epoch_acc = 100. * correct / total
    avg_loss = epoch_loss / len(dataloader)
    print(f'Epoch {epoch}: Average Loss = {avg_loss:.4f}, Accuracy = {epoch_acc:.2f}%')
    
    # 保存最佳模型
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), 'swin_model_best.pth')
        print(f'New best accuracy: {best_acc:.2f}%')
    
    # 更新学习率
    scheduler.step()

print(f'Training completed! Best accuracy: {best_acc:.2f}%')
print('Final model saved as: swin_model.pth')
print('Best model saved as: swin_model_best.pth')