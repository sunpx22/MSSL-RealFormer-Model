import torch
import pandas as pd
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from realformer import RealFormerEncoder
print("# GPU is available")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("# Load dataset")

data_normalized = pd.read_excel("Model2_solubility.xlsx")
data_normalized = data_normalized.values
X = data_normalized[0:, 1:-1].astype(float)
# Y = data_normalized[0:, -1].astype(int)

# 定义Realformer模型
class RealformerModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(RealformerModel, self).__init__()
        # 定义 Realformer 编码器，并指定输入维数和头数
        self.encoder = RealFormerEncoder(d_model=input_size, num_heads=5, expansion_factor=2, dropout=0.5, num_layers=5)
        # 定义全连接层，将 Transformer 编码器的输出映射到分类空间
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        # 在序列的第2个维度（也就是时间步或帧）上添加一维以适应 Transformer 的输入格式
        x = x.unsqueeze(1)
        # 将输入数据流经 Transformer 编码器进行特征提取
        x = self.encoder(x)
        # 通过压缩第2个维度将编码器的输出恢复到原来的形状
        x = x.squeeze(1)
        # 将编码器的输出传入全连接层，获得最终的输出结果
        x = self.fc(x)
        return x

print("Create model")

# 调用模型，定义输入特征及分类数量
model = RealformerModel(input_size=40, num_classes=3).to(device)
model.load_state_dict(torch.load('model2_params_2.pkl'))

print("Define loss and optimizer")
# 定义损失函数，优化器及学习率
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

X_1 = torch.tensor(X).float().to(device)

with torch.no_grad():

    outputs = model(X_1)
    _, predicted = torch.max(outputs.data, 1)

    # print(f'outputs:{predicted},y_test:{Y_very}')

Y_1 = predicted.cpu().numpy()

print("Results:", Y_1)


