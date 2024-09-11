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

data_normalized = pd.read_excel("Model4_zonghe.xlsx")
data_normalized = data_normalized.values
X = data_normalized[0:, 1: -1].astype(float)
Y = data_normalized[0:, -1].astype(int)

# 定义mixup函数
def mixup_data(xx, alpha=1.0):

    lam = np.random.beta(alpha, alpha)

    batch_size = np.size(xx[:, 0])
    index = torch.randperm(batch_size)

    mixed_xx = lam * xx + (1 - lam) * xx[index, :]

    return mixed_xx


# 区分训练集与验证集
X_train, X_very, Y_train, Y_very = train_test_split(X, Y, test_size=0.2)


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

print("Define loss and optimizer")
# 定义损失函数，优化器及学习率
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

print("Train model")
# 进行训练
losses = []
accuracys = []
num_epochs = 3000

X_train2 = torch.tensor(X_train).float().to(device)
Y_train2 = torch.tensor(Y_train).long().to(device)

for epoch in range(num_epochs):

    outputs = model(X_train2)

    loss = criterion(outputs, Y_train2)
    _, out2 = torch.max(outputs.data, 1)
    yy_train = Y_train2.cpu().numpy()
    out3 = out2.cpu().numpy()
    accuracy1 = accuracy_score(out3, yy_train)
    losses.append(loss)
    accuracys.append(accuracy1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, accuracy: {accuracy1.item():.4f}')


X_mixed_1 = mixup_data(X_train, alpha=1.0)
X_mixed_2 = mixup_data(X_train, alpha=1.0)
X_mixed_3 = mixup_data(X_train, alpha=1.0)
X_mixed1 = np.concatenate((X_mixed_1, X_mixed_2, X_mixed_3), axis=0)
X_mixed1 = torch.tensor(X_mixed1).float().to(device)

with torch.no_grad():

    outputs = model(X_mixed1)
    _, predicted = torch.max(outputs.data, 1)

    # print(f'outputs:{predicted},y_test:{Y_very}')

Y_mixed1 = predicted.cpu().numpy()
X_mixed1 = X_mixed1.cpu().numpy()

X_final1 = np.concatenate((X_train, X_mixed1), axis=0)
Y_final1 = np.concatenate((Y_train, Y_mixed1))
X_final11 = torch.tensor(X_final1).float().to(device)
Y_final11 = torch.tensor(Y_final1).long().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.000075)
for epoch in range(num_epochs):

    outputs = model(X_final11)

    loss = criterion(outputs, Y_final11)
    _, out2 = torch.max(outputs.data, 1)
    yy_train = Y_final11.cpu().numpy()
    out3 = out2.cpu().numpy()
    accuracy1 = accuracy_score(out3, yy_train)
    losses.append(loss)
    accuracys.append(accuracy1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, accuracy: {accuracy1.item():.4f}')


X_mixed_4 = mixup_data(X_final1, alpha=1.0)
X_mixed_5 = mixup_data(X_final1, alpha=1.0)
X_mixed_6 = mixup_data(X_final1, alpha=1.0)
X_mixed2 = np.concatenate((X_mixed1, X_mixed_4, X_mixed_5, X_mixed_6), axis=0)
X_mixed2 = torch.tensor(X_mixed2).float().to(device)

with torch.no_grad():

    outputs = model(X_mixed2)
    _, predicted = torch.max(outputs.data, 1)

    # print(f'outputs:{predicted},y_test:{Y_very}')

Y_mixed2 = predicted.cpu().numpy()
X_mixed2 = X_mixed2.cpu().numpy()

X_final2 = np.concatenate((X_train, X_mixed2), axis=0)
Y_final2 = np.concatenate((Y_train, Y_mixed2))

X_final22 = torch.tensor(X_final2).float().to(device)
Y_final22 = torch.tensor(Y_final2).long().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.000025)
for epoch in range(num_epochs):

    outputs = model(X_final22)

    loss = criterion(outputs, Y_final22)
    _, out2 = torch.max(outputs.data, 1)
    yy_train = Y_final22.cpu().numpy()
    out3 = out2.cpu().numpy()
    accuracy1 = accuracy_score(out3, yy_train)
    losses.append(loss)
    accuracys.append(accuracy1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, accuracy: {accuracy1.item():.4f}')



epochs = np.arange(1, 3 * num_epochs+1)
losses2 = torch.tensor(losses, device='cpu')
accuracys2 = torch.tensor(accuracys, device='cpu')
# 绘制loss曲线图
plt.figure()
plt.plot(epochs, losses2.numpy(), label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
# 绘制accurcay曲线图
plt.figure()
plt.plot(epochs, accuracys2.numpy(), label='Loss')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.title('Training accuracy')
plt.legend()
plt.show()
losses3 = pd.DataFrame(losses2.numpy())
accuracys3 = pd.DataFrame(accuracys2.numpy())
# writer1 = pd.ExcelWriter('loss.xlsx')
losses3.to_excel('MSSL-RealFormer_model4_loss5.xlsx')
accuracys3.to_excel('MSSL-RealFormer_model4_accuracy5.xlsx')

print("Test model")

X_very = torch.tensor(X_very).float().to(device)
Y_very = torch.tensor(Y_very).long().to(device)

with torch.no_grad():

    outputs = model(X_very)
    _, predicted = torch.max(outputs.data, 1)

    print(f'outputs:{predicted},y_test:{Y_very}')

YYY = Y_very.cpu().numpy()
predictedd = predicted.cpu().numpy()

result = confusion_matrix(YYY, predictedd)
print("Confusion Matrix:")
print(result)
result1 = classification_report(YYY, predictedd)
print("Classification Report:", )
print(result1)
result2 = accuracy_score(YYY, predictedd)
print("Accuracy:", result2)

XX = torch.tensor(X).float().to(device)
YY = torch.tensor(Y).long().to(device)

with torch.no_grad():

    outputs = model(XX)
    _, predicted = torch.max(outputs.data, 1)

    print(f'outputs:{predicted},y_test:{YY}')

YYY2 = YY.cpu().numpy()
predictedd2 = predicted.cpu().numpy()

result = confusion_matrix(YYY2, predictedd2)
print("Confusion Matrix:")
print(result)
result1 = classification_report(YYY2, predictedd2)
print("Classification Report:", )
print(result1)
result2 = accuracy_score(YYY2, predictedd2)
print("Accuracy:", result2)



torch.save(model.state_dict(), "model1_params_1.pkl")
