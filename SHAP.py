import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import torch
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
import torch.utils.data as Data
from matplotlib import cm
from matplotlib import colors as plt_colors
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from realformer import RealFormerEncoder
import shap
# batch_size = 57
num_epochs = 2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RealformerModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(RealformerModel, self).__init__()

        self.encoder = RealFormerEncoder(d_model=input_size, num_heads=5, expansion_factor=2, dropout=0.5, num_layers=5)

        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.encoder(x)

        x = x.squeeze(1)

        x = self.fc(x)
        return x


def train(model, device, train_X, train_Y, optimizer, epoch):
    model.train()
    # for batch_idx, (data, target) in enumerate(train_loader):
    data = train_X.to(device)
    target = train_Y.to(device)
    output = model(data)
    loss = nn.CrossEntropyLoss(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}'
            )



def test(model, device, test_X, test_Y):
    model.eval()
    test_loss = 0
    correct = 0
    with (torch.no_grad()):
        # for data, target in test_loader:
        data = test_X.to(device)
        target = test_Y.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target).item()  # sum up batch loss
        _, predicted = torch.max(output.data, 1)
        print(f'outputs:{predicted},y_test:{y_test}')


print("# 加载数据集")

def dataset(path):
    data = pd.read_excel(path)
    label = data.iloc[:, -1]
    data = data.iloc[:, 1:-1]
    label = torch.from_numpy(label.values.reshape(len(label))).type(torch.LongTensor)
    data = torch.from_numpy(data.values.astype(float).reshape(len(data), 1, len(data.loc[0]))).type(torch.FloatTensor)
    dataset = Data.TensorDataset(data, label)
    return dataset

path = "Model3_absorption or partition.xlsx"
source_loader = Data.DataLoader(dataset=dataset(path), batch_size=63, shuffle=True, drop_last=True)

batch = next(iter(source_loader))
images, _ = batch
model = RealformerModel(input_size=40, num_classes=3)
model.load_state_dict(torch.load('model3_params_1.pkl'))

background = images[0:63]
test_images = images[0:63]
background = torch.squeeze(background)
test_images = torch.squeeze(test_images)
list1 = ['MaxEStateIndex', 'qed', 'SPS', 'MolWt', 'MinPartialCharge', 'FpDensityMorgan1', 'BCUT2D_MWLOW', 'BCUT2D_CHGLO', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA9', 'SMR_VSA10', 'SMR_VSA4', 'SMR_VSA5', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA8', 'EState_VSA3', 'EState_VSA5', 'EState_VSA6', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState8', 'NHOHCount', 'NumAliphaticHeterocycles', 'NumAromaticHeterocycles', 'fr_Al_OH', 'fr_Imine', 'fr_NH1', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_aniline', 'fr_halogen', 'fr_methoxy']
list2 = ['Class 0', 'Class 1', 'Class 2']
e = shap.DeepExplainer(model, background)
shap_values = e.shap_values(test_images)
# df = pd.DataFrame(shap_values[2])
# df.to_excel("model3values2.xlsx")
shap.summary_plot(shap_values, test_images, feature_names=list1, class_names=list2, color=pl.get_cmap('tab20'), show=False, max_display=40)
plt.xlabel("mean|SHAP_Value|", fontsize=14, fontweight='semibold', fontdict={'family': 'Times New Roman'})
plt.xticks(fontsize=14, fontname='Times New Roman', fontweight='semibold')
plt.yticks(fontsize=14, fontname='Times New Roman', fontweight='semibold')
plt.legend(prop={'family': 'Times New Roman', 'size': 14, 'weight': 'semibold'})
plt.savefig('model3-summary40.jpg', dpi=1200)
plt.show()



a = shap.summary_plot(shap_values[0], test_images, feature_names=list1, class_names=list2, show=False, max_display=40)
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 14
# cb = a.colorbar(a.collections[0])
# cb.set_label()
plt.xlabel("SHAP_Value", fontsize=14, fontweight='semibold', fontdict={'family': 'Times New Roman'})
plt.xticks(fontsize=14, fontname='Times New Roman', fontweight='semibold')
plt.yticks(fontsize=14, fontname='Times New Roman', fontweight='semibold')
cbar = plt.gcf().axes[-1]
cbar.set_ylabel('Feature Value', fontsize=14, fontweight='semibold', fontdict={'family': 'Times New Roman'})
cbar.yaxis.set_tick_params(labelsize=14)
# label()
# cbar.yaxis.set_tick_params(labelsize=14, fontname='Times New Roman', fontweight='semibold')
plt.savefig('model3-class0-all.jpg', dpi=1200)
plt.show()



