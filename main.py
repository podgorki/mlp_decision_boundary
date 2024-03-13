import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import ToyDataset
from model import MLP, LitMLP

dataset = ToyDataset()
dataloader = DataLoader(dataset, pin_memory=True, batch_size=512, shuffle=True)
# train model
trainer = pl.Trainer(gpus=1, max_epochs=100)
#
lit_model = LitMLP(MLP())
trainer.fit(model=lit_model, train_dataloader=dataloader)

colour_dict = {0: 'b', 1: 'r', 2: 'g', 3: 'y'}
fig, ax = plt.subplots(1)
ax.scatter(np.array(dataset.data)[:, 0], np.array(dataset.data)[:, 1], c=[colour_dict[l] for l in dataset.labels])
ax.set_title('Ground truths')
ax.set_xlim(-.5, 1.5)
ax.set_ylim(-.5, 1.5)
ax.axis('off')
plt.savefig('images/data.jpg')

fig, ax = plt.subplots(1)
ax.set_title('Predictions and decision boundary')
h = 0.001  # step size of mesh - set high for a nice boundary
x_min, y_min = dataset.data_min[0], dataset.data_min[1]
x_max, y_max = dataset.data_max[0], dataset.data_max[1]
xx, yy = np.meshgrid(np.arange(x_min - 0.5, x_max + 0.5, h),
                     np.arange(y_min - 0.5, y_max + 0.5, h))
y_hat = lit_model.model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float)).argmax(1)
y_hat = y_hat.reshape(xx.shape)
ax.contourf(xx, yy, y_hat, cmap=plt.cm.binary_r)
ax.axis('off')
for data, label in dataloader:
    pred = lit_model.model(data).argmax(1)
    data.cpu().numpy()
    colours = [colour_dict[l] for l in list(pred.cpu().int().numpy())]
    ax.scatter(data[:, 0], data[:, 1], c=colours)

ax.set_xlim(-.5, 1.5)
ax.set_ylim(-.5, 1.5)

plt.savefig('images/data_with_boundary.jpg')
ax.scatter(np.array(dataset.data_out)[:, 0],
           np.array(dataset.data_out)[:, 1],
           c=[colour_dict[l] for l in dataset.labels_out])
plt.savefig('images/data_with_boundary_and_outlier.jpg')
