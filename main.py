import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import flask
import deeplake
from torchvision import datasets, transforms, models

lspTrain = deeplake.load("hub://activeloop/lsp-train")
lspTest = deeplake.load("hub://activeloop/lsp-test")

print(lspTest.summary())

tform = transforms.Compose([
    transforms.Resize(size=[130, 130]),
    transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
])

dataloader = lspTrain.pytorch(num_workers=0, batch_size=4, shuffle=False, transform = {'images': tform})

for data in dataloader:
    print(data)
    break