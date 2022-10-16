import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import flask
import iPython
import deeplake

lspTrain = deeplake.load("hub://activeloop/lsp-train")
lspTest = deeplake.load("hub://activeloop/lsp-test")

lspTrain.visualize()

dataloader = lspTrain.pytorch(num_workers=0, batch_size=4, shuffle=False)

