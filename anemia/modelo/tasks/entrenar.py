import torch
from torch.utils.data import DataLoader
import timm
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
from .dataset import ConjuntivaDataset
from .config import EPOCHS, MODELO_NFNET_PATH

import matplotlib
matplotlib.use('Agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def entrenar_nfnet(x_train, y_train, modelo_nombre='nfnet_f0'):
    dataset = ConjuntivaDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    model = timm.create_model(modelo_nombre, pretrained=False, num_classes=2, in_chans=4).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

    torch.save(model.state_dict(), MODELO_NFNET_PATH)
    return model, device

def cargar_modelo_entrenado(modelo_nombre='nfnet_f0'):
    model = timm.create_model(modelo_nombre, pretrained=False, num_classes=2, in_chans=4)
    model.load_state_dict(torch.load(MODELO_NFNET_PATH, map_location=torch.device('cpu')))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device
