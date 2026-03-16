import torch
from tqdm import tqdm

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for waveforms, labels in tqdm(loader, desc="Training"):
        waveforms, labels = waveforms.to(device), labels.to(device)
        
        optimizer.zero_grad()
        _, logits = model(waveforms)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for waveforms, labels in tqdm(loader, desc="Validating"):
            waveforms, labels = waveforms.to(device), labels.to(device)
            _, logits = model(waveforms)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total