from __future__ import annotations

import torch


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device | str) -> dict[str, float]:
    model.eval()
    device = torch.device(device)
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += float(loss.item())
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))
    accuracy = correct / total if total else 0.0
    loss = loss_sum / total if total else 0.0
    return {"accuracy": accuracy, "loss": loss, "total": total}

