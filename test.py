import os
import numpy as np
import pickle
from tqdm import tqdm
import torch
from sklearn import metrics

def test(model, test_loader, device, model_dir):
    model.eval()
    metrics_dict = {}
    outputs = np.zeros(len(test_loader))
    labels = np.zeros(len(test_loader), dtype=np.uint8)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            data, times, label = (
                batch["img_seq"].to(device),
                batch["times"].to(device),
                batch["label"],
            )
            output = model(data, times)
            output = torch.sigmoid(output)[0,1] # score for class 1
            outputs[i] = output.cpu().numpy()
            labels[i] = label.numpy()

    fpr, tpr, _ = metrics.roc_curve(labels, outputs)
    roc_auc = metrics.auc(fpr, tpr)
    metrics_dict["roc_auc"], metrics_dict["fpr"], metrics_dict["tpr"] = roc_auc, fpr, tpr
    print(f"AUC: {roc_auc}")

    metrics_path = os.path.join(model_dir, f"metrics.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics_dict, f)


