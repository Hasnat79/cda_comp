
# from cda_comp.src.baseline_simple_cnn import train_and_evaluate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from torchvision import transforms, models
import timm  # For ViT
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Data transforms
from torchvision.transforms import GaussianBlur, ColorJitter, RandomErasing
from sklearn.metrics import confusion_matrix
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import RandAugment, RandomAffine
import os
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from data import train_dataset, val_dataset, test_dataset
def train(model,
          model_name,
          train_loader,
          val_loader,
          class_weights,
          output_dir,
          epochs=10,
          ):




    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)  # start higher for scratch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights.astype(np.float32)).to(device),
        label_smoothing=0.1
    )
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights.astype(np.float32)).to(device))
    # optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)  # Replace ReduceLROnPlateau

    # Create output directories
    plots_dir = os.path.join(output_dir, "plots")
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"Training {model_name}...")
    best_val_loss = float('inf')
    train_losses, val_losses, train_accs, val_accs = [], [], [], []



    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_acc += (preds == labels).sum().item() / labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = running_acc / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        # Val
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                val_acc += (preds == labels).sum().item() / labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)
        
        scheduler.step(avg_val_loss)
        # scheduler.step()  # CosineAnnealingLR doesn't use val_loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(models_dir, f'{model_name.lower().replace(" ", "_")}_best.pth')
            torch.save(model.state_dict(), model_save_path)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Acc: {avg_val_acc:.4f}")
    return train_losses, val_losses, train_accs, val_accs,model_save_path


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, model_name, output_dir):
    plots_dir = os.path.join(output_dir, "plots")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{model_name} Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title(f'{model_name} Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    curves_path = os.path.join(plots_dir, f"{model_name.lower().replace(' ', '_')}_training_curves.png")
    plt.savefig(curves_path)
    plt.close()

def create_degraded_loader(dataset_subset, degradation_type='blur',output_dir="/data/hma18/CDA_hackathon/cda_comp/outputs"):
    orig_transform = dataset_subset.transform
    if degradation_type == 'blur':
        deg_transform = transforms.Compose([
            GaussianBlur(kernel_size=15, sigma=10.0),   # very strong fixed blur
            *orig_transform.transforms
        ])
    elif degradation_type == 'noise':
        deg_transform = transforms.Compose(orig_transform.transforms + [lambda x: x + torch.randn_like(x) * 0.2])
    elif degradation_type == 'low_light':
        deg_transform = transforms.Compose([
            transforms.ColorJitter(
                brightness=(0.2, 0.3),   # ← key: very low values → strong darkening
                # contrast=(0.3, 0.7),       # flatten contrast
                saturation=(0.5, 0.9),     # slight desaturation
                hue=0.0
            )])
    dataset_subset.transform = deg_transform
    dataset_subset.save_previews(count=5, save_dir=os.path.join(output_dir, f"{degradation_type}_previews"))
    loader = DataLoader(dataset_subset, batch_size=32, shuffle=False)
    dataset_subset.transform = orig_transform  # Reset
    return loader

def evaluate_on_loader(model, loader, desc):
    
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n{desc} Accuracy: {acc:.4f}")
    # Get unique labels present in this evaluation set
    unique_labels = sorted(np.unique(all_labels))
    target_names = [str(train_dataset.label_to_score[label]) for label in unique_labels]
    print(classification_report(all_labels, all_preds, labels=unique_labels, target_names=target_names))
    return all_preds, all_labels

# def evaluate_model(model, model_name,test_loader, model_save_path, output_dir):
#     models_dir = os.path.join(output_dir, "models")
#     plots_dir = os.path.join(output_dir, "plots")
#     model.load_state_dict(torch.load(model_save_path))

#     #Test evaluation
#     all_preds, all_labels = evaluate_on_loader(model, test_loader, "Standard Test")

#     # confusion matrix plot
#     cm = confusion_matrix(all_labels, all_preds)
#     plt.figure(figsize=(8, 8))
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title(f'{model_name} - Confusion Matrix')
#     plt.colorbar()
#     tick_marks = np.arange(len(train_dataset.unique_scores))
#     plt.xticks(tick_marks, [str(s) for s in train_dataset.unique_scores], rotation=45)
#     plt.yticks(tick_marks, [str(s) for s in train_dataset.unique_scores])
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
#     plt.savefig(os.path.join(plots_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'))
#     plt.close()
def evaluate_model(model, model_name, test_loader, model_save_path, output_dir):
    models_dir = os.path.join(output_dir, "models")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load weights
    checkpoint = torch.load(model_save_path, map_location=device)  # safe: handles CPU/GPU saved models
    
    # Load state dict
    model.load_state_dict(checkpoint)
    
    # CRITICAL: Move model to the correct device (GPU if available)
    model = model.to(device)
    
    # Set to evaluation mode
    model.eval()

    # Test evaluation
    all_preds, all_labels = evaluate_on_loader(model, test_loader, "Standard Test")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalize by row (percentage of true class predicted as each class)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_percent = np.nan_to_num(cm_percent)  # 0% if no samples

    # Plot with counts + percentages
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} - Confusion Matrix\n(Counts | Row-Normalized %)')
    plt.colorbar()

    class_names = [str(s) for s in train_dataset.unique_scores]
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Annotate: count \n XX%
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0 or cm_percent[i, j] > 0:
                text = f"{cm[i, j]}\n{int(round(cm_percent[i, j]))}%"
                color = "white" if cm[i, j] > thresh else "black"
                plt.text(j, i, text,
                         horizontalalignment="center",
                         verticalalignment="center",
                         color=color,
                         fontsize=9,
                         fontweight="bold")

    plt.tight_layout()
    
    cm_path = os.path.join(plots_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to: {cm_path}")



def infer(model,model_path,img_path):
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)  # safe: handles CPU/GPU saved models
    
    # Load state dict
    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    img = Image.open(img_path).convert('RGB')
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    input_tensor = test_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs).item()
        conf = torch.max(probs).item()
    print(f"Inference - Predicted: {pred}, Confidence: {conf:.2f}")





    # evaluate_model(baseline_model,
    #             model_name,
    #             test_loader, 
    #             model_save_path, output_dir="/data/hma18/CDA_hackathon/cda_comp/outputs")

