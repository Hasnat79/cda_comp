

import argparse


import os
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
# from torchvision.transforms import Ran

from data import train_dataset, val_dataset, test_dataset,train_df
from trainer import plot_training_curves, train,evaluate_model,infer
from model import SimpleCNN, FeedBunkClassifier,ResNet50Classification,MiniDenseNet

def parse_args():
    parser = argparse.ArgumentParser(description="FeedBunk Classifier - Train/Eval/Infer")
    parser.add_argument("mode", type=str, choices=["train", "evaluate", "infer"],
                        help="Mode: train | evaluate | infer")

    # Common arguments
    parser.add_argument("--output-dir", type=str, default="/data/hma18/CDA_hackathon/cda_comp/outputs",
                        help="Base output directory")
    parser.add_argument("--model-name", type=str, default="baseline_simple_cnn",
                        help="Name of the model architecture / experiment")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for DataLoader")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (only for train mode)")

    # For evaluate and infer
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to saved model checkpoint (.pth) - required for evaluate & infer")

    # For infer only
    parser.add_argument("--image", type=str, default=None,
                        help="Path to single image for inference (required in infer mode)")

    
    parser.add_argument("--model-type", type=str, default="feedbunk",
                        choices=["simple_cnn", "feedbunk"],
                        help="Which model class to instantiate")

    return parser.parse_args()
# if __name__ == "__main__":
#     # Save preview images for sanity check
#     num_classes = len(train_dataset.unique_scores)
#     # Handle class imbalance for train with sampler
#     train_labels = [train_dataset.score_to_label[row['score']] for _, row in train_df.iterrows()]
#     class_counts = np.bincount(train_labels)
#     class_weights = 1. / class_counts
#     samples_weights = [class_weights[label] for label in train_labels]
#     sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

#     # DataLoaders
#     batch_size = 32
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     # Initialize model
#     model_name = "baseline_simple_cnn"
#     model = SimpleCNN(num_classes)
#     model = FeedBunkClassifier(num_classes)
#     train_losses, val_losses, train_accs, val_accs,model_save_path = train(model, 
#              model_name,
#            train_loader,
#              val_loader, 
#              class_weights, 
#              output_dir="/data/hma18/CDA_hackathon/cda_comp/outputs", epochs=50)



#     # plot training curves
#     # plot_training_curves(train_losses, val_losses, train_accs, val_accs,model_name, output_dir="/data/hma18/CDA_hackathon/cda_comp/outputs")
#     model_save_path = "/data/hma18/CDA_hackathon/cda_comp/outputs/models/baseline_simple_cnn_best.pth"
#     # evaluate_model(baseline_model,
#     #             model_name,
#     #             test_loader, 
#     #             model_save_path, output_dir="/data/hma18/CDA_hackathon/cda_comp/outputs")

#     example_img = '/data/hma18/CDA_hackathon/FBSI/dataset/Score 1/score-1_0.jpg'  # Adjust
#     infer (model,model_save_path, example_img)

def main():
    parser = argparse.ArgumentParser(description="FeedBunk Classifier - Train/Eval/Infer")
    args = parse_args()

    # Create output directories if needed
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Decide which model class to use
    if args.model_type == "simple_cnn":
        model_class = SimpleCNN
    else:
        model_class = FeedBunkClassifier

    # -------------------------------------------------------------------------
    if args.mode == "train":

        num_classes = len(train_dataset.unique_scores)

        # Class weights & sampler (your existing imbalance handling)
        train_labels = [train_dataset.score_to_label[row['score']] for _, row in train_df.iterrows()]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        samples_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
        val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
        # test_loader not needed during training, but you can add if desired

        model = model_class(num_classes)

        # You might want to move class_weights to tensor later in train()
        train_losses, val_losses, train_accs, val_accs, model_save_path = train(
            model=model,
            model_name=args.model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,           # or torch.tensor(class_weights)
            output_dir=args.output_dir,
            epochs=args.epochs
        )

        # Optional: plot
        # plot_training_curves(train_losses, val_losses, train_accs, val_accs,
        #                      args.model_name, args.output_dir)

        print(f"Training finished. Best model saved to: {model_save_path}")

    # -------------------------------------------------------------------------
    elif args.mode == "evaluate":

        if not args.model_path:
            parser.error("--model-path is required in evaluate mode")

        if not os.path.isfile(args.model_path):
            raise FileNotFoundError(f"Model not found: {args.model_path}")

        num_classes = len(test_dataset.unique_scores)  # or hardcode / get from somewhere
        model = model_class(num_classes)

        evaluate_model(
            model=model,
            model_name=args.model_name,
            test_loader=DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False),
            model_save_path=args.model_path,
            output_dir=args.output_dir
        )

    # -------------------------------------------------------------------------
    elif args.mode == "infer":

        if not args.model_path:
            parser.error("--model-path is required in infer mode")
        if not args.image:
            parser.error("--image is required in infer mode")
        if not os.path.isfile(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")

        num_classes = len(train_dataset.unique_scores)  # assuming same as training
        model = model_class(num_classes)

        infer(
            model=model,
            model_path=args.model_path,
            img_path=args.image
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()