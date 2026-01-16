#!/usr/bin/env python3
"""
Train NMJ classifier from annotations.
Uses annotated crops (extracted from HTML base64 images) to train a ResNet18 binary classifier.
"""

import argparse
import json
import re
import base64
import io
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from glob import glob

from segmentation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class NMJDataset(Dataset):
    """Dataset for NMJ classification."""

    def __init__(self, samples, transform=None):
        """
        samples: list of (crop_image, label) tuples
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]

        # Convert to PIL Image if needed
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label


def extract_crops_from_html(html_dirs, slide_prefix=None):
    """Extract all base64 crops from HTML pages.

    Args:
        html_dirs: Single directory path or list of directory paths
        slide_prefix: Prefix for converting short IDs to full IDs (auto-detected if None)

    Returns dict mapping sample_id -> PIL Image
    """
    if isinstance(html_dirs, str):
        html_dirs = [html_dirs]

    # Auto-detect slide prefix from directory names if not provided
    if slide_prefix is None:
        for html_dir in html_dirs:
            # Look for slide name pattern in path (e.g., "20251109_PMCA1...")
            parts = Path(html_dir).parts
            for part in parts:
                if part.startswith("20") and "_" in part and len(part) >= 20:
                    slide_prefix = part + "_tile_"
                    logger.info(f"Auto-detected slide prefix: {slide_prefix}")
                    break
            if slide_prefix:
                break
        if slide_prefix is None:
            slide_prefix = ""
            logger.warning("Could not auto-detect slide prefix - short IDs may not match annotations")

    crops = {}
    html_files = []

    for html_dir in html_dirs:
        # Try multiple naming patterns
        patterns = [
            str(Path(html_dir) / "nmj_page*.html"),
            str(Path(html_dir) / "nmj_results_page_*.html"),
        ]
        for pattern in patterns:
            html_files.extend(glob(pattern))

    html_files = sorted(set(html_files))  # Remove duplicates

    logger.info(f"Found {len(html_files)} HTML pages")

    # Regex to extract card id and base64 image
    # Handles both id="..." and data-id="..." formats
    card_pattern = re.compile(
        r'<div class="card"[^>]*(?:id|data-id)="([^"]+)"[^>]*>.*?'
        r'<img src="data:image/png;base64,([^"]+)"',
        re.DOTALL
    )

    for html_file in tqdm(html_files, desc="Parsing HTML pages"):
        with open(html_file, 'r') as f:
            content = f.read()

        for match in card_pattern.finditer(content):
            sample_id = match.group(1)
            b64_data = match.group(2)

            try:
                # Decode base64 to image
                img_bytes = base64.b64decode(b64_data)
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                crops[sample_id] = img

                # Also store with full ID if this is a short ID (for annotation lookup)
                if not sample_id.startswith("20") and "_nmj_" in sample_id:
                    full_id = slide_prefix + sample_id
                    crops[full_id] = img
            except Exception as e:
                logger.error(f"Error decoding {sample_id}: {e}")

    return crops


def train_model(train_loader, val_loader, num_epochs=20, device='cuda'):
    """Train ResNet18 classifier."""

    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Modify final layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info("-" * 30)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            total += inputs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        logger.info(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += (preds == labels).sum().item()
                val_total += inputs.size(0)

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_corrects / val_total
        logger.info(f"Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}")

        # Save best model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_state = model.state_dict().copy()
            logger.info(f"New best model! Acc: {best_acc:.4f}")

        scheduler.step()

    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_acc


def main():
    parser = argparse.ArgumentParser(description='Train NMJ Classifier')
    parser.add_argument('--annotations', type=str,
                        default='/home/dude/nmj_test_output/nmj_annotations.json',
                        help='Path to annotations JSON')
    parser.add_argument('--html-dir', type=str, nargs='+',
                        default=['/home/dude/nmj_output/html',
                                 '/home/dude/nmj_output/20251109_PMCA1_647_nuc488-EDFvar-stitch/inference/html'],
                        help='Directory(s) containing HTML pages with base64 crops')
    parser.add_argument('--output-dir', type=str,
                        default='/home/dude/nmj_output',
                        help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--slide-prefix', type=str, default=None,
                        help='Slide name prefix for ID matching (auto-detected if not provided)')
    args = parser.parse_args()

    # Initialize logging
    setup_logging()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load annotations
    logger.info("\nLoading annotations...")
    with open(args.annotations) as f:
        annotations = json.load(f)

    positive_ids = set(annotations.get('positive', []))
    negative_ids = set(annotations.get('negative', []))

    logger.info(f"Positive annotations: {len(positive_ids)}")
    logger.info(f"Negative annotations: {len(negative_ids)}")

    # Extract crops from HTML
    logger.info("\nExtracting crops from HTML pages...")
    all_crops = extract_crops_from_html(args.html_dir, slide_prefix=args.slide_prefix)
    logger.info(f"Total crops extracted: {len(all_crops)}")

    # Build training samples
    samples = []
    missing_ids = []

    for sample_id in positive_ids:
        if sample_id in all_crops:
            samples.append((all_crops[sample_id], 1))  # 1 = positive
        else:
            missing_ids.append(('positive', sample_id))

    for sample_id in negative_ids:
        if sample_id in all_crops:
            samples.append((all_crops[sample_id], 0))  # 0 = negative
        else:
            missing_ids.append(('negative', sample_id))

    logger.info(f"\nTotal samples: {len(samples)}")
    logger.info(f"Positive: {len([s for s in samples if s[1] == 1])}")
    logger.info(f"Negative: {len([s for s in samples if s[1] == 0])}")
    if missing_ids:
        logger.warning(f"Missing crops: {len(missing_ids)}")
        # Log first 10 missing IDs for debugging
        for label, sample_id in missing_ids[:10]:
            logger.debug(f"  Missing {label}: {sample_id}")
        if len(missing_ids) > 10:
            logger.debug(f"  ... and {len(missing_ids) - 10} more")

    if len(samples) < 50:
        logger.error("ERROR: Not enough samples for training!")
        return

    # Split into train/val
    train_samples, val_samples = train_test_split(
        samples, test_size=0.2, random_state=42,
        stratify=[s[1] for s in samples]
    )

    logger.info(f"\nTrain samples: {len(train_samples)}")
    logger.info(f"Val samples: {len(val_samples)}")

    # Create transforms - resize to 224x224 for ResNet
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets and loaders
    train_dataset = NMJDataset(train_samples, transform=train_transform)
    val_dataset = NMJDataset(val_samples, transform=val_transform)

    # Create weighted sampler for class balancing
    train_labels = [s[1] for s in train_samples]
    class_counts = [train_labels.count(0), train_labels.count(1)]

    # Issue #3: Check for empty classes before computing weights
    if class_counts[0] == 0 or class_counts[1] == 0:
        logger.error(f"ERROR: One or more classes have no samples! Negative: {class_counts[0]}, Positive: {class_counts[1]}")
        logger.error("Training cannot proceed with imbalanced data. Please provide samples for both classes.")
        return

    class_weights = [1.0 / c for c in class_counts]
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    logger.info(f"\nClass balancing enabled:")
    logger.info(f"  Negative samples: {class_counts[0]} (weight: {class_weights[0]:.4f})")
    logger.info(f"  Positive samples: {class_counts[1]} (weight: {class_weights[1]:.4f})")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Train model
    logger.info("\n" + "="*50)
    logger.info("TRAINING NMJ CLASSIFIER")
    logger.info("="*50)

    model, best_acc = train_model(train_loader, val_loader, num_epochs=args.epochs, device=device)

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "nmj_classifier.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_acc': best_acc,
        'num_positive': len([s for s in samples if s[1] == 1]),
        'num_negative': len([s for s in samples if s[1] == 0]),
    }, model_path)

    logger.info(f"\nModel saved to: {model_path}")
    logger.info(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    main()
