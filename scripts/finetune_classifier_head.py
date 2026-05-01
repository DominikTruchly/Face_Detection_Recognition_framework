"""scripts/finetune_classifier_head.py
---------------------------------------------------------------------------
Fine-tunes the classification head of any frozen timm model on a dataset that
follows the standard folder-per-identity layout:

    img_dir/
        person_a/
            img1.jpg
            img2.jpg
        person_b/
            img1.jpg
        ...

The backbone weights are kept frozen; only the final linear projection layer
(head) is replaced and trained from scratch.  Saves the model state-dict and
a class-to-index JSON mapping to --output_dir.

Usage:
    python scripts/finetune_classifier_head.py --img_dir path/to/dataset --output_dir results/
    python scripts/finetune_classifier_head.py --img_dir path/to/dataset --output_dir results/ \\
        --model_id vit_base_patch16_224.augreg_in21k_ft_in1k --epochs 10
"""
import argparse
import json
import os

import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def finetune_classifier_head(image_dir, output_dir, model_id, num_epochs, batch_size, lr):
    """Loads a pre-trained timm model, freezes its backbone, and fine-tunes the
    classification head on the identity classes found in image_dir."""
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "finetuned_model.pth")
    class_map_path = os.path.join(output_dir, "class_to_idx.json")

    print("--- Step 1: Discovering identity classes ---")
    identity_names = sorted([
        d for d in os.listdir(image_dir)
        if os.path.isdir(os.path.join(image_dir, d))
    ])
    if not identity_names:
        print(f"ERROR: No subdirectories found in {image_dir}")
        return
    class_to_idx = {name: i for i, name in enumerate(identity_names)}
    num_classes = len(identity_names)
    print(f"Found {num_classes} unique identities.")

    class IdentityFolderDataset(Dataset):
        def __init__(self, image_dir, identity_names, class_to_idx, transform):
            self.image_paths, self.labels = [], []
            self.transform = transform
            print("Scanning for training images...")
            for identity_name in tqdm(identity_names):
                person_folder = os.path.join(image_dir, identity_name)
                for img_file in os.listdir(person_folder):
                    if os.path.splitext(img_file)[1].lower() in IMAGE_EXTENSIONS:
                        self.image_paths.append(os.path.join(person_folder, img_file))
                        self.labels.append(class_to_idx[identity_name])

        def __len__(self): return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx]).convert("RGB")
            return self.transform(image), self.labels[idx]

    temp_model = timm.create_model(model_id, pretrained=True)
    data_config = resolve_data_config({}, model=temp_model)
    transform = create_transform(**data_config)
    del temp_model
    
    train_dataset = IdentityFolderDataset(image_dir, identity_names, class_to_idx, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset prepared with {len(train_dataset)} images.")


    print("\n--- Step 2: Preparing the Model for Fine-Tuning ---")
    
    model = timm.create_model(model_id, pretrained=True, num_classes=num_classes)

    for name, param in model.named_parameters():
        if not name.startswith('head.'):
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded. Trainable parameters (classifier head): {trainable_params}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"\n--- Step 3: Starting fine-tuning on {device.upper()} for {num_epochs} epochs ---")
    optimizer = AdamW(model.head.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        avg_loss = total_loss / len(train_loader)
        print(f"End of Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")

    print("\n--- Step 4: Saving the Fine-Tuned Model ---")
    torch.save(model.state_dict(), model_save_path)
    with open(class_map_path, 'w') as f:
        json.dump(class_to_idx, f)
    print(f"Model fine-tuned and saved successfully to: {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune the classification head of a frozen timm model on a folder-per-identity dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--img_dir", required=True,
        help="Root directory containing one subdirectory per identity class.")
    parser.add_argument("--output_dir", default="results",
        help="Directory to save the trained model and class-to-index map.")
    parser.add_argument("--model_id", default="vit_base_patch16_224.augreg_in21k_ft_in1k",
        help="timm model identifier.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW.")
    args = parser.parse_args()
    finetune_classifier_head(
        image_dir=args.img_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )