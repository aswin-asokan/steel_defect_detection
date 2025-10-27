import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import SamModel, SamProcessor
from peft import get_peft_model, LoraConfig, TaskType
# Dataset Class
class SteelDefectDataset(Dataset):
    def __init__(self, root_dir, processor):
        self.image_dir = os.path.join(root_dir, "source_images")
        self.mask_dir = os.path.join(root_dir, "ground_truth")
        self.processor = processor

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.images = [f for f in os.listdir(self.image_dir) if f.lower().endswith(valid_exts)]
        print(f"Found {len(self.images)} images in {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(img_name)[0] + ".png")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for {img_name}")

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # SAM processor (handles resizing / normalization)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # [3,1024,1024]

        # Resize mask to match SAM processor resolution (1024x1024)
        mask = mask.resize((1024, 1024), resample=Image.NEAREST)
        mask = np.array(mask) > 0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1,1024,1024]

        return pixel_values, mask
# IoU Metric
def iou_score_from_logits(logits, true_mask, threshold=0.5):
    """
    logits: torch.Tensor with shape [B, H, W] or [H, W]
    true_mask: torch.Tensor with same H,W (float 0/1)
    """
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()
    intersection = (pred * true_mask).sum()
    union = (pred + true_mask - pred * true_mask).sum()
    if union == 0:
        return 1.0
    return (intersection / union).item()
# Configuration
DATASET_DIR = "dataset"
SAVE_DIR = "sam_steel_lora"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training on:", device)

BATCH_SIZE = 2
EPOCHS = 6
LR = 1e-4
VAL_SPLIT = 0.1
POS_WEIGHT = 5.0  # handles class imbalance
# Load Model & Processor
print("🔹 Loading SAM model and processor...")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
# Apply LoRA (Low-Rank Adaptation)
print("🔹 Applying LoRA adaptation...")
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]
)
sam_model = get_peft_model(sam_model, peft_config)
# Freeze base parameters (train only LoRA adapters & heads)
for name, p in sam_model.named_parameters():
    # peft adapter parameters usually contain "lora" in their name; keep them trainable
    if "lora" in name.lower() or "adapter" in name.lower():
        p.requires_grad = True
    else:
        # keep any other parameter that should be trained if you want; default freeze to save memory
        p.requires_grad = False

trainable_params = [p for p in sam_model.parameters() if p.requires_grad]
print(f"Number of trainable parameters (LoRA/adapters): {sum(p.numel() for p in trainable_params):,}")

sam_model.to(device)
# Main Training Logic
if __name__ == "__main__":
    # Dataset Split
    full_dataset = SteelDefectDataset(DATASET_DIR, processor)
    if len(full_dataset) == 0:
        raise RuntimeError(f"No images found in {full_dataset.image_dir}. Please check your dataset path.")

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    # Optimizer & Scheduler
    # Only pass parameters that require gradients to optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, sam_model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS))
    scaler = torch.amp.GradScaler()  # no device argument
    # Training Loop
    print("Starting training...")

    loss_history = []
    val_iou_history = []

    try:
        for epoch in range(EPOCHS):
            sam_model.train()
            total_loss = 0.0
            batch_count = 0
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

            for imgs, masks in progress:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                optimizer.zero_grad()

                with torch.amp.autocast(device_type="cuda" if device.startswith("cuda") else "cpu"):
                    # forward
                    outputs = sam_model(pixel_values=imgs, multimask_output=False)

                    # robust handling for pred_masks shape:
                    # outputs.pred_masks expected shape: [B, 1, H, W] or [B, H, W]
                    if hasattr(outputs, "pred_masks"):
                        pred_masks = outputs.pred_masks
                        # unify to [B, H, W]
                        if pred_masks.ndim == 4 and pred_masks.shape[1] == 1:
                            pred_masks = pred_masks.squeeze(1)
                        elif pred_masks.ndim == 3:
                            pass
                        else:
                            # fallback: try to reshape if possible
                            pred_masks = pred_masks.reshape(imgs.shape[0], pred_masks.shape[-2], pred_masks.shape[-1])
                    else:
                        raise RuntimeError("Model output doesn't contain 'pred_masks' attribute.")

                    # resize ground truth to prediction resolution
                    masks_resized = F.interpolate(masks, size=pred_masks.shape[-2:], mode="nearest")

                    # weighted BCE to handle class imbalance
                    pos_weight_tensor = torch.tensor([POS_WEIGHT], device=device)
                    loss = F.binary_cross_entropy_with_logits(pred_masks, masks_resized, pos_weight=pos_weight_tensor)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                batch_count += 1
                progress.set_postfix({"loss": f"{loss.item():.4f}"})

            # Average training loss
            avg_loss = total_loss / max(1, batch_count)
            loss_history.append(avg_loss)
            scheduler.step()
            # Validation Step
            sam_model.eval()
            iou_scores = []
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs = imgs.to(device)
                    masks = masks.to(device)
                    with torch.amp.autocast(device_type="cuda" if device.startswith("cuda") else "cpu"):
                        outputs = sam_model(pixel_values=imgs, multimask_output=False)
                        if hasattr(outputs, "pred_masks"):
                            pred_masks = outputs.pred_masks
                            if pred_masks.ndim == 4 and pred_masks.shape[1] == 1:
                                pred_masks = pred_masks.squeeze(1)
                        else:
                            raise RuntimeError("Model output doesn't contain 'pred_masks' attribute.")

                        masks_resized = F.interpolate(masks, size=pred_masks.shape[-2:], mode="nearest")
                        # compute IoU per-sample (pred_masks shape: [1, H, W])
                        iou = iou_score_from_logits(pred_masks[0], masks_resized[0], threshold=0.5)
                        iou_scores.append(iou)

            avg_iou = float(np.mean(iou_scores)) if len(iou_scores) > 0 else 0.0
            val_iou_history.append(avg_iou)

            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val IoU: {avg_iou:.4f}")

            # Save checkpoint every 2 epochs
            if (epoch + 1) % 2 == 0 or (epoch + 1) == EPOCHS:
                checkpoint_dir = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                sam_model.save_pretrained(checkpoint_dir)
                processor.save_pretrained(checkpoint_dir)
                print(f"Checkpoint saved at {checkpoint_dir}")

    except KeyboardInterrupt:
        print("⏸Training interrupted by user. Saving last checkpoint...")
        safe_dir = os.path.join(SAVE_DIR, "interrupted_checkpoint")
        os.makedirs(safe_dir, exist_ok=True)
        sam_model.save_pretrained(safe_dir)
        processor.save_pretrained(safe_dir)
        raise
    # Save Final Model
    sam_model.save_pretrained(SAVE_DIR)
    processor.save_pretrained(SAVE_DIR)
    print(f"Final model and processor saved successfully at: {SAVE_DIR}")
    # Plot Training Curves
    plt.figure(figsize=(8,5))
    plt.plot(loss_history, marker='o', label="Training Loss")
    plt.plot(val_iou_history, marker='x', label="Validation IoU")
    plt.title("Training Progress")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "training_progress.png"))
    plt.show()

    print("Training completed successfully!")
