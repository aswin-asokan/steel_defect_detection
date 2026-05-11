# ==========================================================
# FINAL MOBILESAM TRAINING - DEFECT DETECTION OPTIMIZED
# ==========================================================

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
from tqdm import tqdm
from mobile_sam import sam_model_registry
import cv2
import albumentations as A
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURATION - EDIT THESE TO MATCH YOUR SETUP
# ==========================================================

# Dataset paths
DATASET_DIR = "dataset"
IMAGE_SUBDIR = "source_images"
MASK_SUBDIR = "ground_truth"
SAVE_DIR = "mobilesam_defect_optimized"
CHECKPOINT = "mobile_sam.pt"

# Training parameters
BATCH_SIZE = 2              # Reduce to 1 if GPU memory error
EPOCHS = 50                 # Increase to 70-100 for small datasets (<30 images)
LR = 1e-4                   # Learning rate
VAL_SPLIT = 0.15            # 15% for validation
IMG_SIZE = 1024             # Do not change (MobileSAM requirement)

# Mask processing
MASK_BINARIZATION_THRESHOLD = 128  # Threshold to convert mask to binary (0 or 255)

# Loss weights - BALANCED for best results
LOSS_WEIGHTS = {
    'dice': 0.6,        # Primary metric for overlap
    'focal': 0.3,       # Handles class imbalance (background vs defect)
    'boundary': 0.1     # Fine-tunes edges
}

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "validation_samples"), exist_ok=True)

print("="*70)
print("MOBILESAM DEFECT DETECTION TRAINING")
print("="*70)
print(f"Device: {device}")
print(f"Dataset: {DATASET_DIR}/{IMAGE_SUBDIR}")
print(f"Output: {SAVE_DIR}")
print(f"Batch Size: {BATCH_SIZE} | Epochs: {EPOCHS} | LR: {LR}")

# ==========================================================
# LOSS FUNCTIONS
# ==========================================================

def dice_loss(pred, target, smooth=1.0):
    """
    Dice loss - measures overlap between prediction and ground truth
    Lower is better (0 = perfect overlap)
    """
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1,2))
    union = pred.sum(dim=(1,2)) + target.sum(dim=(1,2))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal loss - handles class imbalance
    Focuses learning on hard examples
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pred_prob = torch.sigmoid(pred)
    p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
    focal_weight = (1 - p_t) ** gamma
    focal = alpha * focal_weight * bce
    return focal.mean()


def boundary_loss(pred, target):
    """
    Boundary loss - sharpens edges by matching gradients
    """
    pred_sigmoid = torch.sigmoid(pred)
    
    # Sobel-like gradient kernels
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    
    # Compute gradients
    pred_grad_x = F.conv2d(pred_sigmoid.unsqueeze(1), kernel_x, padding=1)
    pred_grad_y = F.conv2d(pred_sigmoid.unsqueeze(1), kernel_y, padding=1)
    target_grad_x = F.conv2d(target.unsqueeze(1), kernel_x, padding=1)
    target_grad_y = F.conv2d(target.unsqueeze(1), kernel_y, padding=1)
    
    # L1 loss on gradients
    loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
    return loss


def combined_loss(pred, target, weights=LOSS_WEIGHTS):
    """
    Combined loss function
    """
    loss = (
        weights['dice'] * dice_loss(pred, target) +
        weights['focal'] * focal_loss(pred, target) +
        weights['boundary'] * boundary_loss(pred, target)
    )
    return loss


# ==========================================================
# DATA AUGMENTATION
# ==========================================================

def get_training_augmentation():
    """
    Data augmentation for training
    Helps model generalize better
    """
    return A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.2, 
            rotate_limit=45, 
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.5
        ),
        
        # Color/intensity augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    ])


# ==========================================================
# DATASET CLASS
# ==========================================================

class DefectDataset(Dataset):
    """
    Dataset loader for defect detection
    Loads images and corresponding masks
    """
    def __init__(self, root_dir, img_subdir, mask_subdir, augmentation=None):
        self.img_dir = os.path.join(root_dir, img_subdir)
        self.mask_dir = os.path.join(root_dir, mask_subdir)
        self.augmentation = augmentation
        
        # Verify directories exist
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"❌ Image folder not found: {self.img_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"❌ Mask folder not found: {self.mask_dir}")
        
        # Get list of images
        self.images = sorted([
            f for f in os.listdir(self.img_dir) 
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))
        ])
        
        if len(self.images) == 0:
            raise ValueError(f"❌ No images found in {self.img_dir}")
        
        print(f"✅ Loaded {len(self.images)} images from {img_subdir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Find corresponding mask
        base_name = os.path.splitext(img_name)[0]
        mask_path = None
        for ext in ['.png', '.bmp', '.jpg', '.jpeg']:
            candidate = os.path.join(self.mask_dir, base_name + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        
        if mask_path is None:
            raise FileNotFoundError(f"❌ No mask found for {img_name}")
        
        # Read image and mask
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize to model input size
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        # Apply augmentation if provided
        if self.augmentation:
            augmented = self.augmentation(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        # Binarize mask (0 or 1)
        mask = (mask > MASK_BINARIZATION_THRESHOLD).astype(np.float32)
        
        # Normalize image (ImageNet statistics)
        img = img.astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        img = (img - mean) / std
        
        # Convert to PyTorch tensors
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return img, mask, img_name


# ==========================================================
# VALIDATION VISUALIZATION
# ==========================================================

def save_validation_samples(sam, val_loader, epoch, save_dir):
    """
    Save visual comparison of predictions during validation
    Helps monitor training progress
    """
    sam.eval()
    samples_to_save = min(4, len(val_loader))
    
    fig, axes = plt.subplots(samples_to_save, 4, figsize=(16, samples_to_save*4))
    if samples_to_save == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for idx, (imgs, masks, names) in enumerate(val_loader):
            if idx >= samples_to_save:
                break
                
            imgs, masks = imgs.to(device), masks.to(device)
            
            # Get prediction
            img_emb = sam.image_encoder(imgs)
            sparse, dense = sam.prompt_encoder(points=None, boxes=None, masks=None)
            low_res, _ = sam.mask_decoder(
                image_embeddings=img_emb,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False
            )
            pred = F.interpolate(low_res, (IMG_SIZE, IMG_SIZE), 
                               mode="bilinear", align_corners=False)
            pred_binary = (torch.sigmoid(pred) > 0.5).float()
            
            # Convert to numpy for visualization
            img_np = imgs[0].cpu().permute(1, 2, 0).numpy()
            # Denormalize
            mean = np.array([123.675, 116.28, 103.53])
            std = np.array([58.395, 57.12, 57.375])
            img_np = (img_np * std + mean).clip(0, 255).astype(np.uint8)
            
            mask_np = masks[0, 0].cpu().numpy()
            pred_np = pred_binary[0, 0].cpu().numpy()
            
            # Create overlay
            overlay = img_np.copy()
            overlay[pred_np > 0.5] = [255, 0, 0]
            blended = cv2.addWeighted(overlay, 0.5, img_np, 0.5, 0)
            
            # Plot
            axes[idx, 0].imshow(img_np)
            axes[idx, 0].set_title(f'Input: {names[0][:20]}', fontsize=10)
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(mask_np, cmap='gray')
            axes[idx, 1].set_title('Ground Truth', fontsize=10)
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(pred_np, cmap='gray')
            axes[idx, 2].set_title('Prediction', fontsize=10)
            axes[idx, 2].axis('off')
            
            axes[idx, 3].imshow(blended)
            axes[idx, 3].set_title('Overlay', fontsize=10)
            axes[idx, 3].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_samples.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"   💾 Saved validation samples: {save_path}")


# ==========================================================
# MAIN TRAINING
# ==========================================================

def main():
    print("\n" + "="*70)
    print("STEP 1: LOADING MODEL")
    print("="*70)
    
    # Load MobileSAM
    sam = sam_model_registry["vit_t"](checkpoint=CHECKPOINT)
    sam.to(device)
    
    # Freeze image encoder (only train decoder)
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = True
    for p in sam.mask_decoder.parameters():
        p.requires_grad = True
    
    trainable = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    total = sum(p.numel() for p in sam.parameters())
    print(f"✅ Model loaded: {total:,} params ({trainable:,} trainable, {100*trainable/total:.1f}%)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(sam.prompt_encoder.parameters()) + list(sam.mask_decoder.parameters()),
        lr=LR,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Mixed precision training
    scaler = torch.amp.GradScaler("cuda")
    
    # ==========================================================
    print("\n" + "="*70)
    print("STEP 2: PREPARING DATA")
    print("="*70)
    
    # Create augmentation pipelines
    train_aug = get_training_augmentation()
    
    # Load full dataset
    full_dataset = DefectDataset(DATASET_DIR, IMAGE_SUBDIR, MASK_SUBDIR, augmentation=None)
    
    # Split into train/val
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    
    train_indices, val_indices = random_split(
        range(len(full_dataset)), 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create separate datasets with different augmentations
    train_dataset = torch.utils.data.Subset(
        DefectDataset(DATASET_DIR, IMAGE_SUBDIR, MASK_SUBDIR, augmentation=train_aug),
        train_indices.indices
    )
    val_dataset = torch.utils.data.Subset(
        DefectDataset(DATASET_DIR, IMAGE_SUBDIR, MASK_SUBDIR, augmentation=None),
        val_indices.indices
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    print(f"✅ Dataset split: {train_size} train | {val_size} validation")
    
    # ==========================================================
    print("\n" + "="*70)
    print("STEP 3: TRAINING")
    print("="*70)
    
    best_iou = 0.0
    best_dice = 0.0
    patience_counter = 0
    max_patience = 15
    
    train_losses = []
    val_ious = []
    val_dices = []
    
    for epoch in range(EPOCHS):
        # ==================== TRAINING ====================
        sam.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, masks, _ in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda"):
                img_emb = sam.image_encoder(imgs)
                batch_loss = 0
                
                for i in range(imgs.size(0)):
                    sparse, dense = sam.prompt_encoder(points=None, boxes=None, masks=None)
                    low_res, _ = sam.mask_decoder(
                        image_embeddings=img_emb[i:i+1],
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse,
                        dense_prompt_embeddings=dense,
                        multimask_output=False
                    )
                    
                    pred = F.interpolate(low_res, (IMG_SIZE, IMG_SIZE),
                                       mode="bilinear", align_corners=False)
                    
                    loss = combined_loss(pred[:, 0], masks[i])
                    batch_loss += loss
                
                batch_loss /= imgs.size(0)
            
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(sam.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += batch_loss.item()
            pbar.set_postfix({'loss': f'{batch_loss.item():.4f}'})
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # ==================== VALIDATION ====================
        sam.eval()
        ious, dices = [], []
        
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                
                img_emb = sam.image_encoder(imgs)
                sparse, dense = sam.prompt_encoder(points=None, boxes=None, masks=None)
                low_res, _ = sam.mask_decoder(
                    image_embeddings=img_emb,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False
                )
                
                pred = F.interpolate(low_res, (IMG_SIZE, IMG_SIZE),
                                   mode="bilinear", align_corners=False)
                pred_binary = (torch.sigmoid(pred) > 0.5).float()
                
                # Calculate IoU
                intersection = (pred_binary * masks).sum()
                union = (pred_binary + masks).clamp(0, 1).sum()
                iou = (intersection / (union + 1e-6)).item()
                ious.append(iou)
                
                # Calculate Dice
                dice = (2 * intersection / (pred_binary.sum() + masks.sum() + 1e-6)).item()
                dices.append(dice)
        
        mean_iou = np.mean(ious)
        mean_dice = np.mean(dices)
        val_ious.append(mean_iou)
        val_dices.append(mean_dice)
        
        # Print epoch results
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val IoU:    {mean_iou:.4f} | Val Dice: {mean_dice:.4f}")
        
        # Learning rate scheduling
        scheduler.step(mean_iou)
        
        # Save best model
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_dice = mean_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': sam.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iou': best_iou,
                'dice': best_dice,
            }, os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"  ✅ BEST MODEL SAVED! (IoU: {best_iou:.4f}, Dice: {best_dice:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{max_patience})")
        
        # Save validation samples every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_validation_samples(
                sam, val_loader, epoch+1, 
                os.path.join(SAVE_DIR, "validation_samples")
            )
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\n⚠️  Early stopping triggered (no improvement for {max_patience} epochs)")
            break
    
    # ==========================================================
    print("\n" + "="*70)
    print("STEP 4: SAVING RESULTS")
    print("="*70)
    
    # Save training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(val_ious, label='Val IoU', marker='o', linewidth=2)
    plt.axhline(y=0.7, color='r', linestyle='--', label='Target (0.7)')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Validation IoU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(val_dices, label='Val Dice', marker='s', linewidth=2)
    plt.axhline(y=0.75, color='r', linestyle='--', label='Target (0.75)')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.title('Validation Dice')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"✅ Saved: {SAVE_DIR}/training_curves.png")
    
    # Save final model
    torch.save(sam.state_dict(), os.path.join(SAVE_DIR, "final_model.pth"))
    print(f"✅ Saved: {SAVE_DIR}/final_model.pth")
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Validation IoU:  {best_iou:.4f}")
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"Models saved in: {SAVE_DIR}/")
    print(f"  - best_model.pth (use this for inference)")
    print(f"  - final_model.pth")
    print(f"  - training_curves.png")
    print(f"  - validation_samples/")
    print("="*70)
    
    # Recommendations
    if best_iou < 0.5:
        print("\n⚠️  WARNING: Low IoU (<0.5)")
        print("Possible issues:")
        print("  1. Masks may not match images correctly")
        print("  2. Need more training data")
        print("  3. Try adjusting MASK_BINARIZATION_THRESHOLD")
    elif best_iou < 0.7:
        print("\n⚠️  IoU below target (0.7)")
        print("Suggestions:")
        print("  1. Train for more epochs")
        print("  2. Check validation_samples/ to see what model learned")
        print("  3. May need more diverse training data")
    else:
        print("\n✅ Great results! Model ready for inference.")
        print(f"Next step: Update test_improved.py to use:")
        print(f'  SAM_WEIGHTS = "{SAVE_DIR}/best_model.pth"')
    
    print("\n")


if __name__ == "__main__":
    main()