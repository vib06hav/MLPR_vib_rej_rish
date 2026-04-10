# train.py

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from config import (
    MAX_EPOCHS, EARLY_STOP_PATIENCE, LR_PATIENCE,
    SEEDS, CHECKPOINT_DIR, RESULTS_DIR,
    WEIGHT_DECAY, GRAD_CLIP_NORM, LR_SCHEDULER,
)
import config
from models import BaseModel, get_param_groups
from dann import DANNModel, compute_lambda, get_dann_param_groups

# ── Performance ────────────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True

# NOTE: Per-run seeding is handled by set_seed() in main.py before each experiment.
# Do NOT add module-level torch.manual_seed() here — it only runs once at import time.


# ════════════════════════════════════════════════════════════════════════════
# Shared Utilities
# ════════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")
    return device


def save_checkpoint(
    model:      nn.Module,
    epoch:      int,
    val_loss:   float,
    model_name: str,
    experiment: str,
) -> None:
    """Save model state dict to checkpoints directory."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    current_seed = SEEDS[config.ACTIVE_SEED_INDEX]
    if config.EXP1_ACTIVE:
        path = CHECKPOINT_DIR / f"{model_name}_{experiment}_{config.EXP1_CONFIG_NAME}_seed{current_seed}_best.pth"
    else:
        path = CHECKPOINT_DIR / f"{model_name}_{experiment}_seed{current_seed}_best.pth"
    torch.save({
        "epoch":      epoch,
        "val_loss":   val_loss,
        "state_dict": model.state_dict(),
    }, path)
    print(f"  [Ckpt] Saved checkpoint → {path}")


def load_checkpoint(
    model:      nn.Module,
    model_name: str,
    experiment: str,
) -> nn.Module:
    """
    Load best checkpoint for a given model + experiment into model.
    Used by finetune experiment to load source_only weights before
    continuing training on night data.
    """
    current_seed = SEEDS[config.ACTIVE_SEED_INDEX]
    if config.EXP1_ACTIVE:
        path = CHECKPOINT_DIR / f"{model_name}_{experiment}_{config.EXP1_CONFIG_NAME}_seed{current_seed}_best.pth"
    else:
        path = CHECKPOINT_DIR / f"{model_name}_{experiment}_seed{current_seed}_best.pth"
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Run the '{experiment}' experiment first."
        )
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    print(f"  [Ckpt] Loaded checkpoint from {path}  (epoch {ckpt['epoch']})")
    return model


class EarlyStopping:
    """
    Stops training when val loss has not improved for `patience` epochs.
    Tracks the best loss seen so far and signals when to stop.
    """

    def __init__(self, patience: int = EARLY_STOP_PATIENCE):
        self.patience   = patience
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Call once per epoch with current val loss.
        Returns True if training should stop.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ════════════════════════════════════════════════════════════════════════════
# Standard Trainer
# Used for: target_only, source_only, finetune
# ════════════════════════════════════════════════════════════════════════════

def train_standard(
    model:      BaseModel,
    loaders:    dict[str, DataLoader],
    model_name: str,
    experiment: str,
) -> dict:
    """
    Standard classification training loop.
    Used for target_only, source_only, and finetune experiments.

    For finetune: main.py loads source_only checkpoint into the model
    before calling this function. This function just sees a model and
    trains it — it doesn't need to know it's a finetune run.

    Parameters
    ----------
    model      : instantiated BaseModel (not DANN)
    loaders    : dict with keys "train", "val", "test"
    model_name : for checkpoint naming
    experiment : for checkpoint naming

    Returns
    -------
    history : dict with train_loss, val_loss, train_acc, val_acc per epoch
    """
    device    = get_device()
    model     = model.to(device)
    criterion      = nn.CrossEntropyLoss()
    criterion_none = nn.CrossEntropyLoss(reduction='none')
    optimizer      = torch.optim.Adam(get_param_groups(model, model_name), weight_decay=WEIGHT_DECAY)
    
    if LR_SCHEDULER == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=LR_PATIENCE, factor=0.5
        )
    elif LR_SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
        )
    else:
        raise ValueError(f"Unknown scheduler: {LR_SCHEDULER}")

    scaler = torch.amp.GradScaler("cuda")
    early_stop   = EarlyStopping()

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    
    # Per-sample loss tracking (placeholder for Phase 0.3)
    # Stores: epoch -> {sample_id: loss_value}
    per_sample_losses = {}

    best_val_loss = float("inf")

    print(f"\n[Train] Starting standard training — {model_name} / {experiment}")
    print(f"        Max epochs: {MAX_EPOCHS}  |  Early stop patience: {EARLY_STOP_PATIENCE}")

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        # ── Training phase ────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total   = 0

        for imgs, class_labels, _, indices in loaders["train"]:
            # domain_labels ignored in standard training
            imgs         = imgs.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                logits = model(imgs)                        # (B, NUM_CLASSES)
                loss   = criterion(logits, class_labels)
                
                # Per-sample loss tracking
                losses_none = criterion_none(logits, class_labels)
                for i in range(len(indices)):
                    sample_idx = indices[i].item()
                    loss_val   = losses_none[i].item()
                    per_sample_losses[sample_idx] = loss_val

            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            
            scaler.step(optimizer)
            scaler.update()

            train_loss    += loss.item() * imgs.size(0)
            preds          = logits.argmax(dim=1)
            train_correct += (preds == class_labels).sum().item()
            train_total   += imgs.size(0)

        train_loss /= train_total
        train_acc   = train_correct / train_total

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for imgs, class_labels, _, _ in loaders["val"]:
                imgs         = imgs.to(device, non_blocking=True)
                class_labels = class_labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
                    loss   = criterion(logits, class_labels)

                val_loss    += loss.item() * imgs.size(0)
                preds        = logits.argmax(dim=1)
                val_correct += (preds == class_labels).sum().item()
                val_total   += imgs.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total

        # ── Logging ───────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:>3}/{MAX_EPOCHS} | "
              f"train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
              f"val loss: {val_loss:.4f}  acc: {val_acc:.4f} | "
              f"time: {elapsed:.1f}s")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # ── Checkpoint + scheduler + early stopping ───────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, epoch, val_loss, model_name, experiment)

        # Scheduler step logic
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if early_stop.step(val_loss):
            print(f"  [EarlyStop] No improvement for {EARLY_STOP_PATIENCE} "
                  f"epochs — stopping at epoch {epoch}.")
            break

    # Save top 100 hard samples
    import json
    # Sort by loss descending
    sorted_samples = sorted(per_sample_losses.items(), key=lambda x: x[1], reverse=True)[:100]
    hard_samples = [{"crop_name": str(idx), "loss_value": float(loss)} for idx, loss in sorted_samples]
    
    current_seed = SEEDS[config.ACTIVE_SEED_INDEX]
    json_path = RESULTS_DIR / f"{model_name}_{experiment}_seed{current_seed}_hard_samples.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(hard_samples, f, indent=4)
    print(f"  [Train] Saved 100 hard samples to {json_path}")

    print(f"[Train] Standard training complete. Best val loss: {best_val_loss:.4f}")
    return history


# ════════════════════════════════════════════════════════════════════════════
# DANN Trainer
# Used for: dann experiment only
# ════════════════════════════════════════════════════════════════════════════

def train_dann(
    model:      DANNModel,
    loaders:    dict[str, DataLoader],
    model_name: str,
    experiment_tag: str = "dann",
) -> dict:
    """
    DANN training loop.

    Each step:
        1. Sample a batch from day loader   (labelled — has class labels)
        2. Sample a batch from night loader (labelled — has class labels)
        3. Combine both for domain loss
        4. Use only labelled data for classification loss
           (in true semi-supervised DANN, night would be unlabelled —
            here both are labelled so we use both for class loss too,
            which gives a stronger signal)
        5. Total loss = class_loss + lambda * domain_loss
        6. Update lambda via scheduler every step

    Parameters
    ----------
    model      : DANNModel instance
    loaders    : dict with keys "train", "val", "test",
                 "train_day", "train_night"
    model_name : for checkpoint naming

    Returns
    -------
    history : dict with losses and accuracies per epoch
    """
    device    = get_device()
    model     = model.to(device)
    class_criterion  = nn.CrossEntropyLoss()
    class_criterion_none = nn.CrossEntropyLoss(reduction='none')
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(get_dann_param_groups(model, model_name), weight_decay=WEIGHT_DECAY)
    
    if LR_SCHEDULER == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=LR_PATIENCE, factor=0.5
        )
    elif LR_SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
        )
    else:
        raise ValueError(f"Unknown scheduler: {LR_SCHEDULER}")

    scaler = torch.amp.GradScaler("cuda")
    early_stop = EarlyStopping()

    history = {
        "train_class_loss":  [],
        "train_domain_loss": [],
        "train_total_loss":  [],
        "val_loss":          [],
        "val_acc":           [],
    }

    # Per-sample loss tracking (placeholder for Phase 0.3)
    per_sample_losses = {}

    best_val_loss = float("inf")

    # Total steps for lambda scheduling
    # Use length of the shorter domain loader to avoid StopIteration
    steps_per_epoch = min(
        len(loaders["train_day"]),
        len(loaders["train_night"]),
    )
    total_steps = MAX_EPOCHS * steps_per_epoch
    global_step = 0

    print(f"\n[Train] Starting DANN training — {model_name}")
    print(f"        Max epochs: {MAX_EPOCHS}  |  Steps/epoch: {steps_per_epoch}")
    print(f"        Total steps for lambda schedule: {total_steps}")

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()

        epoch_class_loss  = 0.0
        epoch_domain_loss = 0.0
        epoch_total_loss  = 0.0
        epoch_correct     = 0
        epoch_total       = 0

        # Zip day and night loaders — stops at the shorter one
        day_iter   = iter(loaders["train_day"])
        night_iter = iter(loaders["train_night"])

        for step in range(steps_per_epoch):

            # ── Update lambda before this step ────────────────────────────────
            lam = compute_lambda(global_step, total_steps)
            model.set_lambda(lam)
            global_step += 1

            # ── Fetch batches ─────────────────────────────────────────────────
            day_imgs,   day_class,   day_domain,   day_indices   = next(day_iter)
            night_imgs, night_class, night_domain, night_indices = next(night_iter)

            day_imgs    = day_imgs.to(device, non_blocking=True)
            night_imgs  = night_imgs.to(device, non_blocking=True)
            day_class   = day_class.to(device, non_blocking=True)
            night_class = night_class.to(device, non_blocking=True)

            # Domain labels — float for BCELoss
            # day=0, night=1
            day_domain_labels   = torch.zeros(
                day_imgs.size(0), 1, device=device, dtype=torch.float32
            )
            night_domain_labels = torch.ones(
                night_imgs.size(0), 1, device=device, dtype=torch.float32
            )

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                # ── Classification loss ────────────────────────────────────────
                day_class_logits,   day_domain_probs   = model(day_imgs)
                night_class_logits, night_domain_probs = model(night_imgs)

                # Semi-supervised toggle + partial target labels for Direction 1
                day_loss = class_criterion(day_class_logits, day_class)
                night_label_mask = night_class != config.UNLABELED_CLASS_INDEX
                has_target_labels = bool(night_label_mask.any().item())

                if config.DANN_SEMI_SUPERVISED or not has_target_labels:
                    night_loss = torch.tensor(0.0, device=device)
                else:
                    night_loss = class_criterion(
                        night_class_logits[night_label_mask],
                        night_class[night_label_mask],
                    )

                class_loss = (day_loss + night_loss) / 2.0

                # Per-sample loss tracking
                day_losses_none = class_criterion_none(day_class_logits, day_class)

                for i in range(len(day_indices)):
                    per_sample_losses[day_indices[i].item()] = day_losses_none[i].item()
                if not config.DANN_SEMI_SUPERVISED and has_target_labels:
                    night_losses_none = class_criterion_none(
                        night_class_logits[night_label_mask],
                        night_class[night_label_mask],
                    )
                    labelled_night_indices = night_indices[night_label_mask.cpu()]
                    for i in range(len(labelled_night_indices)):
                        per_sample_losses[labelled_night_indices[i].item()] = night_losses_none[i].item()

                # ── Domain loss (both domains combined) ───────────────────────
                domain_loss = (
                    domain_criterion(day_domain_probs,   day_domain_labels) +
                    domain_criterion(night_domain_probs, night_domain_labels)
                ) / 2.0

                # ── Total loss ────────────────────────────────────────────────
                total_loss = class_loss + lam * domain_loss

            scaler.scale(total_loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            
            scaler.step(optimizer)
            scaler.update()

            # ── Accumulate stats ──────────────────────────────────────────────
            epoch_class_loss  += class_loss.item()
            epoch_domain_loss += domain_loss.item()
            epoch_total_loss  += total_loss.item()

            # Track classification accuracy on day batch
            preds          = day_class_logits.argmax(dim=1)
            epoch_correct += (preds == day_class).sum().item()
            epoch_total   += day_imgs.size(0)

        # ── Epoch averages ────────────────────────────────────────────────────
        epoch_class_loss  /= steps_per_epoch
        epoch_domain_loss /= steps_per_epoch
        epoch_total_loss  /= steps_per_epoch
        train_acc          = epoch_correct / epoch_total

        # ── Validation (classification only, on night val) ────────────────────
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for imgs, class_labels, _, _ in loaders["val"]:
                imgs         = imgs.to(device, non_blocking=True)
                class_labels = class_labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    # Only need class logits for validation
                    class_logits, _ = model(imgs)
                    loss = class_criterion(class_logits, class_labels)

                val_loss    += loss.item() * imgs.size(0)
                preds        = class_logits.argmax(dim=1)
                val_correct += (preds == class_labels).sum().item()
                val_total   += imgs.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total

        # ── Logging ───────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:>3}/{MAX_EPOCHS} | "
              f"class: {epoch_class_loss:.4f}  "
              f"domain: {epoch_domain_loss:.4f}  "
              f"total: {epoch_total_loss:.4f}  "
              f"train_acc: {train_acc:.4f} | "
              f"val loss: {val_loss:.4f}  acc: {val_acc:.4f} | "
              f"λ: {lam:.4f}  "
              f"time: {elapsed:.1f}s")

        history["train_class_loss"].append(epoch_class_loss)
        history["train_domain_loss"].append(epoch_domain_loss)
        history["train_total_loss"].append(epoch_total_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # ── Checkpoint + scheduler + early stopping ───────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, epoch, val_loss, model_name, experiment_tag)

        # Save epoch features for animation (every N epochs + epoch 1)
        if config.SAVE_EPOCH_FEATURES and (
            epoch == 1 or epoch % config.SAVE_FEATURES_EVERY_N == 0
        ):
            from visualise import save_epoch_features
            save_epoch_features(
                model, 
                loaders.get("viz", loaders["val"]), 
                model_name, 
                epoch, 
                SEEDS[config.ACTIVE_SEED_INDEX],
                run_label=experiment_tag,
            )

        # Scheduler step logic
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if early_stop.step(val_loss):
            print(f"  [EarlyStop] No improvement for {EARLY_STOP_PATIENCE} "
                  f"epochs — stopping at epoch {epoch}.")
            break

    # Save top 100 hard samples
    import json
    sorted_samples = sorted(per_sample_losses.items(), key=lambda x: x[1], reverse=True)[:100]
    hard_samples = [{"crop_name": str(idx), "loss_value": float(loss)} for idx, loss in sorted_samples]
    
    current_seed = SEEDS[config.ACTIVE_SEED_INDEX]
    json_path = RESULTS_DIR / f"{model_name}_{experiment_tag}_seed{current_seed}_hard_samples.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(hard_samples, f, indent=4)
    print(f"  [Train] Saved 100 hard samples to {json_path}")

    print(f"[Train] DANN training complete. Best val loss: {best_val_loss:.4f}")
    return history

"""Lambda is updated every step, not every epoch — this is important. If you updated per epoch you'd have only 50 lambda values across training. Updating per step gives a smooth continuous curve across all batches, which is what the original paper does.
Validation uses only class logits — during validation we still call model(imgs) which returns a tuple, but we unpack and discard the domain probs. Validation is purely about classification quality on the night val set.
Both domains contribute to class loss — in the original DANN paper the target domain is unlabelled so only source contributes to class loss. Here both day and night are labelled so we use both, which gives a stronger classification signal. This is a deliberate choice for your setup.
Early stopping watches val loss — same as standard trainer, val loss here is the classification loss on night val only. Domain loss is not included in the early stopping signal since it doesn't directly reflect classification quality."""
