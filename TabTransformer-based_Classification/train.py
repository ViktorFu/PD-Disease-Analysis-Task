import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
from pathlib import Path
import numpy as np
import random
# import wandb
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from tab_transformer_pytorch import TabTransformer

# --- 1. Configuration & Reproducibility ---
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'tab_transformer_model.pt'
SEED = 42

def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 2. Data Loading and Splitting ---
def load_train_val_from_single_file(data_path, val_ratio=0.2, random_state=42):
    """Loads data and splits it into training and validation sets stratifiably."""
    data = torch.load(data_path)
    x_categ, x_cont, y = data['x_categ'], data['x_cont'], data['y']

    idx_all = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        idx_all,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y.numpy()
    )

    train_dataset = TensorDataset(x_categ[train_idx], x_cont[train_idx], y[train_idx])
    val_dataset = TensorDataset(x_categ[val_idx], x_cont[val_idx], y[val_idx])
    return train_dataset, val_dataset

# --- 3. Training Function ---
def train_model():
    """Main function to train the TabTransformer model with wandb tracking."""
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    # --- Load Data and Metadata ---
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / 'TabTransformer-based_Classification' / 'data'
    model_dir = project_root / 'TabTransformer-based_Classification'
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / MODEL_NAME

    train_dataset, val_dataset = load_train_val_from_single_file(
        data_dir / 'train_processed.pt',
        val_ratio=0.2,
        random_state=SEED
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    preprocessors = joblib.load(data_dir / 'preprocessors.joblib')

    # --- 4. Initialize Model (Robustly) ---
    print("\n--- Initializing TabTransformer Model ---")
    cat_feats = preprocessors['categorical_features']
    label_encoders = preprocessors['label_encoders']
    categories_counts = tuple(len(label_encoders[feat].classes_) for feat in cat_feats)
    num_continuous = len(preprocessors['numerical_features'])
    num_classes = len(preprocessors['target_encoder'].classes_)

    model_config = {
        'categories': categories_counts,
        'num_continuous': num_continuous,
        'dim': 32,
        'dim_out': num_classes,
        'depth': 6,
        'heads': 8,
        'attn_dropout': 0.1,
        'ff_dropout': 0.1,
        'mlp_hidden_mults': (4, 2)
    }
    model = TabTransformer(**model_config).to(DEVICE)

    # --- WANDB SETUP ---
    # full_config = {
    #     **model_config,
    #     **{'epochs': EPOCHS, 'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE, 'seed': SEED}
    # }

    # run = wandb.init(
    #     project="PD prediction task",
    #     entity="ld77-universit-paris-dauphine-psl",
    #     config=full_config
    # )

    print(
        f"Model initialized with "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
        f"trainable parameters."
    )

    # --- 5. Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- 6. Training Loop with Rich Metrics & Wandb Logging ---
    best_val_f1 = 0.0
    print("\n--- Starting Training ---")

    for epoch in range(EPOCHS):
        # ------------------
        # Training phase
        # ------------------
        model.train()
        train_loss, train_preds, train_targets = 0.0, [], []

        for x_categ, x_cont, y in train_loader:
            x_categ, x_cont, y = x_categ.to(DEVICE), x_cont.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x_categ, x_cont)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_categ.size(0)
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_targets.extend(y.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='weighted')

        # ------------------
        # Validation phase
        # ------------------
        model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []
        with torch.no_grad():
            for x_categ, x_cont, y in val_loader:
                x_categ, x_cont, y = x_categ.to(DEVICE), x_cont.to(DEVICE), y.to(DEVICE)
                outputs = model(x_categ, x_cont)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x_categ.size(0)
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}"
        )

        # --- WANDB LOGGING ---
        # wandb.log({
        #     "epoch": epoch,
        #     "train_loss": train_loss,
        #     "train_accuracy": train_acc,
        #     "train_f1": train_f1,
        #     "val_loss": val_loss,
        #     "val_accuracy": val_acc,
        #     "val_f1": val_f1
        # })

        # --- Checkpointing best model ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"  -> New best model found! Saving to {model_path}")
            print("  Validation Classification Report:")
            target_names = [str(c) for c in preprocessors['target_encoder'].classes_]
            print(classification_report(val_targets, val_preds, target_names=target_names, digits=4))

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'config': model_config,
                'train_script_config': {
                    'epochs': EPOCHS,
                    'lr': LEARNING_RATE,
                    'batch_size': BATCH_SIZE
                }
            }, model_path)

            # wandb.summary['best_val_f1'] = best_val_f1
            # wandb.summary['best_epoch'] = epoch

    print("\n--- Training Completed ---")
    print(f"Best validation F1-score: {best_val_f1:.4f}")

    # --- 7. Final Test Evaluation ---
    print("\n--- Loading best checkpoint for final test evaluation ---")
    best_ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()

    # Build test loader from the held-out test set
    test_data = torch.load(data_dir / 'test_processed.pt')
    test_dataset = TensorDataset(test_data['x_categ'], test_data['x_cont'], test_data['y'])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    test_preds, test_targets = [], []
    with torch.no_grad():
        for x_categ, x_cont, y in test_loader:
            x_categ, x_cont, y = x_categ.to(DEVICE), x_cont.to(DEVICE), y.to(DEVICE)
            logits = model(x_categ, x_cont)
            pred = torch.argmax(logits, dim=1)
            test_preds.extend(pred.cpu().numpy())
            test_targets.extend(y.cpu().numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average='weighted')

    print("\n--- Final Test Evaluation ---")
    print(f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    target_names = [str(c) for c in preprocessors['target_encoder'].classes_]
    print("Test Set Classification Report:")
    print(classification_report(
        test_targets,
        test_preds,
        target_names=target_names,
        digits=4
    ))

    # Log final test metrics to wandb summary for easy comparison
    # wandb.summary['test_acc'] = test_acc
    # wandb.summary['test_f1'] = test_f1

    # wandb.finish()


if __name__ == '__main__':
    train_model()
