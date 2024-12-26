import optuna
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from constants import *
from VGGM_16_custom import *

# Early stopping implementation
class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Define the Optuna objective function
def objective(trial):
    """
    Objective function for Optuna to optimize hyperparameters.
    """

    # Define the hyperparameter search space
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-2)  # Learning rate
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])  # Batch size
    dropout = trial.suggest_uniform('dropout', 0.3, 0.7)  # Dropout probability
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)  # Weight decay for regularization
    step_size = trial.suggest_int('step_size', 5, 20)  # Step size for LR scheduler
    gamma = trial.suggest_uniform('gamma', 0.1, 0.9)  # Gamma (LR decay factor)

    # Initialize the model and apply the dynamic dropout
    model = DeepFakeDetection(epochs=10, batch_size=batch_size, learning_rate=learning_rate).to(DEVICE)
    for layer in model.children():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = dropout  # Update dropout dynamically

    # Loss function: Binary Cross-Entropy Loss
    criterion = torch.nn.BCELoss()

    # Optimizer: Adam with weight decay (regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler: StepLR
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Mixed precision scaler for faster training
    scaler = GradScaler()

    # Early stopping
    early_stopping = EarlyStopping(patience=5)

    # TensorBoard writer for real-time monitoring
    writer = SummaryWriter(log_dir=f'logs/trial_{trial.number}')

    # Training and validation
    train_loss, val_loss = 0, 0
    for epoch in range(10):  # Train for a limited number of epochs during optimization
        model.train()  # Set the model to training mode
        train_loss = 0  # Reset training loss

        # Training loop
        for i in range(0, len(x_paths), batch_size):
            # Load training data in batches
            x_batch, y_batch = create_tensors_from_csv(x_paths, labels, i, batch_size)
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)  # Move data to GPU/CPU

            # Zero the gradient
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast():
                y_pred = model(x_batch).squeeze()  # Predict
                loss = criterion(y_pred, y_batch.float())  # Compute loss

            # Backward pass and optimizer step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()  # Accumulate training loss

        # Update learning rate
        scheduler.step()

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0  # Reset validation loss
        all_y_true, all_y_pred = [], []  # To store true and predicted values

        with torch.no_grad():  # Disable gradient computation for validation
            for i in range(0, len(x_paths), batch_size):
                # Load validation data in batches
                x_batch, y_batch = create_tensors_from_csv(x_paths, labels, i, batch_size)
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

                # Forward pass with mixed precision
                with autocast():
                    y_pred = model(x_batch).squeeze()
                    val_loss += criterion(y_pred, y_batch.float()).item()  # Compute validation loss

                # Collect true and predicted labels for metrics
                all_y_true.extend(y_batch.cpu().numpy())
                all_y_pred.extend(y_pred.cpu().numpy())

        # Compute validation metrics
        accuracy, recall, f1 = calculate_metrics(np.array(all_y_true), np.array(all_y_pred))

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/accuracy', accuracy, epoch)
        writer.add_scalar('Metrics/recall', recall, epoch)
        writer.add_scalar('Metrics/f1', f1, epoch)

        # Print metrics for debugging
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Accuracy={accuracy:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    writer.close()  # Close TensorBoard writer

    # Return the validation loss for optimization
    return val_loss

# Run the Optuna optimization
study = optuna.create_study(direction='minimize')  # Minimize validation loss
study.optimize(objective, n_trials=50)  # Perform 50 trials

# Print the best hyperparameters
print("Best hyperparameters:", study.best_params)
