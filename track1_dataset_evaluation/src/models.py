"""
Corrected model definitions and factory for tabular network intrusion detection (Version 2).
Fixes XGBoost class weighting by calculating and applying balanced sample weights,
and standardizes model interfaces.
"""

from typing import Any, Literal, Optional
import numpy as np
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BalancedXGBClassifier(xgb.XGBClassifier):
    """
    Subclass of XGBClassifier that natively supports class_weight='balanced'.
    Automatically computes balanced sample weights on y_train during fit().
    """

    def __init__(self, class_weight: Optional[str] = "balanced", **kwargs):
        self.class_weight = class_weight
        super().__init__(**kwargs)

    def get_xgb_params(self):
        params = super().get_xgb_params()
        params.pop("class_weight", None)
        return params

    def fit(self, X, y, sample_weight=None, **kwargs):
        if sample_weight is None and self.class_weight == "balanced":
            # Compute balanced sample weights: n_samples / (n_classes * np.bincount(y))
            sample_weight = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)


class PyTorchTabularDNN(nn.Module):
    """
    Multi-Layer Perceptron (DNN) for tabular flow features.
    Architecture: Input(D) -> Dense(128) -> BN -> ReLU -> Dropout ->
                  Dense(64) -> BN -> ReLU -> Dropout -> Output(num_classes)
    """

    def __init__(self, input_dim: int = 39, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PyTorchDNNWrapper:
    """
    Scikit-learn compatible wrapper around PyTorchTabularDNN.
    Supports balanced class-weighted CrossEntropyLoss.
    Records training and validation loss for every epoch.
    """

    def __init__(
        self,
        input_dim: int = 39,
        num_classes: int = 2,
        epochs: int = 5,
        batch_size: int = 2048,
        lr: float = 0.002,
        weight_decay: float = 1e-4,
        class_weight: Optional[str] = "balanced",
        device: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.class_weight = class_weight
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PyTorchTabularDNN(input_dim=input_dim, num_classes=num_classes).to(self.device)
        self.history: dict = {"train_loss": [], "val_loss": []}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ):
        self.model.train()
        self.history = {"train_loss": [], "val_loss": []}

        class_weights_tensor = None
        if self.class_weight == "balanced":
            classes = np.unique(y_train)
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
            full_weights = np.ones(self.num_classes, dtype=np.float32)
            for cls_idx, w in zip(classes, weights):
                full_weights[cls_idx] = w
            class_weights_tensor = torch.tensor(full_weights, dtype=torch.float32).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        tensor_x = torch.tensor(X_train, dtype=torch.float32)
        tensor_y = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        val_loader = None
        if X_val is not None and y_val is not None:
            tensor_vx = torch.tensor(X_val, dtype=torch.float32)
            tensor_vy = torch.tensor(y_val, dtype=torch.long)
            val_dataset = TensorDataset(tensor_vx, tensor_vy)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            total_samples = 0

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item()) * len(batch_y)
                total_samples += len(batch_y)

            epoch_train_loss = running_loss / total_samples if total_samples > 0 else 0.0
            self.history["train_loss"].append(round(epoch_train_loss, 5))

            # Validation loss evaluation strictly on validation set
            epoch_val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss_acc = 0.0
                val_total = 0
                with torch.no_grad():
                    for batch_vx, batch_vy in val_loader:
                        batch_vx = batch_vx.to(self.device)
                        batch_vy = batch_vy.to(self.device)
                        v_out = self.model(batch_vx)
                        v_loss = criterion(v_out, batch_vy)
                        val_loss_acc += float(v_loss.item()) * len(batch_vy)
                        val_total += len(batch_vy)
                epoch_val_loss = val_loss_acc / val_total if val_total > 0 else 0.0
                self.history["val_loss"].append(round(epoch_val_loss, 5))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        tensor_x = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(tensor_x)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        probas = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                probs = torch.softmax(logits, dim=-1)
                probas.append(probs.cpu().numpy())

        return np.vstack(probas)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


def get_model_v2(
    model_name: Literal[
        "logistic_regression", "random_forest", "lightgbm", "xgboost", "pytorch_dnn"
    ],
    num_classes: int = 2,
    input_dim: int = 39,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
    **kwargs: Any,
) -> Any:
    """
    Factory function returning an initialized model instance with verified class weighting.
    """
    if model_name == "logistic_regression":
        max_iter = kwargs.get("max_iter", 100)
        return SGDClassifier(
            loss="log_loss",
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
        )

    elif model_name == "random_forest":
        n_estimators = kwargs.get("n_estimators", 50)
        max_depth = kwargs.get("max_depth", 15)
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    elif model_name == "lightgbm":
        n_estimators = kwargs.get("n_estimators", 100)
        learning_rate = kwargs.get("learning_rate", 0.05)
        num_leaves = kwargs.get("num_leaves", 31)
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    elif model_name == "xgboost":
        n_estimators = kwargs.get("n_estimators", 100)
        learning_rate = kwargs.get("learning_rate", 0.05)
        max_depth = kwargs.get("max_depth", 6)

        if num_classes == 2:
            return BalancedXGBClassifier(
                class_weight=class_weight,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
                eval_metric="logloss",
                tree_method="hist",
            )
        else:
            return BalancedXGBClassifier(
                class_weight=class_weight,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
                eval_metric="mlogloss",
                objective="multi:softprob",
                num_class=num_classes,
                tree_method="hist",
            )

    elif model_name == "pytorch_dnn":
        epochs = kwargs.get("epochs", 5)
        batch_size = kwargs.get("batch_size", 2048)
        lr = kwargs.get("lr", 0.002)
        return PyTorchDNNWrapper(
            input_dim=input_dim,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            class_weight=class_weight,
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")
