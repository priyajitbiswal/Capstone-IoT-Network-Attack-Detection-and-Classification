"""
Model definitions and factory for tabular network intrusion detection.
Supports Logistic Regression, Random Forest, LightGBM, XGBoost, and PyTorch DNN.
"""

from typing import Any, Dict, Literal, Optional
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PyTorchTabularDNN(nn.Module):
    """
    Multi-Layer Perceptron (DNN) for tabular network flow features.
    Architecture: Input(39) -> Dense(128) -> BN -> ReLU -> Dropout ->
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
    Implements fit, predict, and predict_proba methods.
    """

    def __init__(
        self,
        input_dim: int = 39,
        num_classes: int = 2,
        epochs: int = 3,
        batch_size: int = 2048,
        lr: float = 0.002,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = PyTorchTabularDNN(
            input_dim=input_dim, num_classes=num_classes
        ).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model.train()
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device)
            if class_weights is not None
            else None
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        tensor_x = torch.tensor(X_train, dtype=torch.float32)
        tensor_y = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

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


def get_model(
    model_name: Literal[
        "logistic_regression", "random_forest", "lightgbm", "xgboost", "pytorch_dnn"
    ],
    num_classes: int = 2,
    input_dim: int = 39,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
) -> Any:
    """
    Factory function returning an initialized model instance.
    """
    if model_name == "logistic_regression":
        # Scalable logistic regression via stochastic gradient descent (log-loss)
        return SGDClassifier(
            loss="log_loss",
            max_iter=100,
            class_weight=class_weight,
            random_state=random_state,
        )

    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    elif model_name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    elif model_name == "xgboost":
        if num_classes == 2:
            return xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                random_state=random_state,
                n_jobs=-1,
                eval_metric="logloss",
            )
        else:
            return xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                random_state=random_state,
                n_jobs=-1,
                eval_metric="mlogloss",
                objective="multi:softprob",
                num_class=num_classes,
            )

    elif model_name == "pytorch_dnn":
        return PyTorchDNNWrapper(
            input_dim=input_dim,
            num_classes=num_classes,
            epochs=3,
            batch_size=2048,
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")
