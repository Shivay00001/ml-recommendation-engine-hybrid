"""Training pipeline for recommendation models."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Dict, Any
import numpy as np


class InteractionDataset(Dataset):
    """Dataset for user-item interactions."""
    
    def __init__(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray, item_features: np.ndarray = None):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.ratings = torch.FloatTensor(ratings)
        self.item_features = torch.FloatTensor(item_features) if item_features is not None else None
    
    def __len__(self) -> int:
        return len(self.users)
    
    def __getitem__(self, idx: int) -> Tuple:
        if self.item_features is not None:
            return self.users[idx], self.items[idx], self.ratings[idx], self.item_features[self.items[idx]]
        return self.users[idx], self.items[idx], self.ratings[idx]


class Trainer:
    """Training manager for recommendation models."""
    
    def __init__(self, model: nn.Module, lr: float = 0.001, weight_decay: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            batch = [b.to(self.device) for b in batch]
            users, items, ratings = batch[0], batch[1], batch[2]
            
            self.optimizer.zero_grad()
            
            if len(batch) == 4:
                predictions = self.model(users, items, batch[3])
            else:
                predictions = self.model(users, items)
            
            loss = self.criterion(predictions, ratings)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        predictions, actuals = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = [b.to(self.device) for b in batch]
                users, items, ratings = batch[0], batch[1], batch[2]
                
                if len(batch) == 4:
                    preds = self.model(users, items, batch[3])
                else:
                    preds = self.model(users, items)
                
                predictions.extend(preds.cpu().numpy())
                actuals.extend(ratings.cpu().numpy())
        
        predictions, actuals = np.array(predictions), np.array(actuals)
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))
        
        return {'rmse': rmse, 'mae': mae}
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10) -> Dict[str, list]:
        history = {'train_loss': [], 'val_rmse': [], 'val_mae': []}
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_rmse'].append(val_metrics['rmse'])
            history['val_mae'].append(val_metrics['mae'])
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val RMSE: {val_metrics['rmse']:.4f}")
        
        return history
