"""
Hybrid Recommendation Model combining collaborative and content-based filtering.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MatrixFactorization(nn.Module):
    """Classic matrix factorization with embeddings."""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        dot_product = (user_emb * item_emb).sum(dim=1)
        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()
        
        return dot_product + user_b + item_b + self.global_bias


class NeuralCollaborativeFiltering(nn.Module):
    """Neural CF with MLP layers on concatenated embeddings."""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, hidden_dims: list = [128, 64, 32]):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        concat = torch.cat([user_emb, item_emb], dim=1)
        return self.mlp(concat).squeeze()


class TwoTowerModel(nn.Module):
    """Two-tower architecture for efficient retrieval."""
    
    def __init__(self, user_features: int, item_features: int, embedding_dim: int = 128):
        super().__init__()
        
        self.user_tower = nn.Sequential(
            nn.Linear(user_features, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        self.item_tower = nn.Sequential(
            nn.Linear(item_features, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def encode_user(self, user_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_tower(user_features), dim=1)
    
    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_tower(item_features), dim=1)
    
    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        user_emb = self.encode_user(user_features)
        item_emb = self.encode_item(item_features)
        return (user_emb * item_emb).sum(dim=1)


class HybridRecommender(nn.Module):
    """Hybrid model combining collaborative and content-based signals."""
    
    def __init__(self, num_users: int, num_items: int, item_feature_dim: int, embedding_dim: int = 64):
        super().__init__()
        self.cf_model = NeuralCollaborativeFiltering(num_users, num_items, embedding_dim)
        self.content_model = nn.Sequential(
            nn.Linear(item_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        self.user_content_proj = nn.Linear(embedding_dim, embedding_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        cf_score = self.cf_model(user_ids, item_ids).unsqueeze(1)
        
        user_emb = self.cf_model.user_embedding(user_ids)
        user_proj = self.user_content_proj(user_emb)
        item_content = self.content_model(item_features)
        content_score = (user_proj * item_content).sum(dim=1, keepdim=True)
        
        combined = torch.cat([cf_score, content_score], dim=1)
        return self.fusion(combined).squeeze()
