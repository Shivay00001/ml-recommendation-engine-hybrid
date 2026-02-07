"""Tests for recommendation models."""
import pytest
import torch
from src.models.recommender import MatrixFactorization, NeuralCollaborativeFiltering, HybridRecommender


class TestMatrixFactorization:
    def test_forward_pass(self):
        model = MatrixFactorization(num_users=100, num_items=50, embedding_dim=32)
        users = torch.LongTensor([0, 1, 2])
        items = torch.LongTensor([0, 5, 10])
        output = model(users, items)
        assert output.shape == (3,)
    
    def test_embedding_shapes(self):
        model = MatrixFactorization(num_users=100, num_items=50, embedding_dim=32)
        assert model.user_embedding.weight.shape == (100, 32)
        assert model.item_embedding.weight.shape == (50, 32)


class TestNeuralCF:
    def test_forward_pass(self):
        model = NeuralCollaborativeFiltering(num_users=100, num_items=50)
        users = torch.LongTensor([0, 1, 2])
        items = torch.LongTensor([0, 5, 10])
        output = model(users, items)
        assert output.shape == (3,)


class TestHybridRecommender:
    def test_forward_pass(self):
        model = HybridRecommender(num_users=100, num_items=50, item_feature_dim=20)
        users = torch.LongTensor([0, 1, 2])
        items = torch.LongTensor([0, 5, 10])
        features = torch.randn(3, 20)
        output = model(users, items, features)
        assert output.shape == (3,)
