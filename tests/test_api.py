"""
Test suite for Breast Cancer Detection API
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
import numpy as np
from model_training import ModelManager


class TestModelManager(unittest.TestCase):
    def setUp(self):
        self.manager = ModelManager(model_path="models/test_model.pkl")

    def test_load_data(self):
        X, y = self.manager.load_data()
        self.assertEqual(X.shape[1], 30)  # 30 features
        self.assertGreater(len(y), 0)

    def test_prediction_shape(self):
        # Train a model first
        self.manager.train_models()

        # Test with correct number of features
        sample_features = [1.0] * 30
        result = self.manager.predict(sample_features)

        self.assertIn("prediction", result)
        self.assertIn("probability", result)
        self.assertIn("diagnosis", result)


class TestAPIFeatures(unittest.TestCase):
    def test_feature_count(self):
        # Breast cancer dataset has 30 features
        manager = ModelManager()
        X, _ = manager.load_data()
        self.assertEqual(X.shape[1], 30)


if __name__ == "__main__":
    unittest.main()
