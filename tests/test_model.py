import unittest
import joblib
from sklearn.ensemble import RandomForestClassifier
class TestModelTraining(unittest.TestCase):
    def test_model_training(self):
        model = joblib.load('model/iris_model.pkl')
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertGreaterEqual(len(model.feature_importances_), 4)


if __name__ == '__main__':
    unittest.main()

