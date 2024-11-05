import joblib
import numpy as np
import pytest

def load_model():
    return joblib.load('model/linear_regression_model.pkl')

def test_model():
    model = load_model()
    # Test a sample input
    sample_input = np.array([[6]])  # Test input
    prediction = model.predict(sample_input)
    
    assert np.isclose(prediction[0], 12.0, atol=1e-5), f"Expected 12, but got {prediction[0]}"

def test_prediction():
    model = load_model()
    sample_input = np.array([[5]])
    prediction = model.predict(sample_input)
    
    assert np.isclose(prediction[0], 10.0, atol=1e-5), f"Expected 10, but got {prediction[0]}"
