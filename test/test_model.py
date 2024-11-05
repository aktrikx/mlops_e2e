import pickle
import numpy as np
import pandas as pd  # Import pandas
import pytest

def load_model():
    with open('model/linear_regression_model.pkl', 'rb') as f:
        return pickle.load(f)

def test_model():
    model = load_model()
    sample_input = pd.DataFrame([[6]], columns=['X'])  # Use a DataFrame
    prediction = model.predict(sample_input)
    
    assert np.isclose(prediction[0], 12.0, atol=1e-5), f"Expected 12, but got {prediction[0]}"

def test_prediction():
    model = load_model()
    sample_input = pd.DataFrame([[5]], columns=['X'])  # Use a DataFrame
    prediction = model.predict(sample_input)
    
    assert np.isclose(prediction[0], 10.0, atol=1e-5), f"Expected 10, but got {prediction[0]}"

if __name__ == "__main__":
    pytest.main()
    print("All tests passed!")