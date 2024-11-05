import joblib
import numpy as np

# Load the saved model
model = joblib.load('model/linear_regression_model.pkl')

def test_prediction():
    # Test prediction for input value [6] -> should be close to [12]
    input_data = np.array([6]).reshape(1, -1)
    prediction = model.predict(input_data)
    
    # Use np.isclose to compare with a tolerance for floating-point errors
    assert np.isclose(prediction[0], 12.), f"Expected 12, but got {prediction[0]}"

def test_model():
    test_prediction()
    print("All tests passed!")
    
if __name__ == "__main__":
    test_model()
