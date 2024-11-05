# MLOps Linear Regression Project

This project demonstrates a simple MLOps pipeline for a linear regression model using Python, Docker, Flask, and GitHub Actions.

## Project Structure

- **app.py**: Flask app to serve the trained model as an API.
- **Dockerfile**: Instructions to containerize the application.
- **requirements.txt**: Python dependencies.
- **model/**: Contains the training script and the saved model.
- **test/**: Contains unit tests for the model.
- **.github/**: Contains CI/CD pipeline setup using GitHub Actions.

## Getting Started

### Local Setup

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
