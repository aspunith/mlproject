# Student Performance Prediction

## Project Overview
This project aims to predict student performance based on features such as gender, parental education, and test scores. The project includes a machine learning model, a Streamlit web app for user interaction, and MLflow for experiment tracking and model management.

## Project Structure
- `app.py`: Streamlit web app for user interaction and predictions.
- `mlruns/`: Directory containing MLflow experiment tracking files.
- `notebook/data/stud.csv`: Dataset used for training and evaluation.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aspunith/mlproject.git
   ```
2. Navigate to the project directory:
   ```bash
   cd mlproject
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Use the web interface to input features and get predictions.

## Key Components
- **Streamlit App**: Provides a user-friendly interface for predictions.
- **MLflow**: Tracks experiments and manages models.

## Dataset
The dataset (`stud.csv`) contains features like gender, parental education, and test scores. It is located in the `notebook/data/` directory.

## Logging
MLflow is used for logging experiments and tracking model performance.

## Deployment
The model can be deployed as a REST API for broader accessibility (pending implementation).

## Future Enhancements
- Add unit tests for pipelines and models.
- Deploy the model as a REST API.
- Improve the Streamlit app UI.

## Pipelines
The project uses separate pipelines for training and testing:

- **Training Pipeline**:
  - Located in `src/pipeline/train_pipeline.py`.
  - Handles data ingestion, transformation, and model training.
  - Saves the trained model and preprocessor as artifacts for future use.

- **Prediction Pipeline**:
  - Located in `src/pipeline/predict_pipeline.py`.
  - Loads the saved model and preprocessor to make predictions on new data.

## Machine Learning Models
The project explores and uses the following machine learning models:

- **Logistic Regression**: Used for initial baseline predictions.
- **XGBoost**: Utilized for improved performance and handling complex relationships in the data.
- **CatBoost**: Explored for its ability to handle categorical features effectively.

## Key Libraries
The following libraries are used in the project:

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For machine learning algorithms and preprocessing.
- **XGBoost**: For gradient boosting models.
- **CatBoost**: For categorical boosting models.
- **Streamlit**: For building the web app interface.
- **MLflow**: For experiment tracking and model management.