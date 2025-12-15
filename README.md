# Heart Disease Predictor - Supervised Classification

A comprehensive machine learning project for predicting heart disease using supervised classification algorithms.

## Project Overview

This project implements multiple classification algorithms to predict heart disease based on patient features. It includes:

- **Data Loading & Preprocessing**: Automatic dataset detection and preprocessing pipeline
- **Multiple ML Models**: Logistic Regression, Random Forest, SVM, KNN, Decision Tree
- **Model Evaluation**: Comprehensive metrics, confusion matrices, and ROC curves
- **Exploratory Data Analysis**: Statistical analysis and visualizations
- **Prediction Pipeline**: Easy-to-use prediction interface

## Project Structure

```
heart disease predictor/
├── data/                    # Place your dataset here
│   └── heart_disease.csv
├── models/                  # Saved models and reports
│   ├── best_model.joblib
│   ├── preprocessor.joblib
│   └── evaluation_plots/
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── data_loader.py      # Dataset loading
│   ├── preprocessor.py     # Data preprocessing
│   ├── train_model.py      # Model training
│   ├── evaluate_model.py   # Model evaluation
│   ├── predict.py          # Prediction module
│   └── explore_data.py     # EDA module
├── templates/              # Web UI templates
│   └── index.html
├── static/                 # Web UI static files
│   ├── style.css
│   └── script.js
├── notebooks/              # Jupyter notebooks (optional)
├── app.py                  # Web application (Flask)
├── main.py                 # CLI entry point
├── requirements.txt        # Dependencies
├── API_DOCS.md            # API documentation
└── README.md              # This file
```

## Installation

1. **Clone or download this project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your dataset** in the `data/` folder:
   - Supported formats: CSV, Excel (.xlsx, .xls)
   - Common filenames: `heart_disease.csv`, `heart.csv`, `dataset.csv`
   - Or specify the filename when running commands

## Quick Start - Complete Workflow

To train and test your models, you only need **one command**:

```bash
python main.py train
```

This single command will:
1. ✅ Load and preprocess your dataset
2. ✅ Split data into training and test sets
3. ✅ Train 5 different classification models
4. ✅ **Test all models on the test set** (automatic evaluation)
5. ✅ Compare model performance and select the best one
6. ✅ Generate evaluation reports and visualizations
7. ✅ Save the best model for future predictions

**Optional**: Before training, you can explore your data:
```bash
python main.py explore  # Optional EDA step
```

## Web Interface

A modern, user-friendly web interface is available for easy interaction with the prediction model.

### Starting the Web Server

1. **Make sure you have trained a model first:**
   ```bash
   python main.py train
   ```

2. **Install Flask dependencies (if not already installed):**
   ```bash
   pip install flask flask-cors
   ```

3. **Start the web server:**
   ```bash
   python app.py
   ```

4. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

### Web UI Features

- ✅ **Beautiful, modern interface** with gradient design
- ✅ **Easy-to-use form** with all 13 patient features
- ✅ **Real-time predictions** with instant results
- ✅ **Visual confidence indicators** with progress bars
- ✅ **Probability breakdowns** showing both outcomes
- ✅ **Mobile responsive** design
- ✅ **Form validation** with helpful error messages

### API Integration

The web app exposes RESTful APIs that can be integrated into fullstack applications:

**Available Endpoints:**
- `POST /api/predict` - Single patient prediction
- `POST /api/predict/batch` - Batch predictions for multiple patients
- `GET /api/model/info` - Get model information
- `GET /health` - Health check endpoint

**Example API Call:**
```javascript
// JavaScript/Node.js example
const response = await fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    age: 63, sex: 1, cp: 3, trestbps: 145, chol: 233,
    fbs: 1, restecg: 0, thalach: 150, exang: 0,
    oldpeak: 2.3, slope: 0, ca: 0, thal: 1
  })
});
const result = await response.json();
```

See `API_DOCS.md` for detailed API documentation and more examples.

## Usage

### 1. Explore Your Data (EDA) - Optional

Before training, explore your dataset:

```bash
python main.py explore
```

This will:
- Display dataset statistics
- Generate visualizations (distributions, correlations, etc.)
- Save plots to `models/` folder

### 2. Train and Test Models

**Important**: The `train` command automatically trains AND tests all models. Testing is built into the training pipeline.

Train all models, evaluate them on the test set, and select the best one:

```bash
python main.py train
```

**Options**:
- `--dataset`: Specify dataset filename (default: auto-detect)
- `--target`: Specify target column name (default: auto-detect)
- `--test-size`: Test set proportion (default: 0.2)
- `--random-state`: Random seed (default: 42)
- `--cv`: Cross-validation folds (default: 5)
- `--save-all`: Save all models, not just the best

**Example**:
```bash
python main.py train --dataset data/my_heart_data.csv --test-size 0.3
```

This will:
- Load and preprocess the dataset (splits into train/test sets)
- Train 5 different classification models on the training set
- **Test all models on the test set** (evaluation happens automatically)
- Compare model performance (accuracy, precision, recall, F1-score)
- Generate evaluation plots (confusion matrices, ROC curves)
- Save detailed classification reports
- Save the best model to `models/best_model.joblib`

### 3. Make Predictions

After training, make predictions on new data:

```bash
python main.py predict --file data/new_samples.csv
```

**Options**:
- `--file`: CSV file with features to predict
- `--model`: Path to model file (default: `models/best_model.joblib`)
- `--preprocessor`: Path to preprocessor file (default: `models/preprocessor.joblib`)

## Programmatic Usage

You can also use the modules directly in Python:

### Training

```python
from src.data_loader import load_dataset
from src.preprocessor import DataPreprocessor
from src.train_model import ModelTrainer
from src.evaluate_model import ModelEvaluator

# Load data
df = load_dataset()

# Preprocess
preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.preprocess(df)

# Train models
trainer = ModelTrainer()
trained_models = trainer.train_all_models(X_train, y_train, X_test, y_test)

# Evaluate
evaluator = ModelEvaluator()
evaluator.generate_all_plots(trained_models, X_test, y_test)

# Save best model
trainer.save_best_model()
```

### Making Predictions

```python
from src.predict import HeartDiseasePredictor, predict_from_dict

# Load predictor
predictor = HeartDiseasePredictor()

# Make prediction
features = {
    'age': 63,
    'sex': 1,
    'cp': 3,
    # ... other features
}

result = predict_from_dict(features)
print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['probability']:.2%}")
```

## Dataset Requirements

Your dataset should:
- Be in CSV or Excel format
- Have a target column (automatically detected)
- Contain numerical and/or categorical features
- Be placed in the `data/` folder

**Common target column names** (auto-detected):
- `target`, `Target`, `TARGET`
- `label`, `Label`, `LABEL`
- `heart_disease`, `HeartDisease`
- `disease`, `Disease`
- Or the last column will be used as target

## Model Algorithms

The project trains and compares:

1. **Logistic Regression**: Linear classification model
2. **Random Forest**: Ensemble of decision trees
3. **Support Vector Machine (SVM)**: Kernel-based classifier
4. **K-Nearest Neighbors (KNN)**: Instance-based learning
5. **Decision Tree**: Rule-based classifier

The best model (highest test accuracy) is automatically selected and saved.

## Evaluation Metrics

Each model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions
- **ROC Curve & AUC**: Receiver Operating Characteristic analysis

## Output Files

After training, the following files are generated in `models/`:

- `best_model.joblib`: Best performing model
- `preprocessor.joblib`: Preprocessing pipeline
- `*_confusion_matrix.png`: Confusion matrices for each model
- `*_roc_curve.png`: ROC curves for each model
- `*_report.txt`: Detailed classification reports
- `target_distribution.png`: Target variable distribution
- `feature_distributions.png`: Feature distributions
- `correlation_heatmap.png`: Feature correlation matrix
- `feature_target_relationships.png`: Feature-target relationships

## Requirements

- Python 3.7+
- pandas >= 2.1.4
- numpy >= 1.26.2
- scikit-learn >= 1.3.2
- matplotlib >= 3.8.2
- seaborn >= 0.13.0
- joblib >= 1.3.2

## Troubleshooting

### Dataset Not Found
- Ensure your dataset is in the `data/` folder
- Check the filename matches common names or specify with `--dataset`
- Supported formats: CSV, Excel (.xlsx, .xls)

### Model Not Found (for prediction)
- Train a model first using `python main.py train`
- Ensure `models/best_model.joblib` exists

### Missing Values
- The preprocessor automatically handles missing values
- Numeric: mean imputation
- Categorical: mode imputation

## Contributing

Feel free to extend this project:
- Add more classification algorithms
- Implement feature selection
- Add hyperparameter tuning
- Create a web interface
- Add more visualizations

## License

This project is open source and available for educational purposes.

## Contact

For questions or issues, please open an issue or contact the project maintainer.

---

**Note**: This is a supervised classification project. Ensure your dataset has labeled examples (target variable) for training.

"# semester-project-AI" 
