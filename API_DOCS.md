# Heart Disease Predictor - API Documentation

## Base URL
```
http://localhost:5000
```

## Endpoints

### 1. Health Check
**GET** `/health`

Check if the service is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Random Forest"
}
```

### 2. Single Prediction
**POST** `/api/predict`

Make a prediction for a single patient.

**Request Body:**
```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

**Response:**
```json
{
  "success": true,
  "prediction": 1,
  "prediction_label": "Heart Disease",
  "probability": 0.85,
  "probabilities": {
    "No Heart Disease": 0.15,
    "Heart Disease": 0.85
  }
}
```

### 3. Batch Predictions
**POST** `/api/predict/batch`

Make predictions for multiple patients.

**Request Body:**
```json
{
  "patients": [
    {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1},
    {"age": 45, "sex": 0, "cp": 1, "trestbps": 128, "chol": 204, "fbs": 0, "restecg": 0, "thalach": 172, "exang": 0, "oldpeak": 1.4, "slope": 2, "ca": 0, "thal": 2}
  ]
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "patient_id": 1,
      "prediction": 1,
      "prediction_label": "Heart Disease",
      "probability": 0.85,
      "probabilities": {
        "No Heart Disease": 0.15,
        "Heart Disease": 0.85
      }
    },
    {
      "patient_id": 2,
      "prediction": 0,
      "prediction_label": "No Heart Disease",
      "probability": 0.78,
      "probabilities": {
        "No Heart Disease": 0.78,
        "Heart Disease": 0.22
      }
    }
  ],
  "total": 2
}
```

### 4. Model Information
**GET** `/api/model/info`

Get information about the loaded model.

**Response:**
```json
{
  "success": true,
  "model_name": "Random Forest",
  "feature_count": 13,
  "features": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
}
```

## Feature Descriptions

| Feature | Description | Valid Values |
|---------|-------------|--------------|
| `age` | Patient's age | 1-120 (years) |
| `sex` | Gender | 0 = Female, 1 = Male |
| `cp` | Chest pain type | 0-3 (see below) |
| `trestbps` | Resting blood pressure | 80-250 (mm Hg) |
| `chol` | Serum cholesterol | 100-600 (mg/dl) |
| `fbs` | Fasting blood sugar > 120 | 0 = No, 1 = Yes |
| `restecg` | Resting ECG results | 0-2 (see below) |
| `thalach` | Max heart rate achieved | 60-220 (bpm) |
| `exang` | Exercise induced angina | 0 = No, 1 = Yes |
| `oldpeak` | ST depression | 0-10 (decimal) |
| `slope` | Slope of peak exercise ST | 0-2 (see below) |
| `ca` | Number of major vessels | 0-3 |
| `thal` | Thalassemia | 1-3 (see below) |

### Chest Pain Type (cp)
- 0: Typical Angina
- 1: Atypical Angina
- 2: Non-anginal Pain
- 3: Asymptomatic

### Resting ECG (restecg)
- 0: Normal
- 1: ST-T Wave Abnormality
- 2: Left Ventricular Hypertrophy

### Slope
- 0: Upsloping
- 1: Flat
- 2: Downsloping

### Thalassemia (thal)
- 1: Normal
- 2: Fixed Defect
- 3: Reversible Defect

## Error Responses

All endpoints may return error responses in this format:

```json
{
  "success": false,
  "error": "Error message here"
}
```

Common HTTP status codes:
- `400`: Bad Request (missing/invalid data)
- `500`: Internal Server Error (model not loaded or prediction failed)

## Example Usage

### JavaScript/Node.js
```javascript
const response = await fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    age: 63,
    sex: 1,
    cp: 3,
    trestbps: 145,
    chol: 233,
    fbs: 1,
    restecg: 0,
    thalach: 150,
    exang: 0,
    oldpeak: 2.3,
    slope: 0,
    ca: 0,
    thal: 1
  })
});

const result = await response.json();
console.log(result);
```

### Python
```python
import requests

url = 'http://localhost:5000/api/predict'
data = {
    'age': 63,
    'sex': 1,
    'cp': 3,
    'trestbps': 145,
    'chol': 233,
    'fbs': 1,
    'restecg': 0,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 2.3,
    'slope': 0,
    'ca': 0,
    'thal': 1
}

response = requests.post(url, json=data)
result = response.json()
print(result)
```

### cURL
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }'
```

