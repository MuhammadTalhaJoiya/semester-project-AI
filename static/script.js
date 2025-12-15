// Heart Disease Predictor - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resultsContainer = document.getElementById('resultsContainer');
    const resultsContent = document.getElementById('resultsContent');
    const predictBtn = document.getElementById('predictBtn');

    // Form submission handler
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Get form data
        const formData = new FormData(form);
        const patientData = {
            age: parseInt(formData.get('age')),
            sex: parseInt(formData.get('sex')),
            cp: parseInt(formData.get('cp')),
            trestbps: parseInt(formData.get('trestbps')),
            chol: parseInt(formData.get('chol')),
            fbs: parseInt(formData.get('fbs')),
            restecg: parseInt(formData.get('restecg')),
            thalach: parseInt(formData.get('thalach')),
            exang: parseInt(formData.get('exang')),
            oldpeak: parseFloat(formData.get('oldpeak')),
            slope: parseInt(formData.get('slope')),
            ca: parseInt(formData.get('ca')),
            thal: parseInt(formData.get('thal'))
        };

        // Show loading state
        setLoadingState(true);
        resultsContainer.style.display = 'none';

        try {
            // Make API call
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(patientData)
            });

            const result = await response.json();

            if (result.success) {
                displayResults(result);
            } else {
                displayError(result.error || 'Prediction failed. Please try again.');
            }
        } catch (error) {
            displayError('Network error: ' + error.message);
        } finally {
            setLoadingState(false);
        }
    });

    function setLoadingState(loading) {
        if (loading) {
            predictBtn.disabled = true;
            predictBtn.querySelector('.btn-text').style.display = 'none';
            predictBtn.querySelector('.btn-loader').style.display = 'inline-block';
            form.classList.add('loading');
        } else {
            predictBtn.disabled = false;
            predictBtn.querySelector('.btn-text').style.display = 'inline-block';
            predictBtn.querySelector('.btn-loader').style.display = 'none';
            form.classList.remove('loading');
        }
    }

    function displayResults(result) {
        const isPositive = result.prediction === 1;
        const probability = result.probability;
        const probNoDisease = result.probabilities['No Heart Disease'];
        const probDisease = result.probabilities['Heart Disease'];

        // Determine confidence level
        let confidenceClass = 'medium';
        if (probability >= 0.75) confidenceClass = 'high';
        else if (probability >= 0.6) confidenceClass = 'medium';
        else confidenceClass = 'low';

        const html = `
            <div class="result-card ${isPositive ? 'result-positive' : 'result-negative'}">
                <div class="result-title ${isPositive ? 'positive' : 'negative'}">
                    ${isPositive ? '⚠️' : '✅'}
                    ${result.prediction_label}
                </div>
                
                <div class="confidence-meter">
                    <div class="confidence-label">Confidence Level: ${(probability * 100).toFixed(1)}%</div>
                    <div class="confidence-bar-container">
                        <div class="confidence-bar ${confidenceClass}" style="width: ${probability * 100}%">
                            ${(probability * 100).toFixed(1)}%
                        </div>
                    </div>
                </div>

                <div class="probabilities">
                    <div class="prob-item">
                        <div class="prob-label">No Heart Disease</div>
                        <div class="prob-value">${(probNoDisease * 100).toFixed(1)}%</div>
                    </div>
                    <div class="prob-item">
                        <div class="prob-label">Heart Disease</div>
                        <div class="prob-value">${(probDisease * 100).toFixed(1)}%</div>
                    </div>
                </div>
            </div>
        `;

        resultsContent.innerHTML = html;
        resultsContainer.style.display = 'block';
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function displayError(message) {
        resultsContent.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> ${message}
            </div>
        `;
        resultsContainer.style.display = 'block';
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Form reset handler
    form.addEventListener('reset', function() {
        resultsContainer.style.display = 'none';
        resultsContent.innerHTML = '';
    });
});

