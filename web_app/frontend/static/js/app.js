async function submitPrediction() {
    const formData = {
        student_id: document.getElementById('student_id').value,
        // Agrega aquí todos los campos necesarios
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        const result = await response.json();
        showPredictionResult(result.prediccion);
    } catch (error) {
        console.error('Error:', error);
        showPredictionResult('Error al realizar la predicción', true);
    }
}

function showPredictionResult(message, isError = false) {
    const resultDiv = document.getElementById('predictionResult');
    resultDiv.innerHTML = message;
    resultDiv.className = isError ? 'error-message' : 'success-message';
}