const form = document.getElementById('uploadForm');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    resultDiv.innerHTML = "Processing...";
    const formData = new FormData(form);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (data.error) {
            resultDiv.innerHTML = `<p class="error">${data.error}</p>`;
        } else {
            resultDiv.innerHTML = `
                <p><strong>Predicted Angle:</strong> ${data.class}</p>
                <p><strong>Confidence:</strong> ${data.confidence}</p>
            `;
        }
    } catch (error) {
        resultDiv.innerHTML = `<p class="error">Error: Unable to process the request.</p>`;
    }
});
