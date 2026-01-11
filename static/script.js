let selectedFile = null;

// File input event listener
document.getElementById('imageInput').addEventListener('change', function(e) {
    selectedFile = e.target.files[0];
    if (selectedFile) {
        displayFileInfo(selectedFile);
        displayPreview(selectedFile);
        document.getElementById('predictBtn').disabled = false;
    }
});

// Drag and drop functionality
const uploadBox = document.querySelector('.upload-box');

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#764ba2';
    uploadBox.style.background = '#f0f2ff';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '#f8f9ff';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '#f8f9ff';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        selectedFile = files[0];
        displayFileInfo(selectedFile);
        displayPreview(selectedFile);
        document.getElementById('predictBtn').disabled = false;
    }
});

function displayFileInfo(file) {
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = (file.size / 1024 / 1024).toFixed(2);
    
    fileName.textContent = `📄 ${file.name} (${fileSize} MB)`;
    fileInfo.style.display = 'block';
}

function displayPreview(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        const preview = document.getElementById('imagePreview');
        const noPreview = document.getElementById('noPreview');
        
        preview.src = e.target.result;
        preview.style.display = 'block';
        noPreview.style.display = 'none';
    };
    
    reader.readAsDataURL(file);
}

function predictImage() {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }
    
    // Show loading
    document.getElementById('loadingSection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('errorSection').style.display = 'none';
    
    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    // Send to backend
    fetch('/api/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Prediction failed');
            });
        }
        return response.json();
    })
    .then(data => {
        displayResults(data);
    })
    .catch(error => {
        showError(error.message);
    })
    .finally(() => {
        document.getElementById('loadingSection').style.display = 'none';
    });
}

function displayResults(data) {
    // Display predicted class (only the diagnosis — no percentages)
    document.getElementById('predictedClass').textContent = data.predicted_class;

    // Display a short description for the diagnosis
    document.getElementById('predictedDescription').textContent = data.description || '';

    // Display filename or source info
    document.getElementById('resultFilename').textContent = data.filename || '';

    // Show results
    document.getElementById('resultsSection').style.display = 'block';
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('errorSection').style.display = 'block';
}

function hideError() {
    document.getElementById('errorSection').style.display = 'none';
}

function resetForm() {
    selectedFile = null;
    document.getElementById('imageInput').value = '';
    document.getElementById('imagePreview').style.display = 'none';
    document.getElementById('noPreview').style.display = 'block';
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('errorSection').style.display = 'none';
    document.getElementById('predictBtn').disabled = true;
}

// Check model health on page load
window.addEventListener('load', () => {
    fetch('/api/health')
        .then(response => response.json())
        .then(data => {
            if (!data.model_loaded) {
                console.warn('⚠️ Model not loaded. Make sure oct_diagnosis_model.keras exists.');
            }
        })
        .catch(error => console.error('Health check failed:', error));
});
