# OCT Image Diagnosis Web Application

## Setup Instructions

### 1. Folder Structure
Ensure your project has this structure:
```
project-folder/
├── app.py
├── requirements.txt
├── oct_diagnosis_model.keras  (← Place your model here!)
├── uploads/  (← Created automatically for uploaded images)
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Model
- Place the `oct_diagnosis_model.keras` file in the **project root folder** (same level as `app.py`)
- This file should be exported from your Colab notebook using:
  ```python
  model.save('oct_diagnosis_model.keras')
  ```

### 4. Update Label Mapping (If Needed)
If your model's classes are different from [CNV, DME, DRUSEN, NORMAL], update the `LABEL_MAP` in `app.py`:

```python
LABEL_MAP = {
    'CLASS_NAME_1': 0,
    'CLASS_NAME_2': 1,
    'CLASS_NAME_3': 2,
    'CLASS_NAME_4': 3
}
```

### 5. Update Image Size (If Needed)
If your model expects a different input size, update `IMG_SIZE` in `app.py`:
```python
IMG_SIZE = (224, 224)  # Change to your model's expected size
```

### 6. Run the Application
```bash
python app.py
```

The app will be available at: **http://localhost:5000**

## Features

✅ **Image Upload** - Drag & drop or click to upload OCT images
✅ **Real-time Preview** - See your uploaded image before prediction
✅ **Model Inference** - Uses your trained TensorFlow model
✅ **Probability Display** - Shows confidence for all classes
✅ **Beautiful UI** - Modern, responsive design
✅ **Error Handling** - Graceful error messages

## Supported Image Formats
- PNG, JPG, JPEG, BMP, TIFF
- Maximum file size: 16MB

## How It Works

1. **Upload**: Select or drag-drop an OCT image
2. **Preview**: See the image before sending it
3. **Process**: Image is preprocessed (grayscale, resized, normalized)
4. **Predict**: Model makes inference on preprocessed image
5. **Results**: See predicted class and confidence scores

## Image Preprocessing Pipeline
The app uses the same preprocessing as your Colab notebook:
- Read as grayscale
- Resize to model input size (e.g., 224x224)
- Normalize to [0, 1] range
- Add channel dimension (1 for grayscale)
- Add batch dimension

## Troubleshooting

**Error: "Model not loaded"**
- Ensure `oct_diagnosis_model.keras` is in the project root
- Check file name spelling matches exactly

**Error: "File type not allowed"**
- Only PNG, JPG, JPEG, BMP, TIFF are supported
- Ensure file extension is correct

**Error: "Failed to read image"**
- Image file may be corrupted
- Try uploading a different image

**Model takes too long to load**
- Large models can take time on first load
- Subsequent predictions will be faster

## API Endpoints

### POST /api/predict
Upload image and get prediction
- Request: Form data with 'file' parameter
- Response: JSON with predicted_class, confidence, all_predictions

### GET /api/health
Check if model is loaded
- Response: JSON with model status and available classes

## Notes
- Uploaded images are saved to the `uploads/` folder
- The model is loaded once on app startup
- All predictions use the same preprocessing pipeline
