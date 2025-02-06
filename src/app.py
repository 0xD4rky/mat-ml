from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from model import get_efficientnet_model  

best_model_path = "/Users/darky/Documents/mat-ml/src/best_model.pth" 
input_size = 224 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  

class_to_angle = {
    0: 0,
    1: 130,
    2: 180,
    3: 230,
    4: 270,
    5: 320,
    6: 40,
    7: 90
}

app = Flask(__name__)

model = get_efficientnet_model(num_classes=8)  
model.load_state_dict(torch.load(best_model_path, map_location=device))
model = model.to(device)
model.eval()

transform = Compose([
    Resize((input_size, input_size)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file:
        return jsonify({'error': 'Invalid file'}), 400

    try:
        image = Image.open(file).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)  

        
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)  
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            predicted_class = class_to_angle[predicted_class]
            
        return jsonify({
            'class': predicted_class,
            'confidence': f"{confidence * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': f"Error processing the file: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
