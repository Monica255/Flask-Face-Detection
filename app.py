from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import io

app = Flask(__name__)

# --- 1) Load the saved state_dict onto CPU ---
state_dict = torch.load('face_model.pth', map_location='cpu')

# Determine number of output classes from state_dict
num_classes = state_dict['classifier.6.weight'].shape[0]

# --- 2) Build matching VGG architecture ---
model = models.vgg19(pretrained=False)
in_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(in_features, num_classes)

# --- 3) Load weights and set to eval mode ---
model.load_state_dict(state_dict)
model.eval()

# --- 4) Class labels ---
class_labels = ['heart', 'oblong', 'oval', 'round', 'square']

# --- 5) Define preprocessing pipeline ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- 6) Hairstyle recommendations ---
def recommend_hairstyle(face_shape):
    recommendations = {
        'heart': ["Crew cut", "Side part", "Undercut"],
        'oblong': ["French crop", "Messy fringe", "Undercut"],
        'oval': ["Buzz cut", "Quiff", "Comma hair"],
        'round': ["Buzz cut", "Curtain", "Comma hair"],
        'square': ["Buzz cut", "Quiff", "Curtain"]
    }
    return recommendations.get(face_shape, "Bentuk wajah tidak ditemukan.")

# --- 7) Flask inference endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'photo' not in request.files:
        return jsonify({
            'error': True,
            'message': 'No image file part named "image"'
        }), 400

    # Read and preprocess image
    img_bytes = request.files['photo'].read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    x = transform(img).unsqueeze(0)  # Shape: [1, 3, 224, 224]

    # Forward pass and get probabilities
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]  # Get the 1D tensor

    # Get the top prediction (first one)
    top_class = torch.argmax(probs).item()  # Get the class with highest probability
    top_score = probs[top_class].item()  # Get the corresponding score

    # Build the response for the top prediction
    label = class_labels[top_class]
    result = {
        'class_index': top_class,
        'class_label': label,
        'confidence': top_score,
        'recommendation': recommend_hairstyle(label)
    }

    print(result)
    return jsonify({
        'error': False,
        'message': "Success",
        'top_prediction': result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
