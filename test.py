import os 
import torch
from torchvision import models, transforms
from PIL import Image

# Use GPU if available, else fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
vgg19 = models.vgg19(pretrained=False)
vgg19.classifier[6] = torch.nn.Linear(4096, 5)  # 5 classes
vgg19.load_state_dict(torch.load('face_model.pth', map_location=device))
vgg19.to(device)

#Daftar kelas (pastikan sesuai dengan urutan pelatihan)
class_names = ['heart', 'oblong', 'oval', 'round', 'square']

# Rekomendasi Model Rambut Berdasarkan Bentuk Wajah
def recommend_hairstyle(face_shape):
    recommendations = {
        'heart': "test heart",
        'oblong': "test oblong",
        'oval': "test oval",
        'round': "test round",
        'square': "test square"
    }
    return recommendations.get(face_shape, "Bentuk wajah tidak ditemukan.")

# Fungsi Prediksi untuk Gambar Baru
def predict_face_shape(image_path, model, class_names):
    # Preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

# Contoh Prediksi
image_path = 'sample/heart.png'  # Ganti dengan path gambar wajah
predicted_shape = predict_face_shape(image_path, vgg19, class_names)
print(f"Predicted Face Shape: {predicted_shape}")
print(f"Recommended Hairstyle: {recommend_hairstyle(predicted_shape)}")