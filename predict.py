import sys

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

MODEL_PATH = 'pneumonia_classifier.pth'
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model():
    model = models.resnet18(weights=None)  # architecture only, we load our own trained weights
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict(image_path, model):
    image = Image.open(image_path).convert('RGB')
    input_tensor = data_transform(image).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()

    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probabilities[predicted_idx].item()

    return predicted_class, confidence, probabilities.cpu().numpy()


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model()

    print(f"Running prediction on {image_path}...")
    predicted_class, confidence, all_probs = predict(image_path, model)

    print(f"\nPrediction: {predicted_class}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("\nFull breakdown:")
    for class_name, prob in zip(CLASS_NAMES, all_probs):
        print(f"  {class_name}: {prob * 100:.2f}%")

    if predicted_class == 'PNEUMONIA':
        print("\nNote: This is an educational model, not a diagnostic tool. "
              "Any real concern about pneumonia should be evaluated by a doctor.")


if __name__ == "__main__":
    main()
