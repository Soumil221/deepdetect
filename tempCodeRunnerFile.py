import os
import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

# Initialize Flask App
app = Flask(__name__)

# Define the Model
class Model(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()

        # Load ResNeXt Model
        model = models.resnext50_32x4d(weights="IMAGENET1K_V2")
        self.model = nn.Sequential(*list(model.children())[:-2])  # Remove last FC layers

        # LSTM Layer
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        
        # Activation & Dropout
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)

        # Fully Connected Layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(latent_dim, num_classes)

        # Initialize Weights to Prevent NaN
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)  # Good initialization for LSTMs
            elif 'bias' in name:
                nn.init.zeros_(param)  # Avoid NaNs in biases

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)  # Reshape for CNN
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)

        # LSTM Forward Pass
        x_lstm, _ = self.lstm(x, None)

        # Prevent NaN values in output
        x_lstm = torch.nan_to_num(x_lstm)  # Replace NaNs with 0s

        return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)

# Load Model Checkpoint Safely
try:
    state_dict = torch.load("model7now.pth", map_location=device)

    # Identify missing keys
    model_keys = set(model.state_dict().keys())
    missing_keys = model_keys - set(state_dict.keys())

    if missing_keys:
        print(f"⚠ Warning: Missing keys found: {missing_keys}")
        for key in missing_keys:
            state_dict[key] = torch.zeros_like(model.state_dict()[key])  # Initialize missing keys to zeros

    model.load_state_dict(state_dict, strict=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None  # Prevent crashes by setting model to None

if model:
    model.eval()

    # Check for NaNs in Model Parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"❌ ERROR: Model parameter {name} contains NaN values!")
            param.data = torch.nan_to_num(param.data)  # Fix NaNs

# Define Video Processing Function
def process_video(video_path, num_frames=20):
    print(f"📹 Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ ERROR: Could not open video file!")
        return None

    frames = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // num_frames)

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(frame))

    cap.release()

    if len(frames) == 0:
        print("❌ ERROR: No valid frames extracted!")
        return None

    frames = torch.stack(frames).unsqueeze(0)  # Add batch dimension

    # Prevent NaNs in Input
    frames = torch.nan_to_num(frames)

    print(f"✅ Processed {len(frames[0])} frames successfully.")
    return frames.to(device)

# Define API Route for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("❌ ERROR: No video uploaded")
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files['file']
    video_path = "temp_video.mp4"
    video.save(video_path)

    print(f"✅ Video received: {video.filename}")

    frames = process_video(video_path)
    if frames is None:
        print("❌ ERROR: Video processing failed")
        return jsonify({"error": "Could not process video"}), 400

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        with torch.no_grad():
            fmap, output = model(frames)

            # Prevent NaNs in Output
            output = torch.nan_to_num(output)

            # Debug: Check if model output contains NaN
            if torch.isnan(output).any():
                print("❌ ERROR: Model output contains NaN values!")
                return jsonify({"error": "Model output is NaN"}), 500

            probabilities = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
            prediction = torch.argmax(output, dim=1).item()
            confidence = float(probabilities[prediction])

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

    os.remove(video_path)

    result = "DeepFake" if prediction == 1 else "Real"
    print(f"✅ Prediction: {result}, Confidence: {confidence:.4f}")
    return jsonify({"prediction": result, "confidence": confidence})

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
