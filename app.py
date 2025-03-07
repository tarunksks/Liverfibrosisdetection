from flask import Flask, render_template, request
import torch
from PIL import Image
import torchvision.transforms as transforms
import pickle

app = Flask(__name__)

# Load the model from the .pkl file
with open('liver_fibrosis_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set the model to evaluation mode
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        
        # Get the uploaded image
        file = request.files["file"]
        if file:
            image = Image.open(file)
            
            # Preprocess the image
            img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

            # Make prediction
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                fibrosis_stage = predicted.item()

            return render_template("index.html", result=f"Predicted Fibrosis Stage: F{fibrosis_stage}")
    return render_template("index.html", result="")

if __name__ == "__main__":
    app.run(debug=True)
