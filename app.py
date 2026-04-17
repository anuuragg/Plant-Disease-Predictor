import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import json

#labels
with open("model/labels.json", "r") as f:
    labels = json.load(f)

num_classes = len(labels)

#model
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 38)
    model.load_state_dict(torch.load("model/plant_disease_model.pt", map_location="cpu"))
    model.eval()
    return model.cpu()

model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# UI
st.title("🌿 Plant Disease Predictor")
st.write("Upload a leaf image to detect disease.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top3 = torch.topk(probs, 3)

    st.subheader("Results")
    for i in range(3):
        idx = int(top3.indices[i].item())
        label = labels[idx].replace("_", " ") if idx < len(labels) else f"Class {idx}"
        confidence = top3.values[i].item() * 100
        st.write(f"**{label}** — {confidence:.1f}%")