import os
import streamlit as st

st.write("CURRENT WORKING DIRECTORY:")
st.write(os.getcwd())

st.write("FILES IN DIRECTORY:")
st.write(os.listdir())



import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# ---------------- CONFIG ----------------
MODEL_PATH = "brain_tumor_resnet18_cpu.pth"
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
DEVICE = torch.device("cpu")  # model was trained on CPU

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ---------------- TRANSFORMS (MUST MATCH TRAINING) ----------------
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# ---------------- STREAMLIT UI ----------------
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI image to classify the tumor type")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", width=300)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    st.subheader("Prediction")
    st.success(f"{CLASSES[pred_idx]}")

    st.subheader("Class Probabilities")
    for i, cls in enumerate(CLASSES):
        st.write(f"{cls}: {probs[i].item():.4f}")
