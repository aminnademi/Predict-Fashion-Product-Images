import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import json
from collections import OrderedDict
from groq import Groq

# ------------ CONFIG ------------
API_KEY        = "your_api_key"
MODEL_A_PATH   = "fashion_model1.pth"  # model A: full 7-task ResNet18
MODEL_B_PATH   = "fashion_model2.pth"  # model B: full 7-task ResNet50
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATTRS          = ['gender', 'masterCategory','subCategory','articleType', 'baseColour', 'season', 'usage']
ATTRS_A        = ['gender', 'baseColour']
ATTRS_B        = ['masterCategory','subCategory','articleType','season', 'usage']

TARGET_SIZE    = (60, 80)
MEAN           = [0.8540, 0.8365, 0.8305]
STD            = [0.2300, 0.2450, 0.2495]

client = Groq(api_key=API_KEY)

# ------------ MODEL CLASSES -----
class ResNet18MultiTask(nn.Module):
    def __init__(self, num_classes_list):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifiers = nn.ModuleList([nn.Linear(in_features, nc) for nc in num_classes_list])
    def forward(self, x):
        feats = self.backbone(x)
        return [clf(feats) for clf in self.classifiers]

class ResNet50MultiTask(nn.Module):
    def __init__(self, num_classes_list):
        super(ResNet50MultiTask, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.fc = nn.Identity()
        self.classifiers = nn.ModuleList([nn.Linear(2048, nc) for nc in num_classes_list])
    def forward(self, x):
        feats = self.backbone(x).view(x.size(0), -1)
        return [clf(feats) for clf in self.classifiers]

# ------------ LABEL MAPS --------
with open("label2idx.json") as f:
    label2idx = json.load(f)
idx2label = {a: {v: k for k, v in label2idx[a].items()} for a in ATTRS}

# ------------ TRANSFORMS --------
transform = T.Compose([
    T.Resize(TARGET_SIZE),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])

# ------------ LOAD MODELS -------
def _safe_load(model, path):
    sd = torch.load(path, map_location=DEVICE, weights_only=True)
    if any(k.startswith("module.") for k in sd):
        sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())
    model.load_state_dict(sd, strict=True)

@st.cache_resource
def load_models():
    # A: ResNet18 with 7 heads (we use only 2)
    num_all = [len(label2idx[a]) for a in ATTRS]
    model_a = ResNet18MultiTask(num_all, backbone_weights=None)
    _safe_load(model_a, MODEL_A_PATH)
    model_a.to(DEVICE).eval()

    # B: ResNet50 with 7 heads (we use only 5)
    model_b = ResNet50MultiTask(num_all)
    _safe_load(model_b, MODEL_B_PATH)
    model_b.to(DEVICE).eval()

    return model_a, model_b

model_a, model_b = load_models()

# ------------ PREDICT -----------
@torch.no_grad()
def predict(img: Image.Image):
    x = transform(img).unsqueeze(0).to(DEVICE)

    out_a = model_a(x)
    out_b = model_b(x)

    logits = {a: o for a, o in zip(ATTRS, out_a) if a in ATTRS_A}
    logits |= {a: o for a, o in zip(ATTRS, out_b) if a in ATTRS_B}

    tags = {}
    for attr, logit in logits.items():
        probs = torch.softmax(logit, dim=-1).cpu().squeeze()
        top   = probs.argmax().item()
        tags[attr] = {"label": idx2label[attr][top], "confidence": float(probs[top])}
    return tags

def caption(tags):
    return " ⟶ ".join(f"{a}: {d['label']} ({d['confidence']:.2f})" for a, d in tags.items())

# ------------ STREAMLIT UI ------
st.title("👕 Fashion Tagger & Stylist")

file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    tags = predict(img)
    st.subheader("Predicted Attributes")
    st.write(caption(tags))

    # Styling question input
    question = st.text_input("Ask a styling question (e.g. What goes well with this?)")
    if question:
        prompt = "I have a piece of clothing with these attributes:\n"
        for a, d in tags.items():
            prompt += f"- {a}: {d['label']} (conf: {d['confidence']:.2f})\n"
        prompt += f"\nUser: {question}\nAssistant:"

        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        st.subheader("Styling Advice")
        st.write(resp.choices[0].message.content)

    # Generate image caption from predicted tags
    if st.button("Generate Image Caption"):
        prompt = "Generate a short, creative and stylish caption for a fashion item with the following attributes:\n"
        for a, d in tags.items():
            prompt += f"- {a}: {d['label']}\n"
        prompt += "\nCaption:"

        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        st.subheader("Image Caption")
        st.write(resp.choices[0].message.content)