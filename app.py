import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import datetime

# --- Configuration & Metadata ---

# Actual classes from the dataset (alphabetically sorted as per dataset.py):
# 0: Damaged concrete structures
# 1: DamagedElectricalPoles  
# 2: DamagedRoadSigns
# 3: DeadAnimalsPollution
# 4: FallenTrees
# 5: Garbage
# 6: Graffitti
# 7: IllegalParking
# 8: Potholes and RoadCracks

URBAN_ISSUES_METADATA = {
    0: {"label": "Damaged Concrete Structures", "priority": "Medium", "dept": "Civil Infrastructure", "action": "Structural inspection required", "color": "orange"},
    1: {"label": "Damaged Electrical Poles", "priority": "High", "dept": "Power & Utilities", "action": "EMERGENCY: Cut power and repair", "color": "red"},
    2: {"label": "Damaged Road Signs", "priority": "High", "dept": "Traffic Control", "action": "Replace signage immediately", "color": "red"},
    3: {"label": "Dead Animals Pollution", "priority": "Low", "dept": "Sanitation Bureau", "action": "Specialized removal request", "color": "green"},
    4: {"label": "Fallen Trees", "priority": "High", "dept": "Parks & Recreation", "action": "Dispatch tree removal unit", "color": "red"},
    5: {"label": "Garbage", "priority": "Medium", "dept": "Sanitation Bureau", "action": "Schedule cleanup", "color": "orange"},
    6: {"label": "Graffiti", "priority": "Low", "dept": "Urban Maintenance", "action": "Log for graffiti removal", "color": "green"},
    7: {"label": "Illegal Parking", "priority": "Low", "dept": "Traffic Enforcement", "action": "Issue citation ticket", "color": "green"},
    8: {"label": "Potholes and Road Cracks", "priority": "High", "dept": "Roads & Transit", "action": "Dispatch rapid repair crew", "color": "red"},
}

MODEL_PATH = "best_model.pth"

# --- Model Loading ---

@st.cache_resource
def load_model():
    """Loads the ResNet18 model with modified final layer."""
    try:
        model = models.resnet18(weights=None) # Load structure only
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 9) # 9 classes in the dataset
        
        # Load weights
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at {MODEL_PATH}. Please ensure the model weights are present.")
            return None
            
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        # Fix for state dict mismatch (strip 'backbone.' prefix if present)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                new_state_dict[k.replace("backbone.", "")] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

@st.cache_resource
def get_transforms():
    """Returns the standard ImageNet validation transforms."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# --- UI & Logic ---

def main():
    st.set_page_config(page_title="Smart Urban Triage Assistant", page_icon="🏙️", layout="wide")
    
    st.title("🏙️ Smart Urban Triage Assistant")
    st.markdown("### Intelligent Dispatch & Priority System")
    st.markdown("---")

    # --- Sidebar ---
    st.sidebar.header("Input")
    uploaded_file = st.sidebar.file_uploader("Upload an image of the urban issue", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display Image
        image = Image.open(uploaded_file).convert('RGB')
        st.sidebar.image(image, caption="Uploaded Image", width="stretch")
        
        # Load Model and Run Inference
        with st.spinner('🔍 Loading model and analyzing image...'):
            model = load_model()
            if model is None:
                return

            preprocess = get_transforms()
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0) # Create mini-batch

            with torch.no_grad():
                output = model(input_batch)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                confidence, predicted_class_idx = torch.max(probabilities, 0)
                
        class_id = predicted_class_idx.item()
        conf_score = confidence.item()
        
        metadata = URBAN_ISSUES_METADATA.get(class_id, {})
        label = metadata.get("label", "Unknown")
        priority = metadata.get("priority", "Unknown")
        dept = metadata.get("dept", "Unknown")
        action = metadata.get("action", "Unknown")
        color = metadata.get("color", "blue")

        # --- Main Panel ---
        
        # Section 1: AI Analysis
        st.subheader("1. AI Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Class", label)
        with col2:
            st.metric("Confidence Score", f"{conf_score:.2%}")
            
        st.divider()

        # Section 2: Triage Dashboard
        st.subheader("2. Triage Dashboard")
        
        # Determine SLA based on priority (mock logic)
        if priority == "High":
            sla = "24 Hours"
            status_color = "inverse" # Streamlit doesn't have direct color for metric, but we can use st.error/warning/info containers
        elif priority == "Medium":
            sla = "3 Days"
            status_color = "normal"
        else:
            sla = "1 Week"
            status_color = "off"

        # Display metrics with color coding using containers
        c1, c2, c3 = st.columns(3)
        
        with c1:
            if color == "red":
                st.error(f"**Priority Level**\n\n# {priority}")
            elif color == "orange":
                st.warning(f"**Priority Level**\n\n# {priority}")
            else:
                st.info(f"**Priority Level**\n\n# {priority}")
                
        with c2:
            st.info(f"**Assigned Dept**\n\n# {dept}")
            
        with c3:
            st.success(f"**Estimated SLA**\n\n# {sla}")

        st.divider()

        # Section 3: Auto-Generated Report
        st.subheader("3. Auto-Generated Report")
        
        report_date = datetime.date.today().strftime("%Y-%m-%d")
        report_text = f"""To: {dept}
From: Smart Urban Triage System
Date: {report_date}

Subject: Triage Alert - {label} Detected

System has detected a {label} issue with {priority.upper()} priority.

Details:
- Issue Type: {label}
- Confidence: {conf_score:.2%}
- Suggested Action: {action}

Please proceed with {action} within {sla}.
"""
        st.code(report_text, language="text")
        st.caption("Copy the above report for dispatch.")

    else:
        st.info("👋 Welcome! Please upload an image in the sidebar to begin analysis.")

if __name__ == "__main__":
    main()
