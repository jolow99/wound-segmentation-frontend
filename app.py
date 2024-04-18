import streamlit as st
import os
import random
from PIL import Image
from models.models import get_model
import argparse 
import torch
import numpy as np

argparser = argparse.ArgumentParser(description="ADL Deep Learning Project")
argparser.add_argument("--data", type=str, default="../data/azh_wound_care_center_dataset_patches", help="Path to data")
argparser.add_argument("--device", type=str, default="mps", help="Device to use for training")
argparser.add_argument("--logdir", type=str, default="../logs", help="Path to save results")
argparser.add_argument("--checkpoint_path", type=str, default="../checkpoints", help="Path to save model")
argparser.add_argument("--batch_size", type=int, default=1, help="Batch size")

# Set the path to the training images and labels folders
train_images_path = "data/wound_dataset/azh_wound_care_center_dataset_patches/train/images"
train_labels_path = "data/wound_dataset/azh_wound_care_center_dataset_patches/train/labels"

# Get the list of image files in the training folder
image_files = [f for f in os.listdir(train_images_path) if f.endswith(".png")]

# Initialize session state variables
if "selected_images" not in st.session_state:
    st.session_state.selected_images = random.sample(image_files, 3)
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

# Display the selected images and allow the user to choose one
st.title("Image Selection")
st.write("Select one of the following images:")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Select", key="image1"):
        st.session_state.selected_image = st.session_state.selected_images[0]
    img1 = Image.open(os.path.join(train_images_path, st.session_state.selected_images[0]))
    st.image(img1, caption=st.session_state.selected_images[0], use_column_width=True)

with col2:
    if st.button("Select", key="image2"):
        st.session_state.selected_image = st.session_state.selected_images[1]
    img2 = Image.open(os.path.join(train_images_path, st.session_state.selected_images[1]))
    st.image(img2, caption=st.session_state.selected_images[1], use_column_width=True)

with col3:
    if st.button("Select", key="image3"):
        st.session_state.selected_image = st.session_state.selected_images[2]
    img3 = Image.open(os.path.join(train_images_path, st.session_state.selected_images[2]))
    st.image(img3, caption=st.session_state.selected_images[2], use_column_width=True)

# Display the selected image and its label in two columns
if st.session_state.selected_image is not None:
    st.write("You selected:")
    
    # Get the label file path
    label_file = st.session_state.selected_image
    label_path = os.path.join(train_labels_path, label_file)
    
    # Display the image and label in two columns
    col1, col2 = st.columns(2)
    with col1:
        selected_img = Image.open(os.path.join(train_images_path, st.session_state.selected_image))
        st.image(selected_img, caption=st.session_state.selected_image, use_column_width=True)
    with col2:
        label_img = Image.open(label_path)
        st.image(label_img, caption="Label", use_column_width=True)

    # Let user select a model
    models = ["unet", "autoencoder", "circlenet", "mixcirclenet", "pix2pix", "deeplabv3plus", "segnet", "fcn", "segformer"]
    model_weights = {
    "unet": "unet_unet/20240417_031427",
    "autoencoder": "autoencoder_autoencoder/20240417_222558",
    "circlenet": "circlenet_circlenet/20240417_044752",
    "mixcirclenet": "mixcirclenet_mixcirclenet/20240417_221609",
    "pix2pix": "pix2pix_pix2pix/20240417_044752",
    "deeplabv3plus": "deeplabv3plus_deeplabv3plus/20240417_205119",
    "segnet": "segnet_segnet/20240417_212642",
    "fcn": "fcn_fcn/20240417_232501",
    "segformer": "segformer_segformer/20240417_124908"
    }
    model_name = st.selectbox("Select a model", models)

        # Button to load and run the model
    if st.button("Run Model"):
        # Load the selected model
        args = argparser.parse_args()
        model = get_model(model_name, vars(args), device=args.device)
        model.load_state_dict(torch.load('checkpoints/' + model_weights[model_name] + '/best_model.pth'))
        model.eval()
        
        # Preprocess the selected image
        input_image = selected_img.convert("RGB")  # Convert to RGB if needed
        input_tensor = torch.from_numpy(np.array(input_image)).float() / 255.0  # Convert to tensor and normalize
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(args.device)  # Reshape to (1, C, H, W)
        
        # Run the model on the selected image
        with torch.no_grad():
            output_tensor = model(input_tensor).to(args.device)
        
        # Apply sigmoid activation to the output tensor
        output_tensor = torch.sigmoid(output_tensor)
        
        # Postprocess the model output
        if output_tensor.shape[1] == 1:
            # Single-channel output (grayscale)
            output_image = Image.fromarray((output_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8))
        else:
            # Multi-channel output (e.g., RGB)
            output_image = Image.fromarray((output_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        
        # Display the model output
        st.write("Model output:")
        st.image(output_image, caption="Model Output", use_column_width=True)
else:
    st.write("No image selected.")