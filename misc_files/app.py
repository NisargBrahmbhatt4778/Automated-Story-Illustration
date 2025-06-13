import streamlit as st
import os
from PIL import Image
import shutil

# Set page config
st.set_page_config(
    page_title="Character Generator",
    page_icon="🎨",
    layout="wide"
)

# Title and description
st.title("Character Generator")
st.markdown("""
This app helps you generate character images using AI. Follow these steps:
1. Enter your character description
2. Upload your character sheet
3. Wait for processing
4. Generate new images of your character
""")

# Initialize session state
if 'character_description' not in st.session_state:
    st.session_state.character_description = ""
if 'character_name' not in st.session_state:
    st.session_state.character_name = ""

# Step 1: Character Description
st.header("Step 1: Character Description")
character_description = st.text_area(
    "Describe your character in detail:",
    height=150,
    placeholder="Example: A young female elf with silver hair, wearing a blue robe..."
)

if character_description:
    st.session_state.character_description = character_description
    # Extract character name from description (simple version)
    character_name = character_description.split()[0].lower()
    st.session_state.character_name = character_name

# Step 2: Character Sheet Upload
st.header("Step 2: Upload Character Sheet")
uploaded_file = st.file_uploader("Upload your character sheet (PNG or JPG)", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Create directory for character if it doesn't exist
    if st.session_state.character_name:
        os.makedirs(st.session_state.character_name, exist_ok=True)
        
        # Save the uploaded file
        file_path = os.path.join(st.session_state.character_name, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File uploaded successfully to {st.session_state.character_name} folder!")
        
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Character Sheet", use_column_width=True)

# Step 3: Processing Status
st.header("Step 3: Processing")
if uploaded_file is not None and st.session_state.character_description:
    if st.button("Process Character Sheet"):
        with st.spinner("Processing character sheet..."):
            # TODO: Add your character sheet processing logic here
            st.info("This is where the character sheet processing would happen")
            st.info("For now, this is a placeholder for the actual processing logic")

# Step 4: Generate New Images
st.header("Step 4: Generate New Images")
if st.session_state.character_description:
    new_prompt = st.text_area(
        "Enter a new prompt for your character:",
        height=100,
        placeholder="Example: The character standing in a forest..."
    )
    
    if st.button("Generate New Image"):
        with st.spinner("Generating new image..."):
            # TODO: Add your image generation logic here
            st.info("This is where the image generation would happen")
            st.info("For now, this is a placeholder for the actual generation logic")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit") 