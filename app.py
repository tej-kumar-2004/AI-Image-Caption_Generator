import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


# -----------------------------------------
# Page Configuration
# -----------------------------------------

st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)


# -----------------------------------------
# Title
# -----------------------------------------

st.title("🖼️ AI Image Caption Generator")

st.write(
    "Upload an image and the AI will analyze it "
    "and generate a meaningful description."
)


# -----------------------------------------
# Load Pre-trained BLIP Model
# -----------------------------------------

@st.cache_resource
def load_model():

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    return processor, model


# Load model
processor, model = load_model()


# -----------------------------------------
# Image Upload
# -----------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)


# -----------------------------------------
# Process Image
# -----------------------------------------

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Your Uploaded Image",
        use_container_width=True
    )


    # Generate Caption Button
    if st.button("✨ Generate Caption"):

        with st.spinner("AI is analyzing the image..."):

            # Prepare image for model
            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            # Generate caption
            output = model.generate(
                **inputs,
                max_new_tokens=50
            )

            # Convert output to text
            caption = processor.decode(
                output[0],
                skip_special_tokens=True
            )


        # Display caption
        st.subheader("🤖 Generated Caption")

        st.success(caption.capitalize())


else:

    st.info("Please upload an image to generate a caption.")