import streamlit as st
from google import genai
from PIL import Image
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialise the new google-genai client
client = genai.Client(api_key=api_key)

st.set_page_config(page_title="Camera OCR", page_icon="📷")
st.title("📷 Camera OCR with Gemini")

picture = st.camera_input("Take a photo")

if picture:
    image = Image.open(picture)
    st.image(image, caption="Captured Image", width='stretch')

    if st.button("🔍 Extract Text"):
        with st.spinner("Sending image to Gemini…"):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        image,
                        "Extract all text visible in this image. Preserve line breaks and formatting exactly."
                    ]
                )

                st.subheader("Extracted Text")
                st.text_area("OCR Result", response.text, height=300)

                st.download_button(
                    label="📥 Download Text",
                    data=response.text,
                    file_name="extracted_text.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")