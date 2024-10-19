import streamlit as st
import fal_client
from groq import Groq
from transformers import pipeline  # Import the transformers library for Hugging Face
import os

# Initialize APIs
def init_groq():
    return Groq(api_key=os.getenv('GROQ_API_KEY'))

def init_fal():
    fal_client.api_key = os.getenv('FAL_KEY')

# Text to Roman Urdu conversion
def convert_to_roman_urdu(text, use_backup=False):
    if not use_backup:
        try:
            client = init_groq()
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {
                        "role": "user",
                        "content": f"convert the following into roman urdu: {text}"
                    }
                ],
                temperature=1,
                max_tokens=1024,
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.error(f"Groq API error: {str(e)}")
            return None
    else:
        # Backup using Hugging Face model
        try:
            translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ur")
            result = translator(text)[0]['translation_text']
            return result
        except Exception as e:
            st.error(f"HuggingFace model error: {str(e)}")
            return None

# Image generation
def generate_image(prompt):
    try:
        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    st.write(log["message"])

        result = fal_client.subscribe(
            "fal-ai/flux/schnell",
            arguments={
                "prompt": prompt,
                "image_size": "landscape_4_3",
                "num_inference_steps": 4,
                "num_images": 1,
                "enable_safety_checker": True
            },
            with_logs=True,
            on_queue_update=on_queue_update,
        )
        return result["images"][0]["url"]
    except Exception as e:
        st.error(f"FAL API error: {str(e)}")
        return None

# Streamlit UI
def main():
    st.title("AI Converter")

    # Mode selection
    mode = st.radio("Select Mode", ["Text to Roman Urdu", "Text to Image"])

    if mode == "Text to Roman Urdu":
        st.header("Text to Roman Urdu Converter")
        input_text = st.text_area("Enter text to convert:")
        use_backup = st.checkbox("Use backup model(hugging face)")

        if st.button("Convert"):
            if input_text:
                with st.spinner("Converting..."):
                    result = convert_to_roman_urdu(input_text, use_backup)
                    if result:
                        st.success("Conversion Complete!")
                        st.write("Result:")
                        st.write(result)

    else:  # Text to Image mode
        st.header("Text to Image Generator")
        image_prompt = st.text_area("Enter image prompt:")

        if st.button("Generate"):
            if image_prompt:
                with st.spinner("Generating image..."):
                    image_url = generate_image(image_prompt)
                    if image_url:
                        st.success("Image Generated!")
                        st.image(image_url)

if __name__ == "__main__":
    st.set_page_config(page_title="AI Converter", layout="wide")
    main()
