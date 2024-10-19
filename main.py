# import streamlit as st
# import fal_client
# from groq import Groq
# import os
# import logging
# import time
# from datetime import datetime
#
#
# # Configure logging
# def setup_logger():
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.StreamHandler(),  # Console handler
#             logging.FileHandler(f'app_logs_{datetime.now().strftime("%Y%m%d")}.log')  # File handler
#         ]
#     )
#     return logging.getLogger(__name__)
#
#
# logger = setup_logger()
#
#
# # Initialize APIs
# def init_groq():
#     logger.info("Initializing Groq API client")
#     api_key = os.getenv('GROQ_API_KEY')
#     if not api_key:
#         logger.error("GROQ_API_KEY not found in environment variables")
#         raise ValueError("GROQ_API_KEY not found")
#     return Groq(api_key=api_key)
#
#
# def init_fal():
#     logger.info("Initializing FAL API client")
#     api_key = os.getenv('FAL_KEY')
#     if not api_key:
#         logger.error("FAL_KEY not found in environment variables")
#         raise ValueError("FAL_KEY not found")
#     fal_client.api_key = api_key
#
#
# # Text to Roman Urdu conversion
# def convert_to_roman_urdu(text):
#     try:
#         logger.info(f"Starting Roman Urdu conversion for text of length: {len(text)}")
#         start_time = time.time()
#
#         client = init_groq()
#         logger.debug("Making API call to Groq")
#         completion = client.chat.completions.create(
#             model="llama3-8b-8192",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": f"convert the following into roman urdu: {text}"
#                 }
#             ],
#             temperature=1,
#             max_tokens=1024,
#         )
#
#         elapsed_time = time.time() - start_time
#         logger.info(f"Conversion completed in {elapsed_time:.2f} seconds")
#         return completion.choices[0].message.content
#     except Exception as e:
#         logger.error(f"Error in Roman Urdu conversion: {str(e)}", exc_info=True)
#         st.error(f"Groq API error: {str(e)}")
#         return None
#
#
# # Image generation
# def generate_image(prompt):
#     try:
#         logger.info(f"Starting image generation for prompt: {prompt}")
#         start_time = time.time()
#
#         def on_queue_update(update):
#             if isinstance(update, fal_client.InProgress):
#                 for log in update.logs:
#                     logger.info(f"FAL Progress: {log['message']}")
#                     st.write(log["message"])
#
#         logger.debug("Making API call to FAL")
#         result = fal_client.subscribe(
#             "fal-ai/flux/schnell",
#             arguments={
#                 "prompt": prompt,
#                 "image_size": "landscape_4_3",
#                 "num_inference_steps": 4,
#                 "num_images": 1,
#                 "enable_safety_checker": True
#             },
#             with_logs=True,
#             on_queue_update=on_queue_update,
#         )
#
#         elapsed_time = time.time() - start_time
#         logger.info(f"Image generation completed in {elapsed_time:.2f} seconds")
#         logger.debug(f"Generated image URL: {result['images'][0]['url']}")
#         return result["images"][0]["url"]
#     except Exception as e:
#         logger.error(f"Error in image generation: {str(e)}", exc_info=True)
#         st.error(f"FAL API error: {str(e)}")
#         return None
#
#
# # Streamlit UI
# def main():
#     logger.info("Starting application")
#     st.title("AI Converter")
#
#     # Mode selection
#     mode = st.radio("Select Mode", ["Text to Roman Urdu", "Text to Image"])
#     logger.info(f"Mode selected: {mode}")
#
#     if mode == "Text to Roman Urdu":
#         st.header("Text to Roman Urdu Converter")
#         input_text = st.text_area("Enter text to convert:")
#
#         if st.button("Convert"):
#             if input_text:
#                 logger.info("Starting text conversion process")
#                 with st.spinner("Converting..."):
#                     result = convert_to_roman_urdu(input_text)
#                     if result:
#                         logger.info("Text conversion successful")
#                         st.success("Conversion Complete!")
#                         st.write("Result:")
#                         st.write(result)
#                     else:
#                         logger.warning("Text conversion returned no result")
#
#     else:  # Text to Image mode
#         st.header("Text to Image Generator")
#         image_prompt = st.text_area("Enter image prompt:")
#
#         if st.button("Generate"):
#             if image_prompt:
#                 logger.info("Starting image generation process")
#                 with st.spinner("Generating image..."):
#                     image_url = generate_image(image_prompt)
#                     if image_url:
#                         logger.info("Image generation successful")
#                         st.success("Image Generated!")
#                         st.image(image_url)
#                     else:
#                         logger.warning("Image generation returned no result")
#
#
# if __name__ == "__main__":
#     try:
#         st.set_page_config(page_title="AI Converter", layout="wide")
#         logger.info("Application configuration set")
#         main()
#         logger.info("Application ended successfully")
#     except Exception as e:
#         logger.critical(f"Application crashed: {str(e)}", exc_info=True)
#         st.error("An unexpected error occurred. Please check the logs for details.")
import streamlit as st
import fal_client
from groq import Groq
# from transformers import pipeline
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
    # else:
        # Backup using HuggingFace model
        # try:
        #     # translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ur")
        #     # result = translator(text)[0]['translation_text']
        #     # return result
        # except Exception as e:
        #     st.error(f"HuggingFace model error: {str(e)}")
        #     return None


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
        print(result)
        print(result.keys())
        # result {'images': [{'url': 'https://fal.media/files/lion/JM7zNERJLvUecmzVvsO4X.png', 'width': 1024, '
        # height': 768, 'content_type': 'image/jpeg'}], 'timings': {'inference': 0.34186329087242484}, 'seed': 2577146647, 'has_nsfw_concepts': [False], 'prompt': 'horse'}
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
        use_backup = st.checkbox("Use backup model")

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