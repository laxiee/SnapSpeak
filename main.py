import pyautogui
import keyboard
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
from PIL import Image
import matplotlib.pyplot as plt
from gtts import gTTS
import pygame
import tempfile
import io

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

pygame.mixer.init()

def speak_text(text):
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts = gTTS(text=text, lang='en')
        tts.save(f"{fp.name}.mp3")
        pygame.mixer.music.load(f"{fp.name}.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  
            pygame.time.Clock().tick(10)

from PIL import Image
import torch
from torchvision import transforms

def custom_image_preprocessor(image, target_size=(224, 224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """ Manually preprocess the image to be compatible with ViT models. """
    
    # Ensure image is in RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(target_size),  # Resize image to match model input
        transforms.ToTensor(),           # Convert image to tensor
        transforms.Normalize(mean, std)  # Normalize pixel values
    ])

    # Apply transformations
    tensor_image = transform(image)
    
    # Add batch dimension since models typically expect batches
    tensor_image = tensor_image.unsqueeze(0)
    
    return tensor_image

def generate_caption(image, model=model, greedy=True):


    pixel_values = custom_image_preprocessor(image)
    #image_processor(images=image, return_tensors="pt").pixel_values

    if greedy:
        generated_ids = model.generate(pixel_values, max_length=50, num_beams=5, early_stopping=True)
    else:
        generated_ids = model.generate(
            pixel_values,
            do_sample=True,
            max_length=50,
            top_k=5)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(generated_text)
    speak_text(generated_text)
    return generated_text

# def screenshot_and_caption(key='f1'):
#     print(f"Press {key} to capture the screen and generate a caption.")
    
#     while True:
#         if keyboard.is_pressed(key):
#             print("Capturing screen...")
#             screenshot = pyautogui.screenshot()
#             generate_caption(screenshot, greedy=False)
#             break
#     screenshot_and_caption()


# if __name__ == "__main__":
#     screenshot_and_caption()

from flask import Flask, request, jsonify
from io import BytesIO
import base64

app = Flask(__name__)


@app.route('/caption', methods=['GET', 'POST'])
def caption_image():
    if request.method == 'POST':
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        image_data = base64.b64decode(data['image'])
        image = Image.open(BytesIO(image_data))
        caption = generate_caption(image, greedy=False)
        print(jsonify({'caption': caption}))
        return jsonify({'caption': caption})
    elif request.method == 'GET':
        # This is just for debugging!
        return "This route expects a POST request with image data."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)