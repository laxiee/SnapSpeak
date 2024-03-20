import pyautogui
import keyboard
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
from PIL import Image
import matplotlib.pyplot as plt
from gtts import gTTS
import pygame
import tempfile
import io

# Initialize the model and tokenizer
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

def generate_caption(image, model=model, greedy=True):

    width, height = image.size
    left = width * 0.1
    top = height * 0.1
    right = width * 0.9
    bottom = height * 0.9
    central_image = image.crop((left, top, right, bottom))

    pixel_values = image_processor(images=central_image, return_tensors="pt").pixel_values
    # plt.imshow(central_image)
    # plt.show()

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

def screenshot_and_caption(key='f1'):
    print(f"Press {key} to capture the screen and generate a caption.")
    
    while True:
        if keyboard.is_pressed(key):
            print("Capturing screen...")
            screenshot = pyautogui.screenshot()
            generate_caption(screenshot, greedy=False)
            break
    screenshot_and_caption()


if __name__ == "__main__":
    screenshot_and_caption()
