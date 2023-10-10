from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

import gradio as gr

def clip_classifier(img, text1, text2, text3, text4):
    input = []
    if(text1 != ""):
        input.append(text1)
    if(text2 != ""):
        input.append(text2)
    if(text3 != ""):
        input.append(text3)
    if(text4 != ""):
        input.append(text4)
    inputs = processor(text=input, images=img, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    output = {}
    for i, text in enumerate(input):
        output[text] = probs[0][i].item()
    return output

demo = gr.Interface(
    fn=clip_classifier,
    inputs=[
        gr.Image(type= "pil", shape=(512, 512), image_mode="RGB", label= "Input Image"),
        gr.Textbox(lines=1, placeholder="Text 1..."),
        gr.Textbox(lines=1, placeholder="Text 2..."),
        gr.Textbox(lines=1, placeholder="Text 3..."),
        gr.Textbox(lines=1, placeholder="Text 4...")],
        outputs=gr.Label(),
        examples = [["women.png", "women wearing a scarf", "man sitting on a chair", "snake crawling on a road", "black and white picture"],
               ],
        description="OpenAI CLIP image classifier"
)
demo.launch(debug = True)