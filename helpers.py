import torch
import clip
from PIL import Image

def instantiate_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load("ViT-B/32", device=device)

    print("Model instantiated.")

    return model, preprocess, device

def generate_text_embeddings(inputs, model, device): # inputs is a list of strings
    text = clip.tokenize(inputs).to(device)

    with torch.no_grad():
        embeddings = model.encode_text(text)

    return embeddings

def upload_image(file, filename):
    with open (f"images/{filename}", "wb") as buffer:
        buffer.write(file.read())

    print(f"File uploaded to images/{filename}.")

def generate_image_embeddings(filepaths, model, preprocess, device): 
    embeddings = []
    for filepath in filepaths:
        image = preprocess(Image.open(filepath)).unsqueeze(0).to(device)

        with torch.no_grad():
            embeddings.append(model.encode_image(image))

    return embeddings

