import os
import torch
import clip
import faiss
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)


def find_all_images(root_dir="/"):
    """Recursively finds all image files on the system."""
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths


def index_images():
    """Indexes all images found in the system using CLIP."""
    image_paths = find_all_images()
    image_features = []

    if not image_paths:
        print("No images found on the system.")
        return None, None

    for img_path in tqdm(image_paths, desc="Indexing Images"):
        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = model.encode_image(image)
                image_features.append(feature.cpu().numpy())
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    image_features = np.vstack(image_features)
    index = faiss.IndexFlatL2(image_features.shape[1])
    index.add(image_features)
    return index, image_paths


def search_images_by_text(query_text, index, image_paths, top_k=5):
    """Search for images based on text description."""
    text = clip.tokenize([query_text]).to(device)
    with torch.no_grad():
        text_feature = model.encode_text(text).cpu().numpy()

    distances, indices = index.search(text_feature, top_k)

    print("\nTop Matches:")
    for i, idx in enumerate(indices[0]):
        print(f"{i+1}. {image_paths[idx]} (Score: {distances[0][i]:.2f})")


def main():
    parser = argparse.ArgumentParser(description="AI-Powered Image Search")
    parser.add_argument(
        "--query_text", type=str, required=True, help="Text description for search"
    )
    args = parser.parse_args()

    print("Scanning and Indexing Images...")
    index, image_paths = index_images()
    if index is None:
        return

    search_images_by_text(args.query_text, index, image_paths)


if __name__ == "__main__":
    main()
