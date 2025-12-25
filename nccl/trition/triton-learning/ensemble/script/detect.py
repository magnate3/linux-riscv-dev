from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests


def download_model():
    # Check if CUDA is available and set the device accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    # Load the pre-trained DETR model and image processor
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm"
    )
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm"
    ).to(device)
    return model, processor


def download_image():
    # Download an image from the internet
    url = "http://images.cocodataset.org/val2017/000000439715.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image.save("detection_sample_image.jpg")
    return image


def main():
    model, processor = download_model()
    image = download_image()
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.9
    )[0]

    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} "
            f"with confidence {round(score.item(), 3)} at location {box}"
        )


if __name__ == "__main__":
    print("Object detection example...")
    #main()
    download_image()
