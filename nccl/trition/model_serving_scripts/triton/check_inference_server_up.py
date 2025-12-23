import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")
model_metadata = client.get_model_metadata("yolo_v4")
#model_metadata = client.get_model_metadata("resnet18")
print(model_metadata)

