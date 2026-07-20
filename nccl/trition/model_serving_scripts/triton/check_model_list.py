import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")
models = client.get_model_repository_index()
print(models)

