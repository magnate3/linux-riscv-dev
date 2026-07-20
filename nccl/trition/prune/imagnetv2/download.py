from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="vaishaal/ImageNetV2",
    filename="imagenetv2-matched-frequency.tar.gz",
    repo_type="dataset",
    local_dir="./imagenetv2"
)
print(f"download to: {file_path}")
