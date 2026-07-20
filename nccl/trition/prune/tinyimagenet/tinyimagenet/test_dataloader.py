from tinyimagenet import TinyImageNet
from torchvision import transforms as T
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
splits = ["val","test","train"]


normalize_transform = T.Compose([ T.ToTensor(),
                                 T.Normalize(mean=TinyImageNet.mean,
                            std=TinyImageNet.std),
                             # Converting cropped images to tensors
])
transform = T.Compose([ T.Resize(256), # Resize images to 256 x 256
                T.CenterCrop(224), # Center crop image
                T.RandomHorizontalFlip(),
                normalize_transform
                ])


for split in splits:

    dataset = TinyImageNet(Path("~/.torchvision/tinyimagenet/"),split=split,imagenet_idx=False,transform=transform)
    dataloader = DataLoader(dataset,batch_size=32)
    print(f"TinyImageNet, split {split}, loading all samples with dataloader:")
    

    for images,klasses in tqdm(dataloader):
        for klass in klasses:
            klass = klass.item()
            id = dataset.idx_to_class[klass]
            
            imagenet_klass = dataset.class_to_imagenet_idx[id]

        # print(f"Sample of class {klass:3d} (imagenet {imagenet_klass}), image {image}, words {dataset.idx_to_words[klass]}")
        




