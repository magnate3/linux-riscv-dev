import torch
import sys
import time, datetime
import os
import argparse
import function_project2 as fn
from function_project2 import DenseNet, LabelSmoothingCrossEntropy, TinyImageNetDataset
from torch.utils.data import DataLoader


class PrintLog:
    def __init__(self, filepath, mode="w"):
        self.file = open(filepath, mode)
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self.file.write(data)

    def flush(self):
        self._stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self._stdout
        self.file.close()


def start_print_logging(filepath="prints.log", mode="w"):
    logger = PrintLog(filepath, mode)
    sys.stdout = logger
    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for TinyImageNet DenseNet with adaptive num_workers."
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "Number of threads. "
            "default (=None): uses the maximum number of CPU. "
            "If you ask for more than the maximum number of CPU, it will be limited to the maximum. Same with negative values."
        ),
    )


    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size. Defaults (= 8) : uses a batch size of 8."
    )

    return parser.parse_args()


def main():
    logger = start_print_logging("output.txt")

    # -----------------------
    # Hyperparams / config
    # -----------------------
    number_of_iterations = 200000
    learning_rate = 1
    model_save_path = "./model_project2/"
    brestore = False
    restore_iter = 0
    if not brestore:
        restore_iter = 0

    # -----------------------
    # Parse args
    # -----------------------
    args = parse_args()

    # Retrieves the number of logical CPUs
    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 1  # fallback parano

    # If the user did not specify a number of workers, we use the maximum number of logical CPUs
    if args.num_workers is None:
        effective_num_workers = cpu_count
    else:
        # if user specified a negative number of workers, we use the maximum number of logical CPUs
        if args.num_workers < 0:
            effective_num_workers = 0
        #if the user said something, we take the minimum of the number of logical CPUs and the number of workers asked
        else:
            effective_num_workers = min(args.num_workers, cpu_count)

    print(f"CPU used : {cpu_count}")
    print(f"num_workers asked   : {args.num_workers}")
    print(f"num_workers used   : {effective_num_workers}")

    batch_size = args.batch_size
    print(f"batch_size used    : {batch_size}")

    # -----------------------
    # Dataset paths
    # -----------------------
    train_zip_path = "./dataset/train.zip"
    test_zip_path = "./dataset/test.zip"

    train_gt_path = "./dataset/train_gt.txt"
    test_gt_path = "./dataset/test_gt.txt"

    train_extract_path = "./dataset/train/"
    test_extract_path = "./dataset/test/"

    # -----------------------
    # Extraction if needed
    # -----------------------
    fn.extract_zip_if_needed(train_zip_path, train_extract_path)
    fn.extract_zip_if_needed(test_zip_path, test_extract_path)

    # -----------------------
    # loading GT
    # -----------------------
    print("Loading the ground truth...")
    train_gt = fn.load_ground_truth(train_gt_path)
    test_gt = fn.load_ground_truth(test_gt_path)
    print(f"Train GT: {len(train_gt)} labels")
    print(f"Test GT: {len(test_gt)} labels")

    # -----------------------
    # Mapping classes
    # -----------------------
    classes, num_classes = fn.build_class_mapping(train_gt, test_gt)

    # -----------------------
    # Datasets (lazy loading)
    # -----------------------
    print("Creating the datasets (lazy loading)...")
    train_dataset = TinyImageNetDataset(
        train_extract_path,
        train_gt,
        img_size=(128, 128),
        augment=True  # train augmentation
    )

    test_dataset = TinyImageNetDataset(
        test_extract_path,
        test_gt,
        img_size=(128, 128),
        augment=False  # no aug for test
    )

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    print(f"Using the device: {DEVICE}")

    # -----------------------
    # Dataloaders
    # -----------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=True if USE_CUDA else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=True if USE_CUDA else False
    )

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Test dataset: {len(test_dataset)} images")

    # -----------------------
    # Model / opti / loss
    # -----------------------
    model = DenseNet(
        growth_rate=12,
        block_config=(6, 12, 24, 16),
        num_classes=num_classes,
        compression=0.5,
        dropout_rate=0.2,
        num_init_features=64
    ).to(DEVICE)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4
    )

    loss = LabelSmoothingCrossEntropy(smoothing=0.1)

    start = time.time()

    # Iterateur infini sur le dataloader
    train_iter = iter(train_loader)

    for i in range(restore_iter, number_of_iterations):

        # scheduler manuel du LR
        if 100000 <= i < 150000:
            optimizer.param_groups[0]['lr'] = 0.5
        if 160000 <= i < 200000:
            optimizer.param_groups[0]['lr'] = 0.1

        # Batching
        try:
            batch_img, batch_cls = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_img, batch_cls = next(train_iter)

        batch_img = batch_img.to(DEVICE)
        batch_cls = batch_cls.to(DEVICE)

        # Forward / backward
        model.train()
        optimizer.zero_grad()
        pred = model(batch_img)

        train_loss = loss(pred, batch_cls)
        train_loss.backward()
        optimizer.step()

        # Logging / save / eval every 1000 iters
        if i % 1000 == 0:
            sec = (time.time() - start)
            result = str(datetime.timedelta(seconds=sec)).split(".")[0]
            print(f"iter {i} | loss {train_loss.item():.4f} | Time: {result}")
            start = time.time()

            # saving model
            if not os.path.isdir(model_save_path):
                os.makedirs(model_save_path)
            torch.save(model.state_dict(), os.path.join(model_save_path, f"model_{i}.pt"))

            # evaluation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_imgs, batch_targets in test_loader:
                    batch_imgs = batch_imgs.to(DEVICE)
                    batch_targets = batch_targets.to(DEVICE)

                    logits = model(batch_imgs)
                    pred_eval = torch.argmax(logits, dim=1)

                    total += batch_targets.size(0)
                    correct += (pred_eval == batch_targets).sum().item()

            acc = 100.0 * correct / total
            print(f"Accuracy = {acc:.2f}%")

    logger.close()

if __name__ == "__main__":
    main()

