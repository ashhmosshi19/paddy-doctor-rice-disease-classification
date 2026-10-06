from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models


DEFAULT_CHECKPOINT = Path(
    "reports/suggested_runs/image_only_fix/densenet_gpu_no_weights/"
    "best_densenet121_gpu_no_weights.pt"
)
DEFAULT_TEST_DIR = Path("data/test_images")
DEFAULT_OUTPUT = Path("submission.csv")
DEFAULT_DIAGNOSTICS = Path("reports/suggested_runs/image_only_fix/kaggle_test_predictions.csv")

IMG_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class TestImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform: T.Compose):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Kaggle submission.csv from the image-only DenseNet121 checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diagnostics-output", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Use cuda when available by default; cpu is slower but reproducible.",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Only write submission.csv, without confidence diagnostics.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def build_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_model(num_classes: int) -> nn.Module:
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def sorted_image_paths(test_dir: Path) -> list[Path]:
    if not test_dir.exists():
        raise FileNotFoundError(f"Test image directory not found: {test_dir}")
    paths = [
        path
        for path in test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise ValueError(f"No supported test images found in {test_dir}")
    return sorted(paths, key=lambda path: path.name)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint.get("classes")
    if not classes:
        raise ValueError("Checkpoint does not contain the `classes` label order.")

    model = build_model(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = TestImageDataset(sorted_image_paths(args.test_dir), build_transform())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    with torch.no_grad():
        for images, image_ids in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            confidences, indices = probs.max(dim=1)

            for image_id, index, confidence, prob_row in zip(
                image_ids, indices.cpu().tolist(), confidences.cpu().tolist(), probs.cpu().tolist()
            ):
                label = classes[index]
                rows.append({"image_id": image_id, "label": label})
                if not args.no_diagnostics:
                    diagnostic = {
                        "image_id": image_id,
                        "label": label,
                        "confidence": confidence,
                    }
                    diagnostic.update({f"prob_{name}": prob for name, prob in zip(classes, prob_row)})
                    diagnostic_rows.append(diagnostic)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["image_id", "label"]).to_csv(args.output, index=False)

    if not args.no_diagnostics:
        args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(diagnostic_rows).to_csv(args.diagnostics_output, index=False)

    print(f"Wrote {len(rows)} predictions to {args.output}")
    if not args.no_diagnostics:
        print(f"Wrote diagnostics to {args.diagnostics_output}")


if __name__ == "__main__":
    main()
