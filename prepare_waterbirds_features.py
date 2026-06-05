#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


def find_metadata(root: Path) -> Path:
    candidates = [
        root / "metadata.csv",
        root / "metadata_waterbird_complete95_forest2water2.csv",
        root / "metadata_waterbirds.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    csvs = list(root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No metadata CSV found under {root}")
    metadata_csvs = [p for p in csvs if "metadata" in p.name.lower()]
    return metadata_csvs[0] if metadata_csvs else csvs[0]


def pick_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_columns(df: pd.DataFrame):
    filename_col = pick_column(df, ["img_filename", "filename", "filepath", "image", "path", "file"])
    y_col = pick_column(df, ["y", "waterbird_complete95", "target", "label"])
    attr_col = pick_column(df, ["place", "forest2water2", "background", "spurious", "confounder", "a", "attr"])
    split_col = pick_column(df, ["split", "split_id"])
    missing = []
    if filename_col is None: missing.append("filename/path")
    if y_col is None: missing.append("label y")
    if attr_col is None: missing.append("attribute/background")
    if split_col is None: missing.append("split")
    if missing:
        raise ValueError(
            "Could not infer columns: " + ", ".join(missing) +
            f"\nAvailable columns: {list(df.columns)}\n" +
            "Rerun with --filename_col, --y_col, --attr_col, --split_col."
        )
    return filename_col, y_col, attr_col, split_col


def normalize_split_value(x):
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ["train", "tr", "0"]: return "train"
        if s in ["val", "valid", "validation", "dev", "1"]: return "val"
        if s in ["test", "te", "2"]: return "test"
        raise ValueError(f"Unknown split string: {x}")
    xi = int(x)
    if xi == 0: return "train"
    if xi == 1: return "val"
    if xi == 2: return "test"
    raise ValueError(f"Unknown split id: {x}")


class WaterbirdsImageDataset(Dataset):
    def __init__(self, root, df, filename_col, y_col, attr_col, transform):
        self.root = Path(root)
        self.df = df.reset_index(drop=True)
        self.filename_col = filename_col
        self.y_col = y_col
        self.attr_col = attr_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, rel):
        rel = str(rel)
        candidates = [
            self.root / rel,
            self.root / "images" / rel,
            self.root / "CUB_200_2011" / "images" / rel,
        ]
        for p in candidates:
            if p.exists():
                return p
        matches = list(self.root.rglob(Path(rel).name))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Could not find image for metadata path: {rel}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self._resolve_path(row[self.filename_col])
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = int(row[self.y_col])
        a = int(row[self.attr_col])
        g = 2 * y + a
        return x, y, a, g


def build_resnet_feature_extractor(model_name, pretrained=True):
    import torchvision.models as models
    from torchvision import transforms

    model_name = model_name.lower()
    if model_name == "resnet18":
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
                model = models.resnet18(weights=weights)
                transform = weights.transforms()
            except Exception:
                model = models.resnet18(pretrained=True)
                transform = transforms.Compose([
                    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                ])
        else:
            model = models.resnet18(weights=None)
            transform = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
    elif model_name == "resnet50":
        if pretrained:
            try:
                weights = models.ResNet50_Weights.DEFAULT
                model = models.resnet50(weights=weights)
                transform = weights.transforms()
            except Exception:
                model = models.resnet50(pretrained=True)
                transform = transforms.Compose([
                    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                ])
        else:
            model = models.resnet50(weights=None)
            transform = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
    else:
        raise ValueError("model must be resnet18 or resnet50")
    feature_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, transform, feature_dim


@torch.no_grad()
def extract_split_features(split_name, df_split, root, filename_col, y_col, attr_col,
                           model, transform, device, batch_size, num_workers):
    ds = WaterbirdsImageDataset(root, df_split, filename_col, y_col, attr_col, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=(device.type == "cuda"))
    feats_all, y_all, a_all, g_all = [], [], [], []
    model.eval()
    for x, y, a, g in tqdm(loader, desc=f"Extracting {split_name}"):
        x = x.to(device)
        feats = model(x)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        feats_all.append(feats.cpu().float())
        y_all.append(y.long())
        a_all.append(a.long())
        g_all.append(g.long())
    features = torch.cat(feats_all, 0)
    labels = torch.cat(y_all, 0)
    attrs = torch.cat(a_all, 0)
    groups = torch.cat(g_all, 0)
    envs = torch.zeros_like(labels).long()
    return features, labels, envs, groups, attrs


def summarize_tensor_tuple(name, bundle, out):
    features, labels, envs, groups, attrs = bundle
    summary = {
        "n": int(labels.numel()),
        "feature_shape": list(features.shape),
        "label_counts": {str(int(k)): int((labels == k).sum()) for k in torch.unique(labels)},
        "attr_counts": {str(int(k)): int((attrs == k).sum()) for k in torch.unique(attrs)},
        "group_counts": {str(int(k)): int((groups == k).sum()) for k in torch.unique(groups)},
    }
    out[name] = summary
    print("-"*80)
    print(f"{name}: n={summary['n']}, feature_shape={summary['feature_shape']}")
    print(f"label_counts={summary['label_counts']}")
    print(f"attr_counts={summary['attr_counts']}")
    print(f"group_counts={summary['group_counts']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data/cub/data/waterbird_complete95_forest2water2")
    parser.add_argument("--output_dir", default="./data/Waterbirds")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--filename_col", default=None)
    parser.add_argument("--y_col", default=None)
    parser.add_argument("--attr_col", default=None)
    parser.add_argument("--split_col", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = find_metadata(root)
    df = pd.read_csv(metadata_path)
    if args.filename_col and args.y_col and args.attr_col and args.split_col:
        filename_col, y_col, attr_col, split_col = args.filename_col, args.y_col, args.attr_col, args.split_col
    else:
        filename_col, y_col, attr_col, split_col = infer_columns(df)

    print("="*80)
    print(f"Waterbirds root: {root}")
    print(f"Metadata: {metadata_path}")
    print(f"Rows: {len(df)}")
    print(f"filename_col={filename_col}, y_col={y_col}, attr_col={attr_col}, split_col={split_col}")

    df = df.copy()
    df["_split_name"] = df[split_col].apply(normalize_split_value)
    df["_group"] = 2 * df[y_col].astype(int) + df[attr_col].astype(int)
    print("\nMetadata split counts:")
    print(df["_split_name"].value_counts().sort_index())
    print("\nMetadata group counts by split:")
    print(pd.crosstab(df["_split_name"], df["_group"]))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"\nUsing device: {device}")
    model, transform, feature_dim = build_resnet_feature_extractor(args.model, pretrained=(not args.no_pretrained))
    model = model.to(device)
    print(f"Feature extractor: {args.model}, feature_dim={feature_dim}, pretrained={not args.no_pretrained}")

    summary = {
        "root": str(root), "metadata": str(metadata_path), "model": args.model,
        "pretrained": not args.no_pretrained, "feature_dim": feature_dim,
        "columns": {"filename_col": filename_col, "y_col": y_col, "attr_col": attr_col, "split_col": split_col},
    }

    for split_name in ["train", "val", "test"]:
        df_split = df[df["_split_name"] == split_name].reset_index(drop=True)
        if len(df_split) == 0:
            raise ValueError(f"No samples found for split={split_name}. Check split column.")
        bundle = extract_split_features(split_name, df_split, root, filename_col, y_col, attr_col,
                                        model, transform, device, args.batch_size, args.num_workers)
        save_path = output_dir / f"{split_name}.pt"
        torch.save(bundle, save_path)
        print(f"Saved {save_path}")
        summarize_tensor_tuple(split_name, bundle, summary)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("="*80)
    print(f"Done. Saved tensors to: {output_dir}")


if __name__ == "__main__":
    main()
