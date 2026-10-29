import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
PROTOCOLS = HERE / "Protocols"
MANIFESTS = HERE / "Manifests"
OUTPUT = HERE / "Processed" / "deepfunneled_112"

# Published MD5 values for the six official LFW protocols (also in torchvision).https://docs.pytorch.org/vision/0.25/_modules/torchvision/datasets/lfw.html
OFFICIAL_MD5 = {
    "pairsDevTrain.txt": "4f27cbf15b2da4a85c1907eb4181ad21",
    "pairsDevTest.txt": "5132f7440eb68cf58910c8a45a2ac10b",
    "pairs.txt": "9f1ba174e4e1c508ff7cdf10ac338a7d",
    "peopleDevTrain.txt": "54eaac34beb6d042ed3a7d883e247a21",
    "peopleDevTest.txt": "e4bf5be0a43b5dcd9dc5ccfcb8fb19c5",
    "people.txt": "450f0863dd89e85e73936a6d71a3474b",
}


def find_metadata() -> Path:
    """Find the extracted Kaggle CSVs, regardless of the wrapper-folder name."""
    matches = [p.parent for p in HERE.rglob("peopleDevTrain.csv")
               if (p.parent / "people.csv").is_file()
               and (p.parent / "pairs.csv").is_file()]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one folder containing Kaggle CSVs inside {HERE}; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def rows(folder: Path, filename: str) -> list[list[str]]:
    with (folder / filename).open(encoding="utf-8-sig", newline="") as f:
        data = list(csv.reader(f))
    if not data or data[0][0].strip().lower() != "name":
        raise ValueError(f"Unexpected CSV header: {filename}")
    return data[1:]


def write_csv(path: Path, headers, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(values)


def write_official_txt(filename: str, lines: list[str], view: str) -> None:
    text = "\n".join(lines) + "\n"
    actual = hashlib.md5(text.encode("utf-8")).hexdigest()
    if actual != OFFICIAL_MD5[filename]:
        raise ValueError(
            f"{filename} does not match the official LFW file.\n"
            f"Expected MD5: {OFFICIAL_MD5[filename]}\n"
            f"Actual MD5:   {actual}\n"
            "Do not train until the CSV/protocol mismatch is resolved."
        )
    dest = PROTOCOLS / view / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    print(f"  Verified: {filename}")


def image_name(identity: str, index: int) -> str:
    return f"{identity}/{identity}_{int(index):04d}.png"


def prepare_protocols(metadata: Path) -> dict[str, int]:
    """Reconstruct official TXT protocols and model-ready CSV manifests."""
    print("[1/2] Generating and validating View 1 / View 2 protocols...")

    # View 1: the identity lists for unrestricted training (Triplet, Center, etc.).
    people_dev = {}
    for split in ("Train", "Test"):
        filename = f"peopleDev{split}.csv"
        entries = [(r[0].strip(), int(r[1])) for r in rows(metadata, filename)]
        if len(entries) != len(set(name for name, _ in entries)):
            raise ValueError(f"Duplicate identity in {filename}")
        people_dev[split] = dict(entries)
        write_official_txt(
            f"peopleDev{split}.txt",
            [str(len(entries))] + [f"{name}\t{count}" for name, count in entries],
            "View_1",
        )
        write_csv(
            MANIFESTS / f"view1_{'train' if split == 'Train' else 'validation'}_images.csv",
            ["identity", "image_path"],
            ((name, image_name(name, i))
             for name, count in entries for i in range(1, count + 1)),
        )

    if set(people_dev["Train"]) & set(people_dev["Test"]):
        raise ValueError("View 1 train/validation identities overlap")
    if (len(people_dev["Train"]), len(people_dev["Test"])) != (4038, 1711):
        raise ValueError("Unexpected View 1 identity counts")

    # View 1: matched/mismatched pair lists. Preserve the original ordering.
    for split, n, output_name in (
        ("Train", 1100, "view1_train_pairs.csv"),
        ("Test", 500, "view1_validation_pairs.csv"),
    ):
        matched = rows(metadata, f"matchpairsDev{split}.csv")
        mismatched = rows(metadata, f"mismatchpairsDev{split}.csv")
        if len(matched) != n or len(mismatched) != n:
            raise ValueError(f"Incorrect View 1 {split} pair counts")
        if any(len(r) != 3 for r in matched) or any(len(r) != 4 for r in mismatched):
            raise ValueError(f"Incorrect columns in View 1 {split} pair CSVs")
        pair_rows = []
        raw_lines = [str(n)]
        for name, a, b in matched:
            raw_lines.append(f"{name}\t{int(a)}\t{int(b)}")
            pair_rows.append((0, image_name(name, a), image_name(name, b), 1))
        for name, a, other, b in mismatched:
            if name == other:
                raise ValueError(f"Mismatched pair has same identity: {name}")
            raw_lines.append(f"{name}\t{int(a)}\t{other}\t{int(b)}")
            pair_rows.append((0, image_name(name, a), image_name(other, b), 0))
        write_official_txt(f"pairsDev{split}.txt", raw_lines, "View_1")
        write_csv(MANIFESTS / output_name, ["fold", "img1", "img2", "label"], pair_rows)

    # View 2: blank lines in people.csv separate ten identity-disjoint folds.
    folds = [[]]
    for row in rows(metadata, "people.csv"):
        if not row or not row[0].strip():
            if not folds[-1]:
                raise ValueError("Unexpected repeated View 2 fold separator")
            folds.append([])
        else:
            folds[-1].append((row[0].strip(), int(row[1])))
    if len(folds) != 10 or any(not fold for fold in folds):
        raise ValueError(f"Expected 10 nonempty folds, found {len(folds)}")
    people_lines = ["10"]
    view2_images = []
    all_identities = set()
    fold_counts = []
    for fold_no, fold in enumerate(folds, start=1):
        people_lines.append(str(len(fold)))
        current = {}
        for name, count in fold:
            if name in all_identities or count < 1:
                raise ValueError(f"Duplicate/invalid identity in View 2: {name}")
            all_identities.add(name)
            current[name] = count
            people_lines.append(f"{name}\t{count}")
            for i in range(1, count + 1):
                view2_images.append((fold_no, name, image_name(name, i)))
        fold_counts.append(current)
    if len(all_identities) != 5749 or len(view2_images) != 13233:
        raise ValueError("View 2 must contain 5,749 identities and 13,233 images")
    write_official_txt("people.txt", people_lines, "View_2")
    write_csv(MANIFESTS / "view2_images.csv",
              ["fold", "identity", "image_path"], view2_images)

    # View 2: pairs.csv uses 600 consecutive entries per fold:
    # first 300 matching (third value plus blank fourth), then 300 mismatching.
    pair_csv = rows(metadata, "pairs.csv")
    if len(pair_csv) != 6000:
        raise ValueError(f"Expected 6,000 View 2 pairs, found {len(pair_csv)}")
    pair_lines = ["10\t300"]
    view2_pairs = []
    for index, row in enumerate(pair_csv):
        fold = index // 600 + 1
        positive = index % 600 < 300
        if len(row) != 4:
            raise ValueError(f"Expected four CSV fields at View 2 row {index + 2}")
        name, a, third, fourth = row
        if positive:
            if fourth.strip():
                raise ValueError(f"Malformed matched pair at View 2 row {index + 2}")
            other, b, label = name, third, 1
            pair_lines.append(f"{name}\t{int(a)}\t{int(b)}")
        else:
            if not fourth.strip() or third == name:
                raise ValueError(f"Malformed mismatched pair at View 2 row {index + 2}")
            other, b, label = third, fourth, 0
            pair_lines.append(f"{name}\t{int(a)}\t{other}\t{int(b)}")
        identities = fold_counts[fold - 1]
        if (name not in identities or other not in identities
                or int(a) < 1 or int(a) > identities[name]
                or int(b) < 1 or int(b) > identities[other]):
            raise ValueError(f"Pair references identity/image outside fold {fold}: {row}")
        view2_pairs.append((fold, image_name(name, a), image_name(other, b), label))
    write_official_txt("pairs.txt", pair_lines, "View_2")
    write_csv(MANIFESTS / "view2_pairs.csv",
              ["fold", "img1", "img2", "label"], view2_pairs)

    counts = {
        "view1_train_identities": 4038,
        "view1_train_images": sum(people_dev["Train"].values()),
        "view1_validation_identities": 1711,
        "view1_validation_images": sum(people_dev["Test"].values()),
        "view1_train_pairs": 2200,
        "view1_validation_pairs": 1000,
        "view2_folds": 10,
        "view2_identities": 5749,
        "view2_images": 13233,
        "view2_pairs": 6000,
    }
    print("  All six reconstructed TXT files match official LFW MD5 checksums.")
    return counts


def find_images(metadata: Path) -> Path:
    """Handle your nested lfw_deepfuneeled/lfw-deepfunneled/... structure."""
    candidates = [p for p in metadata.rglob("lfw-deepfunneled")
                  if p.is_dir() and (p / "Aaron_Eckhart").is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot uniquely locate the extracted image folders: {candidates}")
    return candidates[0]


def process_images(image_root: Path) -> None:
    """Verify original images and resize them, without any extra cropping."""
    print("[2/2] Checking and resizing LFW Deep-Funneled images...")
    actual = {p.relative_to(image_root).as_posix()
              for p in image_root.glob("*/*.jpg")}
    with (MANIFESTS / "view2_images.csv").open(encoding="utf-8", newline="") as f:
        expected_pngs = [r["image_path"] for r in csv.DictReader(f)]
    expected = {str(Path(p).with_suffix(".jpg")) for p in expected_pngs}
    expected_folders = {Path(rel).parts[0] for rel in expected}
    actual_folders = {p.name for p in image_root.iterdir() if p.is_dir()}
    if actual != expected or actual_folders != expected_folders:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise ValueError(f"Incomplete original images. Missing: {missing[:5]}; "
                         f"unexpected: {extra[:5]}; "
                         f"missing identity folders: {sorted(expected_folders - actual_folders)[:5]}")
    for idx, rel_png in enumerate(expected_pngs, start=1):
        rel_jpg = Path(rel_png).with_suffix(".jpg")
        with Image.open(image_root / rel_jpg) as original:
            original.load()  # Check that the full JPEG can be decoded.
            if original.size != (250, 250):
                raise ValueError(f"Not 250x250: {rel_jpg} ({original.size})")
            # Already aligned. No arbitrary 224x224 crop and no random crop.
            resized = original.convert("RGB").resize((112, 112), Image.Resampling.LANCZOS)
        target = OUTPUT / rel_png
        target.parent.mkdir(parents=True, exist_ok=True)
        resized.save(target, format="PNG")
        if idx % 1000 == 0 or idx == len(expected_pngs):
            print(f"  {idx:,}/{len(expected_pngs):,} images processed")
    print(f"  Cached images: {OUTPUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true",
                        help="Generate protocols/manifests without resizing images")
    args = parser.parse_args()
    metadata = find_metadata()
    print(f"Kaggle metadata folder: {metadata}")
    summary = prepare_protocols(metadata)
    if not args.metadata_only:
        process_images(find_images(metadata))
    summary.update({
        "aligned_input": "LFW Deep-Funneled 250x250 RGB JPEG",
        "processed_output": "112x112 RGB PNG, direct resize (LANCZOS), no additional crop",
        "normalization_for_training": "(pixel - 127.5) / 128.0",
        "augmentation_for_training_only": "random horizontal flip; never pre-cache augmented images",
        "images_processed": not args.metadata_only,
    })
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    (MANIFESTS / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Done. Protocols, model-ready CSVs, and summary saved.")


if __name__ == "__main__":
    main()
