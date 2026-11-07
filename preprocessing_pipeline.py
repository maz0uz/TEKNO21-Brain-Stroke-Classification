from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Folder containing the original dataset.
INPUT_ROOT = Path("processed_dataset")

# Folder where the preprocessed images will be saved.
OUTPUT_ROOT = Path("contrast_dataset")

# CNN input size.
TARGET_SIZE = (224, 224)

# Margin added around the detected head.
MARGIN_RATIO = 0.08

# Background threshold
BACKGROUND_THRESHOLD = 5

# Minimum size of a detected head region relative to the whole image.
MIN_COMPONENT_AREA_RATIO = 0.05

# Whether to use CLAHE contrast enhancement.
USE_CLAHE = True

# CLAHE settings.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Number of random before-and-after samples to display.
NUMBER_OF_PREVIEW_SAMPLES = 6

def load_ct_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        raise ValueError(f"OpenCV could not load: {image_path}")

    return image

def create_head_binary_mask(
    gray_image: np.ndarray,
    background_threshold: int = 5,
) -> np.ndarray:
    """
    Create a binary mask separating the head from the black background.

    Pixels brighter than background_threshold become white.
    Pixels below the threshold remain black.

    Morphological operations are then used to:
        - connect nearby head regions;
        - fill small gaps;
        - remove small isolated regions.
    """

    binary_mask = np.where(
        gray_image > background_threshold,
        255,
        0,
    ).astype(np.uint8)

    kernel = np.ones((5, 5), dtype=np.uint8)

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    return binary_mask


def find_head_bounding_box(
    gray_image: np.ndarray,
    background_threshold: int = 5,
    margin_ratio: float = 0.08,
    min_component_area_ratio: float = 0.05,
) -> tuple[int, int, int, int, bool]:

    image_height, image_width = gray_image.shape

    binary_mask = create_head_binary_mask(
        gray_image,
        background_threshold=background_threshold,
    )

    number_of_labels, labels, statistics, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )
    )
    if number_of_labels <= 1:
        return 0, 0, image_width, image_height, False

    foreground_statistics = statistics[1:]

    component_areas = foreground_statistics[:, cv2.CC_STAT_AREA]

    largest_relative_index = int(np.argmax(component_areas))

    largest_label_index = largest_relative_index + 1

    largest_component_area = statistics[
        largest_label_index,
        cv2.CC_STAT_AREA,
    ]

    total_image_area = image_height * image_width

    component_area_ratio = (
        largest_component_area / float(total_image_area)
    )

    if component_area_ratio < min_component_area_ratio:
        return 0, 0, image_width, image_height, False

    x = statistics[largest_label_index, cv2.CC_STAT_LEFT]
    y = statistics[largest_label_index, cv2.CC_STAT_TOP]
    width = statistics[largest_label_index, cv2.CC_STAT_WIDTH]
    height = statistics[largest_label_index, cv2.CC_STAT_HEIGHT]

    margin_x = int(round(width * margin_ratio))
    margin_y = int(round(height * margin_ratio))

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)

    x2 = min(image_width, x + width + margin_x)
    y2 = min(image_height, y + height + margin_y)

    if x2 <= x1 or y2 <= y1:
        return 0, 0, image_width, image_height, False

    return x1, y1, x2, y2, True


def crop_head(
    gray_image: np.ndarray,
    background_threshold: int = 5,
    margin_ratio: float = 0.08,
    min_component_area_ratio: float = 0.05,
) -> tuple[np.ndarray, tuple[int, int, int, int], bool]:
    
    x1, y1, x2, y2, crop_detected = find_head_bounding_box(
        gray_image,
        background_threshold=background_threshold,
        margin_ratio=margin_ratio,
        min_component_area_ratio=min_component_area_ratio,
    )

    cropped_image = gray_image[y1:y2, x1:x2]

    return (
        cropped_image,
        (x1, y1, x2, y2),
        crop_detected,
    )


def pad_to_square(
    image: np.ndarray,
    padding_value: int = 0,
) -> np.ndarray:

    height, width = image.shape

    square_size = max(height, width)

    square_image = np.full(
        shape=(square_size, square_size),
        fill_value=padding_value,
        dtype=image.dtype,
    )

    # Center the image inside the square.
    y_offset = (square_size - height) // 2
    x_offset = (square_size - width) // 2

    square_image[
        y_offset:y_offset + height,
        x_offset:x_offset + width,
    ] = image

    return square_image


def resize_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    
    current_height, current_width = image.shape

    target_width, target_height = target_size

    if (
        current_width > target_width
        or current_height > target_height
    ):
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    resized_image = cv2.resize(
        image,
        target_size,
        interpolation=interpolation,
    )

    return resized_image

def apply_clahe(
    gray_image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )

    return clahe.apply(gray_image)

def preprocess_ct_image(
    image_path: Path,
    target_size: tuple[int, int] = (224, 224),
    margin_ratio: float = 0.08,
    background_threshold: int = 5,
    min_component_area_ratio: float = 0.05,
    use_clahe: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, int, int, int],
    bool,
]:

    original_image = load_ct_image(image_path)

    cropped_image, bounding_box, crop_detected = crop_head(
        original_image,
        background_threshold=background_threshold,
        margin_ratio=margin_ratio,
        min_component_area_ratio=min_component_area_ratio,
    )

    square_image = pad_to_square(
        cropped_image,
        padding_value=0,
    )

    processed_image = resize_image(
        square_image,
        target_size=target_size,
    )

    if use_clahe:
        processed_image = apply_clahe(
            processed_image,
            clip_limit=CLAHE_CLIP_LIMIT,
            tile_grid_size=CLAHE_TILE_GRID_SIZE,
        )

    return (
        original_image,
        processed_image,
        bounding_box,
        crop_detected,
    )

def find_all_images(root_directory: Path) -> list[Path]:

    image_paths: list[Path] = []

    for path in root_directory.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() == ".png"
        ):
            image_paths.append(path)

    return sorted(image_paths)


def process_entire_dataset(
    input_root: Path,
    output_root: Path,
) -> tuple[int, int]:

    if not input_root.exists():
        raise FileNotFoundError(
            f"Input dataset folder does not exist: {input_root.resolve()}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = find_all_images(input_root)

    if not image_paths:
        raise RuntimeError(
            f"No supported images were found inside {input_root.resolve()}"
        )

    print(f"Found {len(image_paths)} images.")
    print(f"Input folder:  {input_root.resolve()}")
    print(f"Output folder: {output_root.resolve()}")
    print()

    log_path = output_root / "preprocessing_log.csv"

    successful_count = 0
    failed_count = 0

    with log_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        fieldnames = [
            "input_path",
            "output_path",
            "original_width",
            "original_height",
            "crop_x1",
            "crop_y1",
            "crop_x2",
            "crop_y2",
            "crop_detected",
            "output_width",
            "output_height",
            "clahe_used",
            "status",
            "error",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index, image_path in enumerate(image_paths, start=1):
            relative_path = image_path.relative_to(input_root)
            output_relative_path = relative_path.with_suffix(".png")
            output_path = output_root / output_relative_path
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            try:
                (
                    original_image,
                    processed_image,
                    bounding_box,
                    crop_detected,
                ) = preprocess_ct_image(
                    image_path=image_path,
                    target_size=TARGET_SIZE,
                    margin_ratio=MARGIN_RATIO,
                    background_threshold=BACKGROUND_THRESHOLD,
                    min_component_area_ratio=(
                        MIN_COMPONENT_AREA_RATIO
                    ),
                    use_clahe=USE_CLAHE,
                )
                save_successful = cv2.imwrite(
                    str(output_path),
                    processed_image,
                )

                if not save_successful:
                    raise IOError(
                        f"OpenCV failed to save {output_path}"
                    )

                original_height, original_width = (
                    original_image.shape
                )

                output_height, output_width = (
                    processed_image.shape
                )

                x1, y1, x2, y2 = bounding_box

                writer.writerow({
                    "input_path": str(image_path),
                    "output_path": str(output_path),
                    "original_width": original_width,
                    "original_height": original_height,
                    "crop_x1": x1,
                    "crop_y1": y1,
                    "crop_x2": x2,
                    "crop_y2": y2,
                    "crop_detected": crop_detected,
                    "output_width": output_width,
                    "output_height": output_height,
                    "clahe_used": USE_CLAHE,
                    "status": "success",
                    "error": "",
                })

                successful_count += 1

            except Exception as error:
                failed_count += 1

                writer.writerow({
                    "input_path": str(image_path),
                    "output_path": str(output_path),
                    "original_width": "",
                    "original_height": "",
                    "crop_x1": "",
                    "crop_y1": "",
                    "crop_x2": "",
                    "crop_y2": "",
                    "crop_detected": "",
                    "output_width": "",
                    "output_height": "",
                    "clahe_used": USE_CLAHE,
                    "status": "failed",
                    "error": str(error),
                })

                print(
                    f"Failed to process {image_path}: {error}"
                )
            if (
                index % 100 == 0
                or index == len(image_paths)
            ):
                print(
                    f"Processed {index}/{len(image_paths)} images"
                )

    print()
    print("Preprocessing complete.")
    print(f"Successfully processed: {successful_count}")
    print(f"Failed:                 {failed_count}")
    print(f"Log saved to:           {log_path.resolve()}")

    return successful_count, failed_count

def preview_preprocessing_results(
    input_root: Path,
    number_of_samples: int = 6,
) -> None:

    image_paths = find_all_images(input_root)

    if not image_paths:
        print("No images are available for preview.")
        return

    actual_sample_count = min(
        number_of_samples,
        len(image_paths),
    )

    selected_paths = random.sample(
        image_paths,
        actual_sample_count,
    )

    figure, axes = plt.subplots(
        actual_sample_count,
        3,
        figsize=(12, 4 * actual_sample_count),
    )
    if actual_sample_count == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, image_path in enumerate(selected_paths):

        (
            original_image,
            processed_image,
            bounding_box,
            crop_detected,
        ) = preprocess_ct_image(
            image_path=image_path,
            target_size=TARGET_SIZE,
            margin_ratio=MARGIN_RATIO,
            background_threshold=BACKGROUND_THRESHOLD,
            min_component_area_ratio=MIN_COMPONENT_AREA_RATIO,
            use_clahe=USE_CLAHE,
        )

        x1, y1, x2, y2 = bounding_box

        bounding_box_image = cv2.cvtColor(
            original_image,
            cv2.COLOR_GRAY2RGB,
        )

        cv2.rectangle(
            bounding_box_image,
            (x1, y1),
            (x2 - 1, y2 - 1),
            color=(255, 0, 0),
            thickness=2,
        )

        relative_name = image_path.relative_to(input_root)

        axes[row_index, 0].imshow(
            original_image,
            cmap="gray",
        )
        axes[row_index, 0].set_title(
            f"Original\n{relative_name}"
        )
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(
            bounding_box_image,
        )
        axes[row_index, 1].set_title(
            f"Detected crop\nSuccess: {crop_detected}"
        )
        axes[row_index, 1].axis("off")

        axes[row_index, 2].imshow(
            processed_image,
            cmap="gray",
        )
        axes[row_index, 2].set_title(
            f"Processed\n{processed_image.shape[1]} × "
            f"{processed_image.shape[0]}"
        )
        axes[row_index, 2].axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    process_entire_dataset(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
    )

    preview_preprocessing_results(
        input_root=INPUT_ROOT,
        number_of_samples=NUMBER_OF_PREVIEW_SAMPLES,
    )


if __name__ == "__main__":
    main()