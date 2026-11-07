from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers


# Paths to CSV split files.
TRAIN_CSV = Path("train_split.csv")
VAL_CSV = Path("val_split.csv")
TEST_CSV = Path("test_split.csv")

IMAGE_ROOT = Path("nocontrast_dataset")

# Folder for saved models, figures, logs, and predictions.
OUTPUT_DIRECTORY = Path("vgg16_results_nocontrast")

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# Number of images processed at once.
BATCH_SIZE = 16

# Initial training trains only the new classification head.
INITIAL_EPOCHS = 20

# Fine-tuning trains the last convolutional block of VGG16.
FINE_TUNE_EPOCHS = 10

# Learning rate used while the VGG16 base is frozen.
INITIAL_LEARNING_RATE = 1e-4

# learning rate for fine-tuning pretrained layers.
FINE_TUNE_LEARNING_RATE = 1e-5

# Random seed for reproducibility.
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE
def set_random_seeds(seed: int) -> None:

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_random_seeds(SEED)
REQUIRED_COLUMNS = {
    "image_path",
    "label_name",
    "label",
}


def load_split_csv(csv_path: Path, split_name: str) -> pd.DataFrame:

    if not csv_path.exists():
        raise FileNotFoundError(
            f"{split_name} CSV was not found:\n"
            f"{csv_path.resolve()}"
        )

    dataframe = pd.read_csv(csv_path)

    dataframe.columns = dataframe.columns.str.strip()

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{split_name} CSV is missing these columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError(f"{split_name} CSV contains no rows.")

    dataframe = dataframe[
        ["image_path", "label_name", "label"]
    ].copy()

    dataframe["image_path"] = (
        dataframe["image_path"].astype(str).str.strip()
    )

    dataframe["label_name"] = (
        dataframe["label_name"].astype(str).str.strip()
    )

    dataframe["label"] = pd.to_numeric(
        dataframe["label"],
        errors="raise",
    ).astype(np.int32)

    if dataframe.isnull().any().any():
        raise ValueError(
            f"{split_name} CSV contains missing values."
        )

    return dataframe


train_dataframe = load_split_csv(
    TRAIN_CSV,
    "training",
)

validation_dataframe = load_split_csv(
    VAL_CSV,
    "validation",
)

test_dataframe = load_split_csv(
    TEST_CSV,
    "testing",
)

def resolve_image_path(path_text: str) -> str:

    image_path = Path(path_text)

    if image_path.is_absolute():
        return str(image_path)

    return str((IMAGE_ROOT / image_path).resolve())


def resolve_dataframe_paths(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["image_path"] = dataframe["image_path"].apply(
        resolve_image_path
    )

    return dataframe


train_dataframe = resolve_dataframe_paths(train_dataframe)
validation_dataframe = resolve_dataframe_paths(
    validation_dataframe
)
test_dataframe = resolve_dataframe_paths(test_dataframe)


def verify_image_paths(
    dataframe: pd.DataFrame,
    split_name: str,
) -> None:

    missing_paths = [
        path
        for path in dataframe["image_path"]
        if not Path(path).exists()
    ]

    if missing_paths:
        examples = "\n".join(missing_paths[:10])

        raise FileNotFoundError(
            f"{len(missing_paths)} images from the {split_name} "
            f"split could not be found.\n\n"
            f"Examples:\n{examples}\n\n"
            f"Check IMAGE_ROOT and the image_path column."
        )


verify_image_paths(train_dataframe, "training")
verify_image_paths(validation_dataframe, "validation")
verify_image_paths(test_dataframe, "testing")

def create_label_mapping(
    dataframe: pd.DataFrame,
) -> dict[int, str]:

    unique_pairs = dataframe[
        ["label", "label_name"]
    ].drop_duplicates()

    duplicated_numbers = unique_pairs["label"].duplicated(
        keep=False
    )

    if duplicated_numbers.any():
        problematic = unique_pairs[duplicated_numbers]

        raise ValueError(
            "One numeric label is connected to multiple "
            f"label names:\n{problematic}"
        )

    duplicated_names = unique_pairs["label_name"].duplicated(
        keep=False
    )

    if duplicated_names.any():
        problematic = unique_pairs[duplicated_names]

        raise ValueError(
            "One label name is connected to multiple "
            f"numeric labels:\n{problematic}"
        )

    unique_pairs = unique_pairs.sort_values("label")

    return dict(
        zip(
            unique_pairs["label"].astype(int),
            unique_pairs["label_name"],
        )
    )


label_mapping = create_label_mapping(train_dataframe)

print("\nClass mapping:")

for numeric_label, class_name in label_mapping.items():
    print(f"    {numeric_label} -> {class_name}")


class_ids = sorted(label_mapping.keys())
expected_class_ids = list(range(len(class_ids)))

if class_ids != expected_class_ids:
    raise ValueError(
        "The class labels must be consecutive and start at zero.\n"
        f"Found:    {class_ids}\n"
        f"Expected: {expected_class_ids}"
    )


NUM_CLASSES = len(class_ids)

if NUM_CLASSES != 3:
    raise ValueError(
        "This project expects exactly three classes, but found "
        f"{NUM_CLASSES}: {label_mapping}"
    )


CLASS_NAMES = [
    label_mapping[class_id]
    for class_id in class_ids
]


def verify_split_mapping(
    dataframe: pd.DataFrame,
    split_name: str,
    expected_mapping: dict[int, str],
) -> None:

    split_mapping = create_label_mapping(dataframe)

    if split_mapping != expected_mapping:
        raise ValueError(
            f"The label mapping in the {split_name} split is "
            f"different from the training split.\n\n"
            f"Training mapping: {expected_mapping}\n"
            f"{split_name} mapping: {split_mapping}"
        )


verify_split_mapping(
    validation_dataframe,
    "validation",
    label_mapping,
)

verify_split_mapping(
    test_dataframe,
    "testing",
    label_mapping,
)

def display_split_information(
    dataframe: pd.DataFrame,
    split_name: str,
) -> None:

    print(f"\n{split_name} split")
    print("-" * 40)
    print(f"Total images: {len(dataframe)}")

    counts = (
        dataframe.groupby(["label", "label_name"])
        .size()
        .reset_index(name="count")
        .sort_values("label")
    )

    print(counts.to_string(index=False))


display_split_information(train_dataframe, "Training")
display_split_information(validation_dataframe, "Validation")
display_split_information(test_dataframe, "Testing")


def check_path_overlap(
    first_dataframe: pd.DataFrame,
    second_dataframe: pd.DataFrame,
    first_name: str,
    second_name: str,
) -> None:

    first_paths = set(first_dataframe["image_path"])
    second_paths = set(second_dataframe["image_path"])

    overlap = first_paths.intersection(second_paths)

    if overlap:
        examples = "\n".join(list(overlap)[:10])

        raise ValueError(
            f"Data leakage detected between {first_name} and "
            f"{second_name}.\n"
            f"{len(overlap)} exact image paths appear in both.\n\n"
            f"Examples:\n{examples}"
        )


check_path_overlap(
    train_dataframe,
    validation_dataframe,
    "training",
    "validation",
)

check_path_overlap(
    train_dataframe,
    test_dataframe,
    "training",
    "testing",
)

check_path_overlap(
    validation_dataframe,
    test_dataframe,
    "validation",
    "testing",
)

def load_image(
    image_path: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:

    image_bytes = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False,
    )
    image.set_shape([None, None, 1])

    image = tf.cast(image, tf.float32)

    image = tf.image.resize(
        image,
        size=IMAGE_SIZE,
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=True,
    )

    image = tf.image.grayscale_to_rgb(image)

    label = tf.cast(label, tf.int32)

    return image, label


def create_tensorflow_dataset(
    dataframe: pd.DataFrame,
    training: bool,
) -> tf.data.Dataset:

    image_paths = dataframe["image_path"].to_numpy(
        dtype=str
    )

    labels = dataframe["label"].to_numpy(
        dtype=np.int32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE,
    )

    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False,
    )

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


train_dataset = create_tensorflow_dataset(
    train_dataframe,
    training=True,
)

validation_dataset = create_tensorflow_dataset(
    validation_dataframe,
    training=False,
)

test_dataset = create_tensorflow_dataset(
    test_dataframe,
    training=False,
)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip(
            mode="horizontal",
            seed=SEED,
        ),
        layers.RandomRotation(
            factor=0.02,
            fill_mode="constant",
            fill_value=0.0,
            seed=SEED,
        ),
        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05,
            fill_mode="constant",
            fill_value=0.0,
            seed=SEED,
        ),
        layers.RandomZoom(
            height_factor=(-0.10, 0.10),
            width_factor=(-0.10, 0.10),
            fill_mode="constant",
            fill_value=0.0,
            seed=SEED,
        ),
    ],
    name="ct_data_augmentation",
)



def build_vgg16_model(
    number_of_classes: int,
) -> tuple[keras.Model, keras.Model]:

    base_model = keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
    )

    base_model.trainable = False

    inputs = keras.Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        name="ct_image",
    )

    x = data_augmentation(inputs)
    x = keras.applications.vgg16.preprocess_input(x)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling",
    )(x)

    x = layers.Dense(
        units=256,
        activation="relu",
        name="classification_dense",
    )(x)

    x = layers.Dropout(
        rate=0.50,
        name="classification_dropout",
    )(x)

    outputs = layers.Dense(
        units=number_of_classes,
        activation="softmax",
        name="class_probabilities",
    )(x)

    complete_model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="tekno21_vgg16",
    )

    return complete_model, base_model
model, vgg16_base = build_vgg16_model(
    number_of_classes=NUM_CLASSES,
)

print("\nVGG16 model created successfully.")
model.summary()


training_labels = train_dataframe["label"].to_numpy(
    dtype=np.int32,
)

class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=np.asarray(class_ids, dtype=np.int32),
    y=training_labels,
)

CLASS_WEIGHTS = {
    int(class_id): float(weight)
    for class_id, weight in zip(class_ids, class_weight_values)
}

print("\nClass weights:")
for class_id, weight in CLASS_WEIGHTS.items():
    print(
        f"    {class_id} ({CLASS_NAMES[class_id]}): "
        f"{weight:.4f}"
    )


def compile_model(
    model_to_compile: keras.Model,
    learning_rate: float,
) -> None:
    model_to_compile.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(
                name="accuracy",
            )
        ],
    )


compile_model(
    model,
    learning_rate=INITIAL_LEARNING_RATE,
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


def create_callbacks(
    checkpoint_path: Path,
    csv_log_path: Path,
) -> list[keras.callbacks.Callback]:

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=3,
            min_learning_rate=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            filename=str(csv_log_path),
            append=False,
        ),
    ]

stage_one_checkpoint = (
    OUTPUT_DIRECTORY / "vgg16_stage1_best.keras"
)
stage_one_log = (
    OUTPUT_DIRECTORY / "vgg16_stage1_training.csv"
)

print("\n" + "=" * 60)
print("STAGE 1: TRAINING THE NEW CLASSIFICATION HEAD")
print("=" * 60)

history_stage_one = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=INITIAL_EPOCHS,
    class_weight=CLASS_WEIGHTS,
    callbacks=create_callbacks(
        checkpoint_path=stage_one_checkpoint,
        csv_log_path=stage_one_log,
    ),
)
vgg16_base.trainable = True

for layer in vgg16_base.layers:
    layer.trainable = layer.name.startswith("block5_")

compile_model(
    model,
    learning_rate=FINE_TUNE_LEARNING_RATE,
)

stage_two_checkpoint = (
    OUTPUT_DIRECTORY / "vgg16_stage2_best.keras"
)
stage_two_log = (
    OUTPUT_DIRECTORY / "vgg16_stage2_training.csv"
)

completed_stage_one_epochs = len(
    history_stage_one.history["loss"]
)

print("\n" + "=" * 60)
print("STAGE 2: FINE-TUNING VGG16 BLOCK 5")
print("=" * 60)

history_stage_two = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    initial_epoch=completed_stage_one_epochs,
    epochs=completed_stage_one_epochs + FINE_TUNE_EPOCHS,
    class_weight=CLASS_WEIGHTS,
    callbacks=create_callbacks(
        checkpoint_path=stage_two_checkpoint,
        csv_log_path=stage_two_log,
    ),
)

stage_one_model = keras.models.load_model(
    stage_one_checkpoint,
)

stage_two_model = keras.models.load_model(
    stage_two_checkpoint,
)

stage_one_validation_metrics = stage_one_model.evaluate(
    validation_dataset,
    verbose=0,
    return_dict=True,
)

stage_two_validation_metrics = stage_two_model.evaluate(
    validation_dataset,
    verbose=0,
    return_dict=True,
)

print("\nValidation loss comparison:")
print(
    "    Stage 1 validation loss: "
    f"{stage_one_validation_metrics['loss']:.4f}"
)
print(
    "    Stage 2 validation loss: "
    f"{stage_two_validation_metrics['loss']:.4f}"
)

if (
    stage_two_validation_metrics["loss"]
    < stage_one_validation_metrics["loss"]
):
    best_model = stage_two_model
    selected_stage = "stage 2 fine-tuned model"
else:
    best_model = stage_one_model
    selected_stage = "stage 1 frozen-base model"

print(f"Selected: {selected_stage}")

final_model_path = (
    OUTPUT_DIRECTORY / "vgg16_final_best.keras"
)

best_model.save(final_model_path)
print(f"Final model saved to: {final_model_path.resolve()}")

test_results = best_model.evaluate(
    test_dataset,
    verbose=1,
    return_dict=True,
)

print("\n" + "=" * 60)
print("KERAS TEST METRICS")
print("=" * 60)

for metric_name, metric_value in test_results.items():
    print(f"{metric_name}: {metric_value:.4f}")

prediction_probabilities = best_model.predict(
    test_dataset,
    verbose=1,
)

predicted_labels = np.argmax(
    prediction_probabilities,
    axis=1,
)

true_labels = test_dataframe["label"].to_numpy(
    dtype=np.int32
)

if len(predicted_labels) != len(true_labels):
    raise ValueError(
        "The number of predictions does not match the number "
        "of test labels."
    )

test_accuracy = accuracy_score(
    true_labels,
    predicted_labels,
)

balanced_accuracy = balanced_accuracy_score(
    true_labels,
    predicted_labels,
)

print("\n" + "=" * 60)
print("OVERALL TEST METRICS")
print("=" * 60)

print(f"Accuracy:          {test_accuracy:.4f}")
print(f"Balanced accuracy: {balanced_accuracy:.4f}")

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )
)

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )
)

print("\nMacro-average metrics:")
print(f"Precision: {macro_precision:.4f}")
print(f"Recall:    {macro_recall:.4f}")
print(f"F1-score:  {macro_f1:.4f}")

print("\nWeighted-average metrics:")
print(f"Precision: {weighted_precision:.4f}")
print(f"Recall:    {weighted_recall:.4f}")
print(f"F1-score:  {weighted_f1:.4f}")


report = classification_report(
    true_labels,
    predicted_labels,

    labels=list(range(NUM_CLASSES)),

    target_names=CLASS_NAMES,

    digits=4,
    zero_division=0,
)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(report)


try:
    roc_auc_macro = roc_auc_score(
        true_labels,
        prediction_probabilities,
        multi_class="ovr",
        average="macro",
    )

    roc_auc_weighted = roc_auc_score(
        true_labels,
        prediction_probabilities,
        multi_class="ovr",
        average="weighted",
    )

    print("\nROC-AUC metrics:")
    print(f"Macro ROC-AUC:    {roc_auc_macro:.4f}")
    print(f"Weighted ROC-AUC: {roc_auc_weighted:.4f}")

except ValueError as error:
    print("\nROC-AUC could not be calculated.")
    print(error)


matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(NUM_CLASSES)),
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(matrix)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=CLASS_NAMES,
)

display.plot(
    values_format="d",
    cmap="Blues",
)

plt.title("VGG16 Test Confusion Matrix")
plt.tight_layout()

confusion_matrix_path = OUTPUT_DIRECTORY / "confusion_matrix.png"
plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight",
)
plt.show()


report_path = OUTPUT_DIRECTORY / "classification_report.txt"
report_path.write_text(
    report,
    encoding="utf-8",
)

metrics_rows = [
    {"metric": "test_loss", "value": float(test_results["loss"])},
    {
        "metric": "test_accuracy_keras",
        "value": float(test_results["accuracy"]),
    },
    {"metric": "accuracy", "value": float(test_accuracy)},
    {
        "metric": "balanced_accuracy",
        "value": float(balanced_accuracy),
    },
    {"metric": "macro_precision", "value": float(macro_precision)},
    {"metric": "macro_recall", "value": float(macro_recall)},
    {"metric": "macro_f1", "value": float(macro_f1)},
    {
        "metric": "weighted_precision",
        "value": float(weighted_precision),
    },
    {"metric": "weighted_recall", "value": float(weighted_recall)},
    {"metric": "weighted_f1", "value": float(weighted_f1)},
]

if "roc_auc_macro" in globals():
    metrics_rows.append(
        {
            "metric": "macro_roc_auc",
            "value": float(roc_auc_macro),
        }
    )

if "roc_auc_weighted" in globals():
    metrics_rows.append(
        {
            "metric": "weighted_roc_auc",
            "value": float(roc_auc_weighted),
        }
    )

metrics_dataframe = pd.DataFrame(metrics_rows)
metrics_path = OUTPUT_DIRECTORY / "test_metrics.csv"
metrics_dataframe.to_csv(
    metrics_path,
    index=False,
)

predictions_dataframe = test_dataframe.copy()
predictions_dataframe["predicted_label"] = predicted_labels
predictions_dataframe["predicted_label_name"] = [
    CLASS_NAMES[int(label)]
    for label in predicted_labels
]
predictions_dataframe["correct_prediction"] = (
    predictions_dataframe["label"]
    == predictions_dataframe["predicted_label"]
)

for class_index, class_name in enumerate(CLASS_NAMES):
    safe_class_name = (
        class_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    predictions_dataframe[
        f"probability_{safe_class_name}"
    ] = prediction_probabilities[:, class_index]

predictions_path = OUTPUT_DIRECTORY / "test_predictions.csv"
predictions_dataframe.to_csv(
    predictions_path,
    index=False,
)

print("\nSaved outputs:")
print(f"    Final model:           {final_model_path.resolve()}")
print(f"    Metrics CSV:           {metrics_path.resolve()}")
print(f"    Classification report: {report_path.resolve()}")
print(f"    Confusion matrix:      {confusion_matrix_path.resolve()}")
print(f"    Test predictions:      {predictions_path.resolve()}")