import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import time

start_time = time.time()

# --- GPU Configuration for Optimal Utilization ---
# Check TensorFlow and CUDA compatibility
print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Set memory growth to True to avoid allocating all memory at once
        tf.config.experimental.set_memory_growth(gpus[0], True)

        # Set the visible devices to use only the first GPU (if multiple are available)
        tf.config.set_visible_devices(gpus[0], "GPU")
        print("Using GPU:", gpus[0])

        # Enable mixed precision training for potential speedup and reduced memory usage
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Using mixed precision training.")

    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. Training will run on CPU.")

# Data augmentation as a separate function for clarity
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(image, label):

    for layer in data_augmentation_layers:
        image = layer(image)
    return image, label


def visualize_training(model, image_size, history, exp_no, stanford=False):
    # --- Visualize Training History ---
    if history is not None:
        acc = history.history["acc"]
        val_acc = history.history["val_acc"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs_range = range(len(history.history["acc"]))

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label="Training Accuracy")
        plt.plot(epochs_range, val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label="Training Loss")
        plt.plot(epochs_range, val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")
        # plt.show() # commented out to prevent display

        # Save the plot to a file
        if not stanford:
            plt.savefig(f"plots/training_history_{exp_no}.png")
        else:
            plt.savefig(f"plots/training_history_{exp_no}_stanford.png")
        plt.close()  # close the figure to prevent memory issues when running on A100

    # --- Inference on a Single Image ---
    # Load and process the image
    img_path = "PetImages/Cat/6779.jpg"
    if os.path.exists(img_path):
        img = keras.preprocessing.image.load_img(img_path, target_size=image_size)
        # plt.imshow(img)
        # plt.title("Sample Image")
        # plt.axis("off")
        # plt.show()

        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        # Make prediction
        predictions = model.predict(img_array)
        score = float(tf.keras.activations.sigmoid(predictions[0][0]))
        print(f"Raw prediction score: {score:.4f}")
        print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
        if score > 0.5:
            print(f"This image is likely a dog ({100 * score:.2f}% confidence).")
        else:
            print(f"This image is likely a cat ({100 * (1 - score):.2f}% confidence).")
    else:
        print(f"Image not found at: {img_path}")


# --- Model Definition ---
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Rescaling layer (important to normalize pixel values)
    x = layers.Rescaling(1.0 / 255)(inputs)

    # Entry block
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    if num_classes == 2:
        units = 1
        activation = "sigmoid"
    else:
        units = num_classes
        activation = "softmax"

    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


def train_evaluate(model, train_ds, val_ds, callbacks, epochs=25, train=True, evaluate=True):
    history = None

    if train:
        # --- Training the Model ---
        print("Starting training...")
        history = model.fit(
            train_ds,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_ds,
        )
        print("Training finished.")

    if evaluate:
        # --- Evaluate the Model ---
        loss, accuracy = model.evaluate(val_ds)
        print(f"Validation Loss: {loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")

    return history


def compiling(model, num_classes, learning_rate=1e-4):
    if num_classes == 2:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[keras.metrics.BinaryAccuracy(name="acc")],
        )
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
        )


def filter_corrupted_imgs(dataset_path="PetImages"):
    # Filter out corrupted images (recommended for robustness)
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(dataset_path, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = b"JFIF" in fobj.peek(10)
            except Exception as e:
                is_jfif = False
            finally:
                if not hasattr(fobj, "closed") or not fobj.closed:
                    fobj.close()

            if not is_jfif:
                num_skipped += 1
                os.remove(fpath)
    print(f"Deleted {num_skipped} images.")


def load_stanford_dogs_data(image_size, batch_size):
    # Load both training and testing splits
    all_data = tfds.load("stanford_dogs", split="train+test", as_supervised=True)

    # Resize images before splitting
    all_data = all_data.map(
        lambda image, label: (tf.image.resize(image, image_size), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Split into 80% training and 20% validation
    train_size = 0.8
    total_size = len(list(all_data))
    train_size = int(train_size * total_size)

    # Shuffle and split dataset
    all_data = all_data.shuffle(total_size, seed=1337)
    train_ds = all_data.take(train_size)
    val_ds = all_data.skip(train_size)

    # Batch the datasets
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    # Apply `data_augmentation` to the training images.
    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img, label)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def load_cats_dogs_data(image_size, batch_size, dataset_path="PetImages"):

    train_ds = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    # Apply `data_augmentation` to the training images.
    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img, label)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def run_exp_1_stanford_dogs(
    image_size=(180, 180), batch_size=64, epochs=25, learning_rate=1e-4, train=True, evaluate=True
):

    exp_no = "experiment_1"

    print(f"--- Running Experiment: {exp_no} on stanford dogs data! ---")

    train_ds, val_ds = load_stanford_dogs_data(image_size, batch_size)

    model = make_model(input_shape=image_size + (3,), num_classes=120)
    plot_model(model, to_file=f"plots/{exp_no}_stanford_dogs_model.png", show_shapes=True)

    # Callbacks for better training control
    callbacks = [
        keras.callbacks.ModelCheckpoint(f"{exp_no}" + "_stanford_dogs/save_at_{epoch}.keras"),
        keras.callbacks.ModelCheckpoint(
            f"{exp_no}_stanford_dogs/best_model.keras", save_best_only=True, monitor="val_acc"
        ),
    ]

    compiling(model, num_classes=120, learning_rate=learning_rate)

    history = train_evaluate(model, train_ds, val_ds, callbacks, epochs, train, evaluate)

    visualize_training(model, image_size, history, exp_no, stanford=True)

    print(f"--- Ending Experiment: {exp_no} on stanford dogs data! ---")


def run_exp_1_cats_dogs(image_size=(180, 180), batch_size=64, epochs=25, learning_rate=1e-4, train=True, evaluate=True):

    exp_no = "experiment_1"

    print(f"--- Running Experiment: {exp_no} on cats and dogs data! ---")

    train_ds, val_ds = load_cats_dogs_data(image_size, batch_size)

    model = make_model(input_shape=image_size + (3,), num_classes=2)
    plot_model(model, to_file=f"plots/{exp_no}_model.png", show_shapes=True)

    # Callbacks for better training control
    callbacks = [
        keras.callbacks.ModelCheckpoint(f"{exp_no}" + "_cats_and_dogs/save_at_{epoch}.keras"),
        keras.callbacks.ModelCheckpoint(
            f"{exp_no}_cats_and_dogs/best_model.keras", save_best_only=True, monitor="val_acc"
        ),
    ]

    compiling(model, 2, learning_rate)

    history = train_evaluate(model, train_ds, val_ds, callbacks, epochs, train, evaluate)

    visualize_training(model, image_size, history, exp_no)

    print(f"--- Ending Experiment: {exp_no} on cats and dogs data! ---\n")


def run_exp_2(image_size=(180, 180), batch_size=64, epochs=25, learning_rate=1e-4, train=True, evaluate=True):
    exp_no = "experiment_2"

    print(f"--- Running Experiment: {exp_no} on cats and dogs data! ---")

    train_ds, val_ds = load_cats_dogs_data(image_size, batch_size)

    # --- Model Definition ---
    # Load the model
    model_path = "experiment_1_stanford_dogs/best_model.keras"
    model = keras.models.load_model(model_path)

    # --- Replace the output layer ---
    # Get the output of the second to last layer
    x = model.layers[-2].output

    # Create a new output layer
    output = layers.Dense(1, activation="sigmoid")(x)

    # Create a new model
    model = keras.Model(inputs=model.input, outputs=output)

    plot_model(model, to_file=f"plots/{exp_no}_model.png", show_shapes=True)

    # Callbacks for better training control
    callbacks = [
        keras.callbacks.ModelCheckpoint(f"{exp_no}" + "_cats_and_dogs/save_at_{epoch}.keras"),
        keras.callbacks.ModelCheckpoint(
            f"{exp_no}_cats_and_dogs/best_model.keras", save_best_only=True, monitor="val_acc"
        ),
    ]

    compiling(model, 2, learning_rate)

    history = train_evaluate(model, train_ds, val_ds, callbacks, epochs, train, evaluate)

    visualize_training(model, image_size, history, exp_no)

    print(f"--- Ending Experiment: {exp_no} on cats and dogs data! ---\n")


def run_exp_3(image_size=(180, 180), batch_size=64, epochs=25, learning_rate=1e-4, train=True, evaluate=True):
    exp_no = "experiment_3"

    print(f"--- Running Experiment: {exp_no} on cats and dogs data! ---")

    train_ds, val_ds = load_cats_dogs_data(image_size, batch_size)

    # Load original model
    model_path = "experiment_1_stanford_dogs/best_model.keras"
    original_model = keras.models.load_model(model_path)

    # Rebuild the model
    new_model = make_model(input_shape=image_size + (3,), num_classes=2)

    # Transfer weights from original model except layers to reset
    layers_to_reset = {2, 6}
    num_layers = len(new_model.layers)

    # We start from 1 to skip the input layer and end the loop on the layer before the output.
    for i in range(1, num_layers - 1):
        try:
            if i not in layers_to_reset:
                new_model.layers[i].set_weights(original_model.layers[i].get_weights())
                print(f"Copied weights from layer {i}: {new_model.layers[i].name}")
            else:
                print(f"Resetting layer {i}: {new_model.layers[i].name}")
        except (ValueError, IndexError) as e:
            print(f"Could not copy weights for layer {i}: {new_model.layers[i].name} - {e}")
    print(f"Skipping output layer {num_layers - 1}: {new_model.layers[-1].name}")

    new_model.summary()
    plot_model(new_model, to_file=f"plots/{exp_no}_model.png", show_shapes=True)

    # Callbacks for better training control
    callbacks = [
        keras.callbacks.ModelCheckpoint(f"{exp_no}" + "_cats_and_dogs/save_at_{epoch}.keras"),
        keras.callbacks.ModelCheckpoint(
            f"{exp_no}_cats_and_dogs/best_model.keras", save_best_only=True, monitor="val_acc"
        ),
    ]

    compiling(new_model, 2, learning_rate)

    history = train_evaluate(new_model, train_ds, val_ds, callbacks, epochs, train, evaluate)

    visualize_training(new_model, image_size, history, exp_no)

    print(f"--- Ending Experiment: {exp_no} on cats and dogs data! ---\n")


def run_exp_4(image_size=(180, 180), batch_size=64, epochs=25, learning_rate=1e-4, train=True, evaluate=True):
    exp_no = "experiment_4"

    print(f"--- Running Experiment: {exp_no} on cats and dogs data! ---")

    train_ds, val_ds = load_cats_dogs_data(image_size, batch_size)

    # Load original model
    model_path = "experiment_1_stanford_dogs/best_model.keras"
    original_model = keras.models.load_model(model_path)

    # Rebuild the model
    new_model = make_model(input_shape=image_size + (3,), num_classes=2)

    # Transfer weights from original model except layers to reset
    layers_to_reset = {30, 32}
    num_layers = len(new_model.layers)

    # We start from 1 to skip the input layer and end the loop on the layer before the output.
    for i in range(1, num_layers - 1):
        try:
            if i not in layers_to_reset:
                new_model.layers[i].set_weights(original_model.layers[i].get_weights())
                print(f"Copied weights from layer {i}: {new_model.layers[i].name}")
            else:
                print(f"Resetting layer {i}: {new_model.layers[i].name}")
        except (ValueError, IndexError) as e:
            print(f"Could not copy weights for layer {i}: {new_model.layers[i].name} - {e}")
    print(f"Skipping output layer {num_layers - 1}: {new_model.layers[-1].name}")

    new_model.summary()
    plot_model(new_model, to_file=f"plots/{exp_no}_model.png", show_shapes=True)

    # --- Training Configuration ---

    # Callbacks for better training control
    callbacks = [
        keras.callbacks.ModelCheckpoint(f"{exp_no}" + "_cats_and_dogs/save_at_{epoch}.keras"),
        keras.callbacks.ModelCheckpoint(
            f"{exp_no}_cats_and_dogs/best_model.keras", save_best_only=True, monitor="val_acc"
        ),
    ]

    compiling(new_model, 2, learning_rate)

    history = train_evaluate(new_model, train_ds, val_ds, callbacks, epochs, train, evaluate)

    visualize_training(new_model, image_size, history, exp_no)

    print(f"--- Ending Experiment: {exp_no} on cats and dogs data! ---")


def main():

    image_size = (180, 180)
    batch_size = 64
    epochs = 25
    learning_rate = 1e-4

    filter_corrupted_imgs()

    run_exp_1_cats_dogs(image_size, batch_size, epochs, learning_rate)
    run_exp_1_stanford_dogs(image_size, batch_size, epochs, learning_rate)

    epochs = 50
    run_exp_2(image_size, batch_size, epochs, learning_rate)
    run_exp_3(image_size, batch_size, epochs, learning_rate)
    run_exp_4(image_size, batch_size, epochs, learning_rate)

    end_time = time.time()

    running_time = end_time - start_time
    hours = int(running_time // 3600)
    minutes = int((running_time % 3600) // 60)
    seconds = int(running_time % 60)

    print(f"Running time: {hours} hours, {minutes} minutes, {seconds} seconds")


if __name__ == "__main__":
    main()
