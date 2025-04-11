import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os
import warnings
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

"""
Title: Graph attention network (GAT) for node classification
Author: [akensert](https://github.com/akensert)
Date created: 2021/09/13
Last modified: 2021/12/26
Description: An implementation of a Graph Attention Network (GAT) for node classification.
Accelerator: GPU
"""

"""
## Introduction

[Graph neural networks](https://en.wikipedia.org/wiki/Graph_neural_network)
is the preferred neural network architecture for processing data structured as
graphs (for example, social networks or molecule structures), yielding
better results than fully-connected networks or convolutional networks.

In this tutorial, we will implement a specific graph neural network known as a
[Graph Attention Network](https://arxiv.org/abs/1710.10903) (GAT) to predict labels of
scientific papers based on what type of papers cite them (using the
[Cora](https://linqs.soe.ucsc.edu/data) dataset).

### References

For more information on GAT, see the original paper
[Graph Attention Networks](https://arxiv.org/abs/1710.10903) as well as
[DGL's Graph Attention Networks](https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html)
documentation.
"""

"""
### Import packages
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 6)
pd.set_option("display.max_rows", 6)
np.random.seed(2)

"""
## Obtain the dataset

The preparation of the [Cora dataset](https://linqs.soe.ucsc.edu/data) follows that of the
[Node classification with Graph Neural Networks](https://keras.io/examples/graph/gnn_citations/)
tutorial. Refer to this tutorial for more details on the dataset and exploratory data analysis.
In brief, the Cora dataset consists of two files: `cora.cites` which contains *directed links* (citations) between
papers; and `cora.content` which contains *features* of the corresponding papers and one
of seven labels (the *subject* of the paper).
"""

zip_file = keras.utils.get_file(
    fname="cora.tgz",
    origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    extract=True,
)

data_dir = os.path.join(os.path.dirname(zip_file), "cora")

citations = pd.read_csv(
    os.path.join(data_dir, "cora.cites"),
    sep="\t",
    header=None,
    names=["target", "source"],
)

papers = pd.read_csv(
    os.path.join(data_dir, "cora.content"),
    sep="\t",
    header=None,
    names=["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"],
)

class_values = sorted(papers["subject"].unique())
class_idx = {name: id for id, name in enumerate(class_values)}
paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])

print(citations)

print(papers)

"""
### Split the dataset
"""

# Obtain random indices
random_indices = np.random.permutation(range(papers.shape[0]))

# 50/50 split
train_data = papers.iloc[random_indices[: len(random_indices) // 2]]
test_data = papers.iloc[random_indices[len(random_indices) // 2 :]]

"""
### Prepare the graph data
"""

# Obtain paper indices which will be used to gather node states
# from the graph later on when training the model
train_indices = train_data["paper_id"].to_numpy()
test_indices = test_data["paper_id"].to_numpy()

# Obtain ground truth labels corresponding to each paper_id
train_labels = train_data["subject"].to_numpy()
test_labels = test_data["subject"].to_numpy()

# Define graph, namely an edge tensor and a node feature tensor
edges = tf.convert_to_tensor(citations[["target", "source"]])
node_states = tf.convert_to_tensor(papers.sort_values("paper_id").iloc[:, 1:-1])

# Print shapes of the graph
print("Edges shape:\t\t", edges.shape)
print("Node features shape:", node_states.shape)

"""
## Build the model

GAT takes as input a graph (namely an edge tensor and a node feature tensor) and
outputs \[updated\] node states. The node states are, for each target node, neighborhood
aggregated information of *N*-hops (where *N* is decided by the number of layers of the
GAT). Importantly, in contrast to the
[graph convolutional network](https://arxiv.org/abs/1609.02907) (GCN)
the GAT makes use of attention mechanisms
to aggregate information from neighboring nodes (or *source nodes*). In other words, instead of simply
averaging/summing node states from source nodes (*source papers*) to the target node (*target papers*),
GAT first applies normalized attention scores to each source node state and then sums.
"""

"""
### (Multi-head) graph attention layer

The GAT model implements multi-head graph attention layers. The `MultiHeadGraphAttention`
layer is simply a concatenation (or averaging) of multiple graph attention layers
(`GraphAttention`), each with separate learnable weights `W`. The `GraphAttention` layer
does the following:

Consider inputs node states `h^{l}` which are linearly transformed by `W^{l}`, resulting in `z^{l}`.

For each target node:

1. Computes pair-wise attention scores `a^{l}^{T}(z^{l}_{i}||z^{l}_{j})` for all `j`,
resulting in `e_{ij}` (for all `j`).
`||` denotes a concatenation, `_{i}` corresponds to the target node, and `_{j}`
corresponds to a given 1-hop neighbor/source node.
2. Normalizes `e_{ij}` via softmax, so as the sum of incoming edges' attention scores
to the target node (`sum_{k}{e_{norm}_{ik}}`) will add up to 1.
3. Applies attention scores `e_{norm}_{ij}` to `z_{j}`
and adds it to the new target node state `h^{l+1}_{i}`, for all `j`.
"""


class GraphAttention(layers.Layer):
    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
        )
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
        self.built = True

    def call(self, inputs):
        node_states, edges = inputs

        # Linearly transform node states
        node_states_transformed = tf.matmul(node_states, self.kernel)

        # (1) Compute pair-wise attention scores
        node_states_expanded = tf.gather(node_states_transformed, edges)
        node_states_expanded = tf.reshape(node_states_expanded, (tf.shape(edges)[0], -1))
        attention_scores = tf.nn.leaky_relu(tf.matmul(node_states_expanded, self.kernel_attention))
        attention_scores = tf.squeeze(attention_scores, -1)

        # (2) Normalize attention scores
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores,
            segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,
        )
        attention_scores_sum = tf.repeat(attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32")))
        attention_scores_norm = attention_scores / attention_scores_sum

        # (3) Gather node states of neighbors, apply attention scores and aggregate
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        return out


class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        atom_features, pair_indices = inputs

        # Obtain outputs from each attention head
        outputs = [attention_layer([atom_features, pair_indices]) for attention_layer in self.attention_layers]
        # Concatenate or average the node states from each head
        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        # Activate and return node states
        return tf.nn.relu(outputs)


"""
### Implement training logic with custom `train_step`, `test_step`, and `predict_step` methods

Notice, the GAT model operates on the entire graph (namely, `node_states` and
`edges`) in all phases (training, validation and testing). Hence, `node_states` and
`edges` are passed to the constructor of the `keras.Model` and used as attributes.
The difference between the phases are the indices (and labels), which gathers
certain outputs (`tf.gather(outputs, indices)`).

"""


class GraphAttentionNetwork(keras.Model):
    def __init__(
        self,
        node_states,
        edges,
        hidden_units,
        num_heads,
        num_layers,
        output_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.node_states = node_states
        self.edges = edges
        self.preprocess = layers.Dense(hidden_units * num_heads, activation="relu")
        self.attention_layers = [MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)]
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        node_states, edges = inputs
        x = self.preprocess(node_states)
        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges]) + x
        outputs = self.output_layer(x)
        return outputs

    def train_step(self, data):
        indices, labels = data

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self([self.node_states, self.edges])
            # Compute loss
            loss = self.compiled_loss(labels, tf.gather(outputs, indices))
        # Compute gradients
        grads = tape.gradient(loss, self.trainable_weights)
        # Apply gradients (update weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Update metric(s)
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        indices = data
        # Forward pass
        outputs = self([self.node_states, self.edges])
        # Compute probabilities
        return tf.nn.softmax(tf.gather(outputs, indices))

    def test_step(self, data):
        indices, labels = data
        # Forward pass
        outputs = self([self.node_states, self.edges])
        # Compute loss
        loss = self.compiled_loss(labels, tf.gather(outputs, indices))
        # Update metric(s)
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

        return {m.name: m.result() for m in self.metrics}


"""
### Train and evaluate
"""

# Define hyper-parameters
HIDDEN_UNITS = 100
NUM_HEADS = 8
NUM_LAYERS = 3
OUTPUT_DIM = len(class_values)

NUM_EPOCHS = 100
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 3e-1
MOMENTUM = 0.9

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.SGD(LEARNING_RATE, momentum=MOMENTUM)
accuracy_fn = keras.metrics.SparseCategoricalAccuracy(name="acc")
early_stopping = keras.callbacks.EarlyStopping(monitor="val_acc", min_delta=1e-5, patience=5, restore_best_weights=True)

# Build model
gat_model = GraphAttentionNetwork(node_states, edges, HIDDEN_UNITS, NUM_HEADS, NUM_LAYERS, OUTPUT_DIM)

# Compile model
gat_model.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])

gat_model.fit(
    x=train_indices,
    y=train_labels,
    validation_split=VALIDATION_SPLIT,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=[early_stopping],
    verbose=2,
)

_, test_accuracy = gat_model.evaluate(x=test_indices, y=test_labels, verbose=0)

print("--" * 38 + f"\nTest Accuracy {test_accuracy*100:.1f}%")

"""
### Predict (probabilities)
"""
test_probs = gat_model.predict(x=test_indices)

mapping = {v: k for (k, v) in class_idx.items()}

for i, (probs, label) in enumerate(zip(test_probs[:10], test_labels[:10])):
    print(f"Example {i+1}: {mapping[label]}")
    for j, c in zip(probs, class_idx.keys()):
        print(f"\tProbability of {c: <24} = {j*100:7.3f}%")
    print("---" * 20)

"""
## Conclusions

The results look OK! The GAT model seems to correctly predict the subjects of the papers,
based on what they cite, about 80% of the time. Further improvements could be
made by fine-tuning the hyper-parameters of the GAT. For instance, try changing the number of layers,
the number of hidden units, or the optimizer/learning rate; add regularization (e.g., dropout);
or modify the preprocessing step. We could also try to implement *self-loops*
(i.e., paper X cites paper X) and/or make the graph *undirected*.
"""


# def visualize_training(model, image_size, history, exp_no, stanford=False):
#     # --- Visualize Training History ---
#     if history is not None:
#         acc = history.history["acc"]
#         val_acc = history.history["val_acc"]
#         loss = history.history["loss"]
#         val_loss = history.history["val_loss"]
#         epochs_range = range(len(history.history["acc"]))

#         plt.figure(figsize=(12, 6))
#         plt.subplot(1, 2, 1)
#         plt.plot(epochs_range, acc, label="Training Accuracy")
#         plt.plot(epochs_range, val_acc, label="Validation Accuracy")
#         plt.legend(loc="lower right")
#         plt.title("Training and Validation Accuracy")

#         plt.subplot(1, 2, 2)
#         plt.plot(epochs_range, loss, label="Training Loss")
#         plt.plot(epochs_range, val_loss, label="Validation Loss")
#         plt.legend(loc="upper right")
#         plt.title("Training and Validation Loss")
#         # plt.show() # commented out to prevent display

#         # Save the plot to a file
#         if not stanford:
#             plt.savefig(f"plots/training_history_{exp_no}.png")
#         else:
#             plt.savefig(f"plots/training_history_{exp_no}_stanford.png")
#         plt.close()  # close the figure to prevent memory issues when running on A100

#     # --- Inference on a Single Image ---
#     # Load and process the image
#     img_path = "PetImages/Cat/6779.jpg"
#     if os.path.exists(img_path):
#         img = keras.preprocessing.image.load_img(img_path, target_size=image_size)
#         # plt.imshow(img)
#         # plt.title("Sample Image")
#         # plt.axis("off")
#         # plt.show()

#         img_array = keras.preprocessing.image.img_to_array(img)
#         img_array = tf.expand_dims(img_array, 0)  # Create batch axis

#         # Make prediction
#         predictions = model.predict(img_array)
#         score = float(tf.keras.activations.sigmoid(predictions[0][0]))
#         print(f"Raw prediction score: {score:.4f}")
#         print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
#         if score > 0.5:
#             print(f"This image is likely a dog ({100 * score:.2f}% confidence).")
#         else:
#             print(f"This image is likely a cat ({100 * (1 - score):.2f}% confidence).")
#     else:
#         print(f"Image not found at: {img_path}")


# # --- Model Definition ---
# def make_model(input_shape, num_classes):
#     inputs = keras.Input(shape=input_shape)

#     # Rescaling layer (important to normalize pixel values)
#     x = layers.Rescaling(1.0 / 255)(inputs)

#     # Entry block
#     x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)

#     previous_block_activation = x  # Set aside residual

#     for size in [256, 512, 728]:
#         x = layers.Activation("relu")(x)
#         x = layers.SeparableConv2D(size, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)

#         x = layers.Activation("relu")(x)
#         x = layers.SeparableConv2D(size, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)

#         x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

#         # Project residual
#         residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
#         x = layers.add([x, residual])  # Add back residual
#         previous_block_activation = x  # Set aside next residual

#     x = layers.SeparableConv2D(1024, 3, padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)

#     x = layers.GlobalAveragePooling2D()(x)

#     if num_classes == 2:
#         units = 1
#         activation = "sigmoid"
#     else:
#         units = num_classes
#         activation = "softmax"

#     x = layers.Dropout(0.25)(x)
#     outputs = layers.Dense(units, activation=activation)(x)
#     return keras.Model(inputs, outputs)


# def train_evaluate(model, train_ds, val_ds, callbacks, epochs=25, train=True, evaluate=True):
#     history = None

#     if train:
#         # --- Training the Model ---
#         print("Starting training...")
#         history = model.fit(
#             train_ds,
#             epochs=epochs,
#             callbacks=callbacks,
#             validation_data=val_ds,
#         )
#         print("Training finished.")

#     if evaluate:
#         # --- Evaluate the Model ---
#         loss, accuracy = model.evaluate(val_ds)
#         print(f"Validation Loss: {loss:.4f}")
#         print(f"Validation Accuracy: {accuracy:.4f}")

#     return history


# def compiling(model, num_classes, learning_rate=1e-4):
#     if num_classes == 2:
#         model.compile(
#             optimizer=keras.optimizers.Adam(learning_rate),
#             loss=keras.losses.BinaryCrossentropy(from_logits=False),
#             metrics=[keras.metrics.BinaryAccuracy(name="acc")],
#         )
#     else:
#         model.compile(
#             optimizer=keras.optimizers.Adam(learning_rate),
#             loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#             metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
#         )


# def filter_corrupted_imgs(dataset_path="PetImages"):
#     # Filter out corrupted images (recommended for robustness)
#     num_skipped = 0
#     for folder_name in ("Cat", "Dog"):
#         folder_path = os.path.join(dataset_path, folder_name)
#         for fname in os.listdir(folder_path):
#             fpath = os.path.join(folder_path, fname)
#             try:
#                 fobj = open(fpath, "rb")
#                 is_jfif = b"JFIF" in fobj.peek(10)
#             except Exception as e:
#                 is_jfif = False
#             finally:
#                 if not hasattr(fobj, "closed") or not fobj.closed:
#                     fobj.close()

#             if not is_jfif:
#                 num_skipped += 1
#                 os.remove(fpath)
#     print(f"Deleted {num_skipped} images.")


# def load_stanford_dogs_data(image_size, batch_size):
#     # Load both training and testing splits
#     all_data = tfds.load("stanford_dogs", split="train+test", as_supervised=True)

#     # Resize images before splitting
#     all_data = all_data.map(
#         lambda image, label: (tf.image.resize(image, image_size), label),
#         num_parallel_calls=tf.data.AUTOTUNE,
#     )

#     # Split into 80% training and 20% validation
#     train_size = 0.8
#     total_size = len(list(all_data))
#     train_size = int(train_size * total_size)

#     # Shuffle and split dataset
#     all_data = all_data.shuffle(total_size, seed=1337)
#     train_ds = all_data.take(train_size)
#     val_ds = all_data.skip(train_size)

#     # Batch the datasets
#     train_ds = train_ds.batch(batch_size)
#     val_ds = val_ds.batch(batch_size)

#     # Apply `data_augmentation` to the training images.
#     train_ds = train_ds.map(
#         lambda img, label: (data_augmentation(img, label)),
#         num_parallel_calls=tf.data.AUTOTUNE,
#     )
#     # Prefetching samples in GPU memory helps maximize GPU utilization.
#     train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
#     val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

#     return train_ds, val_ds


# def load_cats_dogs_data(image_size, batch_size, dataset_path="PetImages"):

#     train_ds = image_dataset_from_directory(
#         dataset_path,
#         validation_split=0.2,
#         subset="training",
#         seed=1337,
#         image_size=image_size,
#         batch_size=batch_size,
#     )
#     val_ds = image_dataset_from_directory(
#         dataset_path,
#         validation_split=0.2,
#         subset="validation",
#         seed=1337,
#         image_size=image_size,
#         batch_size=batch_size,
#     )

#     # Apply `data_augmentation` to the training images.
#     train_ds = train_ds.map(
#         lambda img, label: (data_augmentation(img, label)),
#         num_parallel_calls=tf.data.AUTOTUNE,
#     )
#     # Prefetching samples in GPU memory helps maximize GPU utilization.
#     train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
#     val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

#     return train_ds, val_ds


# def run_exp_1_stanford_dogs(
#     image_size=(180, 180), batch_size=64, epochs=25, learning_rate=1e-4, train=True, evaluate=True
# ):

#     exp_no = "experiment_1"

#     print(f"--- Running Experiment: {exp_no} on stanford dogs data! ---")

#     train_ds, val_ds = load_stanford_dogs_data(image_size, batch_size)

#     model = make_model(input_shape=image_size + (3,), num_classes=120)
#     plot_model(model, to_file=f"plots/{exp_no}_stanford_dogs_model.png", show_shapes=True)

#     # Callbacks for better training control
#     callbacks = [
#         keras.callbacks.ModelCheckpoint(f"{exp_no}" + "_stanford_dogs/save_at_{epoch}.keras"),
#         keras.callbacks.ModelCheckpoint(
#             f"{exp_no}_stanford_dogs/best_model.keras", save_best_only=True, monitor="val_acc"
#         ),
#     ]

#     compiling(model, num_classes=120, learning_rate=learning_rate)

#     history = train_evaluate(model, train_ds, val_ds, callbacks, epochs, train, evaluate)

#     visualize_training(model, image_size, history, exp_no, stanford=True)

#     print(f"--- Ending Experiment: {exp_no} on stanford dogs data! ---")


# def run_exp_1_cats_dogs(image_size=(180, 180), batch_size=64, epochs=25, learning_rate=1e-4, train=True, evaluate=True):

#     exp_no = "experiment_1"

#     print(f"--- Running Experiment: {exp_no} on cats and dogs data! ---")

#     train_ds, val_ds = load_cats_dogs_data(image_size, batch_size)

#     model = make_model(input_shape=image_size + (3,), num_classes=2)
#     plot_model(model, to_file=f"plots/{exp_no}_model.png", show_shapes=True)

#     # Callbacks for better training control
#     callbacks = [
#         keras.callbacks.ModelCheckpoint(f"{exp_no}" + "_cats_and_dogs/save_at_{epoch}.keras"),
#         keras.callbacks.ModelCheckpoint(
#             f"{exp_no}_cats_and_dogs/best_model.keras", save_best_only=True, monitor="val_acc"
#         ),
#     ]

#     compiling(model, 2, learning_rate)

#     history = train_evaluate(model, train_ds, val_ds, callbacks, epochs, train, evaluate)

#     visualize_training(model, image_size, history, exp_no)

#     print(f"--- Ending Experiment: {exp_no} on cats and dogs data! ---\n")


# def run_exp_2(image_size=(180, 180), batch_size=64, epochs=25, learning_rate=1e-4, train=True, evaluate=True):
#     exp_no = "experiment_2"

#     print(f"--- Running Experiment: {exp_no} on cats and dogs data! ---")

#     train_ds, val_ds = load_cats_dogs_data(image_size, batch_size)

#     # --- Model Definition ---
#     # Load the model
#     model_path = "experiment_1_stanford_dogs/best_model.keras"
#     model = keras.models.load_model(model_path)

#     # --- Replace the output layer ---
#     # Get the output of the second to last layer
#     x = model.layers[-2].output

#     # Create a new output layer
#     output = layers.Dense(1, activation="sigmoid")(x)

#     # Create a new model
#     model = keras.Model(inputs=model.input, outputs=output)

#     plot_model(model, to_file=f"plots/{exp_no}_model.png", show_shapes=True)

#     # Callbacks for better training control
#     callbacks = [
#         keras.callbacks.ModelCheckpoint(f"{exp_no}" + "_cats_and_dogs/save_at_{epoch}.keras"),
#         keras.callbacks.ModelCheckpoint(
#             f"{exp_no}_cats_and_dogs/best_model.keras", save_best_only=True, monitor="val_acc"
#         ),
#     ]

#     compiling(model, 2, learning_rate)

#     history = train_evaluate(model, train_ds, val_ds, callbacks, epochs, train, evaluate)

#     visualize_training(model, image_size, history, exp_no)

#     print(f"--- Ending Experiment: {exp_no} on cats and dogs data! ---\n")


# def run_exp_3(image_size=(180, 180), batch_size=64, epochs=25, learning_rate=1e-4, train=True, evaluate=True):
#     exp_no = "experiment_3"

#     print(f"--- Running Experiment: {exp_no} on cats and dogs data! ---")

#     train_ds, val_ds = load_cats_dogs_data(image_size, batch_size)

#     # Load original model
#     model_path = "experiment_1_stanford_dogs/best_model.keras"
#     original_model = keras.models.load_model(model_path)

#     # Rebuild the model
#     new_model = make_model(input_shape=image_size + (3,), num_classes=2)

#     # Transfer weights from original model except layers to reset
#     layers_to_reset = {2, 6}
#     num_layers = len(new_model.layers)

#     # We start from 1 to skip the input layer and end the loop on the layer before the output.
#     for i in range(1, num_layers - 1):
#         try:
#             if i not in layers_to_reset:
#                 new_model.layers[i].set_weights(original_model.layers[i].get_weights())
#                 print(f"Copied weights from layer {i}: {new_model.layers[i].name}")
#             else:
#                 print(f"Resetting layer {i}: {new_model.layers[i].name}")
#         except (ValueError, IndexError) as e:
#             print(f"Could not copy weights for layer {i}: {new_model.layers[i].name} - {e}")
#     print(f"Skipping output layer {num_layers - 1}: {new_model.layers[-1].name}")

#     new_model.summary()
#     plot_model(new_model, to_file=f"plots/{exp_no}_model.png", show_shapes=True)

#     # Callbacks for better training control
#     callbacks = [
#         keras.callbacks.ModelCheckpoint(f"{exp_no}" + "_cats_and_dogs/save_at_{epoch}.keras"),
#         keras.callbacks.ModelCheckpoint(
#             f"{exp_no}_cats_and_dogs/best_model.keras", save_best_only=True, monitor="val_acc"
#         ),
#     ]

#     compiling(new_model, 2, learning_rate)

#     history = train_evaluate(new_model, train_ds, val_ds, callbacks, epochs, train, evaluate)

#     visualize_training(new_model, image_size, history, exp_no)

#     print(f"--- Ending Experiment: {exp_no} on cats and dogs data! ---\n")


# def run_exp_4(image_size=(180, 180), batch_size=64, epochs=25, learning_rate=1e-4, train=True, evaluate=True):
#     exp_no = "experiment_4"

#     print(f"--- Running Experiment: {exp_no} on cats and dogs data! ---")

#     train_ds, val_ds = load_cats_dogs_data(image_size, batch_size)

#     # Load original model
#     model_path = "experiment_1_stanford_dogs/best_model.keras"
#     original_model = keras.models.load_model(model_path)

#     # Rebuild the model
#     new_model = make_model(input_shape=image_size + (3,), num_classes=2)

#     # Transfer weights from original model except layers to reset
#     layers_to_reset = {30, 32}
#     num_layers = len(new_model.layers)

#     # We start from 1 to skip the input layer and end the loop on the layer before the output.
#     for i in range(1, num_layers - 1):
#         try:
#             if i not in layers_to_reset:
#                 new_model.layers[i].set_weights(original_model.layers[i].get_weights())
#                 print(f"Copied weights from layer {i}: {new_model.layers[i].name}")
#             else:
#                 print(f"Resetting layer {i}: {new_model.layers[i].name}")
#         except (ValueError, IndexError) as e:
#             print(f"Could not copy weights for layer {i}: {new_model.layers[i].name} - {e}")
#     print(f"Skipping output layer {num_layers - 1}: {new_model.layers[-1].name}")

#     new_model.summary()
#     plot_model(new_model, to_file=f"plots/{exp_no}_model.png", show_shapes=True)

#     # --- Training Configuration ---

#     # Callbacks for better training control
#     callbacks = [
#         keras.callbacks.ModelCheckpoint(f"{exp_no}" + "_cats_and_dogs/save_at_{epoch}.keras"),
#         keras.callbacks.ModelCheckpoint(
#             f"{exp_no}_cats_and_dogs/best_model.keras", save_best_only=True, monitor="val_acc"
#         ),
#     ]

#     compiling(new_model, 2, learning_rate)

#     history = train_evaluate(new_model, train_ds, val_ds, callbacks, epochs, train, evaluate)

#     visualize_training(new_model, image_size, history, exp_no)

#     print(f"--- Ending Experiment: {exp_no} on cats and dogs data! ---")


# def main():

#     image_size = (180, 180)
#     batch_size = 64
#     epochs = 25
#     learning_rate = 1e-4

#     filter_corrupted_imgs()

#     run_exp_1_cats_dogs(image_size, batch_size, epochs, learning_rate)
#     run_exp_1_stanford_dogs(image_size, batch_size, epochs, learning_rate)

#     epochs = 50
#     run_exp_2(image_size, batch_size, epochs, learning_rate)
#     run_exp_3(image_size, batch_size, epochs, learning_rate)
#     run_exp_4(image_size, batch_size, epochs, learning_rate)

#     end_time = time.time()

#     running_time = end_time - start_time
#     hours = int(running_time // 3600)
#     minutes = int((running_time % 3600) // 60)
#     seconds = int(running_time % 60)

#     print(f"Running time: {hours} hours, {minutes} minutes, {seconds} seconds")


# if __name__ == "__main__":
#     main()
