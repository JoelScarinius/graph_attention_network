"""
Title: GAT Regression for Pedestrian Future Position Prediction
Description:
    This script demonstrates how to use a Graph Attention Network (GAT)
    for a regression task over pedestrian trajectory data.

    Each scene is treated as a separate graph. The nodes represent
    pedestrians with features (e.g. current position, previous motion, etc.)
    and the edges represent interactions (or connectivity) between them.

    The model learns to predict the pedestrian's future position, namely
    future_x and future_y one second ahead.

Author: Your Name
Date: 2025-04-13
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings


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


warnings.filterwarnings("ignore")
np.random.seed(2)

# ------------------------------------------------------------------------------
# Data Loading and Preprocessing
# ------------------------------------------------------------------------------


# Define the dataset directory
dataset_dir = "dataset"


# Function to find all scene IDs in the dataset directory
def find_all_scene_ids(dataset_dir):
    scene_ids = []
    for file in os.listdir(dataset_dir):
        if file.endswith(".edges"):
            scene_id = file.split(".")[0]
            scene_ids.append(scene_id)
    return scene_ids


# Function to load all subgraphs for the found scene IDs
def load_all_subgraphs(dataset_dir):
    scene_ids = find_all_scene_ids(dataset_dir)
    scenes = []

    for scene_id in scene_ids:

        edges_file = os.path.join(dataset_dir, f"{scene_id}.edges")
        nodes_file = os.path.join(dataset_dir, f"{scene_id}.nodes")

        # Check if both files exist
        if not os.path.exists(edges_file) or not os.path.exists(nodes_file):
            print(f"Skipping scene ID {scene_id}: Missing files.")
            continue

        # Load edges
        edges = pd.read_csv(edges_file, sep=",", header=None, names=["target", "source"])

        # Load nodes
        nodes = pd.read_csv(
            nodes_file,
            sep=",",
            header=None,
            names=["node_id", "current_x", "current_y", "previous_x", "previous_y", "future_x", "future_y"],
        )

        for col in nodes.columns:
            nodes[col] = pd.to_numeric(nodes[col], errors="coerce")

        if nodes.isnull().any().any():
            # Step 1: Identify rows with NaN values in nodes_df
            nan_nodes = nodes[nodes.isnull().any(axis=1)]

            # Step 2: Extract the node_id values of those rows
            nan_node_ids = nan_nodes["node_id"].tolist()

            # Step 3: Filter out edges in edges_df where source or target is in nan_node_ids
            # Display the filtered edges
            print(f"Original edges count: {len(edges)}")
            print(f"Original nodes count: {len(nodes)}")
            edges = edges[~edges["source"].isin(nan_node_ids) & ~edges["target"].isin(nan_node_ids)]

            print(f"Filtered edges count: {len(edges)}")
            nodes = nodes.dropna(subset=["future_x", "future_y"])
            print(f"Filtered nodes count: {len(nodes)}")

        # # Filter out edges with -1 as source value
        # edges = edges[edges["source"] != -1]

        # Check if there are any -1 edges
        if (edges["source"] == -1).any() or (edges["target"] == -1).any():
            print(f"Scene ID {scene_id} contains -1 edges. Processing...")

            # Remove edges with -1 as source or target
            edges = edges[(edges["source"] != -1) & (edges["target"] != -1)]

            # Get unique node IDs from the remaining edges
            connected_nodes = pd.unique(edges[["target", "source"]].values.ravel())

            # Filter nodes to keep only those that are connected
            nodes = nodes[nodes["node_id"].isin(connected_nodes)]

        # Store the subgraph
        scenes.append(
            {"scene_id": scene_id, "edges": edges, "nodes": nodes},
        )

    return scenes


# Example usage
scenes = load_all_subgraphs(dataset_dir)
print(f"Loaded {len(scenes)} scenes.")


def aggregate_scenes(scenes):
    nodes_list = []
    edges_list = []
    scene_node_indices = {}
    node_offset = 0

    for scene in scenes:
        scene_id = scene["scene_id"]
        nodes_df = scene["nodes"].copy().reset_index(drop=True)
        edges_df = scene["edges"].copy().reset_index(drop=True)
        num_nodes = nodes_df.shape[0]
        scene_node_indices[scene_id] = np.arange(node_offset, node_offset + num_nodes)

        # Map original node_id to aggregated index.
        node_id_to_index = dict(zip(nodes_df["node_id"], range(node_offset, node_offset + num_nodes)))
        edges_df["target"] = edges_df["target"].apply(lambda x: node_id_to_index.get(x, -1))
        edges_df["source"] = edges_df["source"].apply(lambda x: node_id_to_index.get(x, -1))
        edges_df = edges_df[(edges_df["target"] != -1) & (edges_df["source"] != -1)]
        nodes_list.append(nodes_df)
        edges_list.append(edges_df)
        node_offset += num_nodes

    all_nodes = pd.concat(nodes_list, ignore_index=True)
    all_edges = pd.concat(edges_list, ignore_index=True).to_numpy().astype(np.int32)
    return all_nodes, all_edges, scene_node_indices


def scene_based_split(scene_node_indices, train_ratio=0.5):
    scene_ids = np.array(list(scene_node_indices.keys()))
    np.random.shuffle(scene_ids)
    n_train = int(len(scene_ids) * train_ratio)
    train_scenes = scene_ids[:n_train]
    test_scenes = scene_ids[n_train:]
    train_indices = np.concatenate([scene_node_indices[sid] for sid in train_scenes])
    test_indices = np.concatenate([scene_node_indices[sid] for sid in test_scenes])
    return train_indices, test_indices


def create_train_val_split(train_indices, val_ratio=0.1):
    # Randomly split the train_indices into training and validation sets.
    np.random.shuffle(train_indices)
    n_val = int(len(train_indices) * val_ratio)
    val_indices = train_indices[:n_val]
    train_indices_new = train_indices[n_val:]
    return train_indices_new, val_indices


# ------------------------------------------------------------------------------
# GAT Model Components for Regression
# ------------------------------------------------------------------------------


class GraphAttention(layers.Layer):
    def __init__(self, units, kernel_initializer="glorot_uniform", kernel_regularizer=None, **kwargs):
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
        self.built = True  # Original

    def call(self, inputs):
        node_states, edges = inputs
        node_states_transformed = tf.matmul(node_states, self.kernel)
        target_states = tf.gather(node_states_transformed, edges[:, 0])
        source_states = tf.gather(node_states_transformed, edges[:, 1])
        concat_features = tf.concat([target_states, source_states], axis=-1)
        e = tf.nn.leaky_relu(tf.matmul(concat_features, self.kernel_attention))
        e = tf.squeeze(e, axis=-1)
        e = tf.exp(tf.clip_by_value(e, -2, 2))
        sum_e = tf.math.unsorted_segment_sum(e, edges[:, 0], num_segments=tf.shape(node_states)[0])
        sum_e_rep = tf.gather(sum_e, edges[:, 0])
        attention = e / (sum_e_rep + 1e-9)
        source_transformed = tf.gather(node_states_transformed, edges[:, 1])
        messages = source_transformed * tf.expand_dims(attention, -1)
        output = tf.math.unsorted_segment_sum(messages, edges[:, 0], num_segments=tf.shape(node_states)[0])
        return output


class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        node_states, edges = inputs
        head_outputs = [att([node_states, edges]) for att in self.attention_layers]
        if self.merge_type == "concat":
            output = tf.concat(head_outputs, axis=-1)
        else:
            output = tf.reduce_mean(tf.stack(head_outputs, axis=-1), axis=-1)
        return tf.nn.relu(output)


class GraphAttentionNetwork(keras.Model):
    def __init__(self, node_states, edges, hidden_units, num_heads, num_layers, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.node_states = node_states
        self.edges = edges
        self.preprocess = layers.Dense(hidden_units * num_heads, activation="relu")
        self.attention_layers = [MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)]
        self.output_layer = layers.Dense(output_dim)

    def call(self, _=None):
        x = self.preprocess(self.node_states)
        for att_layer in self.attention_layers:
            x = att_layer([x, self.edges]) + x  # residual connection
        return self.output_layer(x)

    def train_step(self, data):
        indices, labels = data
        with tf.GradientTape() as tape:
            # outputs = self([None])
            outputs = self()
            predictions = tf.gather(outputs, indices)
            loss = self.compiled_loss(labels, predictions, regularization_losses=self.losses)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compiled_metrics.update_state(labels, predictions)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = loss
        return results

    def test_step(self, data):
        indices, labels = data
        outputs = self([None])
        predictions = tf.gather(outputs, indices)
        loss = self.compiled_loss(labels, predictions, regularization_losses=self.losses)
        self.compiled_metrics.update_state(labels, predictions)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = loss
        return results

    def predict_step(self, data):
        outputs = self([None])
        predictions = tf.gather(outputs, data)
        return predictions


# ------------------------------------------------------------------------------
# Main: Data Preparation, Model Training, and Evaluation
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    all_nodes, all_edges, scene_node_indices = aggregate_scenes(scenes)
    train_indices, test_indices = scene_based_split(scene_node_indices, train_ratio=0.8)
    # Create a validation set from the training nodes (e.g., 10% of training nodes)
    train_indices, val_indices = create_train_val_split(train_indices, val_ratio=0.1)

    # Define features and targets
    feature_cols = [col for col in all_nodes.columns if col not in ["node_id", "future_x", "future_y"]]
    target_cols = ["future_x", "future_y"]
    node_features_np = all_nodes[feature_cols].to_numpy().astype(np.float32)
    targets_np = all_nodes[target_cols].to_numpy().astype(np.float32)

    print("Aggregated nodes shape:", node_features_np.shape)
    print("Aggregated edges shape:", all_edges.shape)
    print(
        "Training nodes:",
        train_indices.shape,
        "Validation nodes:",
        val_indices.shape,
        "Test nodes:",
        test_indices.shape,
    )

    node_features_tensor = tf.convert_to_tensor(node_features_np)
    edges_tensor = tf.convert_to_tensor(all_edges)

    # Define hyper-parameters
    HIDDEN_UNITS = 100
    NUM_HEADS = 8
    NUM_LAYERS = 3
    OUTPUT_DIM = 2  # future_x and future_y
    NUM_EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-2

    # Build the model
    gat_model = GraphAttentionNetwork(
        node_states=node_features_tensor,
        edges=edges_tensor,
        hidden_units=HIDDEN_UNITS,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        output_dim=OUTPUT_DIM,
    )

    # Compile the model with MSE for loss and MAE as a metric.
    gat_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.MeanSquaredError(),
        metrics=[
            keras.metrics.MeanAbsoluteError(),
            keras.metrics.MeanSquaredError(),
            keras.metrics.R2Score(),
        ],
    )

    # Prepare tf.data.Datasets for training, validation, and testing.
    train_dataset = tf.data.Dataset.from_tensor_slices((train_indices, targets_np[train_indices]))
    train_dataset = train_dataset.shuffle(buffer_size=len(train_indices)).batch(BATCH_SIZE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_indices, targets_np[val_indices]))
    val_dataset = val_dataset.batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_indices, targets_np[test_indices]))
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # Set up callbacks: ReduceLROnPlateau and EarlyStopping.
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=10,
        verbose=1,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0.0,
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    )

    print("Training...")
    gat_model.fit(
        train_dataset,
        epochs=NUM_EPOCHS,
        validation_data=val_dataset,
        callbacks=[reduce_lr, early_stopping],
        verbose=2,
    )

    print("Evaluating on test set...")
    results = gat_model.evaluate(test_dataset, verbose=2)
    print(f"\nTest Loss (MSE): {results[0]:.4f}, Test MAE: {results[1]["mean_absolute_error"]:.4f}")

    # Run predictions on test nodes
    print("\nSample predictions for test nodes:")
    predictions = gat_model.predict(tf.convert_to_tensor(test_indices))
    for i, idx in enumerate(test_indices[:5]):
        print(
            f"Node {idx}: True future_x={targets_np[idx,0]:.1f}, future_y={targets_np[idx,1]:.1f} | Predicted future_x={predictions[i,0]:.1f}, future_y={predictions[i,1]:.1f}"
        )
