import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import warnings
import time


def find_all_scene_ids(dataset_dir):
    scene_ids = []
    for file in os.listdir(dataset_dir):
        if file.endswith(".edges"):
            scene_id = file.split(".")[0]
            scene_ids.append(scene_id)
    return scene_ids


def load_all_subgraphs(dataset_dir):
    scene_ids = find_all_scene_ids(dataset_dir)
    scenes = []
    for scene_id in scene_ids:
        edges_file = os.path.join(dataset_dir, f"{scene_id}.edges")
        nodes_file = os.path.join(dataset_dir, f"{scene_id}.nodes")
        if not os.path.exists(edges_file) or not os.path.exists(nodes_file):
            print(f"Skipping scene ID {scene_id}: Missing files.")
            continue

        edges = pd.read_csv(edges_file, sep=",", header=None, names=["target", "source"])
        nodes = pd.read_csv(
            nodes_file,
            sep=",",
            header=None,
            names=["node_id", "current_x", "current_y", "previous_x", "previous_y", "future_x", "future_y"],
        )
        for col in nodes.columns:
            nodes[col] = pd.to_numeric(nodes[col], errors="coerce")

        if nodes.isnull().any().any():
            nan_nodes = nodes[nodes.isnull().any(axis=1)]
            nan_node_ids = nan_nodes["node_id"].tolist()
            print(f"Scene {scene_id}: Filtering {len(nan_node_ids)} nodes with NaN values.")
            edges = edges[~edges["source"].isin(nan_node_ids) & ~edges["target"].isin(nan_node_ids)]
            nodes = nodes.dropna(subset=["future_x", "future_y"])

        if (edges["source"] == -1).any() or (edges["target"] == -1).any():
            print(f"Scene {scene_id} contains -1 edges. Removing these edges.")
            edges = edges[(edges["source"] != -1) & (edges["target"] != -1)]
            connected_nodes = pd.unique(edges[["target", "source"]].values.ravel())
            nodes = nodes[nodes["node_id"].isin(connected_nodes)]
        if len(nodes) > 0:
            scenes.append({"scene_id": scene_id, "edges": edges, "nodes": nodes})
        else:
            print(f"NOTE! Scene {scene_id} skipped: no valid nodes after filtering.")
    return scenes


def convert_scene_to_tensors(scene, feature_cols, target_cols):
    nodes_df = scene["nodes"].reset_index(drop=True)
    edges_df = scene["edges"].reset_index(drop=True)
    node_id_to_idx = {nid: i for i, nid in enumerate(nodes_df["node_id"])}
    edges_df = edges_df.copy()
    edges_df["target"] = edges_df["target"].map(node_id_to_idx)
    edges_df["source"] = edges_df["source"].map(node_id_to_idx)
    edges_df = edges_df.dropna().astype(int)
    features = nodes_df[feature_cols].to_numpy().astype(np.float32)
    targets = nodes_df[target_cols].to_numpy().astype(np.float32)
    edges = edges_df.to_numpy().astype(np.int32)
    return features, edges, targets


def split_scenes(scenes, train_ratio=0.7, val_ratio=0.15):
    np.random.shuffle(scenes)
    n_total = len(scenes)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_scenes = scenes[:n_train]
    val_scenes = scenes[n_train : n_train + n_val]
    test_scenes = scenes[n_train + n_val :]
    return train_scenes, val_scenes, test_scenes


def scene_generator(scene_list, feature_cols, target_cols):
    for scene in scene_list:
        yield convert_scene_to_tensors(scene, feature_cols, target_cols)


def squeeze_batch(features, edges, targets):
    return tf.squeeze(features, axis=0), tf.squeeze(edges, axis=0), tf.squeeze(targets, axis=0)


def mean_euclidean_distance(y_true, y_pred):
    return tf.reduce_mean(tf.norm(y_true - y_pred, axis=-1))


def compile_and_train(gat_model, train_dataset, val_dataset, epochs, learning_rate):
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    metrics = [
        keras.metrics.MeanAbsoluteError(),
        keras.metrics.MeanSquaredError(),
        keras.metrics.RootMeanSquaredError(name="rmse"),
        keras.metrics.R2Score(),
        keras.metrics.MeanMetricWrapper(mean_euclidean_distance, name="mean_euclidean_distance"),
    ]

    gat_model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-5, patience=15, verbose=1, restore_best_weights=True, start_from_epoch=0
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=5, verbose=1, min_delta=1e-4, min_lr=1e-6
    )

    print("Training...")
    history = gat_model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[reduce_lr, early_stopping],
        verbose=2,
    )

    return gat_model, history


def evaluate_and_plot(gat_model, history, test_dataset, task, run="1"):
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    print("Evaluating on test dataset...")
    results = gat_model.evaluate(test_dataset, verbose=2)
    print("Test metrics:", results)

    print("\nSample predictions for test scenes:")
    for features, edges, targets in test_dataset.take(1):
        predictions = gat_model((features, edges), training=False)
        for i in range(min(5, predictions.shape[0])):
            print(
                f"Node {i}: True future_x={targets[i, 0]:.1f}, future_y={targets[i, 1]:.1f} | "
                f"Predicted future_x={predictions[i, 0]:.1f}, future_y={predictions[i, 1]:.1f}"
            )
        plt.figure(figsize=(8, 8))
        plt.scatter(targets[:20, 0], targets[:20, 1], label="True", c="g")
        plt.scatter(predictions[:20, 0], predictions[:20, 1], label="Pred", c="r", marker="x")
        plt.legend()
        targets_np = targets[:20].numpy()
        predictions_np = predictions[:20].numpy()

        x_min = int(np.floor(min(targets_np[:, 0].min(), predictions_np[:, 0].min())))
        x_max = int(np.ceil(max(targets_np[:, 0].max(), predictions_np[:, 0].max())))
        y_min = int(np.floor(min(targets_np[:, 1].min(), predictions_np[:, 1].min())))
        y_max = int(np.ceil(max(targets_np[:, 1].max(), predictions_np[:, 1].max())))

        plt.xticks(np.arange(x_min, x_max + 1, 500), rotation=45)
        plt.yticks(np.arange(y_min, y_max + 1, 500))
        plt.xlabel("future_x")
        plt.ylabel("future_y")
        plt.title("True vs Predicted Future Positions")
        if task == 2:
            plt.savefig(os.path.join(plot_dir, f"task_{task}_run_{run}_scatter.png"))
        else:
            plt.savefig(os.path.join(plot_dir, f"task_{task}_scatter.png"))
        plt.close()

    med = history.history["mean_euclidean_distance"]
    mse = history.history["mean_absolute_error"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(len(history.history["val_loss"]))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, med, label="Mean Euclidean Distance")
    plt.plot(epochs_range, mse, label="Mean Absolute Error")
    plt.xlabel("Epoch")
    plt.ylabel("Distance/Error")
    plt.legend(loc="upper right")
    plt.title("MED and MAE")
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.yscale("log")

    if task == 2:
        plt.savefig(os.path.join(plot_dir, f"task_{task}_run_{run}_history.png"))
    else:
        plt.savefig(os.path.join(plot_dir, f"task_{task}_history.png"))

    plt.close()


# -------------------------
# Define Model Components
# -------------------------
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
        self.built = True

    def call(self, inputs):
        node_states, edges = inputs

        # Linearly transform node states
        node_states_transformed = tf.matmul(node_states, self.kernel)

        # (1) Compute pair-wise attention scores
        target_states = tf.gather(node_states_transformed, edges[:, 0])
        source_states = tf.gather(node_states_transformed, edges[:, 1])
        concat_features = tf.concat([target_states, source_states], axis=-1)
        attention_scores = tf.nn.leaky_relu(tf.matmul(concat_features, self.kernel_attention))
        attention_scores = tf.squeeze(attention_scores, axis=-1)

        # (2) Normalize attention scores
        attention_scores = tf.exp(tf.clip_by_value(attention_scores, -2, 2))
        num_nodes = tf.shape(node_states)[0]
        attention_sum = tf.math.unsorted_segment_sum(attention_scores, segment_ids=edges[:, 0], num_segments=num_nodes)
        normalized_attention = attention_scores / tf.gather(attention_sum, edges[:, 0])

        # (3) Gather node states of neighbors, apply attention scores and aggregate
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * normalized_attention[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=num_nodes,
        )
        return out


class CosineSimilarityGraphAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer="glorot_uniform",
            name="kernel",
        )
        super().build(input_shape)

    def call(self, inputs):
        node_states, edges = inputs

        # Linearly transform node states
        node_states_transformed = tf.matmul(node_states, self.kernel)

        target_states = tf.gather(node_states_transformed, edges[:, 0])
        source_states = tf.gather(node_states_transformed, edges[:, 1])

        # Normalized vectors (safe cosine similarity)
        normalized_target = tf.math.l2_normalize(target_states, axis=-1, epsilon=1e-8)
        normalized_source = tf.math.l2_normalize(source_states, axis=-1, epsilon=1e-8)

        cosine_sim = tf.reduce_sum(normalized_target * normalized_source, axis=-1)

        # Stable softmax over incoming edges
        num_nodes = tf.shape(node_states)[0]
        segment_max = tf.math.unsorted_segment_max(cosine_sim, edges[:, 0], num_nodes)
        shifted_sim = cosine_sim - tf.gather(segment_max, edges[:, 0])
        exp_sim = tf.exp(shifted_sim)

        attention_sum = tf.math.unsorted_segment_sum(exp_sim, edges[:, 0], num_nodes)
        normalized_attention = exp_sim / (tf.gather(attention_sum, edges[:, 0]) + 1e-8)

        # Weighted sum of source node features
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * normalized_attention[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=num_nodes,
        )
        return out


class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        node_features, edges = inputs

        # Obtain outputs from each attention head
        outputs = [attn([node_features, edges]) for attn in self.attention_layers]
        # Concatenate or average the node states from each head
        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        # Activate and return node states
        return tf.nn.relu(outputs)


class MultiHeadCosineGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [CosineSimilarityGraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        node_features, edges = inputs

        # Obtain outputs from each attention head
        outputs = [attention_layer([node_features, edges]) for attention_layer in self.attention_layers]
        # Concatenate or average the node states from each head
        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        # Activate and return node states
        return tf.nn.relu(outputs)


class GraphAttentionNetwork(keras.Model):
    def __init__(self, hidden_units, num_heads, num_layers, output_dim, task, **kwargs):
        super().__init__(**kwargs)
        if task == 2:
            self.preprocess = keras.Sequential(
                [
                    layers.Dense(hidden_units * num_heads, activation="relu"),
                    layers.Dense(hidden_units * num_heads, activation="relu"),
                    layers.Dense(hidden_units * num_heads, activation=None),
                ]
            )
        else:
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
        node_features, edges, targets = data

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self([node_features, edges], training=True)
            # Compute loss
            loss = self.compiled_loss(targets, outputs)
        # Compute gradients
        grads = tape.gradient(loss, self.trainable_weights)
        # Apply gradients (update wights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Update metric(s)
        self.compiled_metrics.update_state(targets, outputs)
        logs = {m.name: m.result() for m in self.metrics}
        logs["loss"] = loss

        return logs

    def predict_step(self, data):
        node_features, edges, _ = data
        # Forward pass
        outputs = self([node_features, edges], training=False)
        return outputs

    def test_step(self, data):
        node_features, edges, targets = data
        # Forward pass
        outputs = self([node_features, edges], training=False)
        # Compute loss
        loss = self.compiled_loss(targets, outputs)
        # Update metric(s)
        self.compiled_metrics.update_state(targets, outputs)
        logs = {m.name: m.result() for m in self.metrics}
        logs["loss"] = loss

        return logs


class CosineGraphAttentionNetwork(keras.Model):
    def __init__(self, hidden_units, num_heads, num_layers, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.preprocess = layers.Dense(hidden_units * num_heads, activation="relu")
        self.attention_layers = [MultiHeadCosineGraphAttention(hidden_units, num_heads) for _ in range(num_layers)]
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        node_features, edges = inputs
        x = self.preprocess(node_features)
        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges]) + x
        outputs = self.output_layer(x)
        return outputs

    def train_step(self, data):
        node_features, edges, targets = data
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self([node_features, edges], training=True)
            # Compute loss
            loss = self.compiled_loss(targets, outputs)
        # Compute gradients
        grads = tape.gradient(loss, self.trainable_weights)
        # Apply gradients (update weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Update metric(s)
        self.compiled_metrics.update_state(targets, outputs)
        logs = {m.name: m.result() for m in self.metrics}
        logs["loss"] = loss

        return logs

    def test_step(self, data):
        node_features, edges, targets = data
        # Forward pass
        outputs = self([node_features, edges], training=False)
        # Compute loss
        loss = self.compiled_loss(targets, outputs)
        # Update metric(s)
        self.compiled_metrics.update_state(targets, outputs)
        logs = {m.name: m.result() for m in self.metrics}
        logs["loss"] = loss

        return logs

    def predict_step(self, data):
        node_features, edges, _ = data
        return self([node_features, edges], training=False)


def main():
    warnings.filterwarnings("ignore")
    np.random.seed(2)
    tf.random.set_seed(2)

    print("TensorFlow Version:", tf.__version__)
    print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], "GPU")
            print("Using GPU:", gpus[0])

        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected. Training will run on CPU.")

    start_time = time.time()

    dataset_dir = "dataset"

    feature_cols = ["current_x", "current_y", "previous_x", "previous_y"]
    target_cols = ["future_x", "future_y"]

    scenes = load_all_subgraphs(dataset_dir)
    print(f"Loaded {len(scenes)} scenes.")
    train_scenes, val_scenes, test_scenes = split_scenes(scenes, train_ratio=0.7, val_ratio=0.15)
    print(f"Train scenes: {len(train_scenes)}, Val scenes: {len(val_scenes)}, Test scenes: {len(test_scenes)}")

    train_dataset = tf.data.Dataset.from_generator(
        lambda: scene_generator(train_scenes, feature_cols, target_cols),
        output_signature=(
            tf.TensorSpec(shape=(None, len(feature_cols)), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
            tf.TensorSpec(shape=(None, len(target_cols)), dtype=tf.float32),
        ),
    )
    val_dataset = tf.data.Dataset.from_generator(
        lambda: scene_generator(val_scenes, feature_cols, target_cols),
        output_signature=(
            tf.TensorSpec(shape=(None, len(feature_cols)), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
            tf.TensorSpec(shape=(None, len(target_cols)), dtype=tf.float32),
        ),
    )
    test_dataset = tf.data.Dataset.from_generator(
        lambda: scene_generator(test_scenes, feature_cols, target_cols),
        output_signature=(
            tf.TensorSpec(shape=(None, len(feature_cols)), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
            tf.TensorSpec(shape=(None, len(target_cols)), dtype=tf.float32),
        ),
    )

    train_dataset = train_dataset.shuffle(100).batch(1).map(squeeze_batch)
    val_dataset = val_dataset.batch(1).map(squeeze_batch)
    test_dataset = test_dataset.batch(1).map(squeeze_batch)

    HIDDEN_UNITS = 100
    NUM_HEADS = 8
    NUM_LAYERS = 3
    OUTPUT_DIM = 2
    LEARNING_RATE = 1e-2
    NUM_EPOCHS = 100

    gat_model = None
    history = None

    tasks = [1, 2, 3]

    for task in tasks:

        if task == 1:
            print("\nRunning Task 1...\n")

            gat_model = GraphAttentionNetwork(
                hidden_units=HIDDEN_UNITS, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, output_dim=OUTPUT_DIM, task=task
            )

        elif task == 2:
            print("\nRunning Task 2...\n")
            num_heads = [2, 4, 8, 16]

            for i, heads in enumerate(num_heads):
                print(f"\nRun: {i + 1}\nHeads: {heads}\n")

                gat_model = GraphAttentionNetwork(
                    hidden_units=HIDDEN_UNITS, num_heads=heads, num_layers=NUM_LAYERS, output_dim=OUTPUT_DIM, task=task
                )

                gat_model, history = compile_and_train(
                    gat_model=gat_model,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    epochs=NUM_EPOCHS,
                    learning_rate=LEARNING_RATE,
                )

                evaluate_and_plot(
                    gat_model=gat_model, history=history, test_dataset=test_dataset, task=task, run=str(i + 1)
                )

        elif task == 3:
            print("\nRunning Task 3...\n")
            NUM_HEADS = 8

            gat_model = CosineGraphAttentionNetwork(
                hidden_units=HIDDEN_UNITS,
                num_heads=NUM_HEADS,
                num_layers=NUM_LAYERS,
                output_dim=OUTPUT_DIM,
            )

        else:
            raise ValueError("Unknown task")

        if task != 2:
            gat_model, history = compile_and_train(
                gat_model=gat_model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE,
            )

            evaluate_and_plot(gat_model=gat_model, history=history, test_dataset=test_dataset, task=task)

    end_time = time.time()
    running_time = end_time - start_time
    hours = int(running_time // 3600)
    minutes = int((running_time % 3600) // 60)
    seconds = int(running_time % 60)
    print(f"Running time: {hours} hours, {minutes} minutes, {seconds} seconds")


if __name__ == "__main__":
    main()
