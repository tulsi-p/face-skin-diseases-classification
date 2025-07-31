import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import psutil

# ✅ Hardware Optimization
tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
tf.config.set_soft_device_placement(True)

# Enable GPU Memory Growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# ✅ Set Base Directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ✅ Load Preprocessed Data
with open(os.path.join(BASE_DIR, "processed_data.pkl"), "rb") as f:
    data = pickle.load(f)

# ✅ Extract Data
class_labels = data["class_labels"]
train_files = data["train_files"]
val_files = data["val_files"]
test_files = data["test_files"]

# ✅ Data Augmentation (Optimized)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='reflect'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# ✅ Define Constants
IMG_SIZE = (300, 300)
BATCH_SIZE = 16

# ✅ Convert File Paths to DataFrame
train_df = pd.DataFrame({
    "filename": train_files,
    "class": [os.path.basename(os.path.dirname(f)) for f in train_files]
})
val_df = pd.DataFrame({
    "filename": val_files,
    "class": [os.path.basename(os.path.dirname(f)) for f in val_files]
})
test_df = pd.DataFrame({
    "filename": test_files,
    "class": [os.path.basename(os.path.dirname(f)) for f in test_files]
})

# ✅ Create Data Generators
train_data = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filename",
    y_col="class",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="filename",
    y_col="class",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_data = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="filename",
    y_col="class",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print(f"✅ Data loaded | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

# ✅ Build EfficientNetB4 Model
def build_model():
    base_model = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
        pooling='avg'
    )
    
    # Freeze first 150 layers
    for layer in base_model.layers[:150]:
        layer.trainable = False
    for layer in base_model.layers[150:]:
        layer.trainable = True

    x = base_model.output
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    predictions = Dense(len(class_labels), activation='softmax', dtype='float32')(x)
    
    return Model(inputs=base_model.input, outputs=predictions)

model = build_model()

# ✅ Compute Class Weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df["class"]),
    y=train_df["class"]
)
class_weights_dict = dict(enumerate(class_weights))
print(f"⚖️ Class Weights: {class_weights_dict}")

# ✅ Compile Model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# ✅ Define Callbacks
callbacks = [
    EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True),
    ModelCheckpoint(
        os.path.join(BASE_DIR, "models", "best_model.h5"),
        monitor='val_auc',
        save_best_only=True,
        mode='max'
    ),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7),
    CSVLogger(os.path.join(BASE_DIR, "models", "training_log.csv"))
]

# ✅ Train Model
print("🔥 Starting training...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=callbacks,
    class_weight=class_weights_dict,
    verbose=1,
    steps_per_epoch=len(train_files) // BATCH_SIZE,
    validation_steps=len(val_files) // BATCH_SIZE
)

# ✅ Save Training Plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Val AUC')
plt.title('Model AUC')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.legend()

plt.savefig(os.path.join(BASE_DIR, "models", "training_metrics.png"))
plt.close()

# ✅ Evaluate Model on Test Set
print("🧪 Evaluating on test set...")
test_results = model.evaluate(
    test_data,
    steps=len(test_files) // BATCH_SIZE,
    verbose=1
)
print(f"\n✅ Final Test Accuracy: {test_results[1]*100:.2f}%")
print(f"✅ Test AUC: {test_results[2]*100:.2f}%")

# ✅ Save Final Model
model.save(os.path.join(BASE_DIR, "models", "final_model.h5"))
print("🎉 Training completed and models saved!")
