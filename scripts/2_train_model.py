import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import psutil

# Hardware Optimization
tf.config.optimizer.set_jit(True)
tf.config.set_soft_device_placement(True)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Set Base Directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load Preprocessed Data
with open(os.path.join(BASE_DIR, "processed_data.pkl"), "rb") as f:
    data = pickle.load(f)

# Extract Data
class_labels = data["class_labels"]
train_files = data["train_files"]
val_files = data["val_files"]
test_files = data["test_files"]

# Optimized Data Augmentation
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

# Create Optimized Data Pipelines
IMG_SIZE = (224, 224)  # Reduced image size for faster training
BATCH_SIZE = 32  # Increased batch size for better performance

def create_flow(dataframe, generator, shuffle):
    return generator.flow_from_dataframe(
        dataframe=dataframe,
        x_col="filename",
        y_col="class",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=shuffle,
        seed=42
    )

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

train_data = create_flow(train_df, train_datagen, shuffle=True)
val_data = create_flow(val_df, val_test_datagen, shuffle=False)
test_data = create_flow(test_df, val_test_datagen, shuffle=False)

print(f"✅ Data loaded | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

# Enhanced Model Architecture
def build_model():
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
        pooling='avg'
    )
    
    for layer in base_model.layers[:150]:
        layer.trainable = False
    for layer in base_model.layers[150:]:
        layer.trainable = True

    x = base_model.output
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)  # Reduced dropout for better learning
    x = BatchNormalization()(x)
    predictions = Dense(len(class_labels), activation='softmax', dtype='float32')(x)
    
    return Model(inputs=base_model.input, outputs=predictions)

model = build_model()

# Optimized Training Setup
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights_dict = dict(enumerate(class_weights))
print(f"⚖️ Class Weights: {class_weights_dict}")

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Smart Callbacks
callbacks = [
    EarlyStopping(monitor='val_auc', patience=7, mode='max', restore_best_weights=True),
    ModelCheckpoint(
        os.path.join(BASE_DIR, "models", "best_model.keras"),  # Changed to .keras format
        monitor='val_auc',
        save_best_only=True,
        mode='max'
    ),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6),
    CSVLogger(os.path.join(BASE_DIR, "models", "training_log.csv"))
]

# Train the Model
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

# Save Final Model
model.save(os.path.join(BASE_DIR, "models", "final_model.keras"))
print("🎉 Training completed and models saved!")
