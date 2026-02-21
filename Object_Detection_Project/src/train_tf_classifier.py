from calendar import EPOCH
from pickletools import optimize
import tensorflow as tf
from pathlib import Path
import json
import numpy as np

DATASET_DIR = Path(__file__).resolve().parents[1]/ "data"/"dataset"
IMG_SIZE = (224,224)
BATCH = 16
EPOCH = 15

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "train",
    image_size = IMG_SIZE,
    batch_size = BATCH,
    label_mode = "int"
    )

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "val",
    image_size = IMG_SIZE,
    batch_size = BATCH,
    label_mode = "int"
    )

class_names = train_ds.class_names
print("Classes:",class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(500).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
    ])

base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top = False,
    weights = "imagenet"
    )
base.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_aug(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training = False)
x = tf.keras.layers.GlobalAveragePooling2D()(x) 
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
    loss = 'sparse_categorical_crossentropy',
    #loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = ['accuracy']
    )

model.summary

callbacks = [
     tf.keras.callbacks.EarlyStopping(patience = 3, restore_best_weights = True),
     tf.keras.callbacks.ModelCheckpoint("best_model.keras",save_best_only = True),
    ]

history = model.fit(
      train_ds,
      validation_data = val_ds,
      epochs = EPOCH,
      #callbacks = callbacks
    )

model.save("tf_model.keras")
with open("labels.json","w",encoding="utf-8") as f:
    json.dump(class_names, f)

print("Saved: tf_model/ and labels.json")
print("Train class_names:", class_names)

for images, labels in val_ds.take(1):
    preds = model.predict(images)
    print("Pred:",np.argmax(preds, axis=1))
    print("True:",labels.numpy())


