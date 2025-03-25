from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
tf.random.set_seed(seed)
file_path = 'your path'
df = pd.read_excel(file_path)

df_cleaned = df.dropna(subset=['trait 1'])

df = df.dropna(subset=['trait 1'])
label_encoder_family = LabelEncoder()
label_encoder_genus = LabelEncoder()
family_encoded = label_encoder_family.fit_transform(df['trait 2'].astype(str))
genus_encoded = label_encoder_genus.fit_transform(df['trait 3)'].astype(str))
genus_encoded += (family_encoded.max() + 1)
combined_encoded = np.vstack((family_encoded, genus_encoded)).T
label_encoder_diet = LabelEncoder()
df['trait name'] = label_encoder_diet.fit_transform(df['your trait'])        # You can add multiple traits
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['trait']])
X = np.hstack((combined_encoded, scaled_features))
X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for Conv1D input
y = to_categorical(df['trait name'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Focal Loss
        alpha_t = y_true * alpha + (tf.keras.backend.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.keras.backend.ones_like(y_true) - y_true) * (1 - y_pred)
        focal_loss = -alpha_t * tf.keras.backend.pow((tf.keras.backend.ones_like(y_true) - p_t),
                                                     gamma) * tf.keras.backend.log(p_t)
        return tf.keras.backend.mean(focal_loss)

    return focal_loss_fixed


def combined_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        ce_loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
        focal_loss_value = focal_loss(gamma=gamma, alpha=alpha)(y_true, y_pred)
        combined = ce_loss + focal_loss_value
        return combined

    return loss
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling1D(pool_size=2, padding='same')(conv1)
    conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling1D(pool_size=2, padding='same')(conv2)
    flatten = Flatten()(pool2)
    dense1 = Dense(64, activation='relu')(flatten)
    output = Dense(num_classes, activation='softmax')(dense1)
    return Model(inputs=inputs, outputs=output)


model = build_model((X_train.shape[1], 1), y_train.shape[1])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model = build_model((X_train.shape[1], 1), y_train.shape[1])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=combined_loss(), metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=150, batch_size=16, validation_split=0.4)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

y_pred_diet = label_encoder_diet.inverse_transform(y_pred_classes)
y_true_diet = label_encoder_diet.inverse_transform(y_true_classes)
comparison_df = pd.DataFrame({'Ture': y_true_diet, 'Pre': y_pred_diet})
print(comparison_df)
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes))
print("Confusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()