import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.optim as optim
from itertools import combinations

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
tf.random.set_seed(seed)


file_path = ' Your path '
df = pd.read_excel(file_path)


task_type_input = int(input("Please select your task category："))
task_type = 'classification' if task_type_input == 0 else 'regression'


if task_type == 'regression':
    print("Detected regression task. Running regression model training...")


    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, y_pred, y_true):
            y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
            loss = -self.alpha * (1 - y_pred) ** self.gamma * (
                    y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
            return loss.mean()

    class RMSELoss(nn.Module):
        def __init__(self):
            super(RMSELoss, self).__init__()

        def forward(self, y_pred, y_true):
            return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


    class RBFNet(nn.Module):
        def __init__(self, k, gamma=None, input_dim=1):
            super(RBFNet, self).__init__()
            self.k = k
            self.gamma = gamma
            self.centers = None
            self.weights = nn.Parameter(torch.randn(k, dtype=torch.float32))
            self.input_dim = input_dim

        def fit(self, X, y, epochs=100, lr=0.001):
            kmeans = KMeans(n_clusters=self.k, random_state=42)
            kmeans.fit(X)
            self.centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

            optimizer = optim.RMSprop(self.parameters(), lr=lr)
            loss_fn = RMSELoss()

            for epoch in range(epochs):
                optimizer.zero_grad()
                y_pred = self.forward(torch.tensor(X, dtype=torch.float32))
                loss = loss_fn(y_pred, torch.tensor(y, dtype=torch.float32))
                loss.backward()
                optimizer.step()

                if epoch % 100 == 0:
                    print(f'Epoch {epoch}, Loss: {loss.item()}')

        def forward(self, X):
            G = self._calculate_interpolation_matrix(X)
            return G @ self.weights

        def _calculate_interpolation_matrix(self, X):
            distances = cdist(X, self.centers.detach().numpy())
            G = torch.exp(-self.gamma * torch.tensor(distances, dtype=torch.float32))
            return G

        def predict(self, X):
            with torch.no_grad():
                return self.forward(torch.tensor(X, dtype=torch.float32)).numpy()
    df_cleaned = df.dropna(subset=['Traits'])

    print(df_cleaned)

    df_cleaned['Aim Traits_log'] = np.log(df_cleaned['Aim Traits'])

    feature_columns = ['Traits']
    encoded_features = []
    for col in feature_columns:
        if df_cleaned[col].dtype == 'object':
            encoder = LabelEncoder()
            encoded_col = encoder.fit_transform(df_cleaned[col].astype(str))
            encoded_features.append(encoded_col)
        else:
            encoded_features.append(df_cleaned[col].values)

    encoded_features = np.array(encoded_features).T
    encoded_features = encoded_features.astype(np.float32)
    total_samples = len(df_cleaned)
    test_size = int(total_samples * 0.3)
    train_size = total_samples - test_size
    train_indices = np.random.choice(total_samples, train_size, replace=False)
    val_indices = np.setdiff1d(np.arange(total_samples), train_indices)
    X_train = np.hstack((encoded_features[train_indices],
                         df_cleaned.iloc[train_indices][['Aim Traits']].values))
    y_train = df_cleaned['Aim Traits_log'].values[train_indices]
    X_val = np.hstack((encoded_features[val_indices],
                       df_cleaned.iloc[val_indices][['Aim Traits']].values))
    y_val = df_cleaned['Aim Traits_log'].values[val_indices]
    best_params = {'k': None, 'gamma': None}
    highest_r2 = float('-inf')

    k_values = [10, 15, 17, 20, 23, 25]
    gamma_values = [0.001, 0.005, 0.009, 0.01, 0.05, 0.09, 0.1, 0.5]

    for k in k_values:
        for gamma in gamma_values:
            print(f"Training with k={k}, gamma={gamma}...")
            rbf_net = RBFNet(k=k, gamma=gamma, input_dim=X_train.shape[1])
            rbf_net.fit(X_train, y_train, epochs=10000, lr=0.01)
            y_pred = rbf_net.predict(X_val)
            r2 = r2_score(y_val, y_pred)

            print(f"R²: {r2}")

            if r2 > highest_r2:
                highest_r2 = r2
                best_params['k'] = k
                best_params['gamma'] = gamma

    print(f"Best Parameters: k={best_params['k']}, gamma={best_params['gamma']}, Highest R²: {highest_r2}")

    rbf_net = RBFNet(k=best_params['k'], gamma=best_params['gamma'], input_dim=X_train.shape[1])
    rbf_net.fit(X_train, y_train, epochs=10000, lr=0.01)
    y_pred = rbf_net.predict(X_val)


    def visualize_results(y_true, y_pred):
        y_true_exp = np.exp(y_true)
        y_pred_exp = np.exp(y_pred)

        plt.figure(figsize=(7, 6))
        plt.scatter(y_true_exp, y_pred_exp, color='blue', label='Predicted')
        plt.plot([y_true_exp.min(), y_true_exp.max()], [y_true_exp.min(), y_true_exp.max()], color='red',
                 linestyle='--', label='Ideal')
        plt.title('True vs Predicted')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.legend()
        plt.tight_layout()
        plt.show()

    visualize_results(y_val, y_pred)

    def save_results_to_excel(y_val, y_pred, file_path):
        df_results = pd.DataFrame({
            "True": np.exp(y_val),
            "Predicted": np.exp(y_pred)
        })
        df_results.to_excel(file_path, index=False)


    output_file_path = 'Your path'
    save_results_to_excel(y_val, y_pred, output_file_path)

elif task_type == 'classification':
    print("Detected classification task. Running classification model training...")

    from tensorflow.keras.models import Model

    
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
    from tensorflow.keras.utils import to_categorical

    df = df.dropna(subset=['Traits'])
    label_encoder_family = LabelEncoder()
    label_encoder_genus = LabelEncoder()
    family_encoded = label_encoder_family.fit_transform(df['Traits'].astype(str))
    genus_encoded = label_encoder_genus.fit_transform(df['Traits'].astype(str))
    genus_encoded += (family_encoded.max() + 1)


    combined_encoded = np.vstack((family_encoded, genus_encoded)).T

    label_encoder_diet = LabelEncoder()
    df['Aim Traits_encoded'] = label_encoder_diet.fit_transform(df['Traits'])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['Traits']])

    X = np.hstack((combined_encoded, scaled_features))
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = to_categorical(df['Aim Traits_encoded'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def focal_loss(gamma=2., alpha=0.25):
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

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

