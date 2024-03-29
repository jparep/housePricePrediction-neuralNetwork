# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Input, Dense, GlobalMaxPooling1D, LeakyReLU
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.callbacks import History

# Set random seed for NumPy and TensorFlow
np.random.seed(42)
tf.random.set_seed(42)


# Load Data Function
def load_data(path) -> pd.DataFrame:
    return pd.read_csv(path)


# Prepare Data Function
def preprocess_data(df, num_col, cat_col) -> ColumnTransformer:
    # Numerical Transformation Pipeline
    num_pipe = Pipeline(
        steps=[
            ("imputer", IterativeImputer(max_iter=10, random_state=42)),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical Transformation Pipeline
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Combine Transformers
    preprocessor = ColumnTransformer(
        transformers=[("num", num_pipe, num_col), ("cat", cat_pipe, cat_col)]
    )

    df_preprocessed = preprocessor.fit_transform(df)
    return df_preprocessed


# Create Model
def create_model(
    input_shape,
    layers=[128, 64],
    activation="leaky_relu",
    dropout_rate=0.2,
    output_activation="linear",
):
    model = Sequential()
    model.add(
        Dense(layers[0], input_shape=(input_shape,), activation=activation)
    )  # Adjust input_shape here
    for layer_size in layers[1:]:
        model.add(Dense(layer_size, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation=output_activation))

    model.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"]
    )
    return model


# Hyperparameter Tuning Function
def hyperparameter_tuning(X_train, y_train, input_shape) -> GridSearchCV:
    # create model
    model = KerasRegressor(model=create_model, model__input_shape=input_shape)
    param_grid = {
        "model__layers": [[128, 64], [64, 32]],
        "batch_size": [32, 64],
        "epochs": [32, 64],
    }

    # Pass model and param_grid to GridSearchCV
    grid = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1
    )

    # Get best model and best parameters
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_params = grid.best_params_

    return best_model, best_params


# Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


# Plot Loss Entropy Function
def plot_loss_entropy(history) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss Entropy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Load data
    df = load_data("../data/house_data.csv")

    # Drop and Split data to features and target
    X = df.drop(["Id", "SalePrice"], axis=1)
    y = df["SalePrice"]

    # Define Numerical and Categorical Features
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Apply the preprocessing to feature data
    X_preprocessed = preprocess_data(X, num_cols, cat_cols)
    X_preprocessed.shape

    # Split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed, y, test_size=0.2, random_state=42
    )

    # Train and evaluate the model before hyperparameter tuning
    # model = create_model(input_shape=X_train.shape[1])
    model = KerasRegressor(model=create_model, input_shape=X_train.shape[1], verbose=0)

    # Create an instance of History callback to capture training history
    history_callback = History()
    # Fit the model with the callback to capture history and metrics
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        verbose=1,
        validation_split=0.2,
        callbacks=[history_callback],
    )

    # Plot the training history
    # plot_training_history(history)

    mse_before, r2_before = evaluate_model(model, X_test, y_test)
    print("Evaluation before hyperparameter tuning:")
    print(f"MSE: {mse_before}")
    print(f"R2: {r2_before}")

    # Hyperparameter tuning
    best_model, best_params = hyperparameter_tuning(
        X_train, y_train, input_shape=X_preprocessed.shape[1]
    )

    # Train the best model to capture the training history
    history = best_model.fit(X_train, y_train, verbose=0)

    # Evaluate the best model after hyperparameter tuning
    mse_after, r2_after = evaluate_model(best_model, X_test, y_test)
    print("\nEvaluation after hyperparameter tuning:")
    print(f"MSE: {mse_after}")
    print(f"R2: {r2_after}")

    # Plot Loss Entropy
    best_model.fit(
        X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0
    )
    history = best_model.model.history
    plot_loss_entropy(history)
