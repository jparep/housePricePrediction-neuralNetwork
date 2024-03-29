
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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

# Set random seed for NumPy and TensorFlow
myID = 42
np.random.seed(myID)
tf.random.set_seed(myID)


# Load Data Function with Exception Handling
def load_data(path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        X = df.drop(['Id', 'SalePrice'], axis=1)
        y = df['SalePrice']
        return X, y
    except FileNotFoundError:
        print(f"The file {path} was not found.")
        return None, None
    except pd.errors.ParserError:
        print(f"Error parsing the file {path}. Check the file format.")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def numerical_transformer(num_cols)->Pipeline:
  num_pipe = Pipeline(steps=[
      ('imputer', IterativeImputer(max_iter=10, random_state=myID)),
      ('scaler', StandardScaler())
  ])

  return num_pipe

  # Categorical Transform
def catergorical_transformer(cat_cols)->Pipeline:
  cat_pipe = Pipeline(steps={
      ('imputer', SimpleImputer(strategy='most-frequent')),
      ('encoder', StandardScaler(handle_unknown='ignore', sparse_output=False))
  })

  return cat_pipe

  # Prepare Data Function
def preprocess_data(df, num_cols, cat_cols) -> ColumnTransformer:
  num_pipe = numerical_transformer(num_cols)
  cat_pipe = catergorical_transformer(cat_cols)

  preprocessor = ColumnTransformer(transformer_weights=[
      ('num', num_pipe, num_cols),
      ('cat', cat_pipe, cat_cols) 
  ])

  df_preprocessed = preprocessor.fit_transform(df)
  return df_preprocessed

# Create Model
def create_model(input_shape, layers=[128, 64], activation="leaky_relu", dropout_rate=0.2, output_activation="linear"):
    model = Sequential()
    model.add(Dense(layers[0], input_shape=(input_shape,), activation=activation))  # Adjust input_shape here
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
    # param_grid = {
    # "model__layers": [[128, 64], [64, 32]],
    # "batch_size": [32, 64],
    # "epochs": [32, 64]
    # }

    # # Pass model and param_grid to GridSearchCV
    # grid = GridSearchCV(
    #     estimator=model,
    #     param_grid=param_grid,
    #     cv=5,
    #     verbose=1,
    #     n_jobs=-1
    # )
    param_dist = {
        "model__layers": [[128, 64], [64, 32]],
        "batch_size": [32, 64],
        "epochs": [32, 64],
    }

    grid = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=8,
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )

    # Get best model and best parameters
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_params = grid.best_params_

    return best_model, best_params

# Evaluate Model
def evaluate_model(model, X_test, y_test) -> pd.DataFrame:
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

# Load data
X, y = load_data("/content/drive/MyDrive/Projects/HousePricePrediction/data/house_data.csv")

# Define Numerical and Categorical Features
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Apply the preprocessing to feature data
X_preprocessed = preprocess_data(X, num_cols, cat_cols)

# Split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

 #Train and evaluate the model before hyperparameter tuning
#model = create_model(input_shape=X_train.shape[1])
model = KerasRegressor(model=create_model, input_shape=X_train.shape[1], verbose=0)

# Fit the model with the callback to capture history and metrics
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_split=0.2)

mse_before, r2_before = evaluate_model(model, X_test, y_test)
print("Evaluation before hyperparameter tuning:")
print(f"MSE: {mse_before}")
print(f"R2: {r2_before}")

# Hyperparameter tuning
best_model, best_params = hyperparameter_tuning(X_train, y_train, input_shape=X_preprocessed.shape[1])

# Evaluate the best model after hyperparameter tuning
mse_after, r2_after = evaluate_model(best_model, X_test, y_test)
print("\nEvaluation after hyperparameter tuning:")
print(f"MSE: {mse_after}")
print(f"R2: {r2_after}")