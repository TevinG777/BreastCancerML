import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical

# Load the data
df = pd.read_csv("brca_metabric_clinical_data.tsv", sep='\t')

# Define targets (treatments to predict)
targets = ['Chemotherapy', 'Type of Breast Surgery', 'Hormone Therapy', 'Radio Therapy']

# Drop rows with missing values in the targets
df = df.dropna(subset=targets)

# Drop non-informative columns
drop_cols = [
    'Study ID', 'Patient ID', 'Sample ID', "Patient's Vital Status",
    'Relapse Free Status (Months)', 'Relapse Free Status',
    'Number of Samples Per Patient', 'Sample Type'
]
df = df.drop(columns=drop_cols, errors='ignore')

# Select relevant features for prediction
relevant_features = ['Age at Diagnosis', 'Cellularity', 'Tumor Size', 'Tumor Stage', 'Sex']
df = df[relevant_features + targets]

# Drop rows with missing values in the relevant features
df = df.dropna()

# Explicitly encode known categorical features and save encoders
categorical_features = ['Cellularity', 'Tumor Stage', 'Sex']
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    joblib.dump(le, f"{col.replace(' ', '_')}_encoder.pkl")

# Normalize numerical features
scaler = StandardScaler()
X = df[relevant_features]
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Function to build the model
def build_model(num_classes):
    model = Sequential([
        Input(shape=(X_scaled.shape[1],)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    model.compile(
        loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

# Train and save models
for target in targets:
    print(f"\nTraining model for: {target}")
    y = df[target]

    # Encode target if it's not already numeric
    if y.dtype == object or y.dtype == 'str':
        le = LabelEncoder()
        y = le.fit_transform(y)
        joblib.dump(le, f"{target.replace(' ', '_')}_encoder.pkl")  # Save target encoder

    # One-hot encode if more than 2 classes
    if len(np.unique(y)) > 2:
        y_encoded = to_categorical(y)
    else:
        y_encoded = y.astype(np.float32)  # Needed for binary_crossentropy

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # Ensure arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Build and train model
    model = build_model(y_train.shape[1] if len(y_train.shape) > 1 else 1)
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

    # Evaluate and save
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"{target} Accuracy: {accuracy:.4f}")
    model.save(f"{target}.h5")
