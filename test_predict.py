import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import tensorflow as tf

# Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Input features and targets
input_features = ['Age at Diagnosis', 'Cellularity', 'Tumor Size', 'Tumor Stage', 'Sex']
targets = ['Chemotherapy', 'Type of Breast Surgery', 'Hormone Therapy', 'Radio Therapy']

# Load encoders and scaler
label_encoders = {}
for col in ['Cellularity', 'Tumor Stage', 'Sex']:
    label_encoders[col] = joblib.load(f"{col.replace(' ', '_')}_encoder.pkl")

scaler = joblib.load("scaler.pkl")

# Patient input
new_patient = {
    'Age at Diagnosis': 62,
    'Cellularity': 'Moderate',
    'Tumor Size': 35,
    'Tumor Stage': 2.0,
    'Sex': 'Female'
}

# Format input
input_df = pd.DataFrame([new_patient])

# Encode with correct type
for col in label_encoders:
    encoder = label_encoders[col]
    expected_type = type(encoder.classes_[0])
    try:
        input_df[col] = input_df[col].astype(expected_type)
        input_df[col] = encoder.transform(input_df[col])
    except ValueError:
        print(f"❌ Invalid value '{input_df[col].values[0]}' for '{col}'")
        print(f"✔ Allowed: {list(encoder.classes_)}")
        exit(1)

# Scale features
X_scaled = scaler.transform(input_df[input_features])

# Predict treatments
print("🩺 Predicted Treatments Needed:")
for target in targets:
    model = load_model(f"{target}.h5", compile=False)
    prediction = model.predict(X_scaled, verbose=0)

    if target == "Type of Breast Surgery":
        # Always treat as multi-class and decode properly
        index = np.argmax(prediction[0])
        encoder = joblib.load("Type_of_Breast_Surgery_encoder.pkl")
        label = encoder.inverse_transform([index])[0]
    else:
        # Binary or multi-class (default)
        if prediction.shape[1] == 1:
            label = "Yes" if prediction[0][0] > 0.5 else "No"
        else:
            index = np.argmax(prediction[0])
            try:
                encoder = joblib.load(f"{target.replace(' ', '_')}_encoder.pkl")
                label = encoder.inverse_transform([index])[0]
            except:
                label = index

    print(f"✅ {target}: {label}")
