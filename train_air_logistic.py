import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import joblib
import json

# 1. Cargar Datos Aumentados
df = pd.read_csv('augmented_air_quality.csv')

# 2. Mapeos
period_map = {
    'Morning Entry': 0, '1st Period': 1, '2nd Period': 2, 'Break': 3,
    '3rd Period': 4, '4th Period': 5, 'Lunch Break': 6, '5th Period': 7,
    'End of Day': 8
}
label_map = {'Good': 0, 'Moderate': 1, 'Poor': 2}

df['school_period'] = df['school_period'].map(period_map)
df['air_quality_label'] = df['air_quality_label'].map(label_map)

# 3. Features
feature_cols = [
    'school_period', 'student_count_estimated', 'co2_ppm', 
    'pm25_ugm3', 'temperature_c', 'humidity_pct', 
    'robot_x_pos', 'robot_y_pos'
]
X = df[feature_cols]
y = df['air_quality_label']

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Modelo (LogisticRegression maneja multiclase automáticamente con multinomial)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Evaluación
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)

# 8. Guardar
joblib.dump(model, 'air_logistic_model.pkl')
joblib.dump(scaler, 'air_scaler.pkl')

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist()
}
with open('air_logistic_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"Logística Multiclase entrenada. Accuracy: {accuracy*100:.2f}%")
