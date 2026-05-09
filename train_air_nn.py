import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import joblib
import json

# 1. Cargar Datos Aumentados
df = pd.read_csv('augmented_air_quality.csv')

# 2. Mapeos (Ahora con 3 Clases)
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

# 6. Train with RandomizedSearchCV
param_grid = {
    "hidden_layer_sizes": [(8, 16, 32), (64,), (128, 64), (50, 50)],
    "activation": ["relu", "tanh"],
    "alpha": [0.001, 0.01, 0.1],
    "learning_rate": ["constant", "adaptive"]
}

mlp = MLPClassifier(max_iter=15000, random_state=42)

grid_search = RandomizedSearchCV(
    mlp, param_grid, cv=3, n_jobs=-1, scoring="accuracy", n_iter=20, random_state=42, verbose=1
)

print("Entrenando Red Neuronal Multiclase...")
grid_search.fit(X_train_scaled, y_train)

# 7. Evaluación Multiclase
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
# Usamos macro para multiclase
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

cm = confusion_matrix(y_test, y_pred)

# 8. Guardar
joblib.dump(best_model, 'air_nn_model.pkl')
joblib.dump(scaler, 'air_scaler.pkl')

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist(), # Matriz 3x3
    "labels": list(label_map.keys()),
    "best_params": grid_search.best_params_
}

with open('air_nn_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nRed Neuronal Multiclase entrenada. Accuracy: {accuracy*100:.2f}%")
