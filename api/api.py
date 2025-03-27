from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Crear la app de Flask
app = Flask(__name__)

# Cargar el modelo entrenado y los transformadores
model = load_model('modelo_final_escalado.h5')  # Modelo de Keras
scaler_X = joblib.load('scaler_X.pkl')  # Escalador de características de entrada
scaler_y = joblib.load('scaler_y.pkl')  # Escalador de la variable objetivo
poly = joblib.load('poly_features.pkl')  # Transformador de características polinómicas
expected_columns = joblib.load('expected_columns.pkl')  # Columnas después de PolynomialFeatures
original_columns = joblib.load('original_columns.pkl')  # Columnas originales antes del escalado

print("✅ Modelos y transformadores cargados correctamente.")

# 🛠 Función de preprocesamiento
def preprocess(data):
    """Convierte los datos JSON en un DataFrame y aplica preprocesamiento."""
    df = pd.DataFrame([data])  # Convertir JSON en DataFrame

    # Generar columnas calculadas
    df['bathrooms_per_area'] = df['bathrooms'] / df['area']
    df['stories_per_area'] = df['stories'] / df['area']
    df['parking_per_area'] = df['parking'] / df['area']
    df['bathrooms_per_bedroom'] = df['bathrooms'] / df['bedrooms']
    df['bed_room_per_area'] = df['bedrooms'] / df['area']
    
    # Añadir nuevas características
    df['rooms_total'] = df['bedrooms'] + df['bathrooms']
    df['avg_room_size'] = df['area'] / df['rooms_total']
    df['room_density'] = df['rooms_total'] / df['area']
    df['bedrooms_per_bathroom'] = df['bedrooms'] / df['bathrooms']
    df['area_per_story'] = df['area'] / df['stories']
    df['parking_per_bedroom'] = df['parking'] / df['bedrooms']
    df['parking_area_ratio'] = df['parking'] * df['area']

    # Convertir valores "sí/no" a 1/0
    yes_no_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().replace({'yes': 1, 'no': 0})
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Manejo de la columna 'furnishingstatus'
    if 'furnishingstatus' in df.columns:
        df = pd.get_dummies(df, columns=['furnishingstatus'], dtype=int)
        for col in ['furnishingstatus_furnished', 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']:
            if col not in df.columns:
                df[col] = 0

    # Crear características adicionales
    df['service_index'] = df[yes_no_cols].sum(axis=1)
    df['parking_service_ratio'] = df['parking'] / (df['service_index'] + 1)
    df['service_area_ratio'] = df['service_index'] / df['area']
    df['guestroom_basement'] = (df['guestroom'] & df['basement']).astype(int)

    # Asegurar que las columnas coincidan con las originales
    df = df.reindex(columns=original_columns, fill_value=0)

    # Escalar los datos con scaler_X
    X_scaled = scaler_X.transform(df)

    # Aplicar PolynomialFeatures
    X_poly = poly.transform(X_scaled)

    # Convertir a DataFrame y alinear con las columnas esperadas
    df_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(input_features=original_columns))
    df_poly = df_poly.reindex(columns=expected_columns, fill_value=0)

    return df_poly

# 🔥 Ruta para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Obtener datos JSON
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Preprocesar los datos
        preprocessed_data = preprocess(data)

        # Hacer la predicción con el modelo
        prediction_scaled = model.predict(preprocessed_data)

        # Desescalar la predicción con scaler_y
        predicted_price = scaler_y.inverse_transform(prediction_scaled)[0][0]
        
        predicted_price = float(predicted_price)

        return jsonify({'predicted_price': round(predicted_price, 2)})

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 🚀 Iniciar la API
if __name__ == '__main__':
    app.run(debug=True)
