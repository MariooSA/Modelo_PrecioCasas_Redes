import requests
import json

# URL de la API local (la ruta /predict)
url = "http://127.0.0.1:5000/predict"

# Datos que vas a enviar (simula las características de una casa)
data = {
    "area": 13000,
    "bedrooms": 15,
    "bathrooms": 11,
    "stories": 11,
    "mainroad": "yes",
    "guestroom": "no",
    "basement": "no",
    "hotwaterheating": "no",
    "airconditioning": "no",
    "parking": 5,
    "prefarea": "no",
    "furnishingstatus": "unfurnished"
}

# Realizar la solicitud POST
response = requests.post(url, json=data)

# Imprimir la respuesta
print(response.json())