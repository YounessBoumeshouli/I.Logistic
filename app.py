from fastapi import FastAPI, WebSocket
import asyncio
import json
import random
from datetime import datetime

app = FastAPI()

# Listes pour générer des données aléatoires réalistes
CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]
SHIPPING_MODES = ["Standard Class", "First Class", "Second Class", "Same Day"]

async def generate_order():
    """Génère une commande logistique aléatoire."""
    return {
        "order_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Order City": random.choice(CITIES),
        "Customer Segment": random.choice(SEGMENTS),
        "Shipping Mode": random.choice(SHIPPING_MODES),
        "Order Item Quantity": random.randint(1, 5),
        "Order Item Total": round(random.uniform(10, 500), 2),
        "Order Profit Per Order": round(random.uniform(-50, 150), 2),
        # Ajoutez ici les autres champs nécessaires à votre modèle ML
        # Assurez-vous qu'ils correspondent aux colonnes attendues par le modèle !
        "Latitude": 0.0, # À remplacer par de vraies valeurs ou geocoding
        "Longitude": 0.0
    }

@app.websocket("/ws/orders")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            order = await generate_order()
            await websocket.send_json(order)
            # Simuler un délai entre les commandes (ex: 1 commande par seconde)
            await asyncio.sleep(1)
    except Exception as e:
        print(f"Erreur WebSocket: {e}")
    finally:
        await websocket.close()