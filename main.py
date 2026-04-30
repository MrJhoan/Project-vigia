# main.py
# Servidor principal de PROJECT_VIGIA.
# Lee la cámara frame a frame, detecta eventos y los envía
# al panel web en tiempo real mediante WebSockets.

import asyncio
import cv2
import base64
import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from detector import analyze_frame
import twilio_sender

app = FastAPI()

# CORS: le permite al frontend React (localhost:3000) conectarse al backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lista de paneles web conectados en este momento
connected_clients: list[WebSocket] = []

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    """
    Punto de conexión WebSocket. Cada vez que el panel web
    abre la página se conecta aquí y queda escuchando alertas.
    """
    await websocket.accept()
    connected_clients.append(websocket)
    print(f"Panel web conectado. Clientes activos: {len(connected_clients)}")

    try:
        while True:
            # Esperamos mensajes del operador (ej: cancelar alerta)
            data = await websocket.receive_text()
            msg  = json.loads(data)

            if msg.get("action") == "cancel_alert":
                print("⚠️  Alerta cancelada por el operador.")

    except Exception:
        connected_clients.remove(websocket)
        print(f"Panel web desconectado. Clientes activos: {len(connected_clients)}")


async def broadcast_alert(event: dict, frame_b64: str):
    """
    Envía el evento detectado a todos los paneles conectados,
    incluyendo el frame (imagen) del momento de la detección.
    """
    message = {
        "type":      "alert",
        "event":     event,
        "frame":     frame_b64,
        "countdown": 10  # Segundos antes del envío automático a Policía
    }
    for client in connected_clients.copy():
        try:
            await client.send_text(json.dumps(message))
        except Exception:
            connected_clients.remove(client)


@app.on_event("startup")
async def start_video_loop():
    """Al arrancar el servidor, lanzamos el loop de video en segundo plano."""
    asyncio.create_task(video_processing_loop())


async def video_processing_loop():
    """
    Loop continuo: lee frames de la cámara y los analiza.
    Cambia el 0 por la URL RTSP de tu cámara IP cuando la tengas:
    cv2.VideoCapture("rtsp://usuario:clave@192.168.1.100:554/stream")
    """
    cap = cv2.VideoCapture(0)  # 0 = cámara integrada del computador

    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara. Verifica que esté conectada.")
        return

    print("✅ Cámara iniciada. Procesando video en tiempo real...")

    while True:
        ret, frame = cap.read()

        if not ret:
            await asyncio.sleep(0.1)
            continue

        # Analizamos el frame con el detector de IA
        events = analyze_frame(frame)

        if events:
            # Convertimos el frame a base64 para enviarlo al navegador
            _, buffer  = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64  = base64.b64encode(buffer).decode("utf-8")

            for event in events:
                print(f"🔴 Evento detectado: {event['label']} ({event['confianza']}%)")
                await broadcast_alert(event, frame_b64)

                # Esperamos 10 segundos antes de notificar a la Policía
                # (el operador puede cancelar desde el panel en ese tiempo)
                await asyncio.sleep(10)
                await twilio_sender.notify_police(event)

        # ~15 frames por segundo para no saturar el procesador
        await asyncio.sleep(0.066)