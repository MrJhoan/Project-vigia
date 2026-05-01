# main.py - VERSION 2.6
# Fix de velocidad: display en hilo propio a FPS completo,
# IA en hilo separado sin bloquear la pantalla.
# El video ahora se ve fluido siempre.

import asyncio
import cv2
import base64
import json
import os
import time
import threading
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from detector import analyze_frame, reset_video_state
import twilio_sender
import alarm

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

connected_clients: list[WebSocket] = []
VIDEOS_FOLDER = "videos"

playback = {
    "paused":    False,
    "speed":     1.0,
    "skip":      False,
    "jump_secs": 0,
}

# Shared state entre hilo de display y hilo de IA
# El hilo de display escribe frames crudos aquí
# El hilo de IA lee frames, analiza, y escribe el resultado anotado
shared = {
    "raw_frame":        None,   # Frame crudo para que la IA analice
    "annotated_frame":  None,   # Frame anotado por la IA para mostrar
    "lock":             threading.Lock(),
    "new_frame_ready":  False,  # Señal: hay frame nuevo para analizar
    "events":           [],
}

alert_queue      = []
alert_queue_lock = threading.Lock()


def get_video_files():
    if not os.path.exists(VIDEOS_FOLDER):
        os.makedirs(VIDEOS_FOLDER)
        return []
    exts = (".mp4", ".avi", ".mov", ".mkv")
    videos = [
        os.path.join(VIDEOS_FOLDER, f)
        for f in sorted(os.listdir(VIDEOS_FOLDER))
        if f.lower().endswith(exts)
    ]
    if videos:
        print(f"📹 {len(videos)} videos:")
        for v in videos:
            print(f"   → {os.path.basename(v)}")
    else:
        print("⚠️  Sin videos. Usando cámara en vivo.")
    return videos


def draw_controls_bar(frame, speed, paused):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-30), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    estado = "PAUSADO" if paused else f"x{speed}"
    txt = (f" {estado} | ESPACIO=pausa  F=x2  S=x0.5  N=normal"
           f"  ,=retroceder 10s  .=adelantar 10s  Q=siguiente")
    cv2.putText(frame, txt, (8, h-9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)


def ai_worker_thread():
    """
    Hilo de trabajo de la IA.
    Lee frames del shared state, los analiza y escribe el resultado anotado.
    Corre independientemente del hilo de display para no bloquearlo.
    """
    print("🤖 Hilo de IA iniciado")
    while True:
        # Esperamos a que haya un frame nuevo para analizar
        with shared["lock"]:
            if not shared["new_frame_ready"] or shared["raw_frame"] is None:
                pass
            else:
                frame_to_analyze       = shared["raw_frame"].copy()
                shared["new_frame_ready"] = False
            frame_to_analyze = shared["raw_frame"].copy() if shared["raw_frame"] is not None else None

        if frame_to_analyze is None:
            time.sleep(0.005)
            continue

        try:
            events, annotated = analyze_frame(frame_to_analyze)
        except Exception as e:
            print(f"Error IA: {e}")
            time.sleep(0.01)
            continue

        with shared["lock"]:
            shared["annotated_frame"] = annotated
            if events:
                shared["events"].extend(events)

        # Encolar alertas para el panel web
        if events:
            _, buf = cv2.imencode(".jpg", annotated,
                                  [cv2.IMWRITE_JPEG_QUALITY, 60])
            fb64 = base64.b64encode(buf).decode("utf-8")
            with alert_queue_lock:
                for ev in events:
                    alert_queue.append({"event": ev, "frame": fb64})

        time.sleep(0.001)


def video_thread_func(video_files):
    """
    Hilo principal de video.
    Lee frames y los muestra a FPS completo.
    Cada N frames envía uno al hilo de IA para análisis.
    El display siempre usa el último frame anotado disponible.
    """
    print("\n🎮 Controles:")
    print("   ESPACIO → Pausar/Reanudar")
    print("   F       → x2 rápido")
    print("   S       → x0.5 lento")
    print("   N       → x1 normal")
    print("   ,       → Retroceder 10 segundos")
    print("   .       → Adelantar 10 segundos")
    print("   Q       → Siguiente video\n")

    # Lanzar hilo de IA
    ai_thread = threading.Thread(target=ai_worker_thread, daemon=True)
    ai_thread.start()

    # Cada cuántos frames enviamos uno a la IA para análisis
    # 3 = analiza 1 de cada 3 frames → display fluido, IA ~10fps
    AI_FRAME_INTERVAL = 3

    while True:
        sources = video_files if video_files else [0]

        for source in sources:
            playback["skip"]      = False
            playback["speed"]     = 1.0
            playback["paused"]    = False
            playback["jump_secs"] = 0

            # Limpiar estado del video anterior
            reset_video_state()
            with shared["lock"]:
                shared["annotated_frame"] = None

            name = (os.path.basename(str(source))
                    if source != 0 else "Camara en vivo")
            print(f"\n▶️  {name}")

            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"❌ No se pudo abrir: {name}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 60:
                fps = 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            base_delay   = 1.0 / fps
            current_pos  = 0
            frame_counter = 0

            print(f"✅ FPS: {fps:.0f} | Duración: {total_frames/fps:.0f}s")

            while True:
                if playback["skip"]:
                    alarm.stop_alarm()
                    break

                # Salto de tiempo
                if playback["jump_secs"] != 0:
                    jump_frames = int(playback["jump_secs"] * fps)
                    current_pos = max(0, min(
                        current_pos + jump_frames, total_frames - 1
                    ))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                    print(f"⏩ Posición: {current_pos/fps:.1f}s")
                    playback["jump_secs"] = 0

                if playback["paused"]:
                    key = cv2.waitKey(50) & 0xFF
                    if key == ord(' '):
                        playback["paused"] = False
                    elif key == ord('q'):
                        playback["skip"] = True
                    elif key == ord(','):
                        playback["jump_secs"] = -10
                    elif key == ord('.'):
                        playback["jump_secs"] = 10
                    time.sleep(0.03)
                    continue

                t0 = time.time()

                # En x2, saltar frames en el video
                if playback["speed"] >= 2.0:
                    current_pos += 2
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                else:
                    current_pos += 1

                ret, frame = cap.read()
                if not ret:
                    print(f"✅ Terminado: {name}")
                    alarm.stop_alarm()
                    cv2.destroyAllWindows()
                    break

                frame_counter += 1

                # Redimensionar
                h, w = frame.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    frame = cv2.resize(frame, (1280, int(h * scale)))

                # Cada AI_FRAME_INTERVAL frames, enviamos uno a la IA
                if frame_counter % AI_FRAME_INTERVAL == 0:
                    with shared["lock"]:
                        shared["raw_frame"]       = frame.copy()
                        shared["new_frame_ready"] = True

                # Mostrar: usamos el frame anotado si está disponible,
                # si no, mostramos el frame crudo con barra de estado básica
                with shared["lock"]:
                    display = (shared["annotated_frame"].copy()
                               if shared["annotated_frame"] is not None
                               else frame.copy())

                draw_controls_bar(display, playback["speed"], playback["paused"])
                cv2.imshow(f"PROJECT_VIGIA | {name}", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    playback["paused"] = True
                elif key == ord('f'):
                    playback["speed"] = 2.0
                    print("⏩ x2")
                elif key == ord('s'):
                    playback["speed"] = 0.5
                    print("⏪ x0.5")
                elif key == ord('n'):
                    playback["speed"] = 1.0
                    print("▶️  Normal")
                elif key == ord(','):
                    playback["jump_secs"] = -10
                    print("⏪ -10s")
                elif key == ord('.'):
                    playback["jump_secs"] = 10
                    print("⏩ +10s")
                elif key == ord('q'):
                    playback["skip"] = True
                    alarm.stop_alarm()
                    cv2.destroyAllWindows()

                # Control de velocidad preciso
                elapsed   = time.time() - t0
                target    = base_delay / playback["speed"]
                remaining = target - elapsed
                if remaining > 0.001:
                    time.sleep(remaining)

            cap.release()

        if not video_files:
            break
        print("\n🔄 Reiniciando...")
        time.sleep(1)


@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg  = json.loads(data)
            if msg.get("action") == "stop_alarm":
                alarm.stop_alarm()
    except Exception:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


async def alert_dispatcher():
    while True:
        if alert_queue:
            with alert_queue_lock:
                items = alert_queue.copy()
                alert_queue.clear()
            for item in items:
                msg = {"type": "alert", "event": item["event"],
                       "frame": item["frame"], "countdown": 10}
                for client in connected_clients.copy():
                    try:
                        await client.send_text(json.dumps(msg))
                    except Exception:
                        connected_clients.remove(client)
                await twilio_sender.notify_police(item["event"])
        await asyncio.sleep(0.1)


@app.on_event("startup")
async def startup():
    video_files = get_video_files()
    t = threading.Thread(
        target=video_thread_func,
        args=(video_files,),
        daemon=True
    )
    t.start()
    asyncio.create_task(alert_dispatcher())