# detector.py - Versión final con ventana deslizante anti-falsos-positivos
#
# Resumen de lo que hace este archivo:
#   1. Detecta rostros con Haar Cascade → dibuja rectángulo verde
#   2. Detecta objetos peligrosos con YOLOv8n → dibuja rectángulo naranja
#   3. Detecta poses del cuerpo con YOLOv8n-pose → dibuja esqueleto verde
#   4. Mide si alguna muñeca se está aproximando rápidamente al rostro
#   5. Usa una VENTANA DESLIZANTE para confirmar el golpe sin falsos positivos:
#      en vez de exigir N frames perfectos consecutivos, mira los últimos
#      8 frames y confirma si al menos 5 de ellos detectaron golpe.
#      Esto tolera los 2-3 frames de "ruido" que antes reiniciaban el contador.

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import math

# ── Modelos de IA ─────────────────────────────────────────────────────────────

# Modelo de poses: detecta 17 puntos del esqueleto humano en cada persona
pose_model = YOLO("yolov8n-pose.pt")

# Modelo general: detecta 80 tipos de objetos, incluyendo "knife" (cuchillo)
# Se descarga automáticamente la primera vez (~6 MB)
object_model = YOLO("yolov8n.pt")

# Detector de rostros Haar Cascade, incluido en OpenCV sin descargar nada
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Clases peligrosas del dataset COCO que nos interesan ─────────────────────
# El modelo yolov8n.pt conoce 80 clases. Aquí filtramos solo las relevantes.
# Cuando tengas tu propio modelo entrenado (weapon_model.pt), agregarás
# pistola, puñal, etc. directamente en este diccionario.
DANGEROUS_CLASSES = {
    43: "Cuchillo"
}

# ── Parámetros de detección (ajusta estos números según tus pruebas) ──────────

# Velocidad mínima de aproximación de la muñeca hacia la cara (px/segundo).
# Bajamos de 400 a 250 para capturar golpes reales sin ser tan estricto.
# Movimientos normales del brazo suelen estar por debajo de 150 px/s.
HIT_VELOCITY_THRESHOLD = 250

# Distancia máxima entre la muñeca y el centro del rostro para considerar
# que el golpe va "hacia la cara" y no en otra dirección.
MAX_DISTANCE_TO_FACE = 350

# ── Parámetros de la ventana deslizante ──────────────────────────────────────
# En vez de exigir N frames consecutivos perfectos (lo que causaba que el
# contador se reiniciara con cualquier frame de ruido), miramos los últimos
# HIT_WINDOW_SIZE frames y preguntamos: ¿cuántos de ellos detectaron golpe?
# Si al menos HIT_WINDOW_MIN lo detectaron, confirmamos la alerta.
HIT_WINDOW_SIZE = 8   # Cuántos frames recientes miramos
HIT_WINDOW_MIN  = 5   # Cuántos de esos frames deben haber detectado golpe

# Cooldown entre alertas consecutivas para no spamear el panel web
ALERT_COOLDOWN_SECONDS = 12

# Área mínima del bounding box de una persona para ignorar ropa y objetos
# pequeños del fondo que el modelo confunde con personas
MIN_PERSON_AREA = 8000

# Confianza mínima para detectar objetos peligrosos (0 a 1)
OBJECT_CONFIDENCE = 0.45

# ── Estado interno del detector ───────────────────────────────────────────────
pose_history      = []     # Posiciones anteriores de muñecas para calcular velocidad
last_alert_time   = 0.0    # Timestamp de la última alerta enviada
last_face_center  = None   # Centro del rostro del frame anterior (para oclusión)
hit_window_buffer = []     # Buffer circular: True/False por frame reciente

# ── Conexiones del esqueleto humano (pares de índices COCO/YOLOv8) ────────────
# Cada tupla (A, B) dibuja una línea entre el punto A y el punto B del cuerpo.
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),           # nariz → ojos
    (1, 3), (2, 4),           # ojos → orejas
    (5, 6),                   # hombro izquierdo ↔ hombro derecho
    (5, 7), (7, 9),           # hombro izq → codo izq → muñeca izq
    (6, 8), (8, 10),          # hombro der → codo der → muñeca der
    (5, 11), (6, 12),         # hombros → caderas
    (11, 12),                 # cadera izquierda ↔ cadera derecha
    (11, 13), (13, 15),       # cadera izq → rodilla izq → tobillo izq
    (12, 14), (14, 16),       # cadera der → rodilla der → tobillo der
]


def draw_skeleton(frame, keypoints, is_aggressive: bool):
    """
    Dibuja el esqueleto de una persona sobre el frame.
    Usa color rojo si hay agresión confirmada, verde si no hay peligro.
    """
    color = (0, 0, 255) if is_aggressive else (0, 255, 0)

    for i, j in SKELETON_CONNECTIONS:
        if i < len(keypoints) and j < len(keypoints):
            pt1, pt2 = keypoints[i], keypoints[j]
            # Solo dibujamos si ambos puntos fueron detectados (no son 0,0)
            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                cv2.line(frame,
                         (int(pt1[0]), int(pt1[1])),
                         (int(pt2[0]), int(pt2[1])),
                         color, 2)

    # Círculo pequeño en cada punto del esqueleto
    for kp in keypoints:
        if kp[0] > 0 and kp[1] > 0:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, -1)


def get_bbox_area(keypoints):
    """
    Calcula el área del bounding box que rodea todos los keypoints de una persona.
    Usamos esto para filtrar detecciones falsas: la ropa colgada en el fondo
    o sillas tienen un área mucho menor que una persona parada frente a la cámara.
    """
    valid = [(kp[0], kp[1]) for kp in keypoints if kp[0] > 0 and kp[1] > 0]
    if len(valid) < 4:
        return 0
    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def euclidean_distance(pt1, pt2):
    """
    Calcula la distancia en píxeles entre dos puntos (x1,y1) y (x2,y2).
    Esta es la fórmula del teorema de Pitágoras aplicada a píxeles de imagen.
    """
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def detect_faces(frame):
    """
    Detecta rostros en el frame usando el clasificador Haar Cascade de OpenCV.
    Dibuja un rectángulo verde con la etiqueta 'Rostro' sobre cada cara detectada.

    Retorna el centro (x, y) del rostro más grande encontrado, o None si no
    hay ninguno visible. El rostro más grande generalmente es el más cercano
    a la cámara, que es la persona de interés principal.

    Trabajar en escala de grises hace que Haar Cascade sea más preciso y rápido
    porque elimina la información de color que no aporta nada a la detección
    de la estructura facial (bordes, sombras, proporciones).
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,    # Cuánto se reduce la imagen en cada escala
        minNeighbors=5,     # Vecinos necesarios para confirmar un rostro
        minSize=(60, 60)    # Tamaño mínimo: evita detectar ruido como caras
    )

    best_face_center = None
    best_face_area   = 0

    for (x, y, w, h) in faces:
        # Rectángulo verde alrededor del rostro
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Etiqueta "Rostro" con fondo verde para que sea legible sobre cualquier fondo
        label_y = y - 25 if y > 30 else y + h + 5
        cv2.rectangle(frame, (x, label_y), (x + 90, label_y + 22),
                      (0, 255, 0), -1)
        cv2.putText(frame, "Rostro",
                    (x + 4, label_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        # Guardamos el rostro más grande (el más cercano a la cámara)
        area = w * h
        if area > best_face_area:
            best_face_area   = area
            best_face_center = (x + w // 2, y + h // 2)

    return best_face_center


def detect_objects(frame):
    """
    Detecta objetos peligrosos en el frame usando yolov8n.pt.
    Actualmente detecta cuchillos (clase 43 del dataset COCO).
    Cuando tengas tu modelo entrenado (weapon_model.pt), simplemente
    agregas las clases nuevas al diccionario DANGEROUS_CLASSES arriba.

    Dibuja un rectángulo naranja con el nombre del objeto y el porcentaje
    de confianza. Retorna una lista de eventos para enviar al panel web.
    """
    events = []

    results = object_model(
        frame,
        classes=list(DANGEROUS_CLASSES.keys()),
        conf=OBJECT_CONFIDENCE,
        verbose=False
    )

    for result in results:
        for box in result.boxes:
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = DANGEROUS_CLASSES.get(class_id, "Objeto peligroso")

            # Rectángulo naranja para diferenciar objetos de personas
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)

            # Etiqueta con nombre y porcentaje de confianza
            display_text = f"{label} {confidence * 100:.0f}%"
            bg_width = len(display_text) * 11
            cv2.rectangle(frame,
                          (x1, y1 - 28), (x1 + bg_width, y1),
                          (0, 140, 255), -1)
            cv2.putText(frame, display_text,
                        (x1 + 3, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

            print(f"🔪 Objeto detectado: {label} ({confidence * 100:.1f}%)")

            events.append({
                "tipo":      "arma",
                "label":     label,
                "confianza": round(confidence * 100, 1),
                "timestamp": datetime.now().isoformat()
            })

    return events


def analyze_frame(frame):
    """
    Función principal llamada desde main.py en cada frame del video.

    El flujo completo es:
      1. Detectar rostros → obtener posición del centro de la cara
      2. Detectar objetos peligrosos (cuchillos, etc.)
      3. Detectar poses corporales con YOLOv8
      4. Para cada persona, calcular si alguna muñeca se aproxima
         rápidamente hacia el rostro detectado
      5. Usar la ventana deslizante para confirmar el golpe sin
         reiniciar el conteo por frames de ruido
      6. Si se confirma agresión o arma, generar evento de alerta
      7. Mostrar todo anotado en la ventana de video
    """
    global pose_history, last_alert_time, last_face_center, hit_window_buffer

    events              = []
    status_text         = "Estado: Monitoreando..."
    status_color        = (0, 200, 0)
    frame_is_aggressive = False

    # ── Paso 1: Detectar rostros ──────────────────────────────────────────────
    face_center = detect_faces(frame)

    # Si no hay cara visible en este frame, usamos la última posición conocida.
    # Esto es importante porque cuando la mano se acerca a la cara justo antes
    # del golpe, puede ocluirla brevemente y hacer que el detector la pierda.
    # Sin este "memoria de posición", el sistema dejaría de funcionar en el
    # momento más crítico de la detección.
    if face_center is not None:
        last_face_center = face_center
    working_face_center = last_face_center

    # ── Paso 2: Detectar objetos peligrosos ───────────────────────────────────
    object_events = detect_objects(frame)
    events.extend(object_events)

    # ── Paso 3: Detectar poses corporales ─────────────────────────────────────
    results = pose_model(frame, verbose=False)

    for result in results:
        if result.keypoints is None:
            continue

        for person_kps in result.keypoints.xy.cpu().numpy():
            if len(person_kps) < 17:
                continue

            # Filtro de tamaño: ignoramos detecciones pequeñas del fondo
            # que no corresponden a personas reales frente a la cámara
            if get_bbox_area(person_kps) < MIN_PERSON_AREA:
                continue

            left_wrist  = person_kps[9]
            right_wrist = person_kps[10]

            # ── Paso 4: Medir aproximación de muñeca hacia el rostro ──────────
            punch_detected = False

            if working_face_center is not None:
                current_pos = {
                    "lw":   left_wrist.tolist(),
                    "rw":   right_wrist.tolist(),
                    "face": list(working_face_center),
                    "time": datetime.now().timestamp()
                }

                if len(pose_history) > 0:
                    prev = pose_history[-1]
                    dt   = current_pos["time"] - prev["time"]

                    if dt > 0 and "face" in prev:
                        # Distancia actual y anterior de cada muñeca al centro del rostro
                        dist_lw_now  = euclidean_distance(current_pos["lw"],
                                                          current_pos["face"])
                        dist_lw_prev = euclidean_distance(prev["lw"], prev["face"])
                        dist_rw_now  = euclidean_distance(current_pos["rw"],
                                                          current_pos["face"])
                        dist_rw_prev = euclidean_distance(prev["rw"], prev["face"])

                        # Velocidad de aproximación en píxeles por segundo.
                        # Valor positivo = la mano se acerca a la cara.
                        # Valor negativo = la mano se aleja de la cara.
                        # Solo nos interesa el caso positivo (acercamiento).
                        approach_left  = (dist_lw_prev - dist_lw_now) / dt
                        approach_right = (dist_rw_prev - dist_rw_now) / dt

                        left_punching = (
                            approach_left  > HIT_VELOCITY_THRESHOLD and
                            dist_lw_now    < MAX_DISTANCE_TO_FACE
                        )
                        right_punching = (
                            approach_right > HIT_VELOCITY_THRESHOLD and
                            dist_rw_now    < MAX_DISTANCE_TO_FACE
                        )

                        punch_detected = left_punching or right_punching

                        # Línea roja visual que muestra la trayectoria del golpe
                        # hacia el rostro, útil para verificar que el sistema
                        # está midiendo correctamente durante las pruebas
                        if punch_detected:
                            active_wrist = (left_wrist if left_punching
                                            else right_wrist)
                            if active_wrist[0] > 0 and active_wrist[1] > 0:
                                cv2.line(frame,
                                         (int(active_wrist[0]),
                                          int(active_wrist[1])),
                                         working_face_center,
                                         (0, 0, 255), 2)

                pose_history.append(current_pos)

            else:
                # Sin cara conocida, guardamos igual para mantener el historial
                pose_history.append({
                    "lw":   left_wrist.tolist(),
                    "rw":   right_wrist.tolist(),
                    "time": datetime.now().timestamp()
                })

            # ── Paso 5: Ventana deslizante para confirmar el golpe ────────────
            #
            # Esta es la mejora clave respecto a la versión anterior.
            # Antes: necesitábamos 6 frames PERFECTAMENTE consecutivos.
            #        Si el frame 4 fallaba, el contador volvía a 0.
            # Ahora: miramos los últimos 8 frames como un grupo y preguntamos
            #        ¿cuántos de ellos detectaron golpe?
            #        Si 5 o más lo hicieron, confirmamos la alerta.
            #        Esto tolera 2-3 frames de ruido sin perder el conteo.

            # Agregamos True o False según si este frame tuvo detección
            hit_window_buffer.append(punch_detected)

            # Mantenemos el buffer con máximo HIT_WINDOW_SIZE entradas.
            # Cuando se llena, el frame más antiguo sale (deslizamiento).
            if len(hit_window_buffer) > HIT_WINDOW_SIZE:
                hit_window_buffer.pop(0)

            # Contamos cuántos frames recientes tuvieron detección positiva
            recent_hits = sum(hit_window_buffer)

            # Confirmamos el golpe solo si:
            # a) La ventana está llena (tenemos suficientes frames para decidir)
            # b) Al menos HIT_WINDOW_MIN frames detectaron golpe
            person_confirmed_hit = (
                len(hit_window_buffer) >= HIT_WINDOW_SIZE and
                recent_hits >= HIT_WINDOW_MIN
            )

            draw_skeleton(frame, person_kps, person_confirmed_hit)

            if person_confirmed_hit:
                frame_is_aggressive = True
                status_text  = "!!! GOLPE DETECTADO"
                status_color = (0, 0, 255)

    # Mantenemos solo los últimos 15 frames en el historial de poses
    if len(pose_history) > 15:
        pose_history.pop(0)

    # ── Paso 6: Generar evento de alerta con cooldown ─────────────────────────
    # El cooldown evita que se generen decenas de alertas por el mismo incidente.
    # Una vez confirmado el golpe, esperamos ALERT_COOLDOWN_SECONDS antes de
    # generar otra alerta, aunque el golpe siga siendo detectado.
    if frame_is_aggressive:
        now = datetime.now().timestamp()
        if now - last_alert_time > ALERT_COOLDOWN_SECONDS:
            last_alert_time = now
            events.append({
                "tipo":      "agresion",
                "label":     "Golpe detectado: muneca aproximandose al rostro",
                "confianza": 82.0,
                "bbox":      None,
                "timestamp": datetime.now().isoformat()
            })

    # ── Texto de estado en la parte superior de la ventana ───────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 45), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, status_text, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)

    # Contador de la ventana deslizante para calibrar durante las pruebas.
    # Te muestra exactamente cuántos de los últimos frames detectaron golpe.
    recent = sum(hit_window_buffer) if hit_window_buffer else 0
    total  = len(hit_window_buffer)
    cv2.putText(frame,
                f"Detecciones recientes: {recent}/{total} (umbral: {HIT_WINDOW_MIN})",
                (10, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.putText(frame, "Presiona 'Q' para cerrar",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # ── Mostrar ventana con todo anotado ──────────────────────────────────────
    cv2.imshow("PROJECT_VIGIA - Monitor en tiempo real", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    return events