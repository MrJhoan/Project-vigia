# detector.py - VERSION 2.6
# Mejoras:
#   1. Período de calentamiento: no alarma en los primeros 3s de cada video
#   2. Detección de grupo atacante: 2+ personas juntas + víctima en suelo
#   3. Rastreo de objeto pequeño para hurto de billetera
#   4. Roles por posición espacial estables (no por índice)
#   5. Falsos positivos reducidos con warmup y umbrales más altos

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import math
import time
import alarm

# ── Modelos ───────────────────────────────────────────────────────────────────
pose_model   = YOLO("yolov8n-pose.pt")
object_model = YOLO("yolov8n.pt")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Clases del dataset COCO que usamos ───────────────────────────────────────
DANGEROUS_CLASSES = {43: "Cuchillo"}
# Objetos pequeños que pueden ser billetera, dinero, cartera
# 67=cell phone (forma similar a billetera), 73=book, 84=book
# Usamos cell phone como proxy para objetos rectangulares pequeños
SMALL_OBJECTS = {67: "Objeto pequeño", 73: "Objeto", 84: "Objeto"}

# ── Colores BGR ───────────────────────────────────────────────────────────────
COLOR_SAFE      = (0, 255, 0)
COLOR_AGGRESSOR = (0, 0, 255)
COLOR_VICTIM    = (0, 165, 255)
COLOR_WEAPON    = (0, 100, 255)
COLOR_SUSPECT   = (0, 200, 255)
COLOR_OBJECT    = (255, 255, 0)   # Cian para objetos rastreados

# ── Parámetros ────────────────────────────────────────────────────────────────
HIT_VELOCITY_THRESHOLD = 480
MAX_DISTANCE_TO_FACE   = 270
MIN_PERSON_AREA        = 2000    # Bajo para detectar personas lejanas (motos)
OBJECT_CONFIDENCE      = 0.38
ALERT_COOLDOWN_SECONDS = 18
WARMUP_SECONDS         = 3.0    # No alarmar en los primeros N segundos del video

# Ventana deslizante golpe
HIT_WINDOW_SIZE = 12
HIT_WINDOW_MIN  = 8

# Ventana hurto
THEFT_WINDOW_SIZE = 30
THEFT_WINDOW_MIN  = 22

# Agresión grupal: distancia máxima entre miembros del grupo
GROUP_DISTANCE = 220    # px

# Roles estables por posición
ROLE_LOCK_FRAMES = 45
ROLE_MATCH_DIST  = 130

# ── Estado interno ────────────────────────────────────────────────────────────
pose_history        = []
last_alert_time     = 0.0
last_face_center    = None
hit_window_buffer   = []
theft_window_buffer = []
tracked_roles       = []   # [{"center": (x,y), "role": str, "frames_left": int}]

# Para rastreo de objetos pequeños (detección de hurto)
# Guardamos posiciones previas de objetos pequeños
prev_small_objects  = []   # lista de (cx, cy) de frames anteriores
object_in_hand_buffer = [] # True/False por frame

# Tiempo de inicio del video actual (para warmup)
video_start_time    = None


# ── Esqueleto ─────────────────────────────────────────────────────────────────
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def reset_video_state():
    """
    Llama esto desde main.py cuando empieza un nuevo video.
    Limpia todo el estado acumulado del video anterior.
    """
    global pose_history, last_face_center, hit_window_buffer
    global theft_window_buffer, tracked_roles, prev_small_objects
    global object_in_hand_buffer, video_start_time, last_alert_time

    pose_history          = []
    last_face_center      = None
    hit_window_buffer     = []
    theft_window_buffer   = []
    tracked_roles         = []
    prev_small_objects    = []
    object_in_hand_buffer = []
    video_start_time      = time.time()
    last_alert_time       = 0.0


def is_in_warmup():
    """Retorna True si todavía estamos en el período de calentamiento."""
    if video_start_time is None:
        return False
    return (time.time() - video_start_time) < WARMUP_SECONDS


def draw_skeleton(frame, kps, color):
    for i, j in SKELETON_CONNECTIONS:
        if i < len(kps) and j < len(kps):
            p1, p2 = kps[i], kps[j]
            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                cv2.line(frame,
                         (int(p1[0]), int(p1[1])),
                         (int(p2[0]), int(p2[1])),
                         color, 2)
    for kp in kps:
        if kp[0] > 0 and kp[1] > 0:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, -1)


def get_bbox_area(kps):
    v = [(k[0], k[1]) for k in kps if k[0] > 0 and k[1] > 0]
    if len(v) < 4:
        return 0
    xs, ys = [p[0] for p in v], [p[1] for p in v]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def get_bbox_center(kps):
    v = [(k[0], k[1]) for k in kps if k[0] > 0 and k[1] > 0]
    if len(v) < 2:
        return None
    xs, ys = [p[0] for p in v], [p[1] for p in v]
    return (int((min(xs)+max(xs))/2), int((min(ys)+max(ys))/2))


def dist(p1, p2):
    if p1 is None or p2 is None:
        return float("inf")
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def get_role_for_person(center):
    if center is None:
        return None
    for tr in tracked_roles:
        if dist(center, tr["center"]) < ROLE_MATCH_DIST:
            return tr["role"]
    return None


def assign_role(center, role):
    if center is None:
        return
    for tr in tracked_roles:
        if dist(center, tr["center"]) < ROLE_MATCH_DIST:
            tr["center"]      = center
            tr["frames_left"] = ROLE_LOCK_FRAMES
            # Regla: VICTIM no se convierte en AGGRESSOR
            if role == "AGGRESSOR" and tr["role"] != "VICTIM":
                tr["role"] = role
            elif role == "VICTIM" and tr["role"] == "SAFE":
                tr["role"] = role
            elif role == "SUSPECT":
                tr["role"] = role
            return
    tracked_roles.append({
        "center":      center,
        "role":        role,
        "frames_left": ROLE_LOCK_FRAMES
    })


def tick_roles():
    global tracked_roles
    for tr in tracked_roles:
        tr["frames_left"] -= 1
    tracked_roles = [tr for tr in tracked_roles if tr["frames_left"] > 0]


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return [(x, y, w, h, (x+w//2, y+h//2)) for (x, y, w, h) in faces]


def draw_face_box(frame, face, color, label="Rostro"):
    x, y, w, h, _ = face
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    label_y = y - 25 if y > 30 else y + h + 5
    tw = len(label) * 10 + 8
    cv2.rectangle(frame, (x, label_y), (x+tw, label_y+22), color, -1)
    cv2.putText(frame, label, (x+4, label_y+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1)


def detect_objects(frame):
    """Detecta armas y objetos pequeños (proxy para billetera)."""
    events, weapon_boxes, small_obj_positions = [], [], []

    # Armas
    results_weapons = object_model(
        frame, classes=list(DANGEROUS_CLASSES.keys()),
        conf=OBJECT_CONFIDENCE, verbose=False
    )
    for result in results_weapons:
        for box in result.boxes:
            cid  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = DANGEROUS_CLASSES.get(cid, "Arma")
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_WEAPON, 3)
            txt = f"{label} {conf*100:.0f}%"
            bw  = len(txt) * 11
            cv2.rectangle(frame, (x1, y1-28), (x1+bw, y1), COLOR_WEAPON, -1)
            cv2.putText(frame, txt, (x1+3, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
            wc = ((x1+x2)//2, (y1+y2)//2)
            weapon_boxes.append({"label": label, "center": wc})
            events.append({"tipo": "arma", "label": label,
                           "confianza": round(conf*100, 1),
                           "timestamp": datetime.now().isoformat()})

    # Objetos pequeños (para rastreo de billetera)
    all_small_classes = list(SMALL_OBJECTS.keys())
    results_small = object_model(
        frame, classes=all_small_classes,
        conf=0.35, verbose=False
    )
    for result in results_small:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1+x2)//2, (y1+y2)//2
            small_obj_positions.append((cx, cy))
            # Dibujamos el objeto detectado en cian
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_OBJECT, 1)

    return events, weapon_boxes, small_obj_positions


def is_person_on_ground(kps, frame_height):
    """
    Persona en el suelo: bounding box más ancho que alto
    y posición en la mitad inferior del frame.
    """
    center = get_bbox_center(kps)
    if center is None:
        return False
    v = [(k[0], k[1]) for k in kps if k[0] > 0 and k[1] > 0]
    if len(v) < 6:
        return False
    xs, ys = [p[0] for p in v], [p[1] for p in v]
    width  = max(xs) - min(xs)
    height = max(ys) - min(ys)
    is_horizontal = height > 0 and width / height > 1.3
    in_lower_half = center[1] > frame_height * 0.45
    return is_horizontal and in_lower_half


def detect_group_attack(persons_centers, frame_height, persons_kps):
    """
    Detecta ataque grupal: 2+ personas muy juntas (< GROUP_DISTANCE px)
    y al menos una persona en el suelo cerca del grupo.

    Este es el algoritmo específico para el video de los motociclistas:
    cuando la víctima cae al suelo y el grupo la rodea, se confirma el ataque.

    Retorna (True, [indices_agresores], indice_victima) o (False, [], None)
    """
    if len(persons_centers) < 2:
        return False, [], None

    # Encontrar grupos de personas cercanas
    groups = []
    used   = set()

    for i, ci in enumerate(persons_centers):
        if i in used or ci is None:
            continue
        group = [i]
        for j, cj in enumerate(persons_centers):
            if j <= i or j in used or cj is None:
                continue
            if dist(ci, cj) < GROUP_DISTANCE:
                group.append(j)
        if len(group) >= 2:
            groups.append(group)
            used.update(group)

    if not groups:
        return False, [], None

    # Para cada grupo, verificar si hay una persona en el suelo cerca
    for group in groups:
        group_center_x = sum(persons_centers[i][0]
                             for i in group if persons_centers[i]) / len(group)
        group_center_y = sum(persons_centers[i][1]
                             for i in group if persons_centers[i]) / len(group)
        group_center = (int(group_center_x), int(group_center_y))

        # Buscar persona en el suelo cerca del grupo
        for vi, vkps in enumerate(persons_kps):
            if vi in group:
                continue
            if is_person_on_ground(vkps, frame_height):
                vc = get_bbox_center(vkps)
                if vc and dist(vc, group_center) < 350:
                    return True, group, vi

    return False, [], None


def detect_object_taken(small_obj_positions, persons_kps, persons_centers):
    """
    Detecta si alguien tomó un objeto que estaba sobre una superficie.

    Lógica:
    - En frames anteriores, el objeto estaba en una posición fija (sobre la mesa)
    - En el frame actual, el objeto está cerca de las manos de una persona
      Y ya no está en su posición original sobre la superficie

    Esto implementa la idea del estudiante:
    "Si el objeto X aparece en sus manos cuando antes no lo tenía = alarma"
    """
    global prev_small_objects

    if not small_obj_positions:
        # No hay objetos detectados en este frame
        prev_small_objects = []
        return False, None

    # Verificar si algún objeto detectado está muy cerca de las manos de una persona
    for person_kps in persons_kps:
        lw = person_kps[9]   # muñeca izquierda
        rw = person_kps[10]  # muñeca derecha

        for obj_pos in small_obj_positions:
            # ¿El objeto está cerca de alguna mano?
            dist_to_lw = dist(obj_pos, (lw[0], lw[1])) if lw[0] > 0 else float("inf")
            dist_to_rw = dist(obj_pos, (rw[0], rw[1])) if rw[0] > 0 else float("inf")
            min_dist_to_hand = min(dist_to_lw, dist_to_rw)

            if min_dist_to_hand < 80:
                # El objeto está en la mano de alguien
                # Verificamos si en frames anteriores este objeto
                # estaba en una posición diferente (sobre una superficie)
                for prev_pos in prev_small_objects:
                    displacement = dist(obj_pos, prev_pos)
                    # Si el objeto se movió más de 60px hacia las manos,
                    # alguien lo tomó
                    if displacement > 60:
                        person_center = get_bbox_center(person_kps)
                        return True, person_center

    # Actualizar posiciones previas
    prev_small_objects = small_obj_positions.copy()
    return False, None


def analyze_frame(frame):
    global pose_history, last_alert_time, last_face_center
    global hit_window_buffer, theft_window_buffer
    global object_in_hand_buffer

    events              = []
    frame_is_aggressive = False
    weapon_detected     = False
    has_punch           = False
    has_theft           = False
    has_group_attack    = False
    punch_person_center = None
    frame_height        = frame.shape[0]

    tick_roles()

    # ── WARMUP: no alarmar en los primeros segundos ───────────────────────────
    in_warmup = is_in_warmup()

    # ── 1. Rostros ────────────────────────────────────────────────────────────
    faces = detect_faces(frame)
    if faces:
        biggest = max(faces, key=lambda f: f[2]*f[3])
        last_face_center = biggest[4]
    working_face = last_face_center

    # ── 2. Objetos ────────────────────────────────────────────────────────────
    obj_events, weapon_boxes, small_obj_pos = detect_objects(frame)
    events.extend(obj_events)
    if weapon_boxes and not in_warmup:
        weapon_detected     = True
        frame_is_aggressive = True

    # ── 3. Poses ──────────────────────────────────────────────────────────────
    results         = pose_model(frame, verbose=False)
    persons_kps     = []
    persons_centers = []

    for result in results:
        if result.keypoints is None:
            continue
        for pkps in result.keypoints.xy.cpu().numpy():
            if len(pkps) < 17:
                continue
            if get_bbox_area(pkps) < MIN_PERSON_AREA:
                continue
            persons_kps.append(pkps)
            persons_centers.append(get_bbox_center(pkps))

    num_persons = len(persons_kps)

    # ── 4. Detección de golpe ─────────────────────────────────────────────────
    punch_this_frame = False

    if working_face is not None and persons_kps and not in_warmup:
        for pi, pkps in enumerate(persons_kps):
            lw  = pkps[9]
            rw  = pkps[10]
            cur = {
                "lw":   lw.tolist(),
                "rw":   rw.tolist(),
                "face": list(working_face),
                "time": time.time()
            }
            if pose_history:
                prev = pose_history[-1]
                dt   = cur["time"] - prev["time"]
                if dt > 0 and "face" in prev:
                    d_lw  = dist(cur["lw"],  cur["face"])
                    d_lw0 = dist(prev["lw"], prev["face"])
                    d_rw  = dist(cur["rw"],  cur["face"])
                    d_rw0 = dist(prev["rw"], prev["face"])
                    ap_l  = (d_lw0 - d_lw) / dt
                    ap_r  = (d_rw0 - d_rw) / dt
                    lp = ap_l > HIT_VELOCITY_THRESHOLD and d_lw < MAX_DISTANCE_TO_FACE
                    rp = ap_r > HIT_VELOCITY_THRESHOLD and d_rw < MAX_DISTANCE_TO_FACE
                    if lp or rp:
                        punch_this_frame    = True
                        punch_person_center = persons_centers[pi]
                        aw = lw if lp else rw
                        if aw[0] > 0 and aw[1] > 0:
                            cv2.line(frame,
                                     (int(aw[0]), int(aw[1])),
                                     working_face, COLOR_AGGRESSOR, 2)
            pose_history.append(cur)

    if len(pose_history) > 15:
        pose_history.pop(0)

    hit_window_buffer.append(punch_this_frame)
    if len(hit_window_buffer) > HIT_WINDOW_SIZE:
        hit_window_buffer.pop(0)
    punch_confirmed = (not in_warmup and
                       len(hit_window_buffer) >= HIT_WINDOW_SIZE and
                       sum(hit_window_buffer) >= HIT_WINDOW_MIN)
    if punch_confirmed:
        frame_is_aggressive = True
        has_punch           = True

    # ── 5. Detección de ataque grupal (video de motos) ────────────────────────
    if not in_warmup and num_persons >= 2:
        group_attack, aggressor_group_indices, victim_idx = detect_group_attack(
            persons_centers, frame_height, persons_kps
        )
        if group_attack:
            has_group_attack    = True
            frame_is_aggressive = True
            # Marcar todos los del grupo como agresores
            for idx in aggressor_group_indices:
                if idx < len(persons_centers):
                    assign_role(persons_centers[idx], "AGGRESSOR")
            # Marcar la víctima en el suelo
            if victim_idx is not None and victim_idx < len(persons_centers):
                assign_role(persons_centers[victim_idx], "VICTIM")

    # ── 6. Detección de hurto por objeto tomado ───────────────────────────────
    object_taken, thief_center = detect_object_taken(
        small_obj_pos, persons_kps, persons_centers
    )

    object_in_hand_buffer.append(object_taken)
    if len(object_in_hand_buffer) > 20:
        object_in_hand_buffer.pop(0)

    # Confirmamos hurto si en 8 de los últimos 20 frames se detectó objeto en mano
    theft_confirmed = (not in_warmup and
                       len(object_in_hand_buffer) >= 15 and
                       sum(object_in_hand_buffer) >= 8)

    if theft_confirmed:
        frame_is_aggressive = True
        has_theft           = True

    # ── 7. Asignar roles ──────────────────────────────────────────────────────
    if frame_is_aggressive and persons_kps:
        if punch_confirmed and punch_person_center:
            assign_role(punch_person_center, "AGGRESSOR")

        if weapon_boxes and persons_centers:
            for wb in weapon_boxes:
                best_d, best_c = float("inf"), None
                for pc in persons_centers:
                    if pc:
                        d = dist(pc, wb["center"])
                        if d < best_d and d < 350:
                            best_d, best_c = d, pc
                if best_c:
                    assign_role(best_c, "AGGRESSOR")

        if theft_confirmed and thief_center:
            assign_role(thief_center, "SUSPECT")

        # Personas en el suelo → VICTIM
        for pkps in persons_kps:
            if is_person_on_ground(pkps, frame_height):
                c = get_bbox_center(pkps)
                assign_role(c, "VICTIM")

        # Resto de personas en escena → VICTIM si no tienen rol
        for pc in persons_centers:
            if pc and get_role_for_person(pc) is None:
                assign_role(pc, "VICTIM")

    # ── 8. Dibujar esqueletos ─────────────────────────────────────────────────
    for pkps in persons_kps:
        center = get_bbox_center(pkps)
        role   = get_role_for_person(center)

        if not frame_is_aggressive or role is None:
            color = COLOR_SAFE
        elif role == "AGGRESSOR":
            color = COLOR_AGGRESSOR
        elif role == "VICTIM":
            color = COLOR_VICTIM
        elif role == "SUSPECT":
            color = COLOR_SUSPECT
        else:
            color = COLOR_SAFE

        draw_skeleton(frame, pkps, color)

        if is_person_on_ground(pkps, frame_height):
            if center:
                cv2.putText(frame, "VICTIMA EN EL SUELO",
                            (center[0]-65, center[1]-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)

    # ── 9. Dibujar rostros ────────────────────────────────────────────────────
    for face in faces:
        fc   = face[4]
        role = get_role_for_person(fc)
        if not frame_is_aggressive or role is None:
            draw_face_box(frame, face, COLOR_SAFE, "Rostro")
        elif role == "AGGRESSOR":
            draw_face_box(frame, face, COLOR_AGGRESSOR, "Agresor")
        elif role == "VICTIM":
            draw_face_box(frame, face, COLOR_VICTIM, "Victima")
        elif role == "SUSPECT":
            draw_face_box(frame, face, COLOR_SUSPECT, "Sospechoso")
        else:
            draw_face_box(frame, face, COLOR_SAFE, "Rostro")

    # ── 10. Alarma ────────────────────────────────────────────────────────────
    if frame_is_aggressive and not in_warmup:
        alarm.start_alarm("AGRESION DETECTADA")
    else:
        alarm.stop_alarm()

    # ── 11. Evento con cooldown ───────────────────────────────────────────────
    if frame_is_aggressive and not in_warmup:
        now = time.time()
        if now - last_alert_time > ALERT_COOLDOWN_SECONDS:
            last_alert_time = now
            events.append({
                "tipo":      "alerta",
                "label":     "AGRESION DETECTADA - Llamando al 123",
                "confianza": 82.0,
                "timestamp": datetime.now().isoformat()
            })

    # ── 12. Barra de estado ───────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 55), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    if in_warmup:
        remaining_warmup = max(0, WARMUP_SECONDS - (time.time() - video_start_time))
        cv2.putText(frame,
                    f"Iniciando sistema... ({remaining_warmup:.1f}s)",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (180, 180, 0), 2)
    elif frame_is_aggressive:
        cv2.putText(frame,
                    "!!! AGRESION DETECTADA  -  ALARMA ACTIVA",
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)
        cv2.putText(frame,
                    "Llamando al 123 - Policia Nacional",
                    (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 140, 255), 2)
    else:
        cv2.putText(frame,
                    "Estado: Monitoreando...",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 0), 2)

    agr_count = sum(1 for tr in tracked_roles if tr["role"] == "AGGRESSOR")
    cv2.putText(frame,
                f"Agresores: {agr_count} | Personas: {num_persons} | "
                f"Alarma: {'ACTIVA' if alarm.is_alarm_active() else 'OFF'}",
                (8, frame.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    return events, frame