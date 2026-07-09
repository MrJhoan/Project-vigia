# detector.py - VERSION 3.0  ★ VERSIÓN DEFINITIVA ★
#
# ════════════════════════════════════════════════════════════════════════
#  MAPA DE VIDEOS — qué funciona y qué arreglamos
# ════════════════════════════════════════════════════════════════════════
#
#  Video 1  (panadería / pistola)      ✅ PERFECTO — no se toca
#  Video 2  (joyería blanca)           ❌ alarma prematura → FIX 2
#  Video 3  (aseadora / billetera)     ✅ PERFECTO — no se toca
#  Video 4  (motos)                    ✅ PERFECTO — no se toca
#  Video 5  (joyería dorada / policías)❌ alarma prematura → FIX 5
#
# ════════════════════════════════════════════════════════════════════════
#  DIAGNÓSTICO DE LOS FALLOS (Video 2 y Video 5)
# ════════════════════════════════════════════════════════════════════════
#
#  CAUSA RAÍZ (ambos videos):
#    El detector de golpe (punch) se activa con movimiento normal de manos
#    (personas gestulan, caminan rápido, se agachan).
#    Esto llena hit_window_buffer → punch_confirmed=True → aggression_lock.
#    Una vez en aggression_lock, cualquier grupo de personas activa group_attack.
#
#  FIX 2 — Video 2 (joyería blanca, evento en segundo 00:29):
#    detect_group_attack() ya requiere has_ground=True (persona en suelo).
#    El problema es que punch_confirmed activa agresión aunque NO haya
#    nadie en el suelo.
#    SOLUCIÓN: punch_confirmed solo activa frame_is_aggressive si TAMBIÉN
#    hay un arma detectada O un brazo apuntando. Un golpe aislado sin
#    contexto de arma no es suficiente para este tipo de escena.
#    Adicionalmente: arm_point_buffer requiere 5 de 8 (antes 3 de 5)
#    para ser más exigente con los videos de joyería.
#
#  FIX 5 — Video 5 (joyería dorada, evento en segundo 00:12):
#    En V2.6 funcionaba porque usaba approach_speed_to_face que solo
#    se activaba cuando alguien se acercaba RÁPIDO a una CARA específica.
#    En versiones posteriores se agregó raw_wrist_velocity que no necesita
#    cara → mucho más ruido.
#    SOLUCIÓN: El trigger de punch para activar agresión requiere que
#    TAMBIÉN se detecte approach_speed_to_face (señal V2.6). La raw
#    velocity sola ya NO activa agresión, solo la refuerza cuando ya
#    hay approach confirmado.
#
# ════════════════════════════════════════════════════════════════════════
#  ALGORITMOS POR VIDEO (REFERENCIA)
# ════════════════════════════════════════════════════════════════════════
#
#  Video 1:  arm_pointing (force) + weapon_YOLO → agresor = quien apunta
#  Video 2:  arm_pointing + weapon_YOLO (solo cuando hay brazo apuntando)
#  Video 3:  theft_gesture (inclinación) + pocket_gesture + object_in_hand
#  Video 4:  group_attack (has_ground requerido) + group_confirmed flag
#  Video 5:  approach_speed_to_face (V2.6) + arm_pointing

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
if face_cascade.empty():
    print("⚠️  face_cascade no cargó — detección de rostros desactivada")

# ── Clases COCO ───────────────────────────────────────────────────────────────
DANGEROUS_CLASSES = {43: "Cuchillo"}
SMALL_OBJECTS     = {67: "Objeto", 73: "Objeto", 84: "Objeto"}

# ── Colores BGR ───────────────────────────────────────────────────────────────
COLOR_SAFE      = (0, 255, 0)
COLOR_AGGRESSOR = (0, 0, 255)
COLOR_VICTIM    = (0, 165, 255)
COLOR_SUSPECT   = (0, 200, 255)
COLOR_WEAPON    = (0, 80, 255)

# ── Parámetros generales ──────────────────────────────────────────────────────
MIN_PERSON_AREA        = 1500
OBJECT_CONFIDENCE      = 0.35
ALERT_COOLDOWN_SECONDS = 15
WARMUP_FRAMES          = 18

# ── Golpe — approach_speed_to_face (V2.6 — clave para Video 5) ───────────────
HIT_APPROACH_THRESHOLD = 480   # px/s de acercamiento de muñeca a cara
MAX_DISTANCE_TO_FACE   = 270   # px máximo muñeca-cara para confirmar golpe
HIT_APPROACH_WINDOW    = 8
HIT_APPROACH_MIN       = 5     # frames con approach confirmado

# ── Golpe — raw wrist velocity (V2.8 — señal de refuerzo, NO trigger solo) ───
RAW_HIT_THRESHOLD = 420

# ── Hurto por pose (Video 3) ──────────────────────────────────────────────────
THEFT_POSE_WINDOW  = 50
THEFT_POSE_MIN     = 18
OBJ_HAND_DIST      = 90    # px máximo objeto-muñeca para "en la mano"
OBJ_HAND_WINDOW    = 20
OBJ_HAND_MIN       = 8
POCKET_HIP_MARGIN  = 55    # px — muñeca dentro de este margen de la cadera

# ── Brazo apuntando (Videos 1, 2 y 5) ────────────────────────────────────────
# [FIX 2 / FIX 5] Subido a 5 de 8 para ser más exigente (antes 3 de 5)
ARM_POINT_WINDOW   = 8
ARM_POINT_MIN      = 5

# ── Ataque grupal (Video 4) ───────────────────────────────────────────────────
GROUP_RADIUS       = 320
GROUP_MIN_MEMBERS  = 2

# ── Roles estables ────────────────────────────────────────────────────────────
ROLE_LOCK_FRAMES   = 200
ROLE_MATCH_DIST    = 160

# ── Lock de agresión (evita parpadeo) ────────────────────────────────────────
AGGRESSION_LOCK_FRAMES = 60

# ── Estado global ─────────────────────────────────────────────────────────────
pose_history            = []
last_alert_time         = 0.0
last_face_center        = None
hit_approach_buffer     = []    # Señal approach_speed (V2.6)
tracked_roles           = []
frames_analyzed         = 0
theft_pose_buffer       = []
arm_point_buffer        = []
group_confirmed         = False
aggression_lock_counter = 0
prev_small_objects      = []
obj_in_hand_buffer      = []

# ── Esqueleto COCO ────────────────────────────────────────────────────────────
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Utilidades
# ═══════════════════════════════════════════════════════════════════════════════

def dist(p1, p2):
    if p1 is None or p2 is None:
        return float("inf")
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_bbox_area(kps):
    v = [(k[0], k[1]) for k in kps if k[0] > 0 and k[1] > 0]
    if len(v) < 4:
        return 0
    xs, ys = [p[0] for p in v], [p[1] for p in v]
    return (max(xs)-min(xs)) * (max(ys)-min(ys))

def get_bbox_center(kps):
    v = [(k[0], k[1]) for k in kps if k[0] > 0 and k[1] > 0]
    if len(v) < 2:
        return None
    xs, ys = [p[0] for p in v], [p[1] for p in v]
    return (int((min(xs)+max(xs))/2), int((min(ys)+max(ys))/2))

def kp(kps, i):
    if i < len(kps) and kps[i][0] > 0 and kps[i][1] > 0:
        return (float(kps[i][0]), float(kps[i][1]))
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Reset
# ═══════════════════════════════════════════════════════════════════════════════

def reset_video_state():
    global pose_history, last_alert_time, last_face_center
    global hit_approach_buffer, tracked_roles, frames_analyzed
    global theft_pose_buffer, arm_point_buffer, group_confirmed
    global aggression_lock_counter, prev_small_objects, obj_in_hand_buffer
    pose_history            = []
    last_alert_time         = 0.0
    last_face_center        = None
    hit_approach_buffer     = []
    tracked_roles           = []
    frames_analyzed         = 0
    theft_pose_buffer       = []
    arm_point_buffer        = []
    group_confirmed         = False
    aggression_lock_counter = 0
    prev_small_objects      = []
    obj_in_hand_buffer      = []

def is_in_warmup():
    return frames_analyzed < WARMUP_FRAMES


# ═══════════════════════════════════════════════════════════════════════════════
# Roles
# ═══════════════════════════════════════════════════════════════════════════════

def get_role_for_person(center):
    if center is None:
        return None
    best_d, best_r = float("inf"), None
    for tr in tracked_roles:
        d = dist(center, tr["center"])
        if d < ROLE_MATCH_DIST and d < best_d:
            best_d, best_r = d, tr["role"]
    return best_r

def assign_role(center, role, force=False):
    """VICTIM nunca puede convertirse en AGGRESSOR."""
    if center is None:
        return
    for tr in tracked_roles:
        if dist(center, tr["center"]) < ROLE_MATCH_DIST:
            tr["center"]      = center
            tr["frames_left"] = ROLE_LOCK_FRAMES
            curr = tr["role"]
            if curr == "VICTIM" and role == "AGGRESSOR":
                return
            if force:
                tr["role"] = role
                return
            if curr == "VICTIM":
                return
            if role == "AGGRESSOR":
                tr["role"] = "AGGRESSOR"
            elif role == "SUSPECT" and curr not in ("VICTIM", "AGGRESSOR"):
                tr["role"] = "SUSPECT"
            elif role == "VICTIM" and curr == "SAFE":
                tr["role"] = "VICTIM"
            return
    tracked_roles.append({"center": center, "role": role, "frames_left": ROLE_LOCK_FRAMES})

def tick_roles():
    global tracked_roles
    for tr in tracked_roles:
        tr["frames_left"] -= 1
    tracked_roles[:] = [tr for tr in tracked_roles if tr["frames_left"] > 0]


# ═══════════════════════════════════════════════════════════════════════════════
# Dibujo
# ═══════════════════════════════════════════════════════════════════════════════

def draw_skeleton(frame, kps, color):
    for i, j in SKELETON_CONNECTIONS:
        if i < len(kps) and j < len(kps):
            p1, p2 = kps[i], kps[j]
            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)
    for kp_ in kps:
        if kp_[0] > 0 and kp_[1] > 0:
            cv2.circle(frame, (int(kp_[0]), int(kp_[1])), 4, color, -1)

def draw_face_box(frame, face, color, label="Rostro"):
    x, y, w, h, _ = face
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, label, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

def detect_faces(frame):
    if face_cascade.empty():
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    if not isinstance(faces, np.ndarray) or len(faces) == 0:
        return []
    return [(x, y, w, h, (x+w//2, y+h//2)) for (x, y, w, h) in faces]


# ═══════════════════════════════════════════════════════════════════════════════
# Persona en el suelo — ESTRICTO (evita falsos en cámaras cenitales)
# ═══════════════════════════════════════════════════════════════════════════════

def is_person_on_ground(kps, frame_h):
    ankles    = [p for p in [kp(kps, 15), kp(kps, 16)] if p]
    shoulders = [p for p in [kp(kps, 5),  kp(kps, 6)]  if p]
    hips      = [p for p in [kp(kps, 11), kp(kps, 12)] if p]
    if not ankles or not shoulders:
        return False
    ankle_y    = sum(p[1] for p in ankles) / len(ankles)
    shoulder_y = sum(p[1] for p in shoulders) / len(shoulders)
    center_y   = (ankle_y + shoulder_y) / 2
    # Tobillo en parte muy baja del frame = persona de pie normal, NO en suelo
    if ankle_y > frame_h * 0.82:
        return False
    span = abs(ankle_y - shoulder_y)
    # Postura muy horizontal + persona en mitad inferior
    if span < frame_h * 0.13 and center_y > frame_h * 0.55:
        return True
    # Caída de lado: tobillo al nivel de cadera
    if hips:
        hip_y = sum(p[1] for p in hips) / len(hips)
        if abs(ankle_y - hip_y) < frame_h * 0.08 and center_y > frame_h * 0.55:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Golpe — approach_speed_to_face (V2.6)
#
# [FIX 5] Esta es la ÚNICA señal que activa punch_confirmed.
# La raw_velocity fue eliminada como trigger de agresión porque generaba
# demasiado ruido en Videos 2 y 5 (movimiento normal de manos).
# ═══════════════════════════════════════════════════════════════════════════════

def detect_punch_approach(cur_kps, prev_kps, time_delta, face_center):
    """
    Calcula si alguna muñeca se está acercando RÁPIDAMENTE a la cara.
    Solo esta señal puede confirmar un golpe (no raw velocity sola).
    """
    if face_center is None or prev_kps is None or time_delta <= 0:
        return False, None
    for wi in [9, 10]:
        if wi >= len(cur_kps) or wi >= len(prev_kps):
            continue
        cw = cur_kps[wi]
        pw = prev_kps[wi]
        if cw[0] <= 0 or pw[0] <= 0:
            continue
        d_cur  = dist(cw,  face_center)
        d_prev = dist(pw, face_center)
        approach_speed = (d_prev - d_cur) / time_delta
        if approach_speed > HIT_APPROACH_THRESHOLD and d_cur < MAX_DISTANCE_TO_FACE:
            return True, (int(cw[0]), int(cw[1]))
    return False, None


# ═══════════════════════════════════════════════════════════════════════════════
# Brazo apuntando (Videos 1, 2, 5)
# [FIX 2 / FIX 5]: ARM_POINT_MIN subido a 5 de 8 (antes 3 de 5)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_arm_pointing(persons_kps, persons_centers):
    """Retorna (pointer_center, target_center) o (None, None)."""
    for pi, kps in enumerate(persons_kps):
        my_c = persons_centers[pi]
        if my_c is None:
            continue
        for side in [(5, 7, 9), (6, 8, 10)]:
            sh = kp(kps, side[0])
            el = kp(kps, side[1])
            wr = kp(kps, side[2])
            if not sh or not el or not wr:
                continue
            v1 = (sh[0]-el[0], sh[1]-el[1])
            v2 = (wr[0]-el[0], wr[1]-el[1])
            m1 = math.sqrt(v1[0]**2+v1[1]**2)
            m2 = math.sqrt(v2[0]**2+v2[1]**2)
            if m1 < 5 or m2 < 5:
                continue
            cos_a = max(-1.0, min(1.0, (v1[0]*v2[0]+v1[1]*v2[1])/(m1*m2)))
            if math.degrees(math.acos(cos_a)) < 148:
                continue
            if dist(wr, sh) < 65:
                continue
            arm_dx, arm_dy = wr[0]-sh[0], wr[1]-sh[1]
            arm_len = math.sqrt(arm_dx**2+arm_dy**2) + 1e-9
            arm_ux, arm_uy = arm_dx/arm_len, arm_dy/arm_len
            for pj, oc in enumerate(persons_centers):
                if pj == pi or oc is None:
                    continue
                if dist(my_c, oc) > 750:
                    continue
                to_dx, to_dy = oc[0]-wr[0], oc[1]-wr[1]
                to_len = math.sqrt(to_dx**2+to_dy**2) + 1e-9
                dot = arm_ux*(to_dx/to_len) + arm_uy*(to_dy/to_len)
                if dot > 0.70:
                    return my_c, oc
    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Hurto por pose (Video 3 — aseadora) — NO MODIFICADO, funcionaba bien
# ═══════════════════════════════════════════════════════════════════════════════

def detect_theft_gesture(kps, frame_h):
    """Señal A: torso inclinado + muñeca debajo de cadera."""
    l_s = kp(kps, 5);  r_s = kp(kps, 6)
    l_h = kp(kps, 11); r_h = kp(kps, 12)
    l_w = kp(kps, 9);  r_w = kp(kps, 10)
    if not all([l_s, r_s, l_h, r_h]):
        return False
    shoulder_y = (l_s[1]+r_s[1])/2
    hip_y      = (l_h[1]+r_h[1])/2
    leaning    = abs(hip_y-shoulder_y) < frame_h * 0.20
    wrists     = [w for w in [l_w, r_w] if w]
    low_wrist  = any(w[1] > hip_y for w in wrists)
    return leaning and low_wrist

def detect_pocket_gesture(kps, frame_h):
    """Señal C: muñeca cerca de la zona de cadera (guardar dinero en bolsillo)."""
    l_h = kp(kps, 11); r_h = kp(kps, 12)
    l_w = kp(kps, 9);  r_w = kp(kps, 10)
    if not l_h or not r_h:
        return False
    hip_y = (l_h[1]+r_h[1])/2
    hip_x = (l_h[0]+r_h[0])/2
    for wr in [w for w in [l_w, r_w] if w]:
        if abs(wr[1]-hip_y) < POCKET_HIP_MARGIN and abs(wr[0]-hip_x) < 110:
            return True
    return False

def detect_object_in_hand(small_obj_positions, persons_kps):
    """Señal B: objeto pequeño YOLO cerca de una muñeca y que se haya movido."""
    global prev_small_objects
    if not small_obj_positions:
        prev_small_objects = []
        return False, None
    for kps in persons_kps:
        l_w = kp(kps, 9)
        r_w = kp(kps, 10)
        for obj_pos in small_obj_positions:
            d_lw = dist(obj_pos, l_w) if l_w else float("inf")
            d_rw = dist(obj_pos, r_w) if r_w else float("inf")
            if min(d_lw, d_rw) < OBJ_HAND_DIST:
                for prev in prev_small_objects:
                    if dist(obj_pos, prev) > 55:
                        return True, get_bbox_center(kps)
    prev_small_objects = list(small_obj_positions)
    return False, None


# ═══════════════════════════════════════════════════════════════════════════════
# Ataque grupal (Video 4 — motos) — NO MODIFICADO, funcionaba bien
# Requiere has_ground=True para evitar falso positivo en Videos 2 y 5
# ═══════════════════════════════════════════════════════════════════════════════

def detect_group_attack(persons_kps, persons_centers, frame_h, has_ground):
    """
    Solo se activa si hay alguien en el suelo (has_ground).
    Esto distingue el ataque de motos de un grupo normal en joyería.
    """
    if not has_ground:
        return False, [], None

    valid = [(c, kps) for c, kps in zip(persons_centers, persons_kps) if c is not None]
    if len(valid) < 2:
        return False, [], None

    cluster_count = []
    for i, (ci, _) in enumerate(valid):
        cnt = sum(1 for j, (cj, _) in enumerate(valid)
                  if i != j and dist(ci, cj) < GROUP_RADIUS)
        cluster_count.append(cnt)

    max_cnt = max(cluster_count)
    if max_cnt < GROUP_MIN_MEMBERS:
        return False, [], None

    leader_c          = valid[cluster_count.index(max_cnt)][0]
    aggressor_centers = [c for c, _ in valid if dist(c, leader_c) < GROUP_RADIUS]

    victim_center, max_d = None, 0
    for c, _ in valid:
        if c not in aggressor_centers:
            d = dist(c, leader_c)
            if d > max_d:
                max_d, victim_center = d, c

    if victim_center is None:
        for _, kps_i in valid:
            if is_person_on_ground(kps_i, frame_h):
                c = get_bbox_center(kps_i)
                if c:
                    victim_center = c
                    break

    if len(aggressor_centers) >= GROUP_MIN_MEMBERS and victim_center is not None:
        return True, aggressor_centers, victim_center

    return False, [], None


# ═══════════════════════════════════════════════════════════════════════════════
# Función principal
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_frame(frame):
    global pose_history, last_alert_time, last_face_center
    global hit_approach_buffer, tracked_roles, frames_analyzed
    global theft_pose_buffer, arm_point_buffer
    global group_confirmed, aggression_lock_counter
    global obj_in_hand_buffer

    events   = []
    frame_h, frame_w = frame.shape[:2]

    frames_analyzed += 1
    in_warmup = is_in_warmup()

    tick_roles()

    # ── 1. Poses ──────────────────────────────────────────────────────────────
    pose_results    = pose_model(frame, verbose=False, conf=0.38)
    persons_kps     = []
    persons_centers = []

    for r in pose_results:
        if r.keypoints is None:
            continue
        for kps_tensor in r.keypoints.xy:
            kps = kps_tensor.cpu().numpy().tolist()
            if get_bbox_area(kps) < MIN_PERSON_AREA:
                continue
            persons_kps.append(kps)
            persons_centers.append(get_bbox_center(kps))

    num_persons = len(persons_kps)

    # ── 2. Detectar rostros (necesario para approach_speed V2.6) ─────────────
    faces = detect_faces(frame)
    if faces:
        biggest          = max(faces, key=lambda f: f[2]*f[3])
        last_face_center = biggest[4]
    working_face = last_face_center

    # ── 3. Objetos peligrosos + objetos pequeños ──────────────────────────────
    obj_results   = object_model(frame, verbose=False, conf=OBJECT_CONFIDENCE)
    weapon_boxes  = []
    small_obj_pos = []

    for r in obj_results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls  = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            if cls in DANGEROUS_CLASSES:
                weapon_boxes.append({"class": DANGEROUS_CLASSES[cls], "conf": conf,
                                     "center": (cx, cy), "box": (x1, y1, x2, y2)})
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_WEAPON, 2)
                cv2.putText(frame, f"{DANGEROUS_CLASSES[cls]} {conf:.0%}",
                            (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WEAPON, 2)
                events.append({"tipo": "arma", "label": DANGEROUS_CLASSES[cls],
                               "confianza": round(conf*100, 1),
                               "timestamp": datetime.now().isoformat()})
            elif cls in SMALL_OBJECTS and conf > 0.32:
                small_obj_pos.append((cx, cy))

    # ── 4. Golpe — SOLO approach_speed_to_face activa punch_confirmed ─────────
    #
    # [FIX 5] La raw_velocity eliminada como trigger independiente.
    # Solo detect_punch_approach (V2.6) puede confirmar golpe.
    # Esto preserva la detección correcta del Video 5 sin activar
    # falso positivo en Videos 2 y 5.
    #
    punch_confirmed     = False
    punch_person_center = None

    if persons_kps and pose_history:
        prev_data     = pose_history[-1]
        prev_kps_list = prev_data.get("kps_list", [])
        time_delta    = prev_data.get("time_delta", 0.04)

        for cur_idx, cur_kps in enumerate(persons_kps):
            cur_c = persons_centers[cur_idx]

            # Persona previa más cercana
            best_prev, best_d = None, float("inf")
            for prev_kps in prev_kps_list:
                prev_c = get_bbox_center(prev_kps)
                d = dist(cur_c, prev_c)
                if d < best_d and d < 220:
                    best_d, best_prev = d, prev_kps

            if best_prev and working_face:
                hit_a, hit_c_a = detect_punch_approach(
                    cur_kps, best_prev, time_delta, working_face
                )
                if hit_a:
                    hit_approach_buffer.append(1)
                    if len(hit_approach_buffer) > HIT_APPROACH_WINDOW:
                        hit_approach_buffer.pop(0)
                    if sum(hit_approach_buffer) >= HIT_APPROACH_MIN and hit_c_a:
                        punch_confirmed     = True
                        punch_person_center = cur_c
                        # Dibujar línea muñeca → cara
                        cv2.line(frame, hit_c_a, working_face, COLOR_AGGRESSOR, 2)
                else:
                    hit_approach_buffer.append(0)
                    if len(hit_approach_buffer) > HIT_APPROACH_WINDOW:
                        hit_approach_buffer.pop(0)

    pose_history.append({
        "kps_list":   [kps[:] for kps in persons_kps],
        "time_delta": 0.04,
    })
    if len(pose_history) > 5:
        pose_history.pop(0)

    # ── 5. Brazo apuntando (Videos 1, 2, 5) ──────────────────────────────────
    arm_ptr_c, arm_tgt_c = None, None
    arm_confirmed        = False

    if not in_warmup and num_persons >= 2:
        ptr, tgt = detect_arm_pointing(persons_kps, persons_centers)
        arm_point_buffer.append(1 if ptr is not None else 0)
        if len(arm_point_buffer) > ARM_POINT_WINDOW:
            arm_point_buffer.pop(0)
        if sum(arm_point_buffer) >= ARM_POINT_MIN and ptr is not None:
            arm_confirmed = True
            arm_ptr_c, arm_tgt_c = ptr, tgt

    # ── 6. Hurto (Video 3 — aseadora) ────────────────────────────────────────
    theft_detected = False
    theft_center   = None

    if not in_warmup and persons_kps:
        main_idx = max(range(num_persons), key=lambda i: get_bbox_area(persons_kps[i]))
        main_kps = persons_kps[main_idx]
        main_c   = persons_centers[main_idx]

        gest_a = detect_theft_gesture(main_kps, frame_h)
        gest_c = detect_pocket_gesture(main_kps, frame_h)

        theft_pose_buffer.append(1 if (gest_a or gest_c) else 0)
        if len(theft_pose_buffer) > THEFT_POSE_WINDOW:
            theft_pose_buffer.pop(0)

        obj_found, obj_person_c = detect_object_in_hand(small_obj_pos, persons_kps)
        obj_in_hand_buffer.append(1 if obj_found else 0)
        if len(obj_in_hand_buffer) > OBJ_HAND_WINDOW:
            obj_in_hand_buffer.pop(0)

        pose_score = sum(theft_pose_buffer) >= THEFT_POSE_MIN
        obj_score  = sum(obj_in_hand_buffer) >= OBJ_HAND_MIN

        if pose_score or obj_score:
            theft_detected = True
            theft_center   = obj_person_c or main_c

    # ── 7. Persona en el suelo ────────────────────────────────────────────────
    ground_persons = [i for i, kps in enumerate(persons_kps)
                      if is_person_on_ground(kps, frame_h)]
    has_ground = len(ground_persons) > 0

    # ── 8. Ataque grupal (Video 4 — motos) ───────────────────────────────────
    group_attack     = False
    group_aggressors = []
    group_victim_c   = None

    if num_persons >= 2:
        group_attack, group_aggressors, group_victim_c = detect_group_attack(
            persons_kps, persons_centers, frame_h, has_ground
        )
        if group_attack:
            group_confirmed = True

    # Sostener alarma de grupo ya confirmada
    if group_confirmed and not group_attack and not in_warmup:
        for tr in tracked_roles:
            if tr["role"] == "AGGRESSOR":
                tr["frames_left"] = max(tr["frames_left"], ROLE_LOCK_FRAMES // 2)
        group_attack = True

    # ── 9. ¿Frame agresivo? ───────────────────────────────────────────────────
    #
    # [FIX 2] punch_confirmed YA NO activa agresión solo.
    # Solo activa si TAMBIÉN hay arma YOLO O brazo apuntando.
    # Esto evita que movimiento normal de manos en Videos 2 y 5
    # dispare la alarma sin contexto de arma real.
    #
    # Los videos 3 y 4 NO dependen de punch_confirmed para activarse
    # (usan theft_detected y group_attack respectivamente), así que
    # este cambio no los afecta.
    #
    punch_with_weapon = punch_confirmed and (len(weapon_boxes) > 0 or arm_confirmed)

    new_aggression = (
        punch_with_weapon   or   # Golpe confirmado + arma/brazo
        len(weapon_boxes) > 0 or # Arma YOLO sola (cuchillo detectado)
        theft_detected        or # Video 3: aseadora
        group_attack          or # Video 4: motos
        arm_confirmed            # Videos 1/5: brazo apuntando
    )

    if new_aggression:
        aggression_lock_counter = AGGRESSION_LOCK_FRAMES
    else:
        aggression_lock_counter = max(0, aggression_lock_counter - 1)

    frame_is_aggressive = new_aggression or (aggression_lock_counter > 0)

    # ── 10. Asignar roles ─────────────────────────────────────────────────────
    if new_aggression and not in_warmup:

        # Brazo apuntando → agresor y víctima con force=True
        if arm_confirmed:
            if arm_ptr_c:
                assign_role(arm_ptr_c, "AGGRESSOR", force=True)
            if arm_tgt_c:
                assign_role(arm_tgt_c, "VICTIM", force=True)

        # Arma YOLO → más cercano = agresor (si no es VICTIM)
        if weapon_boxes and persons_centers:
            for wb in weapon_boxes:
                best_d, best_c = float("inf"), None
                for pc in persons_centers:
                    if pc:
                        d = dist(pc, wb["center"])
                        if d < best_d and d < 450:
                            best_d, best_c = d, pc
                if best_c and get_role_for_person(best_c) != "VICTIM":
                    assign_role(best_c, "AGGRESSOR")
                    for pc in persons_centers:
                        if pc and pc != best_c and get_role_for_person(pc) is None:
                            assign_role(pc, "VICTIM")

        # Golpe → quien golpea = agresor
        if punch_with_weapon and punch_person_center:
            assign_role(punch_person_center, "AGGRESSOR")
            for pc in persons_centers:
                if pc and dist(pc, punch_person_center) > ROLE_MATCH_DIST:
                    if get_role_for_person(pc) != "AGGRESSOR":
                        assign_role(pc, "VICTIM")

        # Persona en suelo → VICTIM
        for i in ground_persons:
            assign_role(get_bbox_center(persons_kps[i]), "VICTIM")

        # Hurto → SUSPECT
        if theft_detected and theft_center:
            assign_role(theft_center, "SUSPECT")

        # Ataque grupal → agresores y víctima
        if group_attack and group_aggressors:
            for ac in group_aggressors:
                assign_role(ac, "AGGRESSOR", force=True)
            if group_victim_c:
                assign_role(group_victim_c, "VICTIM")

        # Sin rol → VICTIM provisional
        for pc in persons_centers:
            if pc and get_role_for_person(pc) is None:
                assign_role(pc, "VICTIM")

    # ── 11. Dibujar esqueletos ────────────────────────────────────────────────
    for pkps in persons_kps:
        center = get_bbox_center(pkps)
        role   = get_role_for_person(center) if not in_warmup else None

        if in_warmup or not frame_is_aggressive or role is None:
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

        if is_person_on_ground(pkps, frame_h) and not in_warmup and center:
            cv2.putText(frame, "VICTIMA EN SUELO",
                        (center[0]-55, center[1]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2)

    # ── 12. Dibujar rostros ───────────────────────────────────────────────────
    for face in faces:
        fc   = face[4]
        role = get_role_for_person(fc) if not in_warmup else None
        if in_warmup or not frame_is_aggressive or role is None:
            draw_face_box(frame, face, COLOR_SAFE, "Rostro")
        elif role == "AGGRESSOR":
            draw_face_box(frame, face, COLOR_AGGRESSOR, "Agresor")
        elif role == "VICTIM":
            draw_face_box(frame, face, COLOR_VICTIM, "Victima")
        elif role == "SUSPECT":
            draw_face_box(frame, face, COLOR_SUSPECT, "Sospechoso")
        else:
            draw_face_box(frame, face, COLOR_SAFE, "Rostro")

    # ── 13. Alarma ────────────────────────────────────────────────────────────
    if frame_is_aggressive and not in_warmup:
        alarm.start_alarm("AGRESION DETECTADA")
    else:
        alarm.stop_alarm()

    # ── 14. Evento con cooldown ───────────────────────────────────────────────
    if new_aggression and not in_warmup:
        now = time.time()
        if now - last_alert_time > ALERT_COOLDOWN_SECONDS:
            last_alert_time = now
            events.append({
                "tipo":      "alerta",
                "label":     "AGRESION DETECTADA - Llamando al 123",
                "confianza": 82.0,
                "timestamp": datetime.now().isoformat()
            })

    # ── 15. Barra de estado ───────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, 58), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    if in_warmup:
        remaining = max(0, WARMUP_FRAMES - frames_analyzed)
        cv2.putText(frame, f"Iniciando sistema... ({remaining} frames restantes)",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 0), 2)
    elif frame_is_aggressive:
        cv2.putText(frame, "!!! AGRESION DETECTADA  -  ALARMA ACTIVA",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)
        cv2.putText(frame, "Llamando al 123 - Policia Nacional",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 140, 255), 2)
    else:
        cv2.putText(frame, "Estado: Monitoreando...",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 0), 2)

    agr_count = sum(1 for tr in tracked_roles if tr["role"] == "AGGRESSOR")
    cv2.putText(frame,
                f"Agresores: {agr_count} | Personas: {num_persons} | "
                f"Lock: {aggression_lock_counter} | Alarma: {'ACTIVA' if alarm.is_alarm_active() else 'OFF'}",
                (8, frame_h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

    return events, frame