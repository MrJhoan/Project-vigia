# alarm.py - VERSION 2.4 con playsound
import threading
import time
import os

ALARM_FILE = "IPhone Alarm Ringtone.mp3"

_alarm_active  = False
_alarm_thread  = None
_current_crime = "EVENTO DELICTIVO"
_playsound_ok  = False

try:
    from playsound import playsound
    if os.path.exists(ALARM_FILE):
        _playsound_ok = True
        print(f"✅ Alarma lista: {ALARM_FILE}")
    else:
        print(f"⚠️  '{ALARM_FILE}' no encontrado en la carpeta backend/")
except Exception as e:
    print(f"⚠️  playsound no disponible ({e}). Usando beeps.")


def _alarm_loop_mp3():
    global _alarm_active
    # Obtenemos la ruta absoluta del archivo
    abs_path = os.path.abspath(ALARM_FILE)
    while _alarm_active:
        try:
            playsound(abs_path, block=True)
        except Exception:
            time.sleep(0.5)


def _alarm_loop_beeps():
    import winsound
    global _alarm_active
    while _alarm_active:
        for _ in range(3):
            if not _alarm_active:
                return
            winsound.Beep(2000, 120)
            time.sleep(0.04)
        time.sleep(0.15)
        for freq in range(1800, 900, -100):
            if not _alarm_active:
                return
            winsound.Beep(freq, 35)
        time.sleep(0.1)


def start_alarm(crime_type: str = "EVENTO DELICTIVO"):
    global _alarm_active, _alarm_thread, _current_crime

    if _alarm_active and _current_crime == crime_type:
        return
    if _alarm_active:
        _current_crime = crime_type
        print(f"🚨 TIPO ACTUALIZADO: {crime_type}")
        return

    _alarm_active  = True
    _current_crime = crime_type

    print(f"\n{'='*55}")
    print(f"  🚨  ALARMA ACTIVADA")
    print(f"  📋  TIPO : {crime_type}")
    print(f"  ⏰  HORA : {time.strftime('%H:%M:%S')}")
    print(f"{'='*55}\n")

    target = _alarm_loop_mp3 if _playsound_ok else _alarm_loop_beeps
    _alarm_thread = threading.Thread(target=target, daemon=True)
    _alarm_thread.start()


def stop_alarm():
    global _alarm_active
    if not _alarm_active:
        return
    _alarm_active = False
    print(f"🔇 Alarma detenida | Evento: {_current_crime}\n")


def is_alarm_active():
    return _alarm_active


def get_current_crime():
    return _current_crime