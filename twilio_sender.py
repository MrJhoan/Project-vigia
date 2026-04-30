# twilio_sender.py
# Módulo para notificar a la Policía cuando se confirma un evento.
# Por ahora solo imprime en consola para pruebas locales.
# Cuando tengas cuenta en twilio.com, descomentas el bloque real.

from datetime import datetime

# --- Credenciales Twilio (completar cuando tengas cuenta) ---
# from twilio.rest import Client
# ACCOUNT_SID   = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# AUTH_TOKEN    = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# TWILIO_NUMBER = "+15550001234"
# POLICE_NUMBER = "+573001234567"
# client = Client(ACCOUNT_SID, AUTH_TOKEN)

async def notify_police(event: dict):
    """
    Envía notificación a la Policía con los detalles del evento.
    En modo de prueba solo imprime en consola.
    """
    label     = event.get("label", "Evento desconocido")
    tipo      = event.get("tipo", "")
    confianza = event.get("confianza", 0)
    timestamp = event.get("timestamp", datetime.now().isoformat())

    mensaje = (
        f"\n🚨 ALERTA PROJECT_VIGIA\n"
        f"Evento    : {label}\n"
        f"Tipo      : {tipo.upper()}\n"
        f"Confianza : {confianza}%\n"
        f"Hora      : {timestamp}\n"
        f"Ubicación : Cámara Piloto UIS - Bucaramanga\n"
    )

    # Modo prueba: imprime en consola del servidor
    print(mensaje)

    # --- Modo producción (descomentar cuando tengas Twilio configurado) ---
    # try:
    #     client.messages.create(
    #         body=mensaje,
    #         from_=TWILIO_NUMBER,
    #         to=POLICE_NUMBER
    #     )
    #     print(f"SMS enviado exitosamente: {label}")
    # except Exception as e:
    #     print(f"Error al enviar SMS: {e}")