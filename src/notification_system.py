import json
import os
import smtplib
import traceback
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional
import streamlit as st

# Configuration file path
CONFIG_FILE = Path(__file__).resolve().parent.parent / "email_config.json"

def load_email_config() -> Dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_email_config(config: Dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

def _json_default(value: object) -> object:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)

def format_history_html(history_entry: Dict) -> str:
    """Formats a GNN history entry into an HTML table."""
    html = """
    <html>
    <head>
    <style>
        table { border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        tr:hover { background-color: #f5f5f5; }
        h2 { color: #333; }
    </style>
    </head>
    <body>
    <h2>Resumen de Ejecución</h2>
    <table>
    """
    
    # Flatten dictionary for simple display or iterate sections
    for key, value in history_entry.items():
        if isinstance(value, dict):
            html += f"<tr><th colspan='2' style='background-color: #e0e0e0;'>{key.upper()}</th></tr>"
            for sub_key, sub_val in value.items():
                 html += f"<tr><td><b>{sub_key}</b></td><td>{sub_val}</td></tr>"
        else:
            html += f"<tr><td><b>{key}</b></td><td>{value}</td></tr>"

    html += """
    </table>
    </body>
    </html>
    """
    return html

def send_notification_email(subject: str, history_entry: Dict) -> bool:
    config = load_email_config()
    smtp_server = config.get("smtp_server", "smtp.gmail.com")
    smtp_port = int(config.get("smtp_port", 587))
    sender_email = config.get("sender_email")
    password = config.get("sender_password")
    recipients = config.get("recipients", [])

    if not sender_email or not password or not recipients:
        print("Falta configuración de email.")
        return False

    # Create message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Script finalizado: {subject}"
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipients)

    text = f"Detalles de la ejecución:\n{json.dumps(history_entry, indent=2, default=_json_default)}"
    html = format_history_html(history_entry)

    part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")

    msg.attach(part1)
    msg.attach(part2)

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, recipients, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Error enviando email: {e}")
        return False


def send_error_notification_email(
    subject: str,
    error: object,
    *,
    context: Optional[Dict] = None,
) -> bool:
    error_text = ""
    if isinstance(error, BaseException):
        error_text = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
    else:
        error_text = str(error)

    payload = {
        "status": "Error",
        "error": error_text,
        "context": context or {},
    }
    return send_notification_email(f"ERROR: {subject}", payload)

def render_notification_config():
    st.subheader("Configuración de Notificaciones por Email")
    
    config = load_email_config()
    
    with st.form("email_config_form"):
        server = st.text_input("Servidor SMTP", value=config.get("smtp_server", "smtp.gmail.com"))
        port = st.number_input("Puerto SMTP", value=int(config.get("smtp_port", 587)))
        email = st.text_input("Email Remitente", value=config.get("sender_email", ""))
        password = st.text_input("Contraseña de Aplicación", value=config.get("sender_password", ""), type="password")
        
        recipients_str = st.text_area("Destinatarios (separados por coma)", value=", ".join(config.get("recipients", [])))
        
        if st.form_submit_button("Guardar Configuración"):
            recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]
            new_config = {
                "smtp_server": server,
                "smtp_port": port,
                "sender_email": email,
                "sender_password": password,
                "recipients": recipients
            }
            save_email_config(new_config)
            st.success("Configuración guardada exitosamente.")

    if st.button("Email de Prueba"):
        test_entry = {
            "timestamp": "Prueba Manual",
            "type": "Test",
            "test_message": "Este es un email de prueba para verificar la configuración.",
            "status": "Success"
        }
        if send_notification_email("Prueba de Configuración", test_entry):
            st.success("Email de prueba enviado correctamente.")
        else:
            st.error("Error al enviar email de prueba. Revise la consola para más detalles.")
