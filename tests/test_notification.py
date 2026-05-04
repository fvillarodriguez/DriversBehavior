import unittest
from unittest.mock import patch, MagicMock
from datetime import date
import os
import sys

# Add repository root to path so `from src ...` is importable.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import notification_system

class TestNotificationSystem(unittest.TestCase):
    
    def setUp(self):
        # Backup config if exists
        self.config_path = notification_system.CONFIG_FILE
        self.original_config = None
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.original_config = f.read()
        
    def tearDown(self):
        # Restore config
        if self.original_config is not None:
            with open(self.config_path, 'w') as f:
                f.write(self.original_config)
        elif self.config_path.exists():
            os.remove(self.config_path)

    def test_save_load_config(self):
        config = {
            "smtp_server": "smtp.test.com",
            "smtp_port": 587,
            "sender_email": "test@test.com",
            "sender_password": "pass",
            "recipients": ["r1@test.com"]
        }
        notification_system.save_email_config(config)
        loaded = notification_system.load_email_config()
        self.assertEqual(loaded, config)

    def test_format_history_html(self):
        entry = {
            "timestamp": "2023-01-01 10:00",
            "type": "Test",
            "details": {"a": 1, "b": 2}
        }
        html = notification_system.format_history_html(entry)
        self.assertIn("<h2>Resumen de Ejecución</h2>", html)
        self.assertIn("Test", html)
        self.assertIn("DETAILS", html)

    @patch("smtplib.SMTP")
    def test_send_notification_email(self, mock_smtp):
        # Setup config
        config = {
            "smtp_server": "smtp.test.com",
            "smtp_port": 587,
            "sender_email": "sender@test.com",
            "sender_password": "password123",
            "recipients": ["recipient@test.com"]
        }
        notification_system.save_email_config(config)
        
        # Setup Mock
        mock_server = MagicMock()
        mock_smtp.return_value = mock_server
        
        # Run
        res = notification_system.send_notification_email("Subject Test", {"status": "ok"})
        
        # Assert
        self.assertTrue(res)
        mock_smtp.assert_called_with("smtp.test.com", 587)
        mock_server.starttls.assert_called()
        mock_server.login.assert_called_with("sender@test.com", "password123")
        mock_server.sendmail.assert_called()
        
        # Verify content of email
        call_args = mock_server.sendmail.call_args
        self.assertEqual(call_args[0][0], "sender@test.com")
        self.assertEqual(call_args[0][1], ["recipient@test.com"])
        msg_str = call_args[0][2]
        self.assertIn("Subject: Script finalizado: Subject Test", msg_str)

    @patch("smtplib.SMTP")
    def test_send_notification_email_accepts_date_values(self, mock_smtp):
        config = {
            "smtp_server": "smtp.test.com",
            "smtp_port": 587,
            "sender_email": "sender@test.com",
            "sender_password": "password123",
            "recipients": ["recipient@test.com"],
        }
        notification_system.save_email_config(config)
        mock_smtp.return_value = MagicMock()

        res = notification_system.send_notification_email(
            "Subject Test",
            {"status": "error", "context": {"start_date": date(2026, 5, 4)}},
        )

        self.assertTrue(res)
        mock_smtp.return_value.sendmail.assert_called()

    @patch("smtplib.SMTP")
    def test_send_notification_email_missing_required_config_returns_false(self, mock_smtp):
        config = {
            "smtp_server": "smtp.test.com",
            "smtp_port": 587,
            "sender_email": "sender@test.com",
            "sender_password": "password123",
            "recipients": [],
        }
        notification_system.save_email_config(config)

        res = notification_system.send_notification_email("Subject Test", {"status": "ok"})

        self.assertFalse(res)
        mock_smtp.assert_not_called()

    @patch("smtplib.SMTP", side_effect=Exception("smtp down"))
    def test_send_notification_email_smtp_exception_returns_false(self, mock_smtp):
        config = {
            "smtp_server": "smtp.test.com",
            "smtp_port": 587,
            "sender_email": "sender@test.com",
            "sender_password": "password123",
            "recipients": ["recipient@test.com"],
        }
        notification_system.save_email_config(config)

        res = notification_system.send_notification_email("Subject Test", {"status": "ok"})

        self.assertFalse(res)
        mock_smtp.assert_called_once_with("smtp.test.com", 587)

if __name__ == '__main__':
    unittest.main()
