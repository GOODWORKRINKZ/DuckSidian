"""Тестовые фикстуры — даём заглушки для обязательных env-переменных."""
import os

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test:token")
os.environ.setdefault("TELEGRAM_CHAT_ID", "-100000000000")
os.environ.setdefault("DEEPSEEK_API_KEY", "test")
