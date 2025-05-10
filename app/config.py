from pydantic_settings import BaseSettings
from pydantic import SecretStr


class Settings(BaseSettings):
    # Обязательные переменные (без default)
    MOEX_SERVICE_TOKEN: str
    API_TOKEN: str
    # Необязательные (с default)
    debug: bool = False

    class Config:
        env_file = "app/.env"  # Указываем файл .env
        env_file_encoding = "utf-8"  # Кодировка файла

# Создаём экземпляр настроек
settings = Settings()