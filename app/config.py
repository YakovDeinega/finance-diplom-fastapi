from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Обязательные переменные (без default)
    MOEX_SERVICE_TOKEN: str
    API_TOKEN: str
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"
    # Необязательные (с default)
    debug: bool = False

    class Config:
        env_file = ".env"  # Указываем файл .env
        env_file_encoding = "utf-8"  # Кодировка файла

# Создаём экземпляр настроек
settings = Settings()