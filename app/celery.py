from celery import Celery
from .config import settings

celery_app = Celery(
    'fastapi_worker',
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=['app.tasks']
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    broker_connection_retry_on_startup=True,
)

celery_app.conf.task_routes = {
    'app.tasks.*': {'queue': 'fastapi_queue'}
}