
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Path
from pydantic import BaseModel

from app.common_utils import predict_cost
from app.config import settings
import logging

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock Price Prediction API",
    description="API для предсказания цен акций с использованием ML",
    version="0.1.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TickerCandles(BaseModel):
    last_date_end: str
    data: list[float]


@app.post("/predict/{ticker}")
async def predict_for_action(request: Request, data: TickerCandles, ticker: str = Path(description="Тикер акции (например: AAPL)")) -> list[float]:
    """Предсказать цены закрытия свечей на следующие 20 часов."""
    token = request.headers.get("Authorization")

    if token != f'Bearer {settings.API_TOKEN}':
        raise HTTPException(status_code=403, detail="Forbidden")

    prediction = predict_cost(data.data, ticker)

    if not prediction:
        raise HTTPException(status_code=400, detail='Ошибки на стороне сервиса, попробуйте позже')

    return prediction
