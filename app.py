import json
import sys
import os
import time
import numpy as np
import logging
import uvicorn
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from models.model_loader import load_models

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

models = load_models()

class ForecastRequest(BaseModel):
    data: List[List[float]]  # Shape will be (24, 24)

class ForecastResponse(BaseModel):
    success: bool
    forecast: List[float] = None
    error_message: str = None
    metadata: dict = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Time Series Forecasting API is fully started!")
    sys.stdout.flush()
    yield

app = FastAPI(lifespan=lifespan)

def validate_input_shape(data: np.ndarray):
    """
    Validate if the input data has the correct shape (24, 24).
    
    Args:
        data (np.ndarray): Input data array
    """
    if data.shape != (24, 24):
        raise ValueError(f"Input data must have shape (24, 24), but got {data.shape}")

def add_batch_dimension(data: np.ndarray) -> np.ndarray:
    """
    Add batch dimension to input data.
    
    Args:
        data (np.ndarray): Input data of shape (24, 24)
    
    Returns:
        np.ndarray: Data with batch dimension of shape (1, 24, 24)
    """
    return data[np.newaxis, ...]


@app.post("/predict/nbeats-cnn")
async def predict_nbeats_cnn(request: ForecastRequest):
    try:
        start_time = time.time()
        
        data = np.array(request.data)
        validate_input_shape(data)
        
        data_with_batch = add_batch_dimension(data)
        
        data_with_batch = tf.convert_to_tensor(data_with_batch)

        forecast = models[0].predict(data_with_batch)

        forecast = np.squeeze(forecast, axis=0)
        
        processing_time = time.time() - start_time
        
        logger.info(f"N-BEATS-CNN forecast completed in {processing_time:.4f} seconds.")
        response = ForecastResponse(
            success=True, 
            forecast=forecast.tolist(),
            metadata={"processing_time": processing_time, "model": "nbeats-cnn"}
        )
        return JSONResponse(status_code=200, content=response.model_dump())
        
    except Exception as e:
        logger.error(f"An error occurred during N-BEATS-CNN forecasting: {e}")
        response = ForecastResponse(
            success=False,
            error_message="An error occurred during N-BEATS-CNN forecasting.",
            metadata={"processing_time": None, "error": str(e), "model": "nbeats-cnn"}
        )
        return JSONResponse(status_code=500, content=response.model_dump())


@app.post("/predict/cnn-nbeats")
async def predict_cnn_nbeats(request: ForecastRequest):
    try:
        start_time = time.time()
        
        data = np.array(request.data)
        validate_input_shape(data)
        
        data_with_batch = add_batch_dimension(data)
        
        data_with_batch = tf.convert_to_tensor(data_with_batch)
        
        forecast = models[1].predict(data_with_batch)
        
        forecast = np.squeeze(forecast, axis=0)
        
        processing_time = time.time() - start_time
        
        logger.info(f"CNN-N-BEATS forecast completed in {processing_time:.4f} seconds.")
        response = ForecastResponse(
            success=True, 
            forecast=forecast.tolist(),
            metadata={"processing_time": processing_time, "model": "cnn-nbeats"}
        )
        return JSONResponse(status_code=200, content=response.model_dump())
        
    except Exception as e:
        logger.error(f"An error occurred during CNN-N-BEATS forecasting: {e}")
        response = ForecastResponse(
            success=False,
            error_message="An error occurred during CNN-N-BEATS forecasting.",
            metadata={"processing_time": None, "error": str(e), "model": "cnn-nbeats"}
        )
        return JSONResponse(status_code=500, content=response.model_dump())
   

@app.post("/predict/nbeats")
async def predict_nbeats(request: ForecastRequest):
    try:
        start_time = time.time()
        
        data = np.array(request.data)
        validate_input_shape(data)
        
        data_with_batch = add_batch_dimension(data)
        
        data_with_batch = tf.convert_to_tensor(data_with_batch)
        
        forecast = models[2].predict(data_with_batch)
        
        forecast = np.squeeze(forecast, axis=0)
        
        processing_time = time.time() - start_time
        
        logger.info(f"N-BEATS forecast completed in {processing_time:.4f} seconds.")
        response = ForecastResponse(
            success=True, 
            forecast=forecast.tolist(),
            metadata={"processing_time": processing_time, "model": "nbeats"}
        )
        return JSONResponse(status_code=200, content=response.model_dump())
        
    except Exception as e:
        logger.error(f"An error occurred during N-BEATS forecasting: {e}")
        response = ForecastResponse(
            success=False,
            error_message="An error occurred during N-BEATS forecasting.",
            metadata={"processing_time": None, "error": str(e), "model": "nbeats"}
        )
        return JSONResponse(status_code=500, content=response.model_dump())
      
@app.get("/healthz", tags=["Health"])
async def health_check():
    return JSONResponse(content={"status": "ok"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

