"""
Chức năng của file schemas là : 
Định nghĩa các schema Response và Request của API

"""

from pydantic import BaseModel, Field, ConfigDict
from fastapi import UploadFile, File

# ============== Response Schema ====================
class PredictResponse(BaseModel):
    """
    Schema cho response của model(Chương trình ta trả về kết quả ntn):
    - prediction: Kết quả dự đoán của model
    - confidence: Độ tin cậy của dự đoán
    """
    prediction: int = Field(..., ge= 0, le=9, description="Dự đoán của mô hình.")
    confidence: float = Field(..., ge = 0.0, le=1.0, description="Độ tin cậy của dự đoán.")
    time_taken: float = Field(..., description="Thời gian dự đoán của mô hình (giây).")
    class Config:
        schema_extra = {
            "example": {
                "prediction": 7,
                "confidence": 0.95,
                "time_taken": 0.01
            }
        }

class ErrorResponse(BaseModel):
    """
    Schema cho response khi có lỗi xảy ra
    """
    detail: str = Field(...,
                        description="Thông báo lỗi.")
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Error!."
            }
        }


# ============== Image Metadata ====================
class ImageMetadata(BaseModel):
    """
    Schema cho metadata của ảnh được gửi lên API, dùng để logging. 
    """
    filename: str
    content_type: str
    file_size: float
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "image.png",
                "content_type": "image/png",
                "file_size": 12345.67
            }
        }
    )

# ============== Health Check Response Schema =============
class HealthCheckResponse(BaseModel):
    """
    Schema cho endpoint /health
    """
    status: str
    model_loaded: bool
    version: str
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }
    )