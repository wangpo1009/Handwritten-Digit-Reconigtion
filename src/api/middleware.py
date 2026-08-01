"""
middleware.py

Đăng ký middleware.
Middleware là phần mềm trung gian chịu trách nhiệm xử lý, check auth, ghi lại log
Đây là nơi để khơi tạo rate limit, logging, authentication, và các middleware khác.
"""
from __future__ import annotations

import time
import uuid
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from collections import defaultdict
from fastapi.middleware.cors import CORSMiddleware


# ================ Logger Middleware ===============

logger = logging.getLogger("mnist")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# ================ Middleware Functions ===============
def register_middleware(app: FastAPI):
    register_request_middleware(app)
    register_timeout_middleware(app)
    register_exception_handler(app)
    register_security_middleware(app)
    register_rate_limit_middleware(app) 
    register_cors(app)

## ================ Request Middleware ===============
def register_request_middleware(app: FastAPI):
    """
    Middleware để log request
    Chức năng:
    - Sinh request_id
    - Đo thời gian xử lý request
    - Logging
    """

    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4()) # sinh chuỗi ngẫu nhiên
        start_time = time.time() # Ghi thời gian bắt đầu xử lý request
        logger.info(f"[{request_id}]" f"{request.method}" f"{request.url.path}") # Ghi lại dòng log với request_id, method, url path

        response = await call_next(request) # Đợi model xử lỹ request và trả response
        end_time = time.time() # Ghi thời gian kết thúc xử lý request

        process_time = end_time - start_time # Tính thời gian xử lý request
        logger.info(f"[{request_id}]" f"Completed" f"{response.status_code}" f"in {process_time:.4f}s")

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"

        return response

## ================ Exception Handler ===============
def register_exception_handler(app: FastAPI):
    """ 
    Middleware để xử lý lỗi
    """

    @app.exception_handler(Exception)
    async def exception_handler(request: Request, exc: Exception):
        logger.exception(exc)
        return JSONResponse(
            status_code = 500,
            content={"success": False,
                     "detail": "Internal Server Error"}
        )

# ================= Security Middleware =================
def register_security_middleware(app: FastAPI):
    """
    Middleware để kiểm tra auth
    """
    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        response = await call_next(request)

        # Chống Clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        # Chống MIME-sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Bảo vệ thông tin Referer
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Tắt quyền truy cập thiết bị
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        # Ép dùng HTTPS (Bật khi sản phẩm chạy HTTPS thực tế)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        # CSP dành riêng cho API trả về JSON
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none';"

        return response

# ================= Rate Limiting Middleware =================
RATE_LIMIT = 50
WINDOW_SIZE = 60

client = defaultdict(list)

def register_rate_limit_middleware(app: FastAPI):
    """
    Middleware để giới hạn số lượng request từ một client trong khoảng thời gian.
    Cơ chế:
    - Mỗi client được lưu bởi một địa chỉ IP, mỗi IP sẽ lưu thời điểm gửi request
    - Khi có request mới:
        - Xóa các request cũ hơn trong WINDOW_SIZE
        - Kiểm tra số lượng request còn lại
        - Nếu vượt RATE_LIMIT, trả lỗi
        - Nếu chưa vượt, thêm thời điểm request mới vào danh sách và tiếp tục xử lý
    """

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        # Lấy địa chỉ IP của client
        ip = request.client.host

        # Lấy thời gian hiện tại
        current_time = time.monotonic()

        # Giữ lại các request trong WINDOW_SIZE
        client[ip] = [t for t in client[ip] if current_time - t < WINDOW_SIZE]

        # Kiểm tra số lượng request
        if len(client[ip]) >= RATE_LIMIT:
            return JSONResponse(
                status_code=429,
                content={"success": False, "detail": "Too Many Requests"},
                headers={"Retry-After": str(WINDOW_SIZE)}
            )

       # Ghi nhận request mới
        client[ip].append(current_time)
        # Chuyển request tới endpoint tiếp theo
        response = await call_next(request)

        return response
    
# ================= Timeout Middleware ==================
import asyncio
TIMEOUT = 20

def register_timeout_middleware(app: FastAPI):
    @app.middleware("http")
    async def timeout(request: Request, call_next):
        try:
            # Chờ endpoint xử lý request
            return await asyncio.wait_for(call_next(request), timeout=TIMEOUT)
        except asyncio.TimeoutError: # Khi vượt quá TIMEOUT: logging, trả lỗi
            logger.warning("Request timed out: %s %s", request.method, request.url.path)
            return JSONResponse(
                status_code=504,
                content={"success": False, "detail": "Request Timeout"}
            )

# ================= CORS Middleware ==================
def register_cors(app: FastAPI):
    """ 
    Middleware để xử lý CORS
    CORS (Cross-Origin Resource Sharing)
    cho phép frontend ở domain khác
    được phép gọi API.

    Ví dụ

    Frontend:
        http://localhost:3000

    Backend:
        http://localhost:8000

    Nếu không bật CORS,
    trình duyệt sẽ chặn request.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"], # Chỉ frontend của dự án được phép gọi API
        allow_credentials=False,
        allow_methods=["GET",
                       "POST",
                       "PUT",
                       "DELETE",
                       "OPTIONS"],  # Cho phép các phương thức HTTP trên
        allow_headers=["*"],  # Cho phép tất cả các header
        # Cho phép frontend đọc các header tự tạo
        expose_headers=[
            "X-Request-ID",
            "X-Process-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining"
        ],

        # Cache kết quả preflight trong 10 phút
        max_age=600
    )