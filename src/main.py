from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from .app.core.config import get_settings
from .app.api.endpoints import router as api_router
from .app.core.logging import get_logger
from .app.core.limiter import limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import secrets

settings = get_settings()
logger = get_logger("main")

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description="AI-powered Resume Analysis Microservice",
        docs_url=None if not settings.DEBUG else "/docs",
        redoc_url=None if not settings.DEBUG else "/redoc",
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Global error: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error. Please contact support."}
        )

    # 1. Security Headers
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        # Relaxed CSP to allow CDN resources (TailwindCSS, AlpineJS, Google Fonts)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://cdn.jsdelivr.net https://unpkg.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; "
            "font-src 'self' https://fonts.gstatic.com https://fonts.googleapis.com data:; "
            "img-src 'self' data: https:; "
            "connect-src 'self' https:"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    # 2. Trusted Hosts
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS
    )

    # 3. CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"], # Restrict methods
        allow_headers=["*"], # Can still be broad or specific
    )

    # 4. Metrics
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app)

    @app.middleware("http")
    async def validate_api_key(request: Request, call_next):
        # DEVELOPMENT MODE: Skip all authentication
        if settings.DEVELOPMENT_MODE:
            return await call_next(request)
        
        # Allow health checks and docs (if enabled) without auth
        if request.url.path.startswith("/api/health") or request.url.path in ["/", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Static files
        if request.url.path.startswith("/static"):
             return await call_next(request)
        
        # Development mode endpoints - protected by DEVELOPMENT_MODE flag instead
        if request.url.path.startswith("/api/dev"):
            return await call_next(request)

        api_key_header = request.headers.get("x-api-key")
        if not api_key_header or not secrets.compare_digest(api_key_header, settings.API_KEY):
            return JSONResponse(
                status_code=403,
                content={"detail": "Could not validate credentials"}
            )
        
        return await call_next(request)

    # Routes
    app.include_router(api_router, prefix="/api")

    # Static Files (Frontend)
    # Mount at root but ensure API routes don't conflict. 
    # API Routes are under /api, so mounting / at the end is safe.
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

    logger.info("Application initialized")
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
