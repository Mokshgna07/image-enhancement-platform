from fastapi import APIRouter

from app.api.v1 import auth, health, images, enhancements


api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(images.router)
api_router.include_router(enhancements.router)
