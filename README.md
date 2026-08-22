# AI Image Enhancement & Super-Resolution Platform

Production-style AI-powered image enhancement and image super-resolution platform.

## Overview

The application accepts degraded or low-resolution images and uses a CNN-based deep learning model to generate enhanced high-resolution images.

## Technology Stack

### Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS

### Backend

- Python
- FastAPI
- SQLAlchemy
- Pydantic

### Machine Learning

- PyTorch
- OpenCV
- Pillow
- NumPy

### Infrastructure

- PostgreSQL
- Redis
- Celery
- Docker

### Payments

- Razorpay Test Mode

## Architecture

```text
Next.js
    |
    v
FastAPI
    |
    +---- PostgreSQL
    |
    +---- Redis
    |
    +---- Background Worker
    |
    +---- ML Inference
    |
    +---- Object Storage