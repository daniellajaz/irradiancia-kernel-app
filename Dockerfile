FROM python:3.12-slim
WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY fkernel_lab ./fkernel_lab
COPY app ./app
COPY data ./data
COPY models ./models
ENV PORT=8000
EXPOSE 8000
# 1 worker: los modelos y la malla se cachean en memoria del proceso
CMD gunicorn -w 1 --threads 4 --timeout 180 -b 0.0.0.0:${PORT} app.app:app
