FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema necessárias para OpenCV e outras libs
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código do serviço
COPY training_service.py .

# Criar diretórios de trabalho
RUN mkdir -p /app/workdir/datasets /app/workdir/models /app/workdir/runs

# Expor porta 5000
EXPOSE 5000

# Variáveis de ambiente para produção
ENV FLASK_ENV=production
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Health check para monitoramento
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Comando para iniciar o serviço
# Usar Gunicorn para produção (mais robusto que Flask dev server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "3600", "--access-logfile", "-", "--error-logfile", "-", "training_service:app"]
