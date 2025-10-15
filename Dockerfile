FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código do serviço
COPY train_service.py .

# Criar diretórios de trabalho
RUN mkdir -p /app/workdir/datasets /app/workdir/models /app/workdir/runs

# Expor porta
EXPOSE 5000

# Variável de ambiente para produção
ENV FLASK_ENV=production

# Comando para iniciar o serviço
CMD ["python", "train_service.py"]
