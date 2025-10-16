# Serviço de Treinamento YOLO Real

Este é o serviço Python/Flask que executa treinamento real de modelos YOLO usando Ultralytics.

## Requisitos

- Python 3.11+
- GPU (NVIDIA) recomendada para treinamento rápido (opcional, funciona com CPU)
- 4GB+ RAM (8GB+ recomendado)
- 10GB+ espaço em disco

## Instalação Local

```bash
# Clone o repositório
cd python-training-service

# Instale as dependências
pip install -r requirements.txt

# Execute o serviço
python training_service.py
```

O serviço estará disponível em `http://localhost:5000`

### Verificar instalação:
```bash
curl http://localhost:5000/health
```

Resposta esperada:
```json
{
  "status": "healthy",
  "device": "cpu",
  "ultralytics_version": "8.3.0"
}
```

## Teste Local com Ngrok

Para testar localmente antes de fazer deploy em produção:

1. Instale o ngrok: https://ngrok.com/download
2. Execute o serviço: `python training_service.py`
3. Em outro terminal, exponha o serviço: `ngrok http 5000`
4. Copie a URL pública do ngrok (ex: `https://abc123.ngrok-free.app`)
5. Configure no Supabase como `TRAINING_WEBHOOK_URL`

**Exemplo de teste:**
```bash
curl -X POST https://abc123.ngrok-free.app/train \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "test-123",
    "callback_url": "https://seu-callback.com/webhook",
    "callback_token": "seu-token",
    "dataset_url": "https://exemplo.com/dataset.zip",
    "base_model": "yolov8n",
    "config": {
      "epochs": 10,
      "batch_size": 16,
      "img_size": 640
    }
  }'
```

## Deploy em Produção

### Opção 1: Railway (Fácil, sem GPU) ⭐ RECOMENDADO PARA TESTES

**Vantagens:** Deploy automático, gratuito (com limites), fácil configuração

1. Crie conta em https://railway.app
2. Clique em "New Project" > "Deploy from GitHub repo"
3. Conecte este repositório
4. Railway detectará automaticamente o Dockerfile
5. **Configure variáveis de ambiente (opcional):**
   - `PORT`: 5000 (já está no Dockerfile)
6. Após deploy, copie a URL do serviço (ex: `https://seu-app.up.railway.app`)
7. Configure no Supabase: 
   ```
   TRAINING_WEBHOOK_URL=https://seu-app.up.railway.app/train
   ```

**Teste o serviço:**
```bash
curl https://seu-app.up.railway.app/health
```

### Opção 2: Render (Fácil, sem GPU)

**Vantagens:** Free tier generoso, SSL automático

1. Crie conta em https://render.com
2. Clique em "New +" > "Web Service"
3. Conecte este repositório
4. **Configurações:**
   - Name: `yolo-training-service`
   - Environment: Docker
   - Region: escolha a mais próxima
   - Instance Type: Standard (ou maior para melhor performance)
   - **Health Check Path:** `/health`
5. Após deploy, copie a URL (ex: `https://seu-app.onrender.com`)
6. Configure no Supabase: 
   ```
   TRAINING_WEBHOOK_URL=https://seu-app.onrender.com/train
   ```

⚠️ **Nota:** No free tier, o serviço "dorme" após inatividade. Primeiro request pode demorar 30-60s.

### Opção 3: Fly.io (Global, sem GPU)

**Vantagens:** Deploy global, escala automática, free tier

1. Instale o CLI: `curl -L https://fly.io/install.sh | sh`
2. Autentique: `fly auth login`
3. Na pasta do projeto, crie `fly.toml`:

```toml
app = "yolo-training-service"

[build]
  dockerfile = "Dockerfile"

[env]
  PORT = "5000"

[[services]]
  internal_port = 5000
  protocol = "tcp"

  [[services.ports]]
    handlers = ["http"]
    port = 80

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443

  [services.concurrency]
    hard_limit = 25
    soft_limit = 20

  [[services.tcp_checks]]
    interval = "15s"
    timeout = "2s"
    grace_period = "10s"
    restart_limit = 0
```

4. Deploy: `fly deploy`
5. Copie a URL: `https://yolo-training-service.fly.dev`
6. Configure no Supabase

### Opção 4: Modal (Serverless GPU) ⭐ RECOMENDADO PARA PRODUÇÃO

**Vantagens:** GPU sob demanda, paga apenas quando usa, auto-scale

1. Crie conta em https://modal.com
2. Instale o cliente: `pip install modal`
3. Configure: `modal token new`
4. Crie arquivo `modal_train.py`:

```python
import modal

stub = modal.Stub("yolo-training")

image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "ultralytics==8.3.0",
        "flask==3.0.0",
        "torch==2.2.0",
        "torchvision==0.17.0",
        "requests==2.31.0",
        "Pillow==10.2.0",
        "opencv-python-headless==4.9.0.80"
    )
)

@stub.function(
    image=image,
    gpu="T4",  # GPU Tesla T4
    timeout=3600,  # 1 hora
    memory=8192,  # 8GB RAM
)
@modal.asgi_app()
def flask_app():
    from training_service import app
    return app
```

5. Deploy: `modal deploy modal_train.py`
6. Copie a URL fornecida (ex: `https://username--yolo-training-web.modal.run`)
7. Configure no Supabase:
   ```
   TRAINING_WEBHOOK_URL=https://username--yolo-training-web.modal.run/train
   ```

### Opção 5: RunPod (GPU Dedicado)

**Vantagens:** GPU potente, melhor custo-benefício para uso intensivo

1. Crie conta em https://runpod.io
2. Vá em "Deploy" > "Serverless"
3. Escolha template Docker ou use nossa imagem
4. Configure:
   - GPU: A4000, RTX 4090, ou similar
   - Container Image: `seu-dockerhub-usuario/yolo-training:latest`
   - Port: 5000
5. Copie o endpoint URL
6. Configure no Supabase

**Build e push da imagem:**
```bash
docker build -t seu-usuario/yolo-training:latest .
docker push seu-usuario/yolo-training:latest
```

### Opção 6: Azure Container Instances (Cloud Microsoft)

**Vantagens:** Integração com Azure, simples, sem Kubernetes

```bash
# Criar resource group
az group create --name yolo-training-rg --location eastus

# Deploy container
az container create \
  --resource-group yolo-training-rg \
  --name yolo-training \
  --image seu-usuario/yolo-training:latest \
  --dns-name-label yolo-training-unique \
  --ports 5000 \
  --cpu 2 \
  --memory 4

# Obter URL
az container show \
  --resource-group yolo-training-rg \
  --name yolo-training \
  --query ipAddress.fqdn
```

### Opção 7: AWS/GCP (Avançado)

Deploy em VM com GPU:
- **AWS EC2:** instância g4dn.xlarge (GPU NVIDIA T4)
- **GCP Compute Engine:** n1-standard-4 + NVIDIA Tesla T4
- **Azure VM:** NC6 ou NV6 (GPU)

## Configuração no Supabase

Após fazer deploy do serviço:

1. Acesse o painel do Supabase: 
   ```
   https://supabase.com/dashboard/project/SEU_PROJECT_ID/settings/functions
   ```
2. Vá em **Edge Functions** > **Environment Variables**
3. Adicione a variável:
   - **Nome:** `TRAINING_WEBHOOK_URL`
   - **Valor:** `https://seu-servico.com/train`
   - **Escopo:** Todas as funções (ou específica)
4. **Salve as alterações**
5. **Redeploy** da Edge Function se necessário

**Importante:** A URL deve incluir `/train` no final!

## Endpoints Disponíveis

### GET /health
Health check do serviço.

**Resposta:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "ultralytics_version": "8.3.0"
}
```

### GET /train
Informações sobre o endpoint de treinamento.

**Resposta:**
```json
{
  "service": "YOLO Training Service",
  "status": "ready",
  "endpoint": "/train",
  "method": "POST",
  "message": "Service is running. Use POST method to start training job."
}
```

### POST /train
Inicia um job de treinamento YOLO.

**Payload:**
```json
{
  "job_id": "uuid-do-job",
  "callback_url": "https://sua-api.com/webhook",
  "callback_token": "seu-bearer-token",
  "dataset_url": "https://url-do-dataset.zip",
  "base_model": "yolov8n",
  "model_name": "meu-modelo",
  "config": {
    "epochs": 100,
    "batch_size": 16,
    "img_size": 640,
    "learning_rate": 0.01
  }
}
```

**Resposta (202 Accepted):**
```json
{
  "success": true,
  "job_id": "uuid-do-job",
  "message": "Training started"
}
```

## Callbacks Durante Treinamento

O serviço envia callbacks para o `callback_url` especificado:

### 1. training_started
Enviado quando o job é aceito e o treinamento está iniciando.

### 2. training_progress
Enviado ao final de cada época com métricas de progresso.

### 3. training_completed
Enviado quando o treinamento é concluído com sucesso (inclui modelo em base64).

### 4. training_failed
Enviado se houver erro durante o treinamento.

## Monitoramento

### Logs da Aplicação
Os logs são impressos no stdout/stderr:
- **Railway:** Logs tab no dashboard
- **Render:** Logs tab no dashboard  
- **Fly.io:** `fly logs`
- **Modal:** Dashboard de logs

### Health Check Automático
Verifique periodicamente:
```bash
curl https://seu-servico.com/health
```

### Métricas Importantes
- Taxa de sucesso/falha dos treinamentos
- Tempo médio de treinamento
- Uso de CPU/GPU/Memória
- Tamanho dos datasets processados

## Troubleshooting

### ❌ Erro: "CUDA not available"
**Causa:** Serviço rodando em CPU (sem GPU disponível)

**Solução:**
- Treinamento funcionará, mas será mais lento
- Para GPU, use Modal, RunPod ou VM com GPU

### ❌ Erro: "data.yaml not found"
**Causa:** Dataset ZIP mal formatado

**Solução:**
Verifique a estrutura do dataset:
```
dataset.zip
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       └── img3.jpg
├── labels/
│   ├── train/
│   │   ├── img1.txt
│   │   └── img2.txt
│   └── val/
│       └── img3.txt
└── data.yaml (opcional - será gerado automaticamente)
```

### ❌ Erro: "Failed to download dataset"
**Causa:** URL do dataset inválida ou inacessível

**Solução:**
- Verifique se a URL está correta e pública
- Teste download manual: `curl -I URL_DO_DATASET`
- Certifique-se que não requer autenticação

### ❌ Timeout durante treinamento
**Causa:** Treinamento demorado + timeout da plataforma

**Solução:**
- Aumente o timeout na configuração da plataforma
- Reduza número de épocas temporariamente
- Reduza batch_size
- Use GPU (muito mais rápido)

### ❌ Erro: "Out of Memory (OOM)"
**Causa:** Dataset/modelo muito grande para memória disponível

**Solução:**
- Reduza `batch_size` (ex: 16 → 8 → 4)
- Use instância com mais RAM
- Reduza `img_size` (ex: 640 → 416)

### ❌ Container não inicia
**Causa:** Erro no Dockerfile ou dependências

**Solução:**
```bash
# Build local para testar
docker build -t yolo-test .
docker run -p 5000:5000 yolo-test

# Ver logs
docker logs <container-id>
```

### ❌ Callback não recebido
**Causa:** URL de callback incorreta ou serviço de callback fora do ar

**Solução:**
- Verifique logs do serviço de treinamento
- Teste o callback manualmente:
```bash
curl -X POST https://seu-callback.com/webhook \
  -H "Authorization: Bearer SEU_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"job_id":"test","type":"test","data":{}}'
```

## Melhorias Futuras

- [ ] Suporte para retomar treinamento pausado
- [ ] Cache de modelos base baixados
- [ ] Fila de jobs (Redis/Celery)
- [ ] Métricas com Prometheus
- [ ] Suporte para múltiplas GPUs
- [ ] Auto-tuning de hiperparâmetros
- [ ] UI para monitoramento

## Suporte

Para problemas ou dúvidas:
- Abra uma issue no GitHub
- Consulte a documentação do Ultralytics: https://docs.ultralytics.com
- Verifique os logs do serviço

## Licença

MIT License
