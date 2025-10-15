# Serviço de Treinamento YOLO Real

Este é o serviço Python/Flask que executa treinamento real de modelos YOLO usando Ultralytics.

## Requisitos

- Python 3.11+
- GPU (NVIDIA) recomendada para treinamento rápido (opcional, funciona com CPU)

## Instalação Local

```bash
cd python-training-service
pip install -r requirements.txt
python train_service.py
```

O serviço estará disponível em `http://localhost:5000`

## Teste Local com Ngrok

Para testar localmente antes de fazer deploy em produção:

1. Instale o ngrok: https://ngrok.com/download
2. Execute o serviço: `python train_service.py`
3. Em outro terminal, exponha o serviço: `ngrok http 5000`
4. Copie a URL pública do ngrok (ex: `https://abc123.ngrok.io`)
5. Configure no Supabase como `TRAINING_WEBHOOK_URL`

## Deploy em Produção

### Opção 1: Railway (Fácil, sem GPU)

1. Crie conta em https://railway.app
2. Clique em "New Project" > "Deploy from GitHub repo"
3. Conecte este repositório
4. Railway detectará automaticamente o Dockerfile
5. Após deploy, copie a URL do serviço
6. Configure no Supabase: `TRAINING_WEBHOOK_URL=https://seu-app.railway.app/train`

### Opção 2: Render (Fácil, sem GPU)

1. Crie conta em https://render.com
2. Clique em "New +" > "Web Service"
3. Conecte este repositório
4. Configurações:
   - Environment: Docker
   - Instance Type: Standard (ou maior)
5. Após deploy, copie a URL
6. Configure no Supabase: `TRAINING_WEBHOOK_URL=https://seu-app.onrender.com/train`

### Opção 3: Modal (Serverless GPU - Recomendado)

1. Crie conta em https://modal.com
2. Instale o cliente: `pip install modal`
3. Configure: `modal token new`
4. Crie arquivo `modal_train.py`:

```python
import modal

stub = modal.Stub("yolo-training")

image = modal.Image.debian_slim().pip_install(
    "ultralytics",
    "flask",
    "torch",
    "torchvision",
    "requests"
)

@stub.function(
    image=image,
    gpu="T4",  # GPU Tesla T4
    timeout=3600,  # 1 hora
)
@modal.web_endpoint(method="POST")
def train(item: dict):
    from train_service import train as train_func
    return train_func(item)
```

5. Deploy: `modal deploy modal_train.py`
6. Copie a URL fornecida
7. Configure no Supabase

### Opção 4: RunPod (GPU Dedicado)

1. Crie conta em https://runpod.io
2. Deploy um pod com GPU
3. Use a imagem Docker deste projeto
4. Configure a URL no Supabase

### Opção 5: AWS/GCP/Azure (Avançado)

Deploy em VM com GPU:
- EC2 (AWS) com instância g4dn.xlarge
- Compute Engine (GCP) com GPU
- Azure VM com GPU

## Configuração no Supabase

Após fazer deploy do serviço:

1. Acesse: https://supabase.com/dashboard/project/pdbqkhutwgtmgaudujwl/settings/functions
2. Adicione a variável de ambiente:
   - Nome: `TRAINING_WEBHOOK_URL`
   - Valor: `https://seu-servico.com/train`
3. Salve as alterações

## Verificação

Teste o health check:
```bash
curl https://seu-servico.com/health
```

Resposta esperada:
```json
{
  "status": "healthy",
  "device": "cuda",
  "ultralytics_version": "8.3.0"
}
```

## Monitoramento

- Logs são impressos no stdout
- Use a plataforma de deploy para visualizar logs
- Callbacks são enviados automaticamente para o Supabase durante o treinamento

## Troubleshooting

### Erro: "CUDA not available"
- Serviço rodando em CPU (mais lento)
- Para usar GPU, faça deploy em plataforma com GPU

### Erro: "data.yaml not found"
- Dataset ZIP mal formatado
- Verifique estrutura do dataset (veja TRAINING_SETUP.md)

### Timeout durante treinamento
- Aumente o timeout da plataforma de deploy
- Reduza número de épocas ou batch size
