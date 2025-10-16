# 🚀 Setup Completo do Serviço de Treinamento YOLO

## 📁 Estrutura de Arquivos Necessários

```
python-training-service/
├── training_service.py          # Código principal (do artefato)
├── requirements.txt              # Dependências Python
├── Dockerfile                    # Configuração Docker
├── .dockerignore                 # Arquivos a ignorar no build
├── docker-compose.yml (opcional) # Para rodar local com Docker
└── README.md                     # Documentação
```

## 🔧 Setup Local (Desenvolvimento)

### 1. Pré-requisitos
```bash
# Python 3.11+
python --version

# pip atualizado
pip install --upgrade pip
```

### 2. Instalar Dependências
```bash
# Criar ambiente virtual (recomendado)
python -m venv venv

# Ativar ambiente virtual
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### 3. Executar Localmente
```bash
# Executar com Python direto
python training_service.py

# OU com Gunicorn (mais próximo da produção)
gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 3600 training_service:app
```

### 4. Testar
```bash
# Health check
curl http://localhost:5000/health

# Info do endpoint
curl http://localhost:5000/train
```

## 🐳 Setup com Docker (Local)

### 1. Build da Imagem
```bash
# Build
docker build -t yolo-training-service:latest .

# Ver imagem criada
docker images | grep yolo-training
```

### 2. Executar Container
```bash
# Rodar container
docker run -d \
  --name yolo-training \
  -p 5000:5000 \
  yolo-training-service:latest

# Ver logs
docker logs -f yolo-training

# Parar container
docker stop yolo-training

# Remover container
docker rm yolo-training
```

### 3. Usar Docker Compose (Mais Fácil)
```bash
# Iniciar serviço
docker-compose up -d

# Ver logs
docker-compose logs -f

# Parar serviço
docker-compose down
```

## ☁️ Deploy em Produção

### Opção 1: Railway (MAIS FÁCIL) ⭐

1. **Criar conta:** https://railway.app
2. **Conectar GitHub:**
   - Faça push do código para um repositório GitHub
   - New Project → Deploy from GitHub
3. **Railway auto-detecta o Dockerfile**
4. **Deploy automático:** Pronto em ~2-5 minutos
5. **Copiar URL:** `https://seu-app.up.railway.app`

**Testar:**
```bash
curl https://seu-app.up.railway.app/health
```

### Opção 2: Render

1. **Criar conta:** https://render.com
2. **New Web Service:**
   - Connect repository
   - Environment: Docker
   - Instance: Standard ou maior
3. **Deploy:** ~5-10 minutos
4. **URL:** `https://seu-app.onrender.com`

### Opção 3: Fly.io (Global)

```bash
# Instalar CLI
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Criar app
fly launch

# Deploy
fly deploy

# Ver logs
fly logs
```

### Opção 4: Push para Docker Hub (Para outras plataformas)

```bash
# Login no Docker Hub
docker login

# Tag da imagem
docker tag yolo-training-service:latest seu-usuario/yolo-training:latest

# Push
docker push seu-usuario/yolo-training:latest

# Usar em qualquer plataforma que suporte Docker
```

## 🔐 Configuração no Supabase

### 1. Acessar Edge Functions
```
https://supabase.com/dashboard/project/SEU_PROJECT_ID/settings/functions
```

### 2. Adicionar Variável de Ambiente
- **Nome:** `TRAINING_WEBHOOK_URL`
- **Valor:** `https://seu-servico.com/train` (URL do seu serviço + `/train`)
- **Scope:** Todas as funções

### 3. Salvar e Redeploy (se necessário)

## ✅ Verificação Final

### 1. Health Check
```bash
curl https://seu-servico.com/health
```

**Resposta esperada:**
```json
{
  "status": "healthy",
  "device": "cpu",
  "ultralytics_version": "8.3.0"
}
```

### 2. Teste Completo do Fluxo
```bash
# Preparar um dataset de teste pequeno
# Fazer upload e iniciar treinamento via interface web

# Monitorar logs do serviço
# Verificar callbacks recebidos
```

## 📊 Monitoramento

### Logs em Tempo Real

**Railway:**
```bash
# No dashboard Railway → Logs tab
```

**Render:**
```bash
# No dashboard Render → Logs tab
```

**Fly.io:**
```bash
fly logs
```

**Docker Local:**
```bash
docker logs -f yolo-training
```

### Métricas Importantes

- ✅ Taxa de sucesso dos treinamentos
- ⏱️ Tempo médio de treinamento
- 💾 Uso de memória/CPU
- 📈 Número de jobs processados

## 🐛 Troubleshooting Rápido

### Container não inicia
```bash
# Ver erro detalhado
docker logs yolo-training

# Testar build local
docker build -t test .
docker run -it --rm test /bin/bash
```

### Porta já em uso
```bash
# Ver o que está usando porta 5000
lsof -i :5000
# OU
netstat -tulpn | grep 5000

# Matar processo
kill -9 <PID>
```

### Dependências não instalam
```bash
# Limpar cache pip
pip cache purge

# Reinstalar
pip install --no-cache-dir -r requirements.txt
```

### Out of Memory
```bash
# Aumentar RAM na plataforma
# OU reduzir batch_size no config:
{
  "config": {
    "batch_size": 4,  # Reduzir de 16 para 4
    "img_size": 416   # Reduzir de 640 para 416
  }
}
```

## 🎯 Checklist Final

- [ ] Código `training_service.py` no lugar
- [ ] `requirements.txt` com todas as dependências
- [ ] `Dockerfile` configurado
- [ ] Build local funcionando
- [ ] Health check respondendo
- [ ] Deploy em produção concluído
- [ ] URL configurada no Supabase
- [ ] Teste de treinamento end-to-end
- [ ] Monitoramento configurado

## 📚 Próximos Passos

1. **Teste com dataset pequeno** (10-20 imagens)
2. **Monitore primeiro treinamento completo**
3. **Ajuste configurações** baseado nos resultados
4. **Configure alertas** para falhas
5. **Documente seus casos de uso específicos**

## 🆘 Precisa de Ajuda?

- **Logs não aparecem:** Verifique configuração de logging na plataforma
- **Timeout:** Aumente timeout nas configs (3600s = 1 hora)
- **GPU não detectada:** Normal em CPU-only deployments
- **Callback não recebe:** Verifique URL e token no payload

---

✅ **Tudo pronto!** Seu serviço de treinamento YOLO está configurado e rodando!
