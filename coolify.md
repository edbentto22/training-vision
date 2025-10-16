# 🚀 Deploy no Coolify - Serviço YOLO Training

## ✅ Pré-requisitos

Seu projeto JÁ ESTÁ PRONTO para Coolify! O Coolify vai:
- ✅ Detectar automaticamente o `Dockerfile`
- ✅ Fazer build da imagem
- ✅ Expor a porta 5000
- ✅ Gerar URL pública automaticamente

## 📋 Passo a Passo Completo

### 1️⃣ Preparar o Repositório Git

**Opção A: GitHub/GitLab (Recomendado)**
```bash
# Inicializar repositório
git init
git add .
git commit -m "Initial commit - YOLO Training Service"

# Adicionar remote (GitHub)
git remote add origin https://github.com/seu-usuario/yolo-training-service.git
git push -u origin main
```

**Opção B: Push direto (Git privado do Coolify)**
- O Coolify também aceita push direto via git

### 2️⃣ Acessar o Coolify

1. Acesse seu painel Coolify: `https://seu-coolify.com`
2. Faça login com suas credenciais

### 3️⃣ Criar Novo Projeto

1. **Click em "+ New"** no menu superior
2. Selecione **"Application"**
3. Escolha o tipo: **"Public Repository"** ou **"Private Repository"**

### 4️⃣ Configurar a Aplicação

#### **Configurações Básicas:**

**Public Repository:**
- **Git Repository URL:** `https://github.com/seu-usuario/yolo-training-service`
- **Branch:** `main` (ou `master`)
- **Build Pack:** Coolify detecta automaticamente o Dockerfile

**Name:** `yolo-training-service`
**Description:** `Serviço de treinamento YOLO com Ultralytics`

#### **Build Settings:**
```
Build Pack: Dockerfile
Dockerfile Location: ./Dockerfile
Docker Compose File Location: (deixar vazio)
Base Directory: / (root)
```

#### **Port Mapping (IMPORTANTE):**
```
Application Port: 5000
Public Port: 80 (ou 443 se SSL)
```

**OU configure manualmente:**
- **Container Port:** `5000`
- **Public:** ✅ (habilitado)
- **Protocol:** HTTP

### 5️⃣ Variáveis de Ambiente (Opcional)

No Coolify, adicione as variáveis:

```bash
PORT=5000
FLASK_ENV=production
PYTHONUNBUFFERED=1
```

**Nota:** Essas variáveis já estão no Dockerfile, mas você pode sobrescrever aqui se necessário.

### 6️⃣ Configurar Health Check

```
Health Check Path: /health
Health Check Port: 5000
Health Check Interval: 30
Health Check Timeout: 10
Health Check Retries: 3
```

### 7️⃣ Configurar Recursos (Importante!)

Dependendo do servidor Coolify:

**CPU e Memória Mínima:**
```
CPU: 1-2 cores
Memory: 2GB (mínimo)
Memory: 4GB (recomendado)
Memory: 8GB (ideal para treinamentos maiores)
```

**Limites de Timeout:**
```
Request Timeout: 3600s (1 hora)
```

### 8️⃣ Deploy

1. Click em **"Deploy"** ou **"Save & Deploy"**
2. Coolify vai:
   - ✅ Clonar o repositório
   - ✅ Detectar o Dockerfile
   - ✅ Fazer build da imagem (pode demorar 5-10 min)
   - ✅ Iniciar o container
   - ✅ Gerar URL pública

3. **Acompanhe o build** na aba "Deployments"

### 9️⃣ Obter a URL Pública

Após o deploy concluído:

1. Vá na aba **"Domains"** ou **"Application"**
2. Copie a URL gerada (exemplo):
   ```
   https://yolo-training-service.seu-coolify.com
   ```

### 🔟 Testar o Serviço

```bash
# Health check
curl https://yolo-training-service.seu-coolify.com/health

# Info
curl https://yolo-training-service.seu-coolify.com/train
```

**Resposta esperada:**
```json
{
  "status": "healthy",
  "device": "cpu",
  "ultralytics_version": "8.3.0"
}
```

---

## 🔧 Configurações Avançadas do Coolify

### Custom Domain (Opcional)

Se quiser usar seu próprio domínio:

1. **Na aplicação, vá em "Domains"**
2. **Add Domain:** `train.seudominio.com`
3. **Configure DNS:**
   ```
   Type: A
   Name: train
   Value: IP-DO-SEU-COOLIFY
   ```
4. **Coolify gera SSL automaticamente** (Let's Encrypt)

### Persistent Storage (Se necessário)

Se quiser persistir os dados de treinamento:

1. **Vá em "Storages"**
2. **Add Volume:**
   ```
   Name: yolo-workdir
   Mount Path: /app/workdir
   ```

**⚠️ Nota:** Para este projeto, **NÃO é necessário** porque:
- Datasets são baixados temporariamente
- Modelos são enviados via callback em base64
- Tudo é limpo após o treino

### Webhooks de Deploy Automático

Coolify pode fazer redeploy automático ao push no Git:

1. **Vá em "Webhooks"**
2. **Enable Automatic Deployment**
3. **Copie a webhook URL**
4. **No GitHub:**
   - Settings → Webhooks → Add webhook
   - Payload URL: (URL do Coolify)
   - Content type: `application/json`
   - Events: `push`

Agora, todo `git push` faz deploy automático! 🚀

---

## 📊 Monitoramento no Coolify

### Ver Logs em Tempo Real

1. **Na aplicação, clique em "Logs"**
2. **Escolha:**
   - Build Logs (logs do build)
   - Application Logs (logs do serviço rodando)

```bash
# Ou via CLI do Coolify (se tiver acesso SSH)
docker logs -f <container-id>
```

### Métricas

Coolify mostra automaticamente:
- ✅ CPU usage
- ✅ Memory usage
- ✅ Network I/O
- ✅ Uptime
- ✅ Health check status

---

## 🔄 Atualizar o Serviço

### Opção 1: Push no Git (Automático)

```bash
# Fazer alterações no código
git add .
git commit -m "Update: nova feature"
git push

# Coolify faz redeploy automático (se webhook configurado)
```

### Opção 2: Manual no Coolify

1. **Vá na aplicação**
2. **Click em "Redeploy"**
3. Coolify faz:
   - Pull do código atualizado
   - Rebuild da imagem
   - Restart do container

---

## 🔒 SSL/HTTPS Automático

Coolify **gera SSL automaticamente** via Let's Encrypt!

Se não estiver funcionando:
1. **Vá em "Domains"**
2. **Click em "Generate SSL"**
3. Aguarde ~1-2 minutos

---

## 🐛 Troubleshooting

### ❌ Build falha

**Ver logs de build:**
```
Coolify → Application → Deployments → Ver build logs
```

**Problemas comuns:**
- Dockerfile com erro de sintaxe
- Dependência faltando no requirements.txt
- Falta de memória durante build

**Solução:**
```bash
# Testar build local primeiro
docker build -t test .
```

### ❌ Container não inicia

**Ver logs da aplicação:**
```
Coolify → Logs → Application Logs
```

**Verificar:**
- Porta 5000 está exposta corretamente
- Variáveis de ambiente estão corretas
- Health check configurado certo

### ❌ Health check falha

**Verificar:**
1. Path está correto: `/health`
2. Porta: `5000`
3. Serviço está respondendo:
   ```bash
   # SSH no servidor Coolify
   curl http://localhost:5000/health
   ```

### ❌ Timeout durante treinamento

**Aumentar timeout:**
1. Coolify → Application → Settings
2. Request Timeout: `3600` (ou mais)
3. Save & Redeploy

### ❌ Out of Memory

**Aumentar limite de memória:**
1. Coolify → Application → Resources
2. Memory Limit: `4096` MB (ou mais)
3. Save & Redeploy

---

## ✅ Checklist Final Coolify

- [ ] Repositório Git criado e código commitado
- [ ] Aplicação criada no Coolify
- [ ] Dockerfile detectado automaticamente
- [ ] Porta 5000 mapeada corretamente
- [ ] Health check configurado (`/health`)
- [ ] Deploy concluído com sucesso
- [ ] URL pública funcionando
- [ ] Health check respondendo OK
- [ ] Teste de treinamento completo realizado

---

## 🎯 Configurar no Supabase

Após deploy concluído:

1. **Copie a URL do Coolify:**
   ```
   https://yolo-training-service.seu-coolify.com
   ```

2. **Vá no Supabase:**
   ```
   Dashboard → Project → Settings → Edge Functions → Environment Variables
   ```

3. **Adicione:**
   ```
   TRAINING_WEBHOOK_URL=https://yolo-training-service.seu-coolify.com/train
   ```

4. **Save e Redeploy** da Edge Function (se necessário)

---

## 🚀 Pronto!

Seu serviço YOLO Training está rodando no Coolify! 🎉

**Próximos passos:**
1. Teste com um dataset pequeno
2. Monitore logs durante primeiro treino
3. Ajuste recursos se necessário (CPU/RAM)
4. Configure webhook de deploy automático
5. Adicione custom domain (opcional)

**URL para configurar no Supabase:**
```
https://seu-app.coolify.com/train
```

---

## 📚 Recursos Úteis

- **Coolify Docs:** https://coolify.io/docs
- **Ultralytics Docs:** https://docs.ultralytics.com
- **Docker Docs:** https://docs.docker.com

---

**Dúvidas?** Verifique os logs no Coolify e teste o health check primeiro!
