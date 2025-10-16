"""
Serviço de Treinamento Real YOLO com Ultralytics
Este serviço recebe requisições do Supabase Edge Function e executa treinamento real de modelos YOLO.
VERSÃO FINAL - Atualizada 2025-10-16
"""

from flask import Flask, request, jsonify
import base64
import zipfile
import io
import os
import shutil
import requests
import logging
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Diretórios de trabalho
WORK_DIR = Path('./workdir')
DATASETS_DIR = WORK_DIR / 'datasets'
MODELS_DIR = WORK_DIR / 'models'
RUNS_DIR = WORK_DIR / 'runs'

# Criar diretórios se não existirem
for dir_path in [WORK_DIR, DATASETS_DIR, MODELS_DIR, RUNS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de health check"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return jsonify({
        'status': 'healthy',
        'device': device,
        'ultralytics_version': YOLO.__version__ if hasattr(YOLO, '__version__') else 'unknown',
        'version': 'v2.0-final'
    })

@app.route('/train', methods=['GET'])
def train_info():
    """Endpoint de informação (GET) - Use POST para treinar"""
    return jsonify({
        'service': 'YOLO Training Service',
        'status': 'ready',
        'endpoint': '/train',
        'method': 'POST',
        'message': 'Service is running. Use POST method to start training job.',
        'version': 'v2.0-final'
    })

@app.route('/debug/dataset', methods=['POST'])
def debug_dataset():
    """DEBUG: Inspecionar estrutura do dataset antes de treinar"""
    try:
        data = request.json
        dataset_url = data.get('dataset_url')
        
        if not dataset_url:
            return jsonify({'error': 'dataset_url required'}), 400
        
        logger.info(f"Debugging dataset from: {dataset_url}")
        
        # Download dataset
        response = requests.get(dataset_url, stream=True, timeout=120)
        response.raise_for_status()
        dataset_bytes = response.content
        
        # Criar diretório temporário
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Extrair ZIP
            with zipfile.ZipFile(io.BytesIO(dataset_bytes)) as zip_ref:
                zip_ref.extractall(tmpdir_path)
            
            # Listar estrutura completa
            all_items = []
            for item in tmpdir_path.rglob('*'):
                rel_path = str(item.relative_to(tmpdir_path))
                is_dir = item.is_dir()
                size = item.stat().st_size if item.is_file() else 0
                all_items.append({
                    'path': rel_path,
                    'type': 'dir' if is_dir else 'file',
                    'size': size
                })
            
            # Contar tipos de arquivo
            images = list(tmpdir_path.rglob('*.jpg')) + list(tmpdir_path.rglob('*.jpeg')) + list(tmpdir_path.rglob('*.png'))
            labels = list(tmpdir_path.rglob('*.txt'))
            
            # Buscar pastas images e labels
            images_dirs = list(tmpdir_path.rglob('images'))
            labels_dirs = list(tmpdir_path.rglob('labels'))
            
            # Buscar data.yaml
            yaml_files = list(tmpdir_path.rglob('data.yaml'))
            
            return jsonify({
                'total_items': len(all_items),
                'structure': all_items[:50],  # Primeiros 50 itens
                'stats': {
                    'total_images': len(images),
                    'total_labels': len(labels),
                    'images_dirs_found': len(images_dirs),
                    'labels_dirs_found': len(labels_dirs),
                    'yaml_files_found': len(yaml_files),
                    'images_dirs_paths': [str(d.relative_to(tmpdir_path)) for d in images_dirs],
                    'labels_dirs_paths': [str(d.relative_to(tmpdir_path)) for d in labels_dirs],
                    'yaml_files_paths': [str(y.relative_to(tmpdir_path)) for y in yaml_files],
                },
                'sample_images': [str(img.relative_to(tmpdir_path)) for img in images[:5]],
                'sample_labels': [str(lbl.relative_to(tmpdir_path)) for lbl in labels[:5]],
            })
            
    except Exception as e:
        logger.error(f"Debug error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    """Endpoint principal de treinamento"""
    
    # Inicializar variáveis no início para escopo correto
    job_id = None
    callback_url = None
    callback_token = None
    job_dir = None
    
    # Definir send_callback NO INÍCIO antes de qualquer uso
    def send_callback(callback_type, callback_data):
        """Envia callback para o Supabase"""
        try:
            # Verificar se variáveis necessárias existem
            if not all([job_id, callback_url, callback_token]):
                logger.warning("Cannot send callback - missing required variables")
                return
                
            payload = {
                'job_id': job_id,
                'type': callback_type,
                'data': callback_data
            }
            response = requests.post(
                callback_url,
                json=payload,
                headers={'Authorization': f'Bearer {callback_token}'},
                timeout=10
            )
            logger.info(f"Callback sent: {callback_type} - Status: {response.status_code}")
        except Exception as e:
            logger.error(f"Error sending callback: {e}")
    
    try:
        data = request.json
        
        if not data:
            logger.error("No JSON data received in request")
            return jsonify({'error': 'No data provided'}), 400
        
        # Log da requisição (sem incluir a URL do dataset completa)
        log_data = {k: v for k, v in data.items() if k != 'dataset_url'}
        if 'dataset_url' in data:
            log_data['dataset_url'] = '***PROVIDED***'
        logger.info(f"Received training request: {log_data}")
        
        job_id = data.get('job_id')
        callback_url = data.get('callback_url')
        callback_token = data.get('callback_token')
        
        if not all([job_id, callback_url, callback_token]):
            error_msg = 'Missing required fields: job_id, callback_url, callback_token'
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400
        
        logger.info(f"Starting training job {job_id}")
        
        # Notificar que o job foi aceito e está iniciando
        send_callback('training_started', {
            'message': 'Training job accepted and starting'
        })
        
    except Exception as e:
        logger.error(f"Error parsing request: {str(e)}", exc_info=True)
        return jsonify({'error': f'Invalid request: {str(e)}'}), 400
    
    # Criar diretório do job
    job_dir = DATASETS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Download dataset from URL
        dataset_url = data.get('dataset_url')
        
        if not dataset_url:
            error_msg = "dataset_url is required in request payload"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Downloading dataset from: {dataset_url}")
        
        try:
            # Download com timeout maior e streaming para datasets grandes
            response = requests.get(dataset_url, stream=True, timeout=600)
            response.raise_for_status()
            
            # Download com progresso
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunks = []
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    chunks.append(chunk)
                    downloaded += len(chunk)
                    # Log a cada 10MB
                    if total_size > 0 and downloaded % (1024 * 1024 * 10) < 8192:
                        progress = int((downloaded / total_size) * 100)
                        logger.info(f"Download progress: {progress}%")
            
            dataset_bytes = b''.join(chunks)
            logger.info(f"Dataset downloaded: {len(dataset_bytes)} bytes")
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to download dataset from URL: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Extrair dataset ZIP
        logger.info("Extracting dataset ZIP")
        try:
            with zipfile.ZipFile(io.BytesIO(dataset_bytes)) as zip_ref:
                zip_ref.extractall(job_dir)
        except zipfile.BadZipFile as e:
            error_msg = f"Invalid ZIP file: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # SEMPRE GERAR NOVO data.yaml (ignorar o que vem no ZIP)
        logger.info("="*60)
        logger.info("🔍 SCANNING DATASET STRUCTURE")
        logger.info("="*60)
        
        # BUSCAR data.yaml antigo para obter informações (mas NÃO usar)
        old_yaml_files = list(job_dir.rglob('data.yaml'))
        
        if old_yaml_files:
            logger.warning(f"⚠️  Found {len(old_yaml_files)} existing data.yaml file(s) in ZIP:")
            for old_yaml in old_yaml_files:
                logger.warning(f"   - {old_yaml.relative_to(job_dir)}")
            
            # Ler o data.yaml existente apenas para informação
            try:
                with open(old_yaml_files[0], 'r') as f:
                    old_data = yaml.safe_load(f)
                logger.info(f"📄 Old data.yaml content:")
                logger.info(f"   path: {old_data.get('path', 'N/A')}")
                logger.info(f"   train: {old_data.get('train', 'N/A')}")
                logger.info(f"   val: {old_data.get('val', 'N/A')}")
                logger.info(f"   nc: {old_data.get('nc', 'N/A')}")
                logger.info(f"   names: {old_data.get('names', 'N/A')}")
            except Exception as e:
                logger.warning(f"Could not read old data.yaml: {e}")
        
        logger.info("🔧 GENERATING NEW data.yaml with CORRECT absolute paths")
        
        # Procurar por diretórios images e labels (recursivamente)
        images_dirs = list(job_dir.rglob('images'))
        labels_dirs = list(job_dir.rglob('labels'))
        
        if not images_dirs or not labels_dirs:
            # Tentar encontrar estrutura alternativa
            logger.warning("⚠️  Standard 'images' and 'labels' directories not found")
            logger.info("🔍 Searching for alternative structure...")
            
            # Procurar por qualquer pasta com imagens
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            found_images = False
            
            for ext in image_extensions:
                img_files = list(job_dir.rglob(ext))
                if img_files:
                    found_images = True
                    # Pegar o diretório pai das imagens encontradas
                    images_dir = img_files[0].parent
                    logger.info(f"✅ Found images in: {images_dir.relative_to(job_dir)}")
                    break
            
            if not found_images:
                # Listar conteúdo para debug
                all_files = list(job_dir.rglob('*'))[:30]
                logger.error(f"❌ No images found! Contents (first 30):")
                for f in all_files:
                    logger.error(f"   - {f.relative_to(job_dir)}")
                raise FileNotFoundError(
                    f"Dataset must contain 'images' directory or image files"
                )
            
            # Procurar labels correspondentes
            labels_dir = images_dir.parent / 'labels'
            if not labels_dir.exists():
                # Tentar encontrar pasta labels em qualquer lugar
                txt_files = list(job_dir.rglob('*.txt'))
                if txt_files:
                    labels_dir = txt_files[0].parent
                    logger.info(f"✅ Found labels in: {labels_dir.relative_to(job_dir)}")
                else:
                    raise FileNotFoundError("Labels directory not found")
        else:
            # Usar o primeiro conjunto encontrado
            images_dir = images_dirs[0]
            labels_dir = labels_dirs[0]
        
        logger.info(f"📂 Images directory: {images_dir}")
        logger.info(f"📂 Labels directory: {labels_dir}")
        
        # Detectar classes únicas dos arquivos .txt
        classes = set()
        label_count = 0
        invalid_labels = []
        
        for label_file in labels_dir.rglob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if parts:
                            try:
                                class_id = int(parts[0])
                                if class_id < 0:
                                    invalid_labels.append(f"{label_file.name}:L{line_num} - negative class {class_id}")
                                classes.add(class_id)
                            except ValueError:
                                invalid_labels.append(f"{label_file.name}:L{line_num} - invalid class format")
                label_count += 1
            except Exception as e:
                logger.warning(f"Error reading label file {label_file}: {e}")
        
        if not classes:
            raise ValueError("No valid class labels found in dataset")
        
        logger.info(f"📊 Found {label_count} label files")
        
        # Validar se classes são sequenciais começando do 0
        max_class = max(classes)
        min_class = min(classes)
        num_classes_detected = len(classes)
        
        logger.info(f"📊 Classes range: {min_class} to {max_class}")
        logger.info(f"📊 Unique classes detected: {num_classes_detected}")
        logger.info(f"📊 Classes found: {sorted(classes)}")
        
        # CRITICAL: Verificar se há classes fora do range esperado
        if max_class >= num_classes_detected:
            logger.warning(f"⚠️  CRITICAL: max class ID ({max_class}) >= num_classes ({num_classes_detected})")
            logger.warning("    This could cause 'index out of bounds' error during training!")
            logger.warning(f"    Expected classes: 0 to {num_classes_detected-1}")
            logger.warning(f"    Found classes: {sorted(classes)}")
            
            # Ajustar num_classes para max_class + 1
            num_classes = max_class + 1
            logger.warning(f"🔧 Auto-adjusting num_classes to {num_classes}")
            
            # Criar nomes para todas as classes até max_class
            class_names = [f"class_{i}" for i in range(num_classes)]
        else:
            # Usar número de classes detectadas
            num_classes = num_classes_detected
            class_names = [f"class_{i}" for i in sorted(classes)]
        
        if invalid_labels:
            logger.warning(f"⚠️  Found {len(invalid_labels)} invalid label entries:")
            for inv in invalid_labels[:10]:  # Log primeiros 10
                logger.warning(f"    - {inv}")
        
        logger.info(f"✅ Final configuration: {num_classes} classes")
        
        # Determinar se há splits train/val
        has_train_val = (images_dir / 'train').exists() and (images_dir / 'val').exists()
        
        logger.info(f"📁 Has train/val split: {has_train_val}")
        
        # Gerar data.yaml com caminhos ABSOLUTOS
        if has_train_val:
            train_path = str((images_dir / 'train').absolute())
            val_path = str((images_dir / 'val').absolute())
            yaml_content = f"""# Auto-generated data.yaml by Training Service
train: {train_path}
val: {val_path}
nc: {num_classes}
names: {class_names}
"""
        else:
            # Sem split, usar mesma pasta para train e val
            images_path = str(images_dir.absolute())
            yaml_content = f"""# Auto-generated data.yaml by Training Service
train: {images_path}
val: {images_path}
nc: {num_classes}
names: {class_names}
"""
        
        # Salvar o NOVO data.yaml na raiz do job_dir
        data_yaml_path = job_dir / 'data.yaml'
        with open(data_yaml_path, 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"✅ Generated NEW data.yaml at: {data_yaml_path}")
        
        # Verificar se o arquivo foi criado
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"Failed to create data.yaml at {data_yaml_path}")
        
        # LOG do conteúdo final do data.yaml
        logger.info("="*60)
        logger.info("📄 FINAL data.yaml content:")
        logger.info("="*60)
        with open(data_yaml_path, 'r') as f:
            final_yaml = f.read()
            for line in final_yaml.split('\n'):
                logger.info(f"  {line}")
        logger.info("="*60)
        
        # 2. Carregar modelo base
        base_model = data.get('base_model', 'yolov8n')
        logger.info(f"Loading base model: {base_model}")
        
        # Validar modelo base (YOLOv8-v11 disponíveis)
        valid_models = [
            'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
            'yolov9t', 'yolov9s', 'yolov9m', 'yolov9c', 'yolov9e',
            'yolov10n', 'yolov10s', 'yolov10m', 'yolov10b', 'yolov10l', 'yolov10x',
            'yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x'
        ]
        
        if base_model not in valid_models:
            error_msg = f"Invalid base_model '{base_model}'. Valid models: {', '.join(valid_models)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Garantir que o modelo tem a extensão .pt
        if not base_model.endswith('.pt'):
            base_model = f"{base_model}.pt"
        
        model = YOLO(base_model)
        
        # 3. Configurar parâmetros de treinamento
        config = data.get('config', {})
        epochs = config.get('epochs', 100)
        batch_size = config.get('batch_size', 16)
        img_size = config.get('img_size', 640)
        learning_rate = config.get('learning_rate', 0.01)
        
        # Detectar device (GPU ou CPU)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Training on device: {device}")
        
        # 4. Callback customizado para cada época
        class TrainingCallback:
            def __init__(self, total_epochs):
                self.total_epochs = total_epochs
                self.current_epoch = 0
            
            def on_train_epoch_end(self, trainer):
                """Chamado ao final de cada época"""
                self.current_epoch = trainer.epoch + 1
                progress = int((self.current_epoch / self.total_epochs) * 100)
                
                # Extrair métricas
                metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
                loss = trainer.loss.item() if hasattr(trainer, 'loss') and trainer.loss is not None else 0.0
                
                callback_data = {
                    'current_epoch': self.current_epoch,
                    'progress': progress,
                    'loss': float(loss),
                    'metrics': {
                        'precision': float(metrics.get('metrics/precision(B)', 0)),
                        'recall': float(metrics.get('metrics/recall(B)', 0)),
                        'mAP50': float(metrics.get('metrics/mAP50(B)', 0)),
                        'mAP50_95': float(metrics.get('metrics/mAP50-95(B)', 0)),
                        'box_loss': float(metrics.get('train/box_loss', 0)),
                        'cls_loss': float(metrics.get('train/cls_loss', 0)),
                        'dfl_loss': float(metrics.get('train/dfl_loss', 0)),
                    }
                }
                
                send_callback('training_progress', callback_data)
        
        # 5. Iniciar treinamento
        callback_handler = TrainingCallback(epochs)
        
        # Adicionar callback ao modelo antes do treinamento
        model.add_callback('on_train_epoch_end', callback_handler.on_train_epoch_end)
        
        logger.info(f"Starting training: {epochs} epochs, batch={batch_size}, imgsz={img_size}")
        
        results = model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            lr0=learning_rate,
            device=device,
            project=str(RUNS_DIR),
            name=job_id,
            exist_ok=True
        )
        
        logger.info(f"Training completed for job {job_id}")
        
        # 6. Ler modelo treinado e converter para base64
        model_path = RUNS_DIR / job_id / 'weights' / 'best.pt'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        
        logger.info(f"Reading trained model from: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
            model_base64 = base64.b64encode(model_bytes).decode('utf-8')
        
        # Extrair métricas finais
        final_metrics = {}
        if hasattr(results, 'results_dict'):
            rd = results.results_dict
            final_metrics = {
                'precision': float(rd.get('metrics/precision(B)', 0)),
                'recall': float(rd.get('metrics/recall(B)', 0)),
                'mAP50': float(rd.get('metrics/mAP50(B)', 0)),
                'mAP50_95': float(rd.get('metrics/mAP50-95(B)', 0)),
            }
        
        # 7. Enviar callback de conclusão
        send_callback('training_completed', {
            'model_base64': model_base64,
            'model_name': data.get('model_name'),
            'final_metrics': final_metrics
        })
        
        # 8. Cleanup
        logger.info(f"Cleaning up job {job_id}")
        try:
            if job_dir and job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
            if job_id and (RUNS_DIR / job_id).exists():
                shutil.rmtree(RUNS_DIR / job_id, ignore_errors=True)
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Training completed successfully'
        })
        
    except Exception as e:
        logger.error(f"Training failed for job {job_id}: {e}", exc_info=True)
        
        # Enviar callback de erro
        send_callback('training_failed', {
            'error': str(e)
        })
        
        # Cleanup em caso de erro com verificação de existência
        try:
            if job_dir and job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
            if job_id and (RUNS_DIR / job_id).exists():
                shutil.rmtree(RUNS_DIR / job_id, ignore_errors=True)
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")
        
        return jsonify({
            'error': str(e),
            'job_id': job_id if job_id else 'unknown'
        }), 500

if __name__ == '__main__':
    # Porta padrão: 5000 (configurável via variável de ambiente PORT)
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting YOLO Training Service on port {port}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Version: v2.0-final")
    app.run(host='0.0.0.0', port=port, debug=False)
