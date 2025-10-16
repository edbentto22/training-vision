"""
Serviço de Treinamento Real YOLO com Ultralytics
Este serviço recebe requisições do Supabase Edge Function e executa treinamento real de modelos YOLO.
"""

from flask import Flask, request, jsonify
import base64
import zipfile
import io
import os
import shutil
import requests
import logging
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
        'ultralytics_version': YOLO.__version__ if hasattr(YOLO, '__version__') else 'unknown'
    })

@app.route('/train', methods=['GET'])
def train_info():
    """Endpoint de informação (GET) - Use POST para treinar"""
    return jsonify({
        'service': 'YOLO Training Service',
        'status': 'ready',
        'endpoint': '/train',
        'method': 'POST',
        'message': 'Service is running. Use POST method to start training job.'
    })

@app.route('/train', methods=['POST'])
def train():
    """Endpoint principal de treinamento"""
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
    except Exception as e:
        logger.error(f"Error parsing request: {str(e)}", exc_info=True)
        return jsonify({'error': f'Invalid request: {str(e)}'}), 400
    
    # Preparar callbacks antes de qualquer operação que possa falhar
    def send_callback(callback_type, callback_data):
        """Envia callback para o Supabase"""
        try:
            payload = {
                'job_id': job_id,
                'type': callback_type,
                'data': callback_data
            }
            requests.post(
                callback_url,
                json=payload,
                headers={'Authorization': f'Bearer {callback_token}'},
                timeout=10
            )
        except Exception as e:
            logger.error(f"Error sending callback: {e}")

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
            # Download com timeout e streaming
            response = requests.get(dataset_url, stream=True, timeout=300)
            response.raise_for_status()
            
            dataset_bytes = response.content
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
        
        # Verificar se data.yaml existe
        data_yaml_path = job_dir / 'data.yaml'
        if not data_yaml_path.exists():
            # Procurar em subdiretórios
            yaml_files = list(job_dir.rglob('data.yaml'))
            if yaml_files:
                data_yaml_path = yaml_files[0]
                logger.info(f"Found data.yaml at: {data_yaml_path}")
            else:
                # data.yaml não encontrado - gerar automaticamente
                logger.info("data.yaml not found, generating automatically from dataset structure")
                
                # Procurar por diretórios images e labels
                images_dirs = list(job_dir.rglob('images'))
                labels_dirs = list(job_dir.rglob('labels'))
                
                if not images_dirs or not labels_dirs:
                    raise FileNotFoundError("Dataset must contain 'images' and 'labels' directories")
                
                # Usar o primeiro conjunto encontrado
                images_dir = images_dirs[0]
                labels_dir = labels_dirs[0]
                
                logger.info(f"Found images dir: {images_dir}")
                logger.info(f"Found labels dir: {labels_dir}")
                
                # Detectar classes únicas dos arquivos .txt
                classes = set()
                for label_file in labels_dir.rglob('*.txt'):
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                classes.add(int(parts[0]))
                
                # Criar lista de nomes de classes
                class_names = [f"class_{i}" for i in sorted(classes)]
                num_classes = len(class_names)
                
                logger.info(f"Detected {num_classes} classes: {class_names}")
                
                # Gerar data.yaml
                yaml_content = f"""# Auto-generated data.yaml
path: {job_dir.absolute()}
train: images
val: images

nc: {num_classes}
names: {class_names}
"""
                
                data_yaml_path = job_dir / 'data.yaml'
                with open(data_yaml_path, 'w') as f:
                    f.write(yaml_content)
                
                logger.info(f"Generated data.yaml at: {data_yaml_path}")
        else:
            logger.info(f"Found existing data.yaml at: {data_yaml_path}")
        
        # 2. Carregar modelo base
        base_model = data.get('base_model', 'yolov8n')
        logger.info(f"Loading base model: {base_model}")
        
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
        
        # 4. Definir callback de progresso
        # Callback function is defined earlier to be available in early error paths

        # Callback customizado para cada época
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
            exist_ok=True,
            callbacks={
                'on_train_epoch_end': callback_handler.on_train_epoch_end
            }
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
        shutil.rmtree(job_dir, ignore_errors=True)
        shutil.rmtree(RUNS_DIR / job_id, ignore_errors=True)
        
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
        
        # Cleanup em caso de erro
        shutil.rmtree(job_dir, ignore_errors=True)
        shutil.rmtree(RUNS_DIR / job_id, ignore_errors=True)
        
        return jsonify({
            'error': str(e),
            'job_id': job_id
        }), 500

if __name__ == '__main__':
    # Porta padrão: 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
