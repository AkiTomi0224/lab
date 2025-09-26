/**
 * Machine Learning Functionality for Equipment Detection System
 */

class MLSystem {
    constructor(apiBaseUrl) {
        this.apiBaseUrl = apiBaseUrl;
        this.annotationCanvas = null;
        this.annotationCtx = null;
        this.isAnnotating = false;
        this.currentAnnotation = null;
        this.selectedEquipmentId = null;
        this.selectedDiagramPath = null;

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadEquipmentOptions();
        this.loadDiagramOptions();
        this.loadTrainedModels();
    }

    setupEventListeners() {
        // Equipment and diagram selection
        document.getElementById('ml-equipment-select').addEventListener('change', (e) => {
            this.selectedEquipmentId = e.target.value;
            this.updateAnnotationControls();
        });

        document.getElementById('ml-diagram-select').addEventListener('change', (e) => {
            this.selectedDiagramPath = e.target.value;
            this.updateAnnotationControls();
        });

        // Annotation controls
        document.getElementById('start-annotation').addEventListener('click', () => {
            this.startAnnotation();
        });

        document.getElementById('save-annotation').addEventListener('click', () => {
            this.saveAnnotation();
        });

        document.getElementById('clear-annotation').addEventListener('click', () => {
            this.clearAnnotation();
        });

        // Training controls
        document.getElementById('start-training').addEventListener('click', () => {
            this.startTraining();
        });

        // Load training equipment options
        this.loadTrainingEquipmentOptions();
    }

    async loadEquipmentOptions() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/equipment`);
            const data = await response.json();

            const select = document.getElementById('ml-equipment-select');
            const trainSelect = document.getElementById('train-equipment-select');

            select.innerHTML = '<option value="">機器を選択してください</option>';
            trainSelect.innerHTML = '<option value="">機器を選択してください</option>';

            data.forEach(equipment => {
                const option = new Option(equipment.name, equipment.id);
                const trainOption = new Option(equipment.name, equipment.id);
                select.add(option);
                trainSelect.add(trainOption);
            });
        } catch (error) {
            console.error('Error loading equipment options:', error);
        }
    }

    async loadDiagramOptions() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/diagrams`);
            const data = await response.json();

            const select = document.getElementById('ml-diagram-select');
            select.innerHTML = '<option value="">図面を選択してください</option>';

            data.forEach(diagram => {
                const option = new Option(diagram.filename, diagram.file_path);
                select.add(option);
            });
        } catch (error) {
            console.error('Error loading diagram options:', error);
        }
    }

    async loadTrainingEquipmentOptions() {
        // Load equipment with training data count
        try {
            const response = await fetch(`${this.apiBaseUrl}/equipment`);
            const data = await response.json();

            for (const equipment of data) {
                const trainingResponse = await fetch(`${this.apiBaseUrl}/training-data/${equipment.id}`);
                const trainingData = await trainingResponse.json();
                equipment.trainingCount = trainingData.count;
            }

            // Update training samples count
            const totalSamples = data.reduce((sum, eq) => sum + eq.trainingCount, 0);
            document.getElementById('training-samples-count').textContent = totalSamples;

        } catch (error) {
            console.error('Error loading training equipment options:', error);
        }
    }

    updateAnnotationControls() {
        const canStart = this.selectedEquipmentId && this.selectedDiagramPath;
        document.getElementById('start-annotation').disabled = !canStart;
    }

    async startAnnotation() {
        if (!this.selectedEquipmentId || !this.selectedDiagramPath) return;

        try {
            // Load the diagram image
            const img = new Image();
            img.crossOrigin = 'anonymous';

            img.onload = () => {
                this.setupAnnotationCanvas(img);
                this.enableAnnotationMode();
            };

            img.onerror = () => {
                this.showNotification('図面画像の読み込みに失敗しました', 'error');
            };

            // Construct image URL
            let imageUrl;
            if (this.selectedDiagramPath.startsWith('static/')) {
                imageUrl = `${this.apiBaseUrl}/${this.selectedDiagramPath}`;
            } else {
                imageUrl = `${this.apiBaseUrl}/uploads/${this.selectedDiagramPath}`;
            }

            img.src = imageUrl;

        } catch (error) {
            console.error('Error starting annotation:', error);
            this.showNotification('アノテーションの開始に失敗しました', 'error');
        }
    }

    setupAnnotationCanvas(img) {
        // Show canvas and hide instructions
        const canvas = document.getElementById('annotation-canvas');
        const instructions = document.getElementById('annotation-instructions');

        canvas.style.display = 'block';
        instructions.style.display = 'none';

        // Calculate canvas size to fit the container while maintaining aspect ratio
        const container = canvas.parentElement;
        const containerWidth = container.clientWidth - 40; // padding
        const containerHeight = 500; // max height

        const scale = Math.min(
            containerWidth / img.width,
            containerHeight / img.height
        );

        canvas.width = img.width * scale;
        canvas.height = img.height * scale;

        // Store scale for coordinate conversion
        this.imageScale = scale;
        this.originalImageWidth = img.width;
        this.originalImageHeight = img.height;

        // Draw image on canvas
        this.annotationCtx = canvas.getContext('2d');
        this.annotationCtx.drawImage(img, 0, 0, canvas.width, canvas.height);
        this.annotationCanvas = canvas;
    }

    enableAnnotationMode() {
        this.isAnnotating = false;
        let startX, startY, isDrawing = false;

        // Mouse events for rectangle selection
        this.annotationCanvas.addEventListener('mousedown', (e) => {
            isDrawing = true;
            const rect = this.annotationCanvas.getBoundingClientRect();
            startX = e.clientX - rect.left;
            startY = e.clientY - rect.top;
        });

        this.annotationCanvas.addEventListener('mousemove', (e) => {
            if (!isDrawing) return;

            const rect = this.annotationCanvas.getBoundingClientRect();
            const currentX = e.clientX - rect.left;
            const currentY = e.clientY - rect.top;

            // Clear canvas and redraw image
            this.redrawCanvas();

            // Draw selection rectangle
            this.annotationCtx.strokeStyle = '#3b82f6';
            this.annotationCtx.lineWidth = 2;
            this.annotationCtx.setLineDash([5, 5]);
            this.annotationCtx.strokeRect(
                startX,
                startY,
                currentX - startX,
                currentY - startY
            );
        });

        this.annotationCanvas.addEventListener('mouseup', (e) => {
            if (!isDrawing) return;
            isDrawing = false;

            const rect = this.annotationCanvas.getBoundingClientRect();
            const endX = e.clientX - rect.left;
            const endY = e.clientY - rect.top;

            // Calculate rectangle coordinates
            const x = Math.min(startX, endX);
            const y = Math.min(startY, endY);
            const width = Math.abs(endX - startX);
            const height = Math.abs(endY - startY);

            // Convert to original image coordinates
            this.currentAnnotation = {
                x: Math.round(x / this.imageScale),
                y: Math.round(y / this.imageScale),
                width: Math.round(width / this.imageScale),
                height: Math.round(height / this.imageScale)
            };

            // Draw final rectangle
            this.redrawCanvas();
            this.annotationCtx.strokeStyle = '#22c55e';
            this.annotationCtx.lineWidth = 3;
            this.annotationCtx.setLineDash([]);
            this.annotationCtx.strokeRect(x, y, width, height);

            // Enable save button
            document.getElementById('save-annotation').disabled = false;
        });

        // Update control buttons
        document.getElementById('start-annotation').disabled = true;
        document.getElementById('clear-annotation').disabled = false;
    }

    redrawCanvas() {
        // Redraw the original image
        const img = new Image();
        img.onload = () => {
            this.annotationCtx.clearRect(0, 0, this.annotationCanvas.width, this.annotationCanvas.height);
            this.annotationCtx.drawImage(img, 0, 0, this.annotationCanvas.width, this.annotationCanvas.height);
        };

        let imageUrl;
        if (this.selectedDiagramPath.startsWith('static/')) {
            imageUrl = `${this.apiBaseUrl}/${this.selectedDiagramPath}`;
        } else {
            imageUrl = `${this.apiBaseUrl}/uploads/${this.selectedDiagramPath}`;
        }
        img.src = imageUrl;
    }

    clearAnnotation() {
        this.currentAnnotation = null;
        this.redrawCanvas();

        // Reset buttons
        document.getElementById('start-annotation').disabled = false;
        document.getElementById('save-annotation').disabled = true;
        document.getElementById('clear-annotation').disabled = true;

        // Hide canvas and show instructions
        const canvas = document.getElementById('annotation-canvas');
        const instructions = document.getElementById('annotation-instructions');
        canvas.style.display = 'none';
        instructions.style.display = 'block';
    }

    async saveAnnotation() {
        if (!this.currentAnnotation || !this.selectedEquipmentId || !this.selectedDiagramPath) {
            this.showNotification('アノテーションデータが不完全です', 'error');
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/training-data`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    equipment_id: parseInt(this.selectedEquipmentId),
                    diagram_path: this.selectedDiagramPath,
                    bbox: [
                        this.currentAnnotation.x,
                        this.currentAnnotation.y,
                        this.currentAnnotation.width,
                        this.currentAnnotation.height
                    ]
                }),
            });

            if (response.ok) {
                this.showNotification('学習データが正常に保存されました', 'success');
                this.clearAnnotation();
                this.loadTrainingEquipmentOptions(); // Update stats
            } else {
                throw new Error('Failed to save annotation');
            }
        } catch (error) {
            console.error('Error saving annotation:', error);
            this.showNotification('学習データの保存に失敗しました', 'error');
        }
    }

    async startTraining() {
        const equipmentId = document.getElementById('train-equipment-select').value;
        const epochs = document.getElementById('training-epochs').value;

        if (!equipmentId) {
            this.showNotification('訓練対象機器を選択してください', 'error');
            return;
        }

        try {
            // Show progress
            const progressContainer = document.getElementById('training-progress');
            const progressBar = document.getElementById('training-progress-bar');
            const progressText = document.getElementById('training-status');
            const startButton = document.getElementById('start-training');

            progressContainer.style.display = 'block';
            startButton.disabled = true;
            startButton.classList.add('training-active');

            // Simulate training progress
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += 2;
                progressBar.style.width = `${Math.min(progress, 95)}%`;
                progressText.textContent = `訓練中... ${Math.min(progress, 95)}%`;
            }, 1000);

            const response = await fetch(`${this.apiBaseUrl}/train-model`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    equipment_id: parseInt(equipmentId),
                    epochs: parseInt(epochs)
                }),
            });

            clearInterval(progressInterval);

            if (response.ok) {
                progressBar.style.width = '100%';
                progressText.textContent = '訓練完了！';
                this.showNotification('モデル訓練が正常に完了しました', 'success');
                this.loadTrainedModels();

                setTimeout(() => {
                    progressContainer.style.display = 'none';
                    progressBar.style.width = '0%';
                }, 3000);
            } else {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Training failed');
            }
        } catch (error) {
            console.error('Error training model:', error);
            this.showNotification(`モデル訓練に失敗しました: ${error.message}`, 'error');
        } finally {
            const startButton = document.getElementById('start-training');
            startButton.disabled = false;
            startButton.classList.remove('training-active');
        }
    }

    async loadTrainedModels() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/trained-models`);
            const data = await response.json();

            const container = document.getElementById('trained-models-list');
            const countElement = document.getElementById('trained-models-count');

            countElement.textContent = data.models.length;

            if (data.models.length === 0) {
                container.innerHTML = `
                    <div class="no-data">
                        <p>学習済みモデルがありません</p>
                        <small>機器の学習データを作成してモデルを訓練してください</small>
                    </div>
                `;
                return;
            }

            container.innerHTML = data.models.map(model => `
                <div class="model-card">
                    <div class="model-name">
                        <i class="fas fa-brain"></i>
                        ${model.equipment_name}
                    </div>
                    <div class="model-stats">
                        <span>学習サンプル: ${model.training_samples}個</span>
                        <span>作成日: ${new Date(model.created_at).toLocaleDateString('ja-JP')}</span>
                    </div>
                    <div class="model-actions">
                        <button class="btn btn-primary btn-small" onclick="mlSystem.testModel(${model.equipment_id})">
                            <i class="fas fa-play"></i>
                            テスト
                        </button>
                    </div>
                </div>
            `).join('');

        } catch (error) {
            console.error('Error loading trained models:', error);
        }
    }

    async testModel(equipmentId) {
        // This would integrate with the main visualization system
        // For now, just show a notification
        this.showNotification(`機器ID ${equipmentId} のモデルをテストします`, 'info');
    }

    showNotification(message, type = 'info') {
        // Use the existing notification system
        if (window.system && window.system.showNotification) {
            window.system.showNotification(message, type);
        } else {
            alert(message);
        }
    }
}

// Initialize ML system when page loads
document.addEventListener('DOMContentLoaded', function() {
    if (typeof window.system !== 'undefined') {
        window.mlSystem = new MLSystem(window.system.apiBaseUrl);
    }
});