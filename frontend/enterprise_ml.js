/**
 * Enterprise Machine Learning System
 * Professional-grade batch training and high-precision detection
 */

class EnterpriseMLSystem {
    constructor(apiBaseUrl) {
        this.apiBaseUrl = apiBaseUrl;
        this.batchImages = [];
        this.batchAnnotations = [];
        this.trainedEquipment = [];
        this.trainingInProgress = false;

        this.init();
    }

    init() {
        console.log('🚀 Initializing Enterprise ML System');
        this.setupEventListeners();
        this.loadTrainedEquipment();
        this.updateEquipmentSelects();
    }

    setupEventListeners() {
        // File upload areas
        this.setupFileUpload('images-upload-area', 'batch-images', 'image/*', this.handleImagesSelected.bind(this));
        this.setupFileUpload('annotations-upload-area', 'batch-annotations', '.json', this.handleAnnotationsSelected.bind(this));

        // Batch upload controls
        document.getElementById('enterprise-batch-upload').addEventListener('click', this.uploadBatch.bind(this));
        document.getElementById('clear-batch').addEventListener('click', this.clearBatch.bind(this));

        // Training controls
        document.getElementById('start-enterprise-training').addEventListener('click', this.startTraining.bind(this));

        // Hyperparameter controls
        const epochsSlider = document.getElementById('enterprise-epochs');
        const epochsValue = document.getElementById('epochs-value');
        epochsSlider.addEventListener('input', (e) => {
            epochsValue.textContent = e.target.value;
        });

        // Equipment selection
        document.getElementById('enterprise-equipment-select').addEventListener('change', this.updateTrainingButton.bind(this));
    }

    setupFileUpload(areaId, inputId, accept, handler) {
        const area = document.getElementById(areaId);
        const input = document.getElementById(inputId);

        // Click to upload
        area.addEventListener('click', () => input.click());
        input.addEventListener('change', handler);

        // Drag and drop
        area.addEventListener('dragenter', this.handleDragEnter.bind(this));
        area.addEventListener('dragover', this.handleDragOver.bind(this));
        area.addEventListener('dragleave', this.handleDragLeave.bind(this));
        area.addEventListener('drop', (e) => this.handleDrop(e, handler));
    }

    handleDragEnter(e) {
        e.preventDefault();
        e.currentTarget.classList.add('drag-over');
    }

    handleDragOver(e) {
        e.preventDefault();
    }

    handleDragLeave(e) {
        e.preventDefault();
        if (!e.currentTarget.contains(e.relatedTarget)) {
            e.currentTarget.classList.remove('drag-over');
        }
    }

    handleDrop(e, handler) {
        e.preventDefault();
        e.currentTarget.classList.remove('drag-over');

        const files = Array.from(e.dataTransfer.files);

        // Create a mock event for the handler
        const mockEvent = {
            target: { files: files }
        };

        handler(mockEvent);
    }

    handleImagesSelected(event) {
        const files = Array.from(event.target.files);
        this.batchImages = files;

        console.log(`📸 Selected ${files.length} images for batch upload`);
        this.displayImagePreviews(files);
        this.updateBatchUploadButton();
    }

    handleAnnotationsSelected(event) {
        const files = Array.from(event.target.files);

        // Validate JSON files
        const validAnnotations = [];
        const promises = files.map(file => {
            return new Promise((resolve) => {
                const reader = new FileReader();
                reader.onload = (e) => {
                    try {
                        const annotation = JSON.parse(e.target.result);
                        if (this.validateAnnotation(annotation)) {
                            validAnnotations.push({
                                file: file,
                                data: annotation
                            });
                        } else {
                            console.warn(`❌ Invalid annotation format in ${file.name}`);
                        }
                        resolve();
                    } catch (error) {
                        console.error(`❌ Invalid JSON in ${file.name}:`, error);
                        resolve();
                    }
                };
                reader.readAsText(file);
            });
        });

        Promise.all(promises).then(() => {
            this.batchAnnotations = validAnnotations;
            console.log(`📋 Loaded ${validAnnotations.length} valid annotations`);
            this.displayAnnotationPreviews(validAnnotations);
            this.updateBatchUploadButton();
        });
    }

    validateAnnotation(annotation) {
        // Validate annotation format
        if (!annotation.bbox || !Array.isArray(annotation.bbox) || annotation.bbox.length !== 4) {
            return false;
        }

        const [x, y, w, h] = annotation.bbox;
        return typeof x === 'number' && typeof y === 'number' &&
               typeof w === 'number' && typeof h === 'number' &&
               x >= 0 && y >= 0 && w > 0 && h > 0;
    }

    displayImagePreviews(files) {
        const previewArea = document.getElementById('images-preview');
        previewArea.innerHTML = '';

        files.forEach((file, index) => {
            const item = document.createElement('div');
            item.className = 'file-preview-item';
            item.innerHTML = `
                <div class="file-info">
                    <i class="fas fa-image file-icon"></i>
                    <div>
                        <div class="file-name">${file.name}</div>
                        <div class="file-size">${this.formatFileSize(file.size)}</div>
                    </div>
                </div>
                <i class="fas fa-times file-remove" onclick="enterpriseML.removeImage(${index})"></i>
            `;
            previewArea.appendChild(item);
        });
    }

    displayAnnotationPreviews(annotations) {
        const previewArea = document.getElementById('annotations-preview');
        previewArea.innerHTML = '';

        annotations.forEach((annotation, index) => {
            const item = document.createElement('div');
            item.className = 'file-preview-item';
            const bbox = annotation.data.bbox;
            item.innerHTML = `
                <div class="file-info">
                    <i class="fas fa-file-code file-icon"></i>
                    <div>
                        <div class="file-name">${annotation.file.name}</div>
                        <div class="file-size">BBox: [${bbox.join(', ')}]</div>
                    </div>
                </div>
                <i class="fas fa-times file-remove" onclick="enterpriseML.removeAnnotation(${index})"></i>
            `;
            previewArea.appendChild(item);
        });
    }

    removeImage(index) {
        this.batchImages.splice(index, 1);
        this.displayImagePreviews(this.batchImages);
        this.updateBatchUploadButton();
    }

    removeAnnotation(index) {
        this.batchAnnotations.splice(index, 1);
        this.displayAnnotationPreviews(this.batchAnnotations);
        this.updateBatchUploadButton();
    }

    updateBatchUploadButton() {
        const button = document.getElementById('enterprise-batch-upload');
        const equipmentName = document.getElementById('enterprise-equipment-name').value.trim();

        const canUpload = equipmentName &&
                         this.batchImages.length > 0 &&
                         this.batchAnnotations.length > 0 &&
                         this.batchImages.length === this.batchAnnotations.length;

        button.disabled = !canUpload;

        if (canUpload) {
            button.innerHTML = `
                <i class="fas fa-upload"></i>
                <span>Upload ${this.batchImages.length} Training Samples</span>
                <div class="btn-glow"></div>
            `;
        } else {
            button.innerHTML = `
                <i class="fas fa-upload"></i>
                <span>Upload Training Batch</span>
                <div class="btn-glow"></div>
            `;
        }
    }

    async uploadBatch() {
        const equipmentName = document.getElementById('enterprise-equipment-name').value.trim();

        if (!equipmentName || this.batchImages.length === 0 || this.batchAnnotations.length === 0) {
            this.showNotification('Please provide equipment name, images, and annotations', 'error');
            return;
        }

        if (this.batchImages.length !== this.batchAnnotations.length) {
            this.showNotification('Number of images must match number of annotations', 'error');
            return;
        }

        try {
            this.showUploadProgress(true);

            // Create FormData for file upload
            const formData = new FormData();
            formData.append('equipment_name', equipmentName);

            // Add images
            this.batchImages.forEach((image, index) => {
                formData.append('images', image);
            });

            // Create annotation files
            this.batchAnnotations.forEach((annotation, index) => {
                const blob = new Blob([JSON.stringify(annotation.data)], { type: 'application/json' });
                formData.append('annotations', blob, `annotation_${index}.json`);
            });

            const response = await fetch(`${this.apiBaseUrl}/enterprise/batch-upload`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.showNotification(`✅ Successfully uploaded ${result.processed_images} training samples for ${equipmentName}`, 'success');
                this.clearBatch();
                this.loadTrainedEquipment();
                this.updateEquipmentSelects();
            } else {
                throw new Error(result.error || 'Upload failed');
            }

        } catch (error) {
            console.error('Batch upload error:', error);
            this.showNotification(`Upload failed: ${error.message}`, 'error');
        } finally {
            this.showUploadProgress(false);
        }
    }

    clearBatch() {
        this.batchImages = [];
        this.batchAnnotations = [];
        document.getElementById('enterprise-equipment-name').value = '';
        document.getElementById('images-preview').innerHTML = '';
        document.getElementById('annotations-preview').innerHTML = '';
        this.updateBatchUploadButton();
    }

    showUploadProgress(show) {
        const progressDiv = document.getElementById('batch-upload-progress');
        const statusText = document.getElementById('upload-status-text');
        const percentage = document.getElementById('upload-percentage');
        const progressFill = document.getElementById('upload-progress-fill');

        if (show) {
            progressDiv.style.display = 'block';
            let progress = 0;

            const interval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress > 95) {
                    clearInterval(interval);
                    progress = 95;
                }

                percentage.textContent = `${Math.round(progress)}%`;
                progressFill.style.width = `${progress}%`;

                if (progress < 30) {
                    statusText.textContent = 'Validating files...';
                } else if (progress < 60) {
                    statusText.textContent = 'Uploading training data...';
                } else {
                    statusText.textContent = 'Processing batch...';
                }
            }, 200);

            // Complete after response
            setTimeout(() => {
                clearInterval(interval);
                percentage.textContent = '100%';
                progressFill.style.width = '100%';
                statusText.textContent = 'Upload completed!';

                setTimeout(() => {
                    progressDiv.style.display = 'none';
                    progressFill.style.width = '0%';
                }, 2000);
            }, 3000);

        } else {
            progressDiv.style.display = 'none';
            progressFill.style.width = '0%';
        }
    }

    async loadTrainedEquipment() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/enterprise/trained-equipment`);
            const result = await response.json();

            if (result.success) {
                this.trainedEquipment = result.trained_equipment;
                this.displayEquipmentRegistry(this.trainedEquipment);
                document.getElementById('total-trained-count').textContent = result.count;
            }
        } catch (error) {
            console.error('Error loading trained equipment:', error);
        }
    }

    displayEquipmentRegistry(equipment) {
        const registry = document.getElementById('enterprise-equipment-registry');

        if (equipment.length === 0) {
            registry.innerHTML = `
                <div class="loading-placeholder">
                    <i class="fas fa-robot"></i>
                    <p>No trained models yet</p>
                    <small>Upload training data and train models to see them here</small>
                </div>
            `;
            return;
        }

        registry.innerHTML = equipment.map(eq => `
            <div class="equipment-card-enterprise">
                <div class="equipment-header">
                    <div>
                        <div class="equipment-name">${eq.equipment_name}</div>
                        <span class="accuracy-badge ${this.getAccuracyClass(eq.validation_accuracy)}">
                            ${eq.accuracy_grade}
                        </span>
                    </div>
                </div>
                <div class="equipment-stats">
                    <div class="equipment-stat">
                        <span class="equipment-stat-value">${(eq.validation_accuracy * 100).toFixed(1)}%</span>
                        <span class="equipment-stat-label">Accuracy</span>
                    </div>
                    <div class="equipment-stat">
                        <span class="equipment-stat-value">${eq.training_samples}</span>
                        <span class="equipment-stat-label">Samples</span>
                    </div>
                </div>
                <div class="equipment-actions" style="margin-top: 16px;">
                    <button class="btn btn-primary btn-small" onclick="enterpriseML.testEquipment(${eq.equipment_id})">
                        <i class="fas fa-search"></i> Test Detection
                    </button>
                </div>
            </div>
        `).join('');
    }

    getAccuracyClass(accuracy) {
        if (accuracy >= 0.9) return 'excellent';
        if (accuracy >= 0.8) return 'good';
        if (accuracy >= 0.7) return 'fair';
        return 'needs-improvement';
    }

    updateEquipmentSelects() {
        // Get equipment with sufficient training data for training
        fetch(`${this.apiBaseUrl}/equipment`)
            .then(response => response.json())
            .then(equipment => {
                const select = document.getElementById('enterprise-equipment-select');
                select.innerHTML = '<option value="">Select equipment with training data...</option>';

                // Add equipment options (in real implementation, filter by training data count)
                equipment.forEach(eq => {
                    const option = new Option(eq.name, eq.id);
                    select.appendChild(option);
                });
            })
            .catch(error => console.error('Error loading equipment:', error));
    }

    updateTrainingButton() {
        const select = document.getElementById('enterprise-equipment-select');
        const button = document.getElementById('start-enterprise-training');

        button.disabled = !select.value || this.trainingInProgress;
    }

    async startTraining() {
        const equipmentId = document.getElementById('enterprise-equipment-select').value;
        const epochs = document.getElementById('enterprise-epochs').value;
        const batchSize = document.getElementById('enterprise-batch-size').value;
        const modelSize = document.getElementById('enterprise-model-size').value;

        if (!equipmentId) {
            this.showNotification('Please select equipment for training', 'error');
            return;
        }

        try {
            this.trainingInProgress = true;
            this.showTrainingMonitor(true);
            this.updateTrainingButton();

            const hyperparameters = {
                epochs: parseInt(epochs),
                batch_size: parseInt(batchSize),
                model_size: modelSize
            };

            console.log('🚀 Starting enterprise training with hyperparameters:', hyperparameters);

            const startTime = Date.now();
            this.updateTrainingProgress(0, epochs, startTime);

            const response = await fetch(`${this.apiBaseUrl}/enterprise/train-model`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    equipment_id: parseInt(equipmentId),
                    hyperparameters: hyperparameters
                })
            });

            const result = await response.json();

            if (result.success) {
                // Simulate training progress
                this.simulateTrainingProgress(epochs, startTime);

                // Wait for training completion
                setTimeout(() => {
                    this.showNotification(`🎉 Enterprise model training completed! Accuracy: ${(result.validation_accuracy * 100).toFixed(1)}%`, 'success');
                    this.showTrainingMonitor(false);
                    this.loadTrainedEquipment();
                    this.trainingInProgress = false;
                    this.updateTrainingButton();
                }, parseInt(epochs) * 100); // Simulate training time

            } else {
                throw new Error(result.error || 'Training failed');
            }

        } catch (error) {
            console.error('Training error:', error);
            this.showNotification(`Training failed: ${error.message}`, 'error');
            this.showTrainingMonitor(false);
            this.trainingInProgress = false;
            this.updateTrainingButton();
        }
    }

    showTrainingMonitor(show) {
        const monitor = document.getElementById('training-monitor');
        monitor.style.display = show ? 'block' : 'none';
    }

    simulateTrainingProgress(totalEpochs, startTime) {
        let currentEpoch = 0;
        const interval = setInterval(() => {
            currentEpoch += Math.ceil(Math.random() * 3);
            if (currentEpoch >= totalEpochs) {
                currentEpoch = totalEpochs;
                clearInterval(interval);
            }

            this.updateTrainingProgress(currentEpoch, totalEpochs, startTime);
        }, 200);
    }

    updateTrainingProgress(currentEpoch, totalEpochs, startTime) {
        const progress = (currentEpoch / totalEpochs) * 100;
        const elapsed = Date.now() - startTime;

        document.getElementById('training-epoch-progress').textContent = `${currentEpoch}/${totalEpochs}`;
        document.getElementById('training-current-accuracy').textContent = (0.3 + (progress / 100) * 0.6).toFixed(3);
        document.getElementById('training-time-elapsed').textContent = this.formatTime(elapsed);
        document.getElementById('enterprise-training-fill').style.width = `${progress}%`;
    }

    testEquipment(equipmentId) {
        // Switch to visualization tab and set up for testing
        this.showNotification(`🧪 Switching to visualization for testing equipment ID: ${equipmentId}`, 'info');

        // Switch tab
        const visualizationTab = document.querySelector('.tab[data-tab="visualization"]');
        if (visualizationTab) {
            visualizationTab.click();
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    formatTime(ms) {
        const seconds = Math.floor(ms / 1000);
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;

        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    showNotification(message, type = 'info') {
        // Use existing notification system
        if (window.system && window.system.showNotification) {
            window.system.showNotification(message, type);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }
}

// Initialize enterprise ML system when page loads
document.addEventListener('DOMContentLoaded', function() {
    if (typeof window.system !== 'undefined') {
        window.enterpriseML = new EnterpriseMLSystem(window.system.apiBaseUrl);
        console.log('🚀 Enterprise ML System initialized');
    }
});