class SimpleMLSystem {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000/api';
        this.currentDiagram = null;
        this.trainedEquipment = [];
        this.trainingSets = [];  // Store training sets
        this.setCounter = 0;     // Counter for unique set IDs
        this.isDetecting = false; // Flag to prevent duplicate detection requests

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadTrainedEquipment();
        this.loadUploadedDiagrams();
        this.updateSetCounters();

        // Monitor equipment name changes
        document.getElementById('equipment-name').addEventListener('input', () => {
            this.updateSetCounters();
        });
    }

    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Training set management
        document.getElementById('add-training-set').addEventListener('click', () => {
            this.addTrainingSet();
        });

        document.getElementById('clear-all-sets').addEventListener('click', () => {
            this.clearAllSets();
        });

        document.getElementById('upload-training-data').addEventListener('click', () => {
            this.uploadTrainingData();
        });

        // Diagram upload
        document.getElementById('diagram-file').addEventListener('change', (e) => {
            this.handleDiagramUpload(e.target.files[0]);
        });

        document.getElementById('upload-diagram').addEventListener('click', () => {
            this.uploadDiagram();
        });

        // Detection
        document.getElementById('detect-equipment').addEventListener('click', () => {
            this.detectEquipment();
        });

        // Clear functions
        document.getElementById('clear-trained-equipment').addEventListener('click', () => {
            this.clearTrainedEquipment();
        });

        document.getElementById('clear-diagrams').addEventListener('click', () => {
            this.clearDiagrams();
        });

        // Drag and drop for diagram
        this.setupDragDrop('diagram-drop', 'diagram-file');
    }

    switchTab(tabName) {
        // Hide all tabs
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });

        // Show selected tab
        document.getElementById(tabName).classList.add('active');
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        if (tabName === 'detection') {
            this.loadTrainedEquipment();
        } else if (tabName === 'training') {
            this.updateSetCounters();
        }
    }

    setupDragDrop(dropAreaId, inputId) {
        const dropArea = document.getElementById(dropAreaId);
        const input = document.getElementById(inputId);

        dropArea.addEventListener('click', () => input.click());

        dropArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropArea.classList.add('dragover');
        });

        dropArea.addEventListener('dragleave', () => {
            dropArea.classList.remove('dragover');
        });

        dropArea.addEventListener('drop', (e) => {
            e.preventDefault();
            dropArea.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (inputId === 'diagram-file') {
                this.handleDiagramUpload(files[0]);
            } else {
                this.handleImageSelection(files, inputId === 'equipment-images' ? 'images' : 'highlights');
            }
        });
    }

    addTrainingSet() {
        const setId = ++this.setCounter;
        const trainingSet = {
            id: setId,
            equipmentImages: [],
            highlightImages: []
        };

        this.trainingSets.push(trainingSet);
        this.renderTrainingSets();
        this.updateSetCounters();
    }

    removeTrainingSet(setId) {
        const index = this.trainingSets.findIndex(set => set.id === setId);
        if (index > -1) {
            this.trainingSets.splice(index, 1);
            this.renderTrainingSets();
            this.updateSetCounters();
        }
    }

    clearAllSets() {
        this.trainingSets = [];
        this.setCounter = 0;
        this.renderTrainingSets();
        this.updateSetCounters();
    }

    renderTrainingSets() {
        const container = document.getElementById('training-sets');

        container.innerHTML = this.trainingSets.map(set => `
            <div class="training-set" data-set-id="${set.id}">
                <div class="training-set-header">
                    <div class="set-title">学習セット ${set.id}</div>
                    <button class="set-remove-btn" onclick="system.removeTrainingSet(${set.id})">削除</button>
                </div>
                <div class="set-uploads">
                    <div class="set-upload-area ${set.equipmentImages.length > 0 ? 'has-file' : ''}" onclick="document.getElementById('equipment-images-${set.id}').click()">
                        <input type="file" id="equipment-images-${set.id}" multiple accept="image/*">
                        <p>${set.equipmentImages.length > 0 ? `機器画像 (${set.equipmentImages.length}枚)` : '機器画像をアップロード'}</p>
                        <div class="set-preview" id="equipment-preview-${set.id}"></div>
                    </div>
                    <div class="set-upload-area ${set.highlightImages.length > 0 ? 'has-file' : ''}" onclick="document.getElementById('highlight-images-${set.id}').click()">
                        <input type="file" id="highlight-images-${set.id}" multiple accept="image/*">
                        <p>${set.highlightImages.length > 0 ? `ハイライト画像 (${set.highlightImages.length}枚)` : 'ハイライト画像をアップロード'}</p>
                        <div class="set-preview" id="highlight-preview-${set.id}"></div>
                    </div>
                </div>
            </div>
        `).join('');

        // Add event listeners for file inputs
        this.trainingSets.forEach(set => {
            const equipmentInput = document.getElementById(`equipment-images-${set.id}`);
            const highlightInput = document.getElementById(`highlight-images-${set.id}`);

            if (equipmentInput) {
                equipmentInput.addEventListener('change', (e) => {
                    this.handleSetImages(set.id, e.target.files, 'equipment');
                });
            }

            if (highlightInput) {
                highlightInput.addEventListener('change', (e) => {
                    this.handleSetImages(set.id, e.target.files, 'highlight');
                });
            }
        });
    }

    handleSetImages(setId, files, type) {
        const set = this.trainingSets.find(s => s.id === setId);
        if (!set) return;

        const targetArray = type === 'equipment' ? set.equipmentImages : set.highlightImages;
        const previewContainer = document.getElementById(`${type === 'equipment' ? 'equipment' : 'highlight'}-preview-${setId}`);

        // Clear existing files and preview
        targetArray.length = 0;
        previewContainer.innerHTML = '';

        Array.from(files).forEach(file => {
            if (file.type.startsWith('image/')) {
                targetArray.push(file);
                this.addSetImagePreview(file, previewContainer);
            }
        });

        this.renderTrainingSets();
        this.updateSetCounters();
    }

    addSetImagePreview(file, container) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.alt = file.name;
            container.appendChild(img);
        };
        reader.readAsDataURL(file);
    }

    updateSetCounters() {
        const setCounterSpan = document.querySelector('.set-counter');
        const totalSetsSpan = document.getElementById('total-sets');
        const uploadButton = document.getElementById('upload-training-data');
        const equipmentName = document.getElementById('equipment-name').value.trim();

        const validSets = this.trainingSets.filter(set =>
            set.equipmentImages.length > 0 && set.highlightImages.length > 0
        ).length;

        setCounterSpan.textContent = `(${this.trainingSets.length}セット)`;
        totalSetsSpan.textContent = validSets;

        uploadButton.disabled = !equipmentName || validSets === 0;
    }

    async uploadTrainingData() {
        const equipmentName = document.getElementById('equipment-name').value.trim();

        if (!equipmentName) {
            this.showNotification('機器名を入力してください', 'error');
            return;
        }

        const validSets = this.trainingSets.filter(set =>
            set.equipmentImages.length > 0 && set.highlightImages.length > 0
        );

        if (validSets.length === 0) {
            this.showNotification('有効な学習セット（機器画像とハイライト画像の両方が必要）を作成してください', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('equipment_name', equipmentName);
        formData.append('set_count', validSets.length.toString());

        validSets.forEach((set, setIndex) => {
            set.equipmentImages.forEach((file, imageIndex) => {
                formData.append(`set_${setIndex}_equipment_${imageIndex}`, file);
            });
            set.highlightImages.forEach((file, imageIndex) => {
                formData.append(`set_${setIndex}_highlight_${imageIndex}`, file);
            });
            formData.append(`set_${setIndex}_equipment_count`, set.equipmentImages.length.toString());
            formData.append(`set_${setIndex}_highlight_count`, set.highlightImages.length.toString());
        });

        try {
            this.showProgress(true, `${validSets.length}セットの学習データをアップロード中...`);

            const response = await fetch(`${this.apiBaseUrl}/training-sets`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                this.showNotification(`${equipmentName}の学習が完了しました！（${validSets.length}セット使用）`, 'success');
                this.clearTrainingForm();
                this.loadTrainedEquipment();
            } else {
                const error = await response.json();
                this.showNotification('学習失敗: ' + (error.error || 'エラーが発生しました'), 'error');
            }
        } catch (error) {
            this.showNotification('ネットワークエラーが発生しました', 'error');
            console.error('Training error:', error);
        } finally {
            this.showProgress(false);
        }
    }

    clearTrainingForm() {
        document.getElementById('equipment-name').value = '';
        this.clearAllSets();
    }

    handleDiagramUpload(file) {
        if (!file || !file.type.startsWith('image/')) {
            this.showNotification('画像ファイルを選択してください', 'error');
            return;
        }

        // Preview the diagram
        const reader = new FileReader();
        reader.onload = (e) => {
            const canvas = document.getElementById('diagram-canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                this.currentDiagram = { file, path: e.target.result };
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    async uploadDiagram() {
        if (!this.currentDiagram) {
            this.showNotification('図面を選択してください', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('diagram', this.currentDiagram.file);

        try {
            const response = await fetch(`${this.apiBaseUrl}/diagrams`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                this.showNotification('図面がアップロードされました', 'success');
                this.loadUploadedDiagrams();
            } else {
                this.showNotification('図面のアップロードに失敗しました', 'error');
            }
        } catch (error) {
            this.showNotification('ネットワークエラーが発生しました', 'error');
        }
    }

    async loadTrainedEquipment() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/trained-equipment`);
            if (response.ok) {
                const data = await response.json();
                this.trainedEquipment = data.equipment || [];
                this.renderTrainedEquipment();
            }
        } catch (error) {
            console.error('Failed to load trained equipment:', error);
        }
    }

    renderTrainedEquipment() {
        const container = document.getElementById('trained-equipment-list');

        if (this.trainedEquipment.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <p>学習済み機器がありません</p>
                    <small>AI学習タブで機器を学習させてください</small>
                </div>
            `;
            return;
        }

        container.innerHTML = this.trainedEquipment.map(equipment => `
            <div class="equipment-item">
                <input type="checkbox" id="eq-${equipment.id}" value="${equipment.id}">
                <label for="eq-${equipment.id}">${equipment.name}</label>
                <span class="accuracy-badge">${equipment.accuracy}%</span>
            </div>
        `).join('');
    }

    async loadUploadedDiagrams() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/diagrams`);
            if (response.ok) {
                const diagrams = await response.json();
                this.renderUploadedDiagrams(diagrams);
            }
        } catch (error) {
            console.error('Failed to load diagrams:', error);
        }
    }

    renderUploadedDiagrams(diagrams) {
        const container = document.getElementById('diagram-list');

        if (diagrams.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <p>アップロード済み図面がありません</p>
                </div>
            `;
            return;
        }

        container.innerHTML = diagrams.map(diagram => `
            <div class="diagram-item ${diagram.path === this.currentDiagram?.path ? 'selected' : ''}" data-diagram-id="${diagram.id}">
                <div class="diagram-controls">
                    <button class="diagram-select-btn ${diagram.path === this.currentDiagram?.path ? 'active' : ''}" onclick="system.selectDiagram('${diagram.path}', ${diagram.id})">
                        ${diagram.path === this.currentDiagram?.path ? '選択中' : '選択'}
                    </button>
                    <button class="diagram-delete-btn" onclick="system.deleteDiagram(${diagram.id})" title="削除">×</button>
                </div>
                <img src="${diagram.path}" alt="${diagram.name}" onclick="system.selectDiagram('${diagram.path}', ${diagram.id})">
                <p>${diagram.name}</p>
            </div>
        `).join('');
    }

    selectDiagram(path, diagramId) {
        const canvas = document.getElementById('diagram-canvas');
        const ctx = canvas.getContext('2d');
        const overlay = document.getElementById('highlight-overlay');

        // Clear previous highlights
        this.clearHighlights();

        const img = new Image();
        img.onload = () => {
            // Set canvas size to match image
            canvas.width = img.width;
            canvas.height = img.height;

            // Position the overlay to match canvas
            overlay.style.width = canvas.width + 'px';
            overlay.style.height = canvas.height + 'px';

            ctx.drawImage(img, 0, 0);
            this.currentDiagram = { path, id: diagramId, width: img.width, height: img.height };

            // Re-render diagrams to show selection
            this.loadUploadedDiagrams();
            this.switchTab('detection');
        };
        img.src = path;
    }

    async deleteDiagram(diagramId) {
        if (!confirm('この図面を削除しますか？')) {
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/diagrams/${diagramId}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                this.showNotification('図面を削除しました', 'success');

                // If deleted diagram was currently selected, clear it
                if (this.currentDiagram && this.currentDiagram.id === diagramId) {
                    this.currentDiagram = null;
                    const canvas = document.getElementById('diagram-canvas');
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    this.clearHighlights();
                }

                this.loadUploadedDiagrams();
            } else {
                const error = await response.json();
                this.showNotification('削除失敗: ' + (error.error || 'エラーが発生しました'), 'error');
            }
        } catch (error) {
            this.showNotification('ネットワークエラーが発生しました', 'error');
            console.error('Delete diagram error:', error);
        }
    }

    async detectEquipment() {
        // Prevent duplicate requests
        if (this.isDetecting) {
            console.log('Detection already in progress, ignoring duplicate request');
            return;
        }

        const selectedEquipment = Array.from(document.querySelectorAll('#trained-equipment-list input[type="checkbox"]:checked'))
            .map(cb => parseInt(cb.value));

        if (selectedEquipment.length === 0) {
            this.showNotification('検出する機器を選択してください', 'warning');
            return;
        }

        if (!this.currentDiagram) {
            this.showNotification('図面を選択してください', 'warning');
            return;
        }

        try {
            // Set detection flag to prevent duplicates
            this.isDetecting = true;
            this.showProgress(true, '機器を検出中...');
            this.clearHighlights();

            const response = await fetch(`${this.apiBaseUrl}/simple-detect`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    equipment_ids: selectedEquipment,
                    diagram_path: this.currentDiagram.path
                })
            });

            if (response.ok) {
                const result = await response.json();
                if (result.detections && result.detections.length > 0) {
                    this.displayDetections(result.detections);
                    this.showNotification(`${result.detections.length}個の機器を検出しました`, 'success');
                } else {
                    this.showNotification('機器が検出されませんでした', 'warning');
                }
            } else {
                this.showNotification('検出に失敗しました', 'error');
            }
        } catch (error) {
            this.showNotification('ネットワークエラーが発生しました', 'error');
            console.error('Detection error:', error);
        } finally {
            // Reset detection flag to allow future requests
            this.isDetecting = false;
            this.showProgress(false);
        }
    }

    displayDetections(detections) {
        const overlay = document.getElementById('highlight-overlay');
        const canvas = document.getElementById('diagram-canvas');

        // Get canvas and container positions
        const canvasRect = canvas.getBoundingClientRect();
        const overlayRect = overlay.getBoundingClientRect();

        detections.forEach((detection, index) => {
            // Calculate scaled coordinates based on canvas display size vs actual size
            const scaleX = canvas.width / canvasRect.width;
            const scaleY = canvas.height / canvasRect.height;

            // Convert detection coordinates to overlay coordinates
            const actualX = detection.x / scaleX;
            const actualY = detection.y / scaleY;
            const actualWidth = detection.width / scaleX;
            const actualHeight = detection.height / scaleY;

            const highlight = document.createElement('div');
            highlight.className = 'detection-highlight';
            highlight.style.position = 'absolute';
            highlight.style.left = `${actualX}px`;
            highlight.style.top = `${actualY}px`;
            highlight.style.width = `${actualWidth}px`;
            highlight.style.height = `${actualHeight}px`;
            highlight.style.border = '3px solid #ff4444';
            highlight.style.background = 'rgba(255, 68, 68, 0.2)';
            highlight.style.borderRadius = '4px';
            highlight.style.pointerEvents = 'none';
            highlight.style.zIndex = '10';
            highlight.style.boxSizing = 'border-box';

            // Add label
            const label = document.createElement('div');
            label.textContent = `${detection.equipment_name} (${(detection.confidence * 100).toFixed(1)}%)`;
            label.style.position = 'absolute';
            label.style.top = '-35px';
            label.style.left = '0';
            label.style.background = '#ff4444';
            label.style.color = 'white';
            label.style.padding = '4px 8px';
            label.style.borderRadius = '4px';
            label.style.fontSize = '12px';
            label.style.fontWeight = 'bold';
            label.style.whiteSpace = 'nowrap';
            label.style.boxShadow = '0 2px 4px rgba(0,0,0,0.3)';

            // Adjust label position if it goes out of bounds
            if (actualY < 35) {
                label.style.top = `${actualHeight + 5}px`;
            }
            if (actualX + 200 > canvas.offsetWidth) {
                label.style.left = 'auto';
                label.style.right = '0';
            }

            highlight.appendChild(label);
            overlay.appendChild(highlight);
        });
    }

    clearHighlights() {
        const overlay = document.getElementById('highlight-overlay');
        overlay.innerHTML = '';
    }

    showProgress(show, text = '処理中...') {
        const progressDiv = document.getElementById('training-progress');
        const progressText = document.getElementById('progress-text');
        const button = document.getElementById('upload-training-data');

        if (show) {
            progressDiv.style.display = 'block';
            progressText.textContent = text;
            button.disabled = true;
            this.animateProgress();
        } else {
            progressDiv.style.display = 'none';
            button.disabled = false;
            document.getElementById('progress-fill').style.width = '0%';
        }
    }

    animateProgress() {
        const fill = document.getElementById('progress-fill');
        let width = 0;
        const interval = setInterval(() => {
            width += Math.random() * 10;
            if (width >= 90) {
                clearInterval(interval);
                width = 90;
            }
            fill.style.width = width + '%';
        }, 200);
    }

    showNotification(message, type = 'info') {
        const notification = document.getElementById('notification');
        notification.textContent = message;
        notification.className = `notification ${type} show`;

        setTimeout(() => {
            notification.classList.remove('show');
        }, 4000);
    }

    async clearTrainedEquipment() {
        if (!confirm('すべての学習済み機器データを削除しますか？この操作は取り消せません。')) {
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/clear-trained-equipment`, {
                method: 'DELETE'
            });

            if (response.ok) {
                this.showNotification('学習済み機器データをクリアしました', 'success');
                this.loadTrainedEquipment();
            } else {
                const error = await response.json();
                this.showNotification('クリア失敗: ' + (error.error || 'エラーが発生しました'), 'error');
            }
        } catch (error) {
            this.showNotification('ネットワークエラーが発生しました', 'error');
            console.error('Clear equipment error:', error);
        }
    }

    async clearDiagrams() {
        if (!confirm('すべての図面を削除しますか？この操作は取り消せません。')) {
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/clear-diagrams`, {
                method: 'DELETE'
            });

            if (response.ok) {
                this.showNotification('図面をクリアしました', 'success');
                this.loadUploadedDiagrams();
                this.currentDiagram = null;
                const canvas = document.getElementById('diagram-canvas');
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                this.clearHighlights();
            } else {
                const error = await response.json();
                this.showNotification('クリア失敗: ' + (error.error || 'エラーが発生しました'), 'error');
            }
        } catch (error) {
            this.showNotification('ネットワークエラーが発生しました', 'error');
            console.error('Clear diagrams error:', error);
        }
    }
}

// Initialize the system
const system = new SimpleMLSystem();