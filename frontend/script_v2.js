/**
 * Advanced Technical Drawing Digitization System
 * Professional Frontend Implementation with YOLOv8 Integration
 */

class AdvancedDiagramSystem {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000/api';
        this.currentDiagram = null;
        this.coordinateMode = false;
        this.equipmentList = [];
        this.diagramList = [];
        this.detectionResults = [];
        this.isProcessing = false;
        this.confidenceThreshold = 0.5;

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadEquipment();
        this.loadDiagrams();
        this.checkSystemHealth();
        this.initializeAdvancedFeatures();
    }

    async checkSystemHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.showNotification(
                    `システム正常動作中 (Device: ${data.device}, YOLO: ${data.yolo_loaded ? '✓' : '✗'})`, 
                    'success'
                );
            } else {
                this.showNotification('システムに問題があります', 'warning');
            }
        } catch (error) {
            this.showNotification('サーバーに接続できません', 'error');
            console.error('Health check failed:', error);
        }
    }

    initializeAdvancedFeatures() {
        // 信頼度スライダーの初期化
        const confidenceSlider = document.createElement('div');
        confidenceSlider.className = 'confidence-slider-container';
        confidenceSlider.innerHTML = `
            <label>検出信頼度閾値: <span id="confidence-value">50%</span></label>
            <input type="range" id="confidence-slider" min="0" max="100" value="50">
        `;
        
        const visualizationTab = document.getElementById('visualization');
        const controlsArea = visualizationTab.querySelector('.diagram-controls');
        if (controlsArea) {
            controlsArea.appendChild(confidenceSlider);
            
            document.getElementById('confidence-slider').addEventListener('input', (e) => {
                this.confidenceThreshold = e.target.value / 100;
                document.getElementById('confidence-value').textContent = `${e.target.value}%`;
                
                // 既存の結果を再フィルタリング
                if (this.detectionResults.length > 0) {
                    this.displayFilteredResults();
                }
            });
        }

        // リアルタイム進捗表示の初期化
        this.initializeProgressIndicator();
    }

    initializeProgressIndicator() {
        const progressContainer = document.createElement('div');
        progressContainer.id = 'progress-container';
        progressContainer.className = 'progress-container hidden';
        progressContainer.innerHTML = `
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill"></div>
            </div>
            <div class="progress-text" id="progress-text">処理中...</div>
        `;
        document.body.appendChild(progressContainer);
    }

    setupEventListeners() {
        // タブ切り替え
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                e.preventDefault();
                this.switchTab(tab.dataset.tab);
            });
        });

        // 機器登録
        document.getElementById('equipment-images').addEventListener('change', this.handleImagePreview.bind(this));
        document.getElementById('register-equipment').addEventListener('click', this.registerEquipment.bind(this));

        // 図面アップロード
        document.getElementById('upload-diagram').addEventListener('click', this.uploadDiagram.bind(this));

        // 可視化コントロール
        document.getElementById('toggle-coordinates').addEventListener('click', this.toggleCoordinates.bind(this));
        document.getElementById('clear-coordinates').addEventListener('click', this.clearCoordinates.bind(this));
        document.getElementById('highlight-selected').addEventListener('click', this.highlightSelectedEquipment.bind(this));

        // Canvas インタラクション
        const canvas = document.getElementById('diagram-canvas');
        canvas.addEventListener('click', this.handleCanvasClick.bind(this));
        canvas.addEventListener('mousemove', this.handleCanvasMouseMove.bind(this));

        // キーボードショートカット
        this.setupKeyboardShortcuts();
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl+D: 検出実行
            if (e.ctrlKey && e.key === 'd') {
                e.preventDefault();
                this.highlightSelectedEquipment();
            }
            // Ctrl+C: クリア
            if (e.ctrlKey && e.key === 'c') {
                e.preventDefault();
                this.clearCoordinates();
            }
            // Escape: 処理キャンセル
            if (e.key === 'Escape' && this.isProcessing) {
                this.cancelProcessing();
            }
        });
    }

    switchTab(tabId) {
        document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

        document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
        document.getElementById(tabId).classList.add('active');

        if (tabId === 'visualization') {
            this.loadEquipmentForSelection();
        }
    }

    handleImagePreview(event) {
        const files = event.target.files;
        const preview = document.getElementById('image-preview');
        preview.innerHTML = '';

        Array.from(files).forEach((file, index) => {
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const container = document.createElement('div');
                    container.className = 'image-preview-item';
                    container.innerHTML = `
                        <img src="${e.target.result}" alt="Preview ${index + 1}">
                        <div class="image-info">
                            <span>${file.name}</span>
                            <span>${(file.size / 1024).toFixed(2)} KB</span>
                        </div>
                    `;
                    preview.appendChild(container);
                };
                reader.readAsDataURL(file);
            }
        });
    }

    async registerEquipment() {
        const name = document.getElementById('equipment-name').value.trim();
        const imagesInput = document.getElementById('equipment-images');

        if (!name) {
            this.showNotification('機器名を入力してください', 'error');
            return;
        }

        if (imagesInput.files.length === 0) {
            this.showNotification('機器画像をアップロードしてください', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('name', name);

        Array.from(imagesInput.files).forEach(file => {
            formData.append('images', file);
        });

        try {
            this.showProgress('機器を登録中...', 0);

            const response = await fetch(`${this.apiBaseUrl}/equipment`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                this.showProgress('登録完了！', 100);
                setTimeout(() => {
                    this.hideProgress();
                    this.showNotification('機器が正常に登録されました', 'success');
                    this.clearEquipmentForm();
                    this.loadEquipment();
                    
                    // カスタムモデルの訓練を提案
                    if (confirm(`${name}用のカスタムAIモデルを訓練しますか？（精度が向上します）`)) {
                        this.trainCustomModel(data.id);
                    }
                }, 1000);
            } else {
                this.hideProgress();
                this.showNotification(data.error || '登録に失敗しました', 'error');
            }
        } catch (error) {
            this.hideProgress();
            this.showNotification('ネットワークエラーが発生しました', 'error');
            console.error('Registration error:', error);
        }
    }

    async trainCustomModel(equipmentId) {
        try {
            this.showProgress('AIモデルを訓練中...', 0);
            
            const response = await fetch(`${this.apiBaseUrl}/train-equipment/${equipmentId}`, {
                method: 'POST'
            });

            const data = await response.json();

            if (response.ok) {
                this.showProgress('訓練完了！', 100);
                setTimeout(() => {
                    this.hideProgress();
                    this.showNotification('カスタムモデルの訓練が完了しました', 'success');
                    this.loadEquipment(); // リストを更新
                }, 1000);
            } else {
                this.hideProgress();
                this.showNotification(data.error || '訓練に失敗しました', 'error');
            }
        } catch (error) {
            this.hideProgress();
            this.showNotification('訓練エラーが発生しました', 'error');
            console.error('Training error:', error);
        }
    }

    async uploadDiagram() {
        const fileInput = document.getElementById('diagram-file');

        if (fileInput.files.length === 0) {
            this.showNotification('図面ファイルを選択してください', 'error');
            return;
        }

        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('diagram', file);

        try {
            this.showProgress(`${file.name}をアップロード中...`, 0);

            const response = await fetch(`${this.apiBaseUrl}/diagrams`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                this.showProgress('アップロード完了！', 100);
                setTimeout(() => {
                    this.hideProgress();
                    this.showNotification('図面が正常にアップロードされました', 'success');
                    fileInput.value = '';
                    this.loadDiagrams();
                }, 1000);
            } else {
                this.hideProgress();
                this.showNotification(data.error || 'アップロードに失敗しました', 'error');
            }
        } catch (error) {
            this.hideProgress();
            this.showNotification('ネットワークエラーが発生しました', 'error');
            console.error('Upload error:', error);
        }
    }

    async loadEquipment() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/equipment`);
            const data = await response.json();

            this.equipmentList = data;
            this.renderEquipmentList();
        } catch (error) {
            console.error('Failed to load equipment:', error);
        }
    }

    async loadDiagrams() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/diagrams`);
            const data = await response.json();

            this.diagramList = data;
            this.renderDiagramList();
        } catch (error) {
            console.error('Failed to load diagrams:', error);
        }
    }

    renderEquipmentList() {
        const container = document.getElementById('equipment-list');
        container.innerHTML = '';

        this.equipmentList.forEach(equipment => {
            const item = document.createElement('div');
            item.className = 'equipment-item';

            const imagesHtml = equipment.images.slice(0, 3).map(imagePath =>
                `<img src="http://localhost:8000/${imagePath}" alt="${equipment.name}">`
            ).join('');

            const modelStatus = equipment.has_custom_model 
                ? '<span class="model-badge">✓ カスタムモデル</span>' 
                : '<span class="model-badge-none">標準モデル</span>';

            item.innerHTML = `
                <div class="equipment-header">
                    <h4>${equipment.name}</h4>
                    ${modelStatus}
                </div>
                <p class="equipment-meta">登録日時: ${new Date(equipment.created_at).toLocaleString()}</p>
                <div class="images">${imagesHtml}</div>
                <div class="equipment-actions">
                    ${!equipment.has_custom_model ? 
                        `<button class="btn btn-primary btn-sm" onclick="system.trainCustomModel(${equipment.id})">
                            AIモデルを訓練
                        </button>` : ''}
                    <button class="btn btn-danger btn-sm" onclick="system.deleteEquipment(${equipment.id}, '${equipment.name}')">
                        削除
                    </button>
                </div>
            `;

            container.appendChild(item);
        });
    }

    renderDiagramList() {
        const container = document.getElementById('diagram-list');
        container.innerHTML = '';

        this.diagramList.forEach(diagram => {
            const item = document.createElement('div');
            item.className = 'diagram-item';

            const thumbnailUrl = `http://localhost:8000/${diagram.image_path}`;

            item.innerHTML = `
                <div class="diagram-thumbnail">
                    <img src="${thumbnailUrl}" alt="${diagram.name}" 
                         onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text x=%2250%%22 y=%2250%%22 text-anchor=%22middle%22 font-size=%2212%22>No Preview</text></svg>'">
                </div>
                <div class="diagram-info">
                    <h4>${diagram.name}</h4>
                    <p>アップロード: ${new Date(diagram.created_at).toLocaleString()}</p>
                </div>
                <div class="diagram-actions">
                    <button class="btn btn-primary" onclick="system.loadDiagramToCanvas('${diagram.image_path}')">
                        使用
                    </button>
                    <button class="btn btn-danger" onclick="system.deleteDiagram(${diagram.id})">
                        削除
                    </button>
                </div>
            `;

            container.appendChild(item);
        });
    }

    loadEquipmentForSelection() {
        const container = document.getElementById('equipment-selection-list');
        container.innerHTML = '';

        this.equipmentList.forEach(equipment => {
            const item = document.createElement('div');
            item.className = 'equipment-checkbox';

            const modelIndicator = equipment.has_custom_model ? '🚀' : '';

            item.innerHTML = `
                <input type="checkbox" id="eq-${equipment.id}" value="${equipment.id}">
                <label for="eq-${equipment.id}">${equipment.name} ${modelIndicator}</label>
            `;

            container.appendChild(item);
        });
    }

    loadDiagramToCanvas(imagePath) {
        const canvas = document.getElementById('diagram-canvas');
        const ctx = canvas.getContext('2d');

        const img = new Image();
        img.onload = () => {
            // Canvasサイズを画像に合わせる
            canvas.width = img.width;
            canvas.height = img.height;

            // 図面を描画
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);

            this.currentDiagram = {
                image: img,
                path: imagePath,
                originalWidth: img.width,
                originalHeight: img.height
            };

            this.showNotification('図面が読み込まれました', 'success');
            
            // 可視化タブに切り替え
            this.switchTab('visualization');
        };

        img.onerror = (error) => {
            console.error('Failed to load image:', error);
            this.showNotification('図面の読み込みに失敗しました', 'error');
        };

        img.src = `http://localhost:8000/${imagePath}`;
    }

    async highlightSelectedEquipment() {
        const selectedEquipment = this.getSelectedEquipment();

        if (selectedEquipment.length === 0) {
            this.showNotification('機器を選択してください', 'warning');
            return;
        }

        if (!this.currentDiagram) {
            this.showNotification('図面を読み込んでください', 'warning');
            return;
        }

        if (this.isProcessing) {
            this.showNotification('処理中です。しばらくお待ちください。', 'info');
            return;
        }

        try {
            this.isProcessing = true;
            this.showProgress('YOLOv8エンジンで検出中...', 10);
            this.clearHighlights();

            // API呼び出し
            const response = await fetch(`${this.apiBaseUrl}/match-equipment`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    diagram_path: this.currentDiagram.path,
                    equipment_ids: selectedEquipment.map(id => parseInt(id)),
                    use_cache: true
                })
            });

            const data = await response.json();

            if (response.ok) {
                this.showProgress('検出完了！結果を表示中...', 90);
                
                if (data.matches && data.matches.length > 0) {
                    this.detectionResults = data.matches;
                    this.displayFilteredResults();
                    
                    this.showProgress('完了！', 100);
                    setTimeout(() => {
                        this.hideProgress();
                        this.showNotification(
                            `${data.total_found}個の機器を検出しました (${data.processing_method})`,
                            'success'
                        );
                    }, 500);
                } else {
                    this.hideProgress();
                    this.showNotification('機器が検出されませんでした', 'warning');
                }
            } else {
                this.hideProgress();
                this.showNotification(data.error || 'ハイライト処理に失敗しました', 'error');
            }
        } catch (error) {
            this.hideProgress();
            this.showNotification('ネットワークエラーが発生しました', 'error');
            console.error('Highlighting error:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    displayFilteredResults() {
        this.clearHighlights();
        
        const filteredResults = this.detectionResults.filter(
            result => result.confidence >= this.confidenceThreshold
        );
        
        filteredResults.forEach(match => {
            this.addAdvancedHighlight(match);
        });
        
        // 統計情報を表示
        this.showDetectionStats(filteredResults);
    }

    showDetectionStats(results) {
        const statsContainer = document.getElementById('detection-stats');
        if (!statsContainer) {
            const container = document.createElement('div');
            container.id = 'detection-stats';
            container.className = 'detection-stats';
            document.querySelector('.visualization-layout').appendChild(container);
        }
        
        const stats = this.calculateStats(results);
        document.getElementById('detection-stats').innerHTML = `
            <h3>検出統計</h3>
            <div class="stats-grid">
                <div class="stat-item">
                    <span class="stat-label">検出数</span>
                    <span class="stat-value">${results.length}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">平均信頼度</span>
                    <span class="stat-value">${(stats.avgConfidence * 100).toFixed(1)}%</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">最高信頼度</span>
                    <span class="stat-value">${(stats.maxConfidence * 100).toFixed(1)}%</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">検出方法</span>
                    <span class="stat-value">${stats.methods.join(', ')}</span>
                </div>
            </div>
        `;
    }

    calculateStats(results) {
        if (results.length === 0) {
            return {
                avgConfidence: 0,
                maxConfidence: 0,
                methods: []
            };
        }
        
        const confidences = results.map(r => r.confidence);
        const methods = [...new Set(results.map(r => r.method))];
        
        return {
            avgConfidence: confidences.reduce((a, b) => a + b, 0) / confidences.length,
            maxConfidence: Math.max(...confidences),
            methods: methods
        };
    }

    addAdvancedHighlight(match) {
        const overlay = document.getElementById('coordinate-overlay');
        const canvas = document.getElementById('diagram-canvas');
        
        // スケーリング計算
        const scaleX = canvas.offsetWidth / canvas.width;
        const scaleY = canvas.offsetHeight / canvas.height;
        
        const scaledX = match.x * scaleX;
        const scaledY = match.y * scaleY;
        const scaledWidth = match.width * scaleX;
        const scaledHeight = match.height * scaleY;
        
        // ハイライト要素を作成
        const highlight = document.createElement('div');
        highlight.className = 'advanced-highlight';
        highlight.style.position = 'absolute';
        highlight.style.left = scaledX + 'px';
        highlight.style.top = scaledY + 'px';
        highlight.style.width = scaledWidth + 'px';
        highlight.style.height = scaledHeight + 'px';
        
        // 信頼度に応じて色を変更
        const hue = match.confidence * 120; // 0(赤) から 120(緑)
        highlight.style.borderColor = `hsl(${hue}, 100%, 50%)`;
        highlight.style.backgroundColor = `hsla(${hue}, 100%, 50%, 0.2)`;
        
        // ラベルを追加
        const label = document.createElement('div');
        label.className = 'highlight-label';
        label.textContent = `${match.equipment_name} (${(match.confidence * 100).toFixed(1)}%)`;
        label.style.backgroundColor = `hsl(${hue}, 100%, 40%)`;
        highlight.appendChild(label);
        
        // ツールチップ
        highlight.title = `
            ${match.equipment_name}
            位置: (${match.x}, ${match.y})
            サイズ: ${match.width}×${match.height}
            信頼度: ${(match.confidence * 100).toFixed(2)}%
            検出方法: ${match.method}
        `.trim();
        
        overlay.appendChild(highlight);
        
        // アニメーション
        highlight.style.animation = 'fadeInScale 0.3s ease-out';
        
        console.log(`✓ Advanced highlight added: ${match.equipment_name} @ (${scaledX}, ${scaledY})`);
    }

    clearHighlights() {
        const overlay = document.getElementById('coordinate-overlay');
        overlay.querySelectorAll('.advanced-highlight').forEach(el => el.remove());
        
        const statsContainer = document.getElementById('detection-stats');
        if (statsContainer) {
            statsContainer.remove();
        }
    }

    showProgress(message, percentage) {
        const container = document.getElementById('progress-container');
        const fill = document.getElementById('progress-fill');
        const text = document.getElementById('progress-text');
        
        container.classList.remove('hidden');
        fill.style.width = percentage + '%';
        text.textContent = message;
    }

    hideProgress() {
        const container = document.getElementById('progress-container');
        container.classList.add('hidden');
    }

    cancelProcessing() {
        this.isProcessing = false;
        this.hideProgress();
        this.showNotification('処理をキャンセルしました', 'info');
    }

    toggleCoordinates() {
        this.coordinateMode = !this.coordinateMode;
        const button = document.getElementById('toggle-coordinates');
        
        if (this.coordinateMode) {
            button.textContent = 'xy座標平面を無効化';
            button.classList.replace('btn-secondary', 'btn-success');
        } else {
            button.textContent = 'xy座標平面を追加';
            button.classList.replace('btn-success', 'btn-secondary');
        }
    }

    clearCoordinates() {
        const overlay = document.getElementById('coordinate-overlay');
        overlay.innerHTML = '';
        this.clearHighlights();
        this.detectionResults = [];
        this.showNotification('クリアしました', 'success');
    }

    handleCanvasClick(event) {
        if (!this.coordinateMode) return;
        
        const canvas = event.target;
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        this.displayCoordinate(x, y);
    }

    handleCanvasMouseMove(event) {
        if (!this.coordinateMode) return;
        
        const canvas = event.target;
        const rect = canvas.getBoundingClientRect();
        const x = Math.round(event.clientX - rect.left);
        const y = Math.round(event.clientY - rect.top);
        
        const display = document.getElementById('coordinate-display');
        display.textContent = `座標: (${x}, ${y})`;
    }

    displayCoordinate(x, y) {
        const overlay = document.getElementById('coordinate-overlay');
        
        const point = document.createElement('div');
        point.className = 'coordinate-point';
        point.style.left = (x - 5) + 'px';
        point.style.top = (y - 5) + 'px';
        point.title = `(${Math.round(x)}, ${Math.round(y)})`;
        
        overlay.appendChild(point);
    }

    getSelectedEquipment() {
        const checkboxes = document.querySelectorAll('#equipment-selection-list input[type="checkbox"]:checked');
        return Array.from(checkboxes).map(cb => cb.value);
    }

    clearEquipmentForm() {
        document.getElementById('equipment-name').value = '';
        document.getElementById('equipment-images').value = '';
        document.getElementById('image-preview').innerHTML = '';
    }

    async deleteEquipment(equipmentId, equipmentName) {
        if (!confirm(`機器「${equipmentName}」を削除してもよろしいですか？`)) {
            return;
        }

        try {
            this.showLoading(true);

            const response = await fetch(`${this.apiBaseUrl}/equipment/${equipmentId}`, {
                method: 'DELETE'
            });

            const data = await response.json();

            if (response.ok) {
                this.showNotification('機器が削除されました', 'success');
                this.loadEquipment();
            } else {
                this.showNotification(data.error || '削除に失敗しました', 'error');
            }
        } catch (error) {
            this.showNotification('ネットワークエラーが発生しました', 'error');
            console.error('Delete error:', error);
        } finally {
            this.showLoading(false);
        }
    }

    async deleteDiagram(diagramId) {
        if (!confirm('この図面を削除してもよろしいですか？')) {
            return;
        }

        try {
            this.showLoading(true);

            const response = await fetch(`${this.apiBaseUrl}/diagrams/${diagramId}`, {
                method: 'DELETE'
            });

            const data = await response.json();

            if (response.ok) {
                this.showNotification('図面が削除されました', 'success');
                this.loadDiagrams();
            } else {
                this.showNotification(data.error || '削除に失敗しました', 'error');
            }
        } catch (error) {
            this.showNotification('ネットワークエラーが発生しました', 'error');
            console.error('Delete error:', error);
        } finally {
            this.showLoading(false);
        }
    }

    showLoading(show) {
        const modal = document.getElementById('loading-modal');
        modal.style.display = show ? 'block' : 'none';
    }

    showNotification(message, type = 'success') {
        const notification = document.getElementById('notification');
        notification.textContent = message;
        notification.className = `notification ${type}`;

        setTimeout(() => {
            notification.classList.add('show');
        }, 100);

        setTimeout(() => {
            notification.classList.remove('show');
        }, 3000);
    }
}

// システム初期化
let system;
document.addEventListener('DOMContentLoaded', () => {
    system = new AdvancedDiagramSystem();
    console.log('Advanced Technical Drawing Digitization System initialized');
});
