class DiagramDigitizationSystem {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000/api';
        this.currentDiagram = null;
        this.coordinateMode = false;
        this.equipmentList = [];
        this.diagramList = [];

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadEquipment();
        this.loadDiagrams();
    }

    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                e.preventDefault();
                this.switchTab(tab.dataset.tab);
            });
        });

        // Equipment registration
        document.getElementById('equipment-images').addEventListener('change', this.handleImagePreview.bind(this));
        document.getElementById('register-equipment').addEventListener('click', this.registerEquipment.bind(this));

        // Diagram upload
        document.getElementById('upload-diagram').addEventListener('click', this.uploadDiagram.bind(this));

        // Visualization controls
        document.getElementById('toggle-coordinates').addEventListener('click', this.toggleCoordinates.bind(this));
        document.getElementById('clear-coordinates').addEventListener('click', this.clearCoordinates.bind(this));
        document.getElementById('highlight-selected').addEventListener('click', this.highlightSelectedEquipment.bind(this));

        // Canvas interactions
        const canvas = document.getElementById('diagram-canvas');
        canvas.addEventListener('click', this.handleCanvasClick.bind(this));
        canvas.addEventListener('mousemove', this.handleCanvasMouseMove.bind(this));
    }

    switchTab(tabId) {
        // Remove active class from all tabs and contents
        document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

        // Add active class to clicked tab and corresponding content
        document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
        document.getElementById(tabId).classList.add('active');

        // Load data when switching to visualization tab
        if (tabId === 'visualization') {
            this.loadEquipmentForSelection();
        }
    }

    handleImagePreview(event) {
        const files = event.target.files;
        const preview = document.getElementById('image-preview');
        preview.innerHTML = '';

        Array.from(files).forEach(file => {
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    preview.appendChild(img);
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
            this.showLoading(true);

            const response = await fetch(`${this.apiBaseUrl}/equipment`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                this.showNotification('機器が正常に登録されました', 'success');
                this.clearEquipmentForm();
                this.loadEquipment();
            } else {
                this.showNotification(data.error || '登録に失敗しました', 'error');
            }
        } catch (error) {
            this.showNotification('ネットワークエラーが発生しました', 'error');
            console.error('Registration error:', error);
        } finally {
            this.showLoading(false);
        }
    }

    async uploadDiagram() {
        const fileInput = document.getElementById('diagram-file');

        if (fileInput.files.length === 0) {
            this.showNotification('図面ファイルを選択してください', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('diagram', fileInput.files[0]);

        try {
            this.showLoading(true);

            const response = await fetch(`${this.apiBaseUrl}/diagrams`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                this.showNotification('図面が正常にアップロードされました', 'success');
                fileInput.value = '';
                this.loadDiagrams();
            } else {
                this.showNotification(data.error || 'アップロードに失敗しました', 'error');
            }
        } catch (error) {
            this.showNotification('ネットワークエラーが発生しました', 'error');
            console.error('Upload error:', error);
        } finally {
            this.showLoading(false);
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

            const imagesHtml = equipment.images.map(imagePath =>
                `<img src="http://localhost:8000/${imagePath}" alt="${equipment.name}">`
            ).join('');

            item.innerHTML = `
                <h4>${equipment.name}</h4>
                <p>登録日時: ${new Date(equipment.created_at).toLocaleString()}</p>
                <div class="images">${imagesHtml}</div>
                <div class="equipment-actions">
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

            item.innerHTML = `
                <h4>${diagram.name}</h4>
                <p>アップロード日時: ${new Date(diagram.created_at).toLocaleString()}</p>
                <div class="diagram-actions">
                    <button class="btn btn-primary" onclick="system.loadDiagramToCanvas('${diagram.image_path}')">
                        可視化で使用
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

            item.innerHTML = `
                <input type="checkbox" id="eq-${equipment.id}" value="${equipment.id}">
                <label for="eq-${equipment.id}">${equipment.name}</label>
            `;

            container.appendChild(item);
        });
    }

    loadDiagramToCanvas(imagePath) {
        const canvas = document.getElementById('diagram-canvas');
        const ctx = canvas.getContext('2d');

        const img = new Image();
        img.onload = () => {
            // Set canvas size to match image
            canvas.width = img.width;
            canvas.height = img.height;

            // Draw the diagram
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);

            this.currentDiagram = {
                image: img,
                path: imagePath
            };

            this.showNotification('図面が読み込まれました', 'success');
        };

        img.onerror = (error) => {
            console.error('Failed to load image:', error);
            this.showNotification('図面の読み込みに失敗しました', 'error');
        };

        img.src = `http://localhost:8000/${imagePath}`;

        // Switch to visualization tab
        this.switchTab('visualization');
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
                this.loadDiagrams(); // Refresh diagram list
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

    toggleCoordinates() {
        this.coordinateMode = !this.coordinateMode;
        const button = document.getElementById('toggle-coordinates');

        if (this.coordinateMode) {
            button.textContent = 'xy座標平面を無効化';
            button.classList.remove('btn-secondary');
            button.classList.add('btn-success');
        } else {
            button.textContent = 'xy座標平面を追加';
            button.classList.remove('btn-success');
            button.classList.add('btn-secondary');
        }
    }

    clearCoordinates() {
        const overlay = document.getElementById('coordinate-overlay');
        overlay.innerHTML = '';
        this.clearHighlights();
        this.showNotification('座標とハイライトをクリアしました', 'success');
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
        point.style.position = 'absolute';
        point.style.left = (x - 5) + 'px';
        point.style.top = (y - 5) + 'px';
        point.style.width = '10px';
        point.style.height = '10px';
        point.style.borderRadius = '50%';
        point.style.backgroundColor = '#ff4444';
        point.style.border = '2px solid white';
        point.style.boxShadow = '0 2px 5px rgba(0,0,0,0.3)';
        point.title = `(${Math.round(x)}, ${Math.round(y)})`;

        overlay.appendChild(point);
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

        try {
            this.showLoading(true);
            this.clearHighlights();

            // 実際の画像マッチングAPIを呼び出し
            const response = await fetch(`${this.apiBaseUrl}/match-equipment`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    diagram_path: this.currentDiagram.path,
                    equipment_ids: selectedEquipment.map(id => parseInt(id))
                })
            });

            const data = await response.json();

            if (response.ok) {
                if (data.matches && data.matches.length > 0) {
                    // 実際の検出結果でハイライト表示
                    data.matches.forEach(match => {
                        this.addRealHighlight(match);
                    });

                    this.showNotification(
                        `${data.total_found}つの機器を検出してハイライトしました`,
                        'success'
                    );
                } else {
                    this.showNotification('図面上で機器が検出されませんでした', 'warning');
                }
            } else {
                this.showNotification(data.error || 'ハイライト処理に失敗しました', 'error');
            }
        } catch (error) {
            this.showNotification('ネットワークエラーが発生しました', 'error');
            console.error('Highlighting error:', error);
        } finally {
            this.showLoading(false);
        }
    }

    getSelectedEquipment() {
        const checkboxes = document.querySelectorAll('#equipment-selection-list input[type="checkbox"]:checked');
        return Array.from(checkboxes).map(cb => cb.value);
    }

    addRealHighlight(match) {
        const overlay = document.getElementById('coordinate-overlay');
        const canvas = document.getElementById('diagram-canvas');
        const container = document.querySelector('.diagram-container');

        console.log('=== DEBUG HIGHLIGHT INFO ===');
        console.log('Match data:', match);
        console.log('Canvas dimensions:', canvas.offsetWidth, 'x', canvas.offsetHeight);
        console.log('Canvas actual size:', canvas.width, 'x', canvas.height);
        console.log('Container dimensions:', container.offsetWidth, 'x', container.offsetHeight);
        console.log('Overlay dimensions:', overlay.offsetWidth, 'x', overlay.offsetHeight);

        // 実際の表示に合わせた座標計算
        const containerRect = container.getBoundingClientRect();
        const canvasRect = canvas.getBoundingClientRect();

        console.log('Container rect:', containerRect);
        console.log('Canvas rect:', canvasRect);

        // Canvas実サイズと表示サイズの比率を計算
        const scaleX = canvas.offsetWidth / canvas.width;
        const scaleY = canvas.offsetHeight / canvas.height;

        const scaledX = match.x * scaleX;
        const scaledY = match.y * scaleY;
        const scaledWidth = match.width * scaleX;
        const scaledHeight = match.height * scaleY;

        console.log('Scale factors:', scaleX, scaleY);
        console.log('Original coords:', match.x, match.y, 'size:', match.width, 'x', match.height);
        console.log('Scaled coords:', scaledX, scaledY, 'size:', scaledWidth, 'x', scaledHeight);

        // テスト用: 左上角に固定ハイライトも追加
        const testHighlight = document.createElement('div');
        testHighlight.className = 'test-highlight';
        testHighlight.style.position = 'absolute';
        testHighlight.style.left = '10px';
        testHighlight.style.top = '10px';
        testHighlight.style.width = '100px';
        testHighlight.style.height = '100px';
        testHighlight.style.border = '5px solid #00ff00';
        testHighlight.style.backgroundColor = 'rgba(0, 255, 0, 0.3)';
        testHighlight.style.zIndex = '1001';
        testHighlight.style.pointerEvents = 'none';
        testHighlight.title = 'テスト用ハイライト（左上）';

        overlay.appendChild(testHighlight);
        console.log('Added test highlight at (10, 10)');

        // 実際の検出結果を使用してハイライト表示
        const highlight = document.createElement('div');
        highlight.className = 'highlight-overlay';
        highlight.style.position = 'absolute';
        highlight.style.left = scaledX + 'px';
        highlight.style.top = scaledY + 'px';
        highlight.style.width = scaledWidth + 'px';
        highlight.style.height = scaledHeight + 'px';
        highlight.title = `${match.equipment_name} (${match.x}, ${match.y}) - 信頼度: ${(match.confidence * 100).toFixed(1)}%`;
        highlight.style.border = '4px solid #ff4444';
        highlight.style.backgroundColor = 'rgba(255, 68, 68, 0.5)';
        highlight.style.zIndex = '1002';
        highlight.style.pointerEvents = 'none';
        highlight.style.boxSizing = 'border-box';

        // さらに目立つようにアニメーション追加
        highlight.style.animation = 'pulse 2s infinite';

        console.log('Highlight element created:', highlight);
        console.log('Highlight computed styles:');
        overlay.appendChild(highlight);

        // 追加後の実際のスタイルを確認
        const computedStyle = window.getComputedStyle(highlight);
        console.log('Computed left:', computedStyle.left);
        console.log('Computed top:', computedStyle.top);
        console.log('Computed width:', computedStyle.width);
        console.log('Computed height:', computedStyle.height);
        console.log('Computed zIndex:', computedStyle.zIndex);
        console.log('Computed border:', computedStyle.border);

        // 中心点の座標情報を表示 (スケール調整)
        this.displayCoordinate(match.center_x * scaleX, match.center_y * scaleY);

        console.log(`✓ ハイライト追加: ${match.equipment_name} at (${scaledX}, ${scaledY}) size=${scaledWidth}x${scaledHeight} 信頼度: ${(match.confidence * 100).toFixed(1)}%`);
        console.log('Total elements in overlay:', overlay.children.length);
        console.log('Overlay innerHTML:', overlay.innerHTML);
        console.log('=== END DEBUG ===');

        // 5秒後にテストハイライトを削除
        setTimeout(() => {
            if (testHighlight.parentNode) {
                testHighlight.parentNode.removeChild(testHighlight);
                console.log('Removed test highlight');
            }
        }, 5000);
    }

    clearHighlights() {
        const overlay = document.getElementById('coordinate-overlay');
        const highlights = overlay.querySelectorAll('.highlight-overlay');
        highlights.forEach(highlight => highlight.remove());
    }

    clearEquipmentForm() {
        document.getElementById('equipment-name').value = '';
        document.getElementById('equipment-images').value = '';
        document.getElementById('image-preview').innerHTML = '';
    }

    showLoading(show) {
        const modal = document.getElementById('loading-modal');
        modal.style.display = show ? 'block' : 'none';
    }

    async deleteEquipment(equipmentId, equipmentName) {
        if (!confirm(`機器「${equipmentName}」を削除してもよろしいですか？この操作は取り消せません。`)) {
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
                this.loadEquipment(); // Refresh equipment list
            } else {
                this.showNotification(data.error || '削除に失敗しました', 'error');
            }
        } catch (error) {
            this.showNotification('ネットワークエラーが発生しました', 'error');
            console.error('Delete equipment error:', error);
        } finally {
            this.showLoading(false);
        }
    }

    showNotification(message, type = 'success') {
        const notification = document.getElementById('notification');
        notification.textContent = message;
        notification.className = `notification ${type}`;

        // Show notification
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);

        // Hide notification after 3 seconds
        setTimeout(() => {
            notification.classList.remove('show');
        }, 3000);
    }
}

// Initialize the system when the page loads
let system;
document.addEventListener('DOMContentLoaded', () => {
    system = new DiagramDigitizationSystem();
});