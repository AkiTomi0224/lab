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

        // Initialize statistics
        setTimeout(() => {
            this.updateStatistics();
        }, 500);
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
                await this.loadEquipment(); // 統計更新のため await を追加
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
            this.updateStatistics();
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
                `<img src="http://localhost:8000/uploads/${imagePath.replace(/.*\//, '')}" alt="${equipment.name}">`
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

        // Update statistics
        this.updateStatistics();
    }

    renderDiagramList() {
        const container = document.getElementById('diagram-list');
        container.innerHTML = '';

        if (this.diagramList.length === 0) {
            container.innerHTML = '<div class="no-data">図面がアップロードされていません</div>';
            return;
        }

        this.diagramList.forEach(diagram => {
            const item = document.createElement('div');
            item.className = 'diagram-item';

            // Create a preview thumbnail if possible
            const thumbnailHtml = diagram.image_path ?
                `<img src="http://localhost:8000/${diagram.image_path}" alt="${diagram.name}" style="width: 100px; height: 70px; object-fit: cover; border-radius: 4px; margin-bottom: 10px;">` :
                '<div style="width: 100px; height: 70px; background: #f0f0f0; border-radius: 4px; margin-bottom: 10px; display: flex; align-items: center; justify-content: center;"><i class="fas fa-file-pdf"></i></div>';

            item.innerHTML = `
                ${thumbnailHtml}
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

        console.log('Rendered diagram list:', this.diagramList);
    }

    async loadEquipmentForSelection() {
        const container = document.getElementById('equipment-selection-list');
        container.innerHTML = '';

        // Load trained equipment from enterprise ML system
        let trainedEquipment = [];
        try {
            const response = await fetch(`${this.apiBaseUrl}/enterprise/trained-equipment`);
            if (response.ok) {
                const data = await response.json();
                trainedEquipment = data.equipment || [];
            }
        } catch (error) {
            console.log('Enterprise ML system not available:', error);
        }

        // Create trained equipment section with ML badges
        if (trainedEquipment.length > 0) {
            const mlSection = document.createElement('div');
            mlSection.className = 'equipment-section';
            mlSection.innerHTML = `
                <div class="section-title">
                    <i class="fas fa-robot"></i>
                    <span>AI学習済み機器 (高精度検出)</span>
                    <span class="ml-badge">ML</span>
                </div>
            `;
            container.appendChild(mlSection);

            trainedEquipment.forEach(equipment => {
                const item = document.createElement('div');
                item.className = 'equipment-checkbox ml-equipment';
                item.innerHTML = `
                    <input type="checkbox" id="ml-eq-${equipment.id}" value="ml-${equipment.id}" data-type="ml">
                    <label for="ml-eq-${equipment.id}">
                        <div class="equipment-info">
                            <span class="equipment-name">${equipment.name}</span>
                            <div class="equipment-badges">
                                <span class="accuracy-badge accuracy-${this.getAccuracyClass(equipment.accuracy)}">${equipment.accuracy}%</span>
                                <span class="ml-indicator">AI</span>
                            </div>
                        </div>
                    </label>
                `;
                container.appendChild(item);
            });
        }

        // Create regular equipment section
        if (this.equipmentList.length > 0) {
            const regularSection = document.createElement('div');
            regularSection.className = 'equipment-section';
            regularSection.innerHTML = `
                <div class="section-title">
                    <i class="fas fa-cog"></i>
                    <span>登録機器 (従来検出)</span>
                </div>
            `;
            container.appendChild(regularSection);

            this.equipmentList.forEach(equipment => {
                const item = document.createElement('div');
                item.className = 'equipment-checkbox regular-equipment';
                item.innerHTML = `
                    <input type="checkbox" id="eq-${equipment.id}" value="${equipment.id}" data-type="regular">
                    <label for="eq-${equipment.id}">
                        <div class="equipment-info">
                            <span class="equipment-name">${equipment.name}</span>
                            <div class="equipment-badges">
                                <span class="detection-type">従来</span>
                            </div>
                        </div>
                    </label>
                `;
                container.appendChild(item);
            });
        }

        // Add empty state if no equipment
        if (trainedEquipment.length === 0 && this.equipmentList.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-inbox"></i>
                    <p>機器が登録されていません</p>
                    <small>機器登録タブまたは機械学習タブで機器を追加してください</small>
                </div>
            `;
        }
    }

    getAccuracyClass(accuracy) {
        if (accuracy >= 95) return 'excellent';
        if (accuracy >= 85) return 'good';
        if (accuracy >= 75) return 'fair';
        return 'poor';
    }

    loadDiagramToCanvas(imagePath) {
        const canvas = document.getElementById('diagram-canvas');
        const ctx = canvas.getContext('2d');

        const img = new Image();
        img.onload = () => {
            // Set canvas size to match image with max dimensions
            const maxWidth = 800;
            const maxHeight = 600;
            let { width, height } = img;

            if (width > maxWidth || height > maxHeight) {
                const ratio = Math.min(maxWidth / width, maxHeight / height);
                width *= ratio;
                height *= ratio;
            }

            canvas.width = width;
            canvas.height = height;
            canvas.style.width = width + 'px';
            canvas.style.height = height + 'px';

            // Draw the diagram
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, width, height);

            this.currentDiagram = {
                image: img,
                path: imagePath,
                width: width,
                height: height
            };

            this.showNotification('図面が読み込まれました', 'success');
        };

        img.onerror = (error) => {
            console.error('Failed to load image:', error);
            console.error('Image path:', imagePath);
            this.showNotification('図面の読み込みに失敗しました', 'error');
        };

        // Use the correct path from backend (static/diagrams/)
        img.src = `http://localhost:8000/${imagePath}`;
        console.log('Loading image from:', img.src);
        console.log('Original imagePath:', imagePath);

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

            // Separate ML-trained and regular equipment
            const mlEquipment = selectedEquipment.filter(eq => eq.type === 'ml');
            const regularEquipment = selectedEquipment.filter(eq => eq.type === 'regular');

            let allMatches = [];
            let mlMatches = [];
            let traditionalMatches = [];

            // Process ML-trained equipment with enterprise ML system
            for (const equipment of mlEquipment) {
                try {
                    const equipmentId = equipment.id.replace('ml-', ''); // Remove ml- prefix
                    const response = await fetch(`${this.apiBaseUrl}/enterprise/predict`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            equipment_id: parseInt(equipmentId),
                            diagram_path: this.currentDiagram.path
                        })
                    });

                    if (response.ok) {
                        const data = await response.json();
                        if (data.detections && data.detections.length > 0) {
                            data.detections.forEach(detection => {
                                const match = {
                                    ...detection,
                                    equipment_name: data.equipment_name || '不明な機器',
                                    detection_type: 'enterprise_ml',
                                    confidence: detection.confidence || 0.95,
                                    model_accuracy: data.model_accuracy || 95
                                };
                                mlMatches.push(match);
                                allMatches.push(match);
                            });
                        }
                    }
                } catch (error) {
                    console.log(`Enterprise ML detection failed for equipment ${equipment.id}:`, error);
                }
            }

            // Process regular equipment with traditional detection
            if (regularEquipment.length > 0) {
                try {
                    const response = await fetch(`${this.apiBaseUrl}/match-equipment`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            diagram_path: this.currentDiagram.path,
                            equipment_ids: regularEquipment.map(eq => parseInt(eq.id))
                        })
                    });

                    if (response.ok) {
                        const data = await response.json();
                        if (data.matches && data.matches.length > 0) {
                            data.matches.forEach(match => {
                                const enhancedMatch = {
                                    ...match,
                                    detection_type: 'traditional',
                                    confidence: match.confidence || 0.7
                                };
                                traditionalMatches.push(enhancedMatch);
                                allMatches.push(enhancedMatch);
                            });
                        }
                    }
                } catch (error) {
                    console.log('Traditional detection failed:', error);
                }
            }

            // Display results with enhanced visual highlighting
            if (allMatches.length > 0) {
                allMatches.forEach(match => {
                    this.addEnhancedHighlight(match);
                });

                // Show comprehensive detection results
                const mlCount = mlMatches.length;
                const traditionalCount = traditionalMatches.length;

                let message = `✅ ${allMatches.length}つの機器を検出しました`;

                if (mlCount > 0 && traditionalCount > 0) {
                    message += `\n🤖 AI検出: ${mlCount}個 (高精度)\n🔍 従来検出: ${traditionalCount}個`;
                } else if (mlCount > 0) {
                    message += `\n🤖 AI検出: ${mlCount}個 (高精度)`;
                    const avgAccuracy = mlMatches.reduce((sum, m) => sum + (m.model_accuracy || 0), 0) / mlCount;
                    message += `\n📊 平均精度: ${avgAccuracy.toFixed(1)}%`;
                } else if (traditionalCount > 0) {
                    message += `\n🔍 従来検出: ${traditionalCount}個`;
                }

                this.showNotification(message, 'success');

                // Show detection details in console for debugging
                console.log('🎯 Detection Results:', {
                    total: allMatches.length,
                    ml: mlMatches.length,
                    traditional: traditionalMatches.length,
                    matches: allMatches
                });

            } else {
                this.showNotification('❌ 図面上で機器が検出されませんでした\n・機器の位置や角度を確認してください\n・AI学習タブで追加学習を検討してください', 'warning');
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
        return Array.from(checkboxes).map(cb => ({
            id: cb.value,
            type: cb.dataset.type || 'regular',
            element: cb
        }));
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

    addEnhancedHighlight(match) {
        const overlay = document.getElementById('coordinate-overlay');
        const canvas = document.getElementById('diagram-canvas');
        const container = document.querySelector('.diagram-container');

        // Calculate scaling factors
        const scaleX = canvas.offsetWidth / canvas.width;
        const scaleY = canvas.offsetHeight / canvas.height;

        const scaledX = match.x * scaleX;
        const scaledY = match.y * scaleY;
        const scaledWidth = match.width * scaleX;
        const scaledHeight = match.height * scaleY;

        // Create main highlight container
        const highlightContainer = document.createElement('div');
        highlightContainer.className = `enhanced-highlight-container ${match.detection_type}-highlight`;
        highlightContainer.style.position = 'absolute';
        highlightContainer.style.left = scaledX + 'px';
        highlightContainer.style.top = scaledY + 'px';
        highlightContainer.style.width = scaledWidth + 'px';
        highlightContainer.style.height = scaledHeight + 'px';
        highlightContainer.style.zIndex = '1000';
        highlightContainer.style.pointerEvents = 'none';

        // Create the main bounding box with enterprise styling
        const boundingBox = document.createElement('div');
        boundingBox.className = 'enhanced-bounding-box';

        // Set colors and styles based on detection type
        let borderColor, backgroundColor, shadowColor;
        if (match.detection_type === 'enterprise_ml') {
            borderColor = '#3b82f6'; // Blue for ML
            backgroundColor = 'rgba(59, 130, 246, 0.15)';
            shadowColor = 'rgba(59, 130, 246, 0.4)';
        } else {
            borderColor = '#f59e0b'; // Orange for traditional
            backgroundColor = 'rgba(245, 158, 11, 0.15)';
            shadowColor = 'rgba(245, 158, 11, 0.4)';
        }

        boundingBox.style.position = 'absolute';
        boundingBox.style.top = '0';
        boundingBox.style.left = '0';
        boundingBox.style.right = '0';
        boundingBox.style.bottom = '0';
        boundingBox.style.border = `3px solid ${borderColor}`;
        boundingBox.style.backgroundColor = backgroundColor;
        boundingBox.style.borderRadius = '8px';
        boundingBox.style.boxShadow = `0 0 20px ${shadowColor}, inset 0 0 20px ${shadowColor}`;
        boundingBox.style.animation = 'enhanced-pulse 2s ease-in-out infinite alternate';

        // Create confidence badge
        const confidenceBadge = document.createElement('div');
        confidenceBadge.className = 'confidence-badge';
        confidenceBadge.textContent = `${(match.confidence * 100).toFixed(1)}%`;
        confidenceBadge.style.position = 'absolute';
        confidenceBadge.style.top = '-12px';
        confidenceBadge.style.right = '-8px';
        confidenceBadge.style.background = match.detection_type === 'enterprise_ml' ?
            'linear-gradient(135deg, #3b82f6, #1e40af)' :
            'linear-gradient(135deg, #f59e0b, #d97706)';
        confidenceBadge.style.color = 'white';
        confidenceBadge.style.padding = '4px 8px';
        confidenceBadge.style.borderRadius = '12px';
        confidenceBadge.style.fontSize = '11px';
        confidenceBadge.style.fontWeight = '700';
        confidenceBadge.style.textShadow = '0 1px 2px rgba(0, 0, 0, 0.3)';
        confidenceBadge.style.boxShadow = '0 2px 6px rgba(0, 0, 0, 0.3)';
        confidenceBadge.style.zIndex = '1001';

        // Create detection type indicator
        const typeIndicator = document.createElement('div');
        typeIndicator.className = 'detection-type-indicator';
        typeIndicator.style.position = 'absolute';
        typeIndicator.style.top = '-12px';
        typeIndicator.style.left = '-8px';
        typeIndicator.style.padding = '3px 6px';
        typeIndicator.style.borderRadius = '8px';
        typeIndicator.style.fontSize = '9px';
        typeIndicator.style.fontWeight = '800';
        typeIndicator.style.textTransform = 'uppercase';
        typeIndicator.style.color = 'white';
        typeIndicator.style.textShadow = '0 1px 2px rgba(0, 0, 0, 0.4)';
        typeIndicator.style.zIndex = '1001';

        if (match.detection_type === 'enterprise_ml') {
            typeIndicator.textContent = 'AI';
            typeIndicator.style.background = 'linear-gradient(135deg, #8b5cf6, #7c3aed)';
            typeIndicator.style.animation = 'ai-glow 1.5s ease-in-out infinite alternate';
        } else {
            typeIndicator.textContent = '従来';
            typeIndicator.style.background = 'linear-gradient(135deg, #6b7280, #4b5563)';
        }

        // Create equipment name label
        const nameLabel = document.createElement('div');
        nameLabel.className = 'equipment-name-label';
        nameLabel.textContent = match.equipment_name || '機器';
        nameLabel.style.position = 'absolute';
        nameLabel.style.bottom = '-28px';
        nameLabel.style.left = '0';
        nameLabel.style.background = 'rgba(0, 0, 0, 0.8)';
        nameLabel.style.color = 'white';
        nameLabel.style.padding = '4px 8px';
        nameLabel.style.borderRadius = '6px';
        nameLabel.style.fontSize = '12px';
        nameLabel.style.fontWeight = '600';
        nameLabel.style.whiteSpace = 'nowrap';
        nameLabel.style.backdropFilter = 'blur(4px)';
        nameLabel.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.3)';
        nameLabel.style.zIndex = '1001';

        // Create accuracy indicator for ML detections
        if (match.detection_type === 'enterprise_ml' && match.model_accuracy) {
            const accuracyIndicator = document.createElement('div');
            accuracyIndicator.className = 'accuracy-indicator';
            accuracyIndicator.textContent = `精度:${match.model_accuracy}%`;
            accuracyIndicator.style.position = 'absolute';
            accuracyIndicator.style.bottom = '-45px';
            accuracyIndicator.style.right = '0';
            accuracyIndicator.style.background = 'linear-gradient(135deg, #10b981, #059669)';
            accuracyIndicator.style.color = 'white';
            accuracyIndicator.style.padding = '3px 6px';
            accuracyIndicator.style.borderRadius = '4px';
            accuracyIndicator.style.fontSize = '10px';
            accuracyIndicator.style.fontWeight = '600';
            accuracyIndicator.style.textShadow = '0 1px 2px rgba(0, 0, 0, 0.3)';
            accuracyIndicator.style.zIndex = '1001';
            highlightContainer.appendChild(accuracyIndicator);
        }

        // Assemble the highlight
        highlightContainer.appendChild(boundingBox);
        highlightContainer.appendChild(confidenceBadge);
        highlightContainer.appendChild(typeIndicator);
        highlightContainer.appendChild(nameLabel);

        // Create center point indicator
        const centerPoint = document.createElement('div');
        centerPoint.className = 'center-point-indicator';
        centerPoint.style.position = 'absolute';
        centerPoint.style.left = '50%';
        centerPoint.style.top = '50%';
        centerPoint.style.width = '8px';
        centerPoint.style.height = '8px';
        centerPoint.style.background = borderColor;
        centerPoint.style.borderRadius = '50%';
        centerPoint.style.transform = 'translate(-50%, -50%)';
        centerPoint.style.boxShadow = `0 0 8px ${shadowColor}`;
        centerPoint.style.zIndex = '1002';
        highlightContainer.appendChild(centerPoint);

        // Add tooltip functionality
        highlightContainer.title = [
            `機器: ${match.equipment_name || '不明'}`,
            `検出方式: ${match.detection_type === 'enterprise_ml' ? 'AI高精度検出' : '従来手法'}`,
            `信頼度: ${(match.confidence * 100).toFixed(1)}%`,
            match.model_accuracy ? `モデル精度: ${match.model_accuracy}%` : '',
            `座標: (${match.x}, ${match.y})`,
            `サイズ: ${match.width}×${match.height}`
        ].filter(Boolean).join('\n');

        overlay.appendChild(highlightContainer);

        // Display coordinate if center point is available
        if (match.center_x !== undefined && match.center_y !== undefined) {
            this.displayCoordinate(match.center_x * scaleX, match.center_y * scaleY);
        }

        console.log(`🎯 Enhanced Highlight: ${match.equipment_name} [${match.detection_type}] - Confidence: ${(match.confidence * 100).toFixed(1)}%`);
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

    updateStatistics() {
        const totalEquipmentEl = document.getElementById('total-equipment');
        const totalImagesEl = document.getElementById('total-images');

        if (totalEquipmentEl && totalImagesEl) {
            const totalEquipment = this.equipmentList.length;
            const totalImages = this.equipmentList.reduce((sum, equipment) => {
                return sum + (equipment.images ? equipment.images.length : 0);
            }, 0);

            totalEquipmentEl.textContent = totalEquipment;
            totalImagesEl.textContent = totalImages;

            // Add animation to updated numbers
            totalEquipmentEl.style.transform = 'scale(1.1)';
            totalImagesEl.style.transform = 'scale(1.1)';

            setTimeout(() => {
                totalEquipmentEl.style.transform = 'scale(1)';
                totalImagesEl.style.transform = 'scale(1)';
            }, 300);
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