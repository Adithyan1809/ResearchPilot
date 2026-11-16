/**
 * ResearchLens — Frontend Application Logic
 * Vanilla JavaScript implementation for PDF ingestion and RAG querying.
 */

const API_BASE = '/api/v1';

class ResearchLensApp {
    constructor() {
        this.documents = [];
        this.isProcessing = false;
        this.healthInterval = null;

        // Cache DOM elements
        this.elements = {
            docCount: document.getElementById('docCount'),
            uploadZone: document.getElementById('uploadZone'),
            fileInput: document.getElementById('fileInput'),
            uploadProgress: document.getElementById('uploadProgress'),
            progressFill: document.getElementById('progressFill'),
            progressText: document.getElementById('progressText'),
            documentList: document.getElementById('documentList'),
            emptyDocState: document.getElementById('emptyDocState'),
            docFilter: document.getElementById('docFilter'),
            questionInput: document.getElementById('questionInput'),
            askBtn: document.getElementById('askBtn'),
            responseArea: document.getElementById('responseArea'),
            initialState: document.getElementById('initialState'),
            loadingState: document.getElementById('loadingState'),
            answerContainer: document.getElementById('answerContainer'),
            answerBody: document.getElementById('answerBody'),
            sourcesSection: document.getElementById('sourcesSection'),
            sourcesList: document.getElementById('sourcesList'),
            metadataBar: document.getElementById('metadataBar'),
            metaTime: document.getElementById('metaTime'),
            metaTokens: document.getElementById('metaTokens'),
            statusIndicator: document.getElementById('statusIndicator'),
            statusText: document.getElementById('statusText'),
            statusDocCount: document.getElementById('statusDocCount'),
            toastContainer: document.getElementById('toastContainer'),
        };

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupDragAndDrop();
        this.loadDocuments();
        this.checkHealth();

        // Periodic health check every 30 seconds
        this.healthInterval = setInterval(() => this.checkHealth(), 30000);
    }

    // ── Event Listeners ──────────────────────────────────────────────────────

    setupEventListeners() {
        // File selection
        this.elements.fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.uploadDocument(file);
            }
        });

        // Question input typing
        this.elements.questionInput.addEventListener('input', () => {
            const hasText = this.elements.questionInput.value.trim().length >= 3;
            this.elements.askBtn.disabled = !hasText || this.isProcessing;
        });

        // Keyboard shortcuts
        this.elements.questionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!this.elements.askBtn.disabled) {
                    this.askQuestion();
                }
            }
        });

        // Ask button click
        this.elements.askBtn.addEventListener('click', () => {
            if (!this.elements.askBtn.disabled) {
                this.askQuestion();
            }
        });
    }

    // ── Drag and Drop ────────────────────────────────────────────────────────

    setupDragAndDrop() {
        const zone = this.elements.uploadZone;

        ['dragenter', 'dragover'].forEach((eventName) => {
            zone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                zone.classList.add('drag-over');
            });
        });

        ['dragleave', 'drop'].forEach((eventName) => {
            zone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                zone.classList.remove('drag-over');
            });
        });

        zone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const file = dt.files[0];
            if (file) {
                if (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')) {
                    this.uploadDocument(file);
                } else {
                    this.showToast('Please upload a PDF document (.pdf)', 'error');
                }
            }
        });
    }

    // ── Document Ingestion ───────────────────────────────────────────────────

    async uploadDocument(file) {
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            this.showToast('Invalid file format. Only PDF files are supported.', 'error');
            return;
        }

        const maxSizeMb = 50;
        if (file.size > maxSizeMb * 1024 * 1024) {
            this.showToast(`File size exceeds maximum limit of ${maxSizeMb} MB.`, 'error');
            return;
        }

        // Show progress UI
        this.elements.uploadProgress.hidden = false;
        this.elements.progressFill.style.width = '30%';
        this.elements.progressText.textContent = `Uploading ${file.name}...`;

        const formData = new FormData();
        formData.append('file', file);

        try {
            this.elements.progressFill.style.width = '60%';
            this.elements.progressText.textContent = 'Processing and embedding chunks...';

            const response = await fetch(`${API_BASE}/ingest`, {
                method: 'POST',
                body: formData,
            });

            this.elements.progressFill.style.width = '100%';

            if (!response.ok) {
                let errorDetail = 'Ingestion failed';
                try {
                    const err = await response.json();
                    errorDetail = err.detail || err.message || errorDetail;
                } catch (_) {}
                throw new Error(errorDetail);
            }

            const data = await response.json();
            this.showToast(`Successfully ingested "${data.filename}" (${data.total_chunks} chunks, ${data.total_pages} pages)`, 'success');

            // Reset upload file input
            this.elements.fileInput.value = '';

            // Refresh document list and health
            await this.loadDocuments();
            await this.checkHealth();
        } catch (error) {
            this.showToast(`Error: ${error.message}`, 'error');
        } finally {
            setTimeout(() => {
                this.elements.uploadProgress.hidden = true;
                this.elements.progressFill.style.width = '0%';
                this.elements.progressText.textContent = 'Uploading...';
            }, 1000);
        }
    }

    // ── Document List & Management ───────────────────────────────────────────

    async loadDocuments() {
        try {
            const response = await fetch(`${API_BASE}/documents`);
            if (!response.ok) throw new Error('Failed to load documents');

            const data = await response.json();
            this.documents = data.documents || [];
            this.renderDocumentList();
            this.updateFilterDropdown();
        } catch (error) {
            console.error('Failed to load documents:', error);
        }
    }

    renderDocumentList() {
        const list = this.elements.documentList;
        const emptyState = this.elements.emptyDocState;
        const count = this.documents.length;

        this.elements.docCount.textContent = `${count} ${count === 1 ? 'document' : 'documents'}`;

        if (count === 0) {
            list.innerHTML = '';
            list.appendChild(emptyState);
            emptyState.hidden = false;
            return;
        }

        emptyState.hidden = true;
        list.innerHTML = '';

        this.documents.forEach((doc) => {
            const item = document.createElement('div');
            item.className = 'document-item';
            item.dataset.id = doc.doc_id;

            const dateStr = doc.ingested_at
                ? new Date(doc.ingested_at).toLocaleDateString(undefined, {
                      month: 'short',
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit',
                  })
                : 'Just now';

            item.innerHTML = `
                <div class="doc-info">
                    <span class="doc-title" title="${this.escapeHtml(doc.filename)}">${this.escapeHtml(doc.filename)}</span>
                    <span class="doc-meta">${doc.total_pages} pages · ${doc.total_chunks} chunks · ${dateStr}</span>
                </div>
                <button class="doc-delete-btn" title="Delete document" data-id="${doc.doc_id}">🗑</button>
            `;

            // Delete event handler
            const deleteBtn = item.querySelector('.doc-delete-btn');
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteDocument(doc.doc_id, doc.filename);
            });

            list.appendChild(item);
        });
    }

    updateFilterDropdown() {
        const select = this.elements.docFilter;
        const currentVal = select.value;

        select.innerHTML = '<option value="">All Documents</option>';

        this.documents.forEach((doc) => {
            const option = document.createElement('option');
            option.value = doc.doc_id;
            option.textContent = doc.filename.length > 30 ? doc.filename.substring(0, 27) + '...' : doc.filename;
            select.appendChild(option);
        });

        // Restore previous selection if still available
        if (this.documents.some((d) => d.doc_id === currentVal)) {
            select.value = currentVal;
        } else {
            select.value = '';
        }
    }

    async deleteDocument(docId, filename) {
        if (!confirm(`Are you sure you want to remove "${filename}" from the index?`)) {
            return;
        }

        try {
            const response = await fetch(`${API_BASE}/documents/${docId}`, {
                method: 'DELETE',
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Delete failed');
            }

            this.showToast(`Removed "${filename}"`, 'info');
            await this.loadDocuments();
            await this.checkHealth();
        } catch (error) {
            this.showToast(`Error deleting document: ${error.message}`, 'error');
        }
    }

    // ── Q&A Query Execution ──────────────────────────────────────────────────

    async askQuestion() {
        const question = this.elements.questionInput.value.trim();
        if (question.length < 3) {
            this.showToast('Please enter a question with at least 3 characters.', 'info');
            return;
        }

        const docId = this.elements.docFilter.value || null;

        this.setProcessing(true);

        // Hide initial state, show loading spinner
        this.elements.initialState.hidden = true;
        this.elements.answerContainer.hidden = true;
        this.elements.loadingState.hidden = false;

        try {
            const payload = {
                question: question,
                doc_id: docId,
                top_k: 5,
            };

            const response = await fetch(`${API_BASE}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                let errorMsg = 'Failed to answer question';
                try {
                    const err = await response.json();
                    errorMsg = err.detail || errorMsg;
                } catch (_) {}
                throw new Error(errorMsg);
            }

            const data = await response.json();
            this.renderAnswer(data);
        } catch (error) {
            this.showToast(`Query failed: ${error.message}`, 'error');
            this.elements.loadingState.hidden = true;
            this.elements.initialState.hidden = false;
        } finally {
            this.setProcessing(false);
        }
    }

    renderAnswer(data) {
        this.elements.loadingState.hidden = true;
        this.elements.answerContainer.hidden = false;

        // 1. Format and display answer
        const formattedAnswer = this.formatMarkdown(data.answer);
        this.elements.answerBody.innerHTML = formattedAnswer;

        // 2. Render source documents
        const sources = data.sources || [];
        if (sources.length > 0) {
            this.elements.sourcesSection.hidden = false;
            this.elements.sourcesList.innerHTML = '';

            sources.forEach((source, index) => {
                const card = document.createElement('div');
                card.className = 'source-card';

                const scorePercent = Math.round(source.relevance_score * 100);
                const scoreClass = scorePercent >= 70 ? 'score-high' : scorePercent >= 40 ? 'score-med' : 'score-low';

                card.innerHTML = `
                    <div class="source-card-header">
                        <span class="source-tag">Source ${index + 1}</span>
                        <span class="source-doc-name" title="${this.escapeHtml(source.filename)}">${this.escapeHtml(source.filename)}</span>
                        <span class="source-page">Page ${source.page_number}</span>
                        <span class="source-score ${scoreClass}">${scorePercent}% match</span>
                    </div>
                    <div class="source-content">${this.escapeHtml(source.content)}</div>
                `;
                this.elements.sourcesList.appendChild(card);
            });
        } else {
            this.elements.sourcesSection.hidden = true;
        }

        // 3. Render metadata bar
        this.elements.metadataBar.hidden = false;

        const timeSpan = this.elements.metaTime.querySelector('.meta-value');
        const formattedTime =
            data.processing_time_ms >= 1000
                ? `${(data.processing_time_ms / 1000).toFixed(2)}s`
                : `${Math.round(data.processing_time_ms)}ms`;
        timeSpan.textContent = `Response time: ${formattedTime}`;

        const tokenSpan = this.elements.metaTokens.querySelector('.meta-value');
        if (data.tokens_used && data.tokens_used.total_tokens) {
            this.elements.metaTokens.hidden = false;
            tokenSpan.textContent = `Tokens: ${data.tokens_used.total_tokens} (prompt: ${data.tokens_used.prompt_tokens}, completion: ${data.tokens_used.completion_tokens})`;
        } else {
            this.elements.metaTokens.hidden = true;
        }

        // Scroll to response smoothly
        this.elements.responseArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    setProcessing(processing) {
        this.isProcessing = processing;
        this.elements.askBtn.disabled = processing || this.elements.questionInput.value.trim().length < 3;
        this.elements.questionInput.disabled = processing;
    }

    // ── System Health ────────────────────────────────────────────────────────

    async checkHealth() {
        try {
            const response = await fetch(`${API_BASE}/health`);
            if (response.ok) {
                const data = await response.json();
                this.elements.statusIndicator.className = 'status-indicator status-online';
                this.elements.statusText.textContent = `System Online · v${data.version}`;
                this.elements.statusDocCount.textContent = `${data.documents_count} documents (${data.chunks_count} chunks indexed)`;
            } else {
                throw new Error('Unhealthy status');
            }
        } catch (_) {
            this.elements.statusIndicator.className = 'status-indicator status-offline';
            this.elements.statusText.textContent = 'Backend Offline';
            this.elements.statusDocCount.textContent = 'Unable to connect to server';
        }
    }

    // ── Formatting & Utilities ───────────────────────────────────────────────

    formatMarkdown(text) {
        if (!text) return '';

        let html = this.escapeHtml(text);

        // Highlight source citations like [Source 1], [Source 2]
        html = html.replace(/\[Source\s*(\d+)\]/gi, '<span class="citation-badge">Source $1</span>');

        // Bold **text**
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Italic *text*
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // Code blocks ```code```
        html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

        // Inline code `code`
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Headers ### Header
        html = html.replace(/^### (.*$)/gim, '<h4>$1</h4>');
        html = html.replace(/^## (.*$)/gim, '<h3>$1</h3>');
        html = html.replace(/^# (.*$)/gim, '<h2>$1</h2>');

        // Bullet lists
        html = html.replace(/^\s*[-*]\s+(.*)$/gim, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>)/gms, '<ul>$1</ul>');
        html = html.replace(/<\/ul>\s*<ul>/g, '');

        // Numbered lists
        html = html.replace(/^\s*\d+\.\s+(.*)$/gim, '<li>$1</li>');

        // Paragraphs (double newlines)
        const paragraphs = html.split(/\n\n+/);
        html = paragraphs
            .map((p) => {
                p = p.trim();
                if (p.startsWith('<h') || p.startsWith('<ul') || p.startsWith('<pre') || p.startsWith('<li')) {
                    return p;
                }
                return p ? `<p>${p.replace(/\n/g, '<br>')}</p>` : '';
            })
            .join('');

        return html;
    }

    escapeHtml(str) {
        if (!str) return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;

        const icon = type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ';

        toast.innerHTML = `
            <span class="toast-icon">${icon}</span>
            <span class="toast-message">${this.escapeHtml(message)}</span>
        `;

        this.elements.toastContainer.appendChild(toast);

        // Animate in
        requestAnimationFrame(() => toast.classList.add('toast-show'));

        // Auto remove
        setTimeout(() => {
            toast.classList.remove('toast-show');
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }
}

// Instantiate on DOM load
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ResearchLensApp();
});
