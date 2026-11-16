/**
 * ResearchLens — Frontend Application Logic (v5.0)
 * ChatGPT-Style 3-Column Interface with Conversational Multi-Turn Memory & Real-Time Token Streaming (SSE)
 */

const API_BASE = '/api/v1';

class ResearchLensApp {
    constructor() {
        this.documents = [];
        this.sessions = [];
        this.isProcessing = false;
        this.healthInterval = null;
        this.sessionId = localStorage.getItem('researchlens_session_id') || null;

        // Cache DOM elements
        this.elements = {
            // Sidebar History
            sidebarHistory: document.getElementById('sidebarHistory'),
            newChatBtn: document.getElementById('newChatBtn'),
            sessionList: document.getElementById('sessionList'),
            emptySessions: document.getElementById('emptySessions'),
            statusIndicator: document.getElementById('statusIndicator'),
            statusText: document.getElementById('statusText'),

            // Center Panel
            chatMain: document.getElementById('chatMain'),
            currentSessionTitle: document.getElementById('currentSessionTitle'),
            sessionIdDisplay: document.getElementById('sessionIdDisplay'),
            retrievalModeSelect: document.getElementById('retrievalModeSelect'),
            streamToggle: document.getElementById('streamToggle'),
            toggleDocsBtn: document.getElementById('toggleDocsBtn'),
            docBadge: document.getElementById('docBadge'),
            responseArea: document.getElementById('responseArea'),
            initialState: document.getElementById('initialState'),
            chatThread: document.getElementById('chatThread'),
            loadingState: document.getElementById('loadingState'),
            loadingText: document.getElementById('loadingText'),

            // Input Bar
            docFilter: document.getElementById('docFilter'),
            questionInput: document.getElementById('questionInput'),
            askBtn: document.getElementById('askBtn'),

            // Right Sidebar Docs
            sidebarDocs: document.getElementById('sidebarDocs'),
            docCount: document.getElementById('docCount'),
            uploadZone: document.getElementById('uploadZone'),
            fileInput: document.getElementById('fileInput'),
            uploadProgress: document.getElementById('uploadProgress'),
            progressFill: document.getElementById('progressFill'),
            progressText: document.getElementById('progressText'),
            documentList: document.getElementById('documentList'),
            emptyDocState: document.getElementById('emptyDocState'),

            // Toast
            toastContainer: document.getElementById('toastContainer'),
        };

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupDragAndDrop();
        this.setupPromptChips();
        this.setupTextareaAutoResize();

        this.loadSessions();
        this.loadDocuments();
        this.checkHealth();
        this.initSession();

        // Periodic health check every 30 seconds
        this.healthInterval = setInterval(() => this.checkHealth(), 30000);
    }

    // ── Session Management ───────────────────────────────────────────────────

    initSession() {
        if (this.sessionId) {
            this.updateSessionDisplay(this.sessionId);
            this.restoreSessionHistory(this.sessionId);
        } else {
            this.updateSessionDisplay('new');
        }
    }

    updateSessionDisplay(id) {
        if (!this.elements.sessionIdDisplay) return;
        const shortId = id && id !== 'new' ? id.substring(0, 8) + '...' : 'new';
        this.elements.sessionIdDisplay.textContent = shortId;
        this.elements.sessionIdDisplay.title = id ? `Session ID: ${id}` : 'New session';
    }

    async loadSessions() {
        try {
            const resp = await fetch(`${API_BASE}/sessions`);
            if (!resp.ok) return;
            const data = await resp.json();
            this.sessions = data.sessions || [];
            this.renderSessionList();
        } catch (err) {
            console.error('Failed to load sessions:', err);
        }
    }

    renderSessionList() {
        const list = this.elements.sessionList;
        const empty = this.elements.emptySessions;

        // Clear existing session items (keep empty placeholder)
        list.querySelectorAll('.session-item').forEach((el) => el.remove());

        if (this.sessions.length === 0) {
            if (empty) empty.hidden = false;
            return;
        }

        if (empty) empty.hidden = true;

        this.sessions.forEach((sess) => {
            const item = document.createElement('div');
            item.className = 'session-item';
            if (sess.session_id === this.sessionId) {
                item.classList.add('active');
            }
            item.dataset.id = sess.session_id;

            const title = sess.title || `Conversation ${sess.session_id.substring(0, 6)}`;
            const countText = `${sess.message_count || 0} messages`;

            item.innerHTML = `
                <div class="session-item-content">
                    <span class="session-item-title" title="${this.escapeHtml(title)}">${this.escapeHtml(title)}</span>
                    <span class="session-item-meta">${countText}</span>
                </div>
                <button class="session-delete-btn" title="Delete conversation" data-id="${sess.session_id}">🗑</button>
            `;

            // Click to switch
            item.addEventListener('click', (e) => {
                if (e.target.closest('.session-delete-btn')) return;
                this.switchSession(sess.session_id);
            });

            // Click to delete
            const delBtn = item.querySelector('.session-delete-btn');
            delBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteSession(sess.session_id);
            });

            list.appendChild(item);
        });
    }

    async switchSession(sessionId) {
        if (this.isProcessing) {
            this.showToast('Please wait for current query to finish.', 'info');
            return;
        }

        this.sessionId = sessionId;
        localStorage.setItem('researchlens_session_id', sessionId);
        this.updateSessionDisplay(sessionId);

        // Highlight in sidebar
        this.elements.sessionList.querySelectorAll('.session-item').forEach((el) => {
            if (el.dataset.id === sessionId) {
                el.classList.add('active');
            } else {
                el.classList.remove('active');
            }
        });

        // Clear chat thread & load messages
        this.elements.chatThread.innerHTML = '';
        await this.restoreSessionHistory(sessionId);
    }

    async restoreSessionHistory(sessionId) {
        try {
            const resp = await fetch(`${API_BASE}/sessions/${sessionId}`);
            if (!resp.ok) {
                this.startNewChat();
                return;
            }
            const data = await resp.json();

            if (data.title && this.elements.currentSessionTitle) {
                this.elements.currentSessionTitle.textContent = data.title;
            }

            if (data.messages && data.messages.length > 0) {
                this.elements.initialState.hidden = true;
                // Group pairs of user + assistant
                for (let i = 0; i < data.messages.length; i += 2) {
                    const userMsg = data.messages[i];
                    const assistMsg = data.messages[i + 1];
                    if (userMsg && userMsg.role === 'user') {
                        this.renderHistoricalTurn(userMsg.content, assistMsg ? assistMsg.content : '');
                    }
                }
                this.elements.responseArea.scrollTop = this.elements.responseArea.scrollHeight;
            } else {
                this.elements.initialState.hidden = false;
            }
        } catch (_) {
            this.elements.initialState.hidden = false;
        }
    }

    renderHistoricalTurn(question, answer) {
        const turnDiv = document.createElement('div');
        turnDiv.className = 'chat-turn';

        const userBubble = document.createElement('div');
        userBubble.className = 'user-message';
        userBubble.textContent = question;

        const assistBubble = document.createElement('div');
        assistBubble.className = 'assistant-message';
        assistBubble.innerHTML = `
            <div class="assistant-message-header">
                <span class="assistant-badge">🔬 Research Assistant</span>
            </div>
            <div class="assistant-body">${this.formatMarkdown(answer)}</div>
        `;

        turnDiv.appendChild(userBubble);
        turnDiv.appendChild(assistBubble);
        this.elements.chatThread.appendChild(turnDiv);
    }

    async startNewChat() {
        try {
            const resp = await fetch(`${API_BASE}/sessions`, { method: 'POST' });
            if (resp.ok) {
                const data = await resp.json();
                this.sessionId = data.session_id;
                localStorage.setItem('researchlens_session_id', this.sessionId);
                this.updateSessionDisplay(this.sessionId);
            }
        } catch (_) {
            this.sessionId = null;
            localStorage.removeItem('researchlens_session_id');
            this.updateSessionDisplay('new');
        }

        // Reset UI
        this.elements.chatThread.innerHTML = '';
        this.elements.initialState.hidden = false;
        this.elements.questionInput.value = '';
        this.elements.questionInput.style.height = 'auto';
        this.elements.askBtn.disabled = true;
        if (this.elements.currentSessionTitle) {
            this.elements.currentSessionTitle.textContent = 'Research Assistant';
        }

        await this.loadSessions();
        this.showToast('Started a new conversation session.', 'info');
    }

    async deleteSession(sessionId) {
        if (!confirm('Are you sure you want to delete this conversation?')) {
            return;
        }

        try {
            const resp = await fetch(`${API_BASE}/sessions/${sessionId}`, { method: 'DELETE' });
            if (!resp.ok) {
                throw new Error('Failed to delete session');
            }

            this.showToast('Conversation deleted.', 'info');

            if (this.sessionId === sessionId) {
                await this.startNewChat();
            } else {
                await this.loadSessions();
            }
        } catch (err) {
            this.showToast(`Error deleting session: ${err.message}`, 'error');
        }
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

        // New Chat button click
        if (this.elements.newChatBtn) {
            this.elements.newChatBtn.addEventListener('click', () => {
                this.startNewChat();
            });
        }

        // Toggle Documents Right Panel
        if (this.elements.toggleDocsBtn && this.elements.sidebarDocs) {
            this.elements.toggleDocsBtn.addEventListener('click', () => {
                this.elements.sidebarDocs.classList.toggle('collapsed');
                this.elements.toggleDocsBtn.classList.toggle('active');
            });
        }
    }

    setupPromptChips() {
        const chips = document.querySelectorAll('.prompt-chips .chip');
        chips.forEach((chip) => {
            chip.addEventListener('click', () => {
                const prompt = chip.dataset.prompt || chip.innerText.trim();
                if (prompt) {
                    this.elements.questionInput.value = prompt;
                    this.elements.askBtn.disabled = false;
                    this.elements.questionInput.focus();
                    this.askQuestion();
                }
            });
        });
    }

    setupTextareaAutoResize() {
        const textarea = this.elements.questionInput;
        if (!textarea) return;

        textarea.addEventListener('input', () => {
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
        });
    }

    // ── Drag and Drop ────────────────────────────────────────────────────────

    setupDragAndDrop() {
        const zone = this.elements.uploadZone;
        if (!zone) return;

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
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
                    this.uploadDocument(file);
                } else {
                    this.showToast('Please upload a PDF file.', 'error');
                }
            }
        });
    }

    // ── Document Management ──────────────────────────────────────────────────

    async loadDocuments() {
        try {
            const response = await fetch(`${API_BASE}/documents`);
            if (!response.ok) throw new Error('Failed to load documents');

            const data = await response.json();
            this.documents = data.documents || [];
            this.renderDocumentList();
            this.updateFilterDropdown();
            this.updateDocCounts(this.documents.length);
        } catch (error) {
            this.showToast(`Error loading documents: ${error.message}`, 'error');
        }
    }

    async uploadDocument(file) {
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            this.showToast('Only PDF files are supported.', 'error');
            return;
        }

        const maxSize = 50 * 1024 * 1024; // 50MB
        if (file.size > maxSize) {
            this.showToast('File size exceeds the 50MB limit.', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        this.showUploadProgress(true, 'Uploading and processing PDF...');

        try {
            const response = await fetch(`${API_BASE}/ingest`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                let errorMsg = 'Upload failed';
                try {
                    const err = await response.json();
                    errorMsg = err.detail || errorMsg;
                } catch (_) {}
                throw new Error(errorMsg);
            }

            const result = await response.json();
            this.showToast(result.message || 'Document ingested successfully!', 'success');

            this.elements.fileInput.value = '';
            await this.loadDocuments();
            await this.checkHealth();
        } catch (error) {
            this.showToast(`Upload failed: ${error.message}`, 'error');
        } finally {
            this.showUploadProgress(false);
        }
    }

    showUploadProgress(show, text = '') {
        const progress = this.elements.uploadProgress;
        const fill = this.elements.progressFill;
        const label = this.elements.progressText;

        if (show) {
            progress.hidden = false;
            label.textContent = text;
            fill.style.width = '30%';
            const timer = setInterval(() => {
                const cur = parseInt(fill.style.width, 10);
                if (cur < 85) {
                    fill.style.width = `${cur + 15}%`;
                } else {
                    clearInterval(timer);
                }
            }, 300);
            progress._timer = timer;
        } else {
            if (progress._timer) clearInterval(progress._timer);
            fill.style.width = '100%';
            setTimeout(() => {
                progress.hidden = true;
                fill.style.width = '0%';
            }, 400);
        }
    }

    renderDocumentList() {
        const list = this.elements.documentList;
        const empty = this.elements.emptyDocState;

        if (this.documents.length === 0) {
            empty.hidden = false;
            list.querySelectorAll('.doc-card').forEach((el) => el.remove());
            return;
        }

        empty.hidden = true;
        list.querySelectorAll('.doc-card').forEach((el) => el.remove());

        this.documents.forEach((doc) => {
            const item = document.createElement('div');
            item.className = 'doc-card';

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
                    <span class="doc-name" title="${this.escapeHtml(doc.filename)}">${this.escapeHtml(doc.filename)}</span>
                    <span class="doc-meta">
                        <span>📄 ${doc.total_pages} pages</span>
                        <span>🧩 ${doc.total_chunks} chunks</span>
                        <span>🕒 ${dateStr}</span>
                    </span>
                </div>
                <button class="doc-delete-btn" title="Delete document" data-id="${doc.doc_id}">🗑</button>
            `;

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
            option.textContent =
                doc.filename.length > 28 ? doc.filename.substring(0, 25) + '...' : doc.filename;
            select.appendChild(option);
        });

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

    // ── Q&A Query Execution (Streaming & History) ───────────────────────────

    async askQuestion() {
        const question = this.elements.questionInput.value.trim();
        if (question.length < 3) {
            this.showToast('Please enter a question with at least 3 characters.', 'info');
            return;
        }

        const docId = this.elements.docFilter.value || null;
        const retrievalMode = this.elements.retrievalModeSelect?.value || 'hybrid';
        const isStreaming = this.elements.streamToggle?.checked ?? true;

        this.setProcessing(true);
        this.elements.questionInput.value = '';
        this.elements.questionInput.style.height = 'auto';
        this.elements.initialState.hidden = true;

        // Create UI Turn in Chat Thread
        const turnDiv = document.createElement('div');
        turnDiv.className = 'chat-turn';

        const userBubble = document.createElement('div');
        userBubble.className = 'user-message';
        userBubble.textContent = question;

        const assistBubble = document.createElement('div');
        assistBubble.className = 'assistant-message';
        assistBubble.innerHTML = `
            <div class="assistant-message-header">
                <span class="assistant-badge">🔬 Research Assistant</span>
            </div>
            <div class="assistant-body"><span class="typing-cursor"></span></div>
            <div class="turn-sources" hidden>
                <details>
                    <summary>📚 Referenced Sources (<span class="sources-count">0</span>)</summary>
                    <div class="sources-list"></div>
                </details>
            </div>
            <div class="turn-meta" hidden>
                <span class="meta-time"></span>
            </div>
        `;

        turnDiv.appendChild(userBubble);
        turnDiv.appendChild(assistBubble);
        this.elements.chatThread.appendChild(turnDiv);
        turnDiv.scrollIntoView({ behavior: 'smooth' });

        const bodyDiv = assistBubble.querySelector('.assistant-body');
        const sourcesContainer = assistBubble.querySelector('.turn-sources');
        const sourcesCount = assistBubble.querySelector('.sources-count');
        const sourcesList = assistBubble.querySelector('.sources-list');
        const metaDiv = assistBubble.querySelector('.turn-meta');
        const metaTime = assistBubble.querySelector('.meta-time');

        const payload = {
            question: question,
            doc_id: docId,
            top_k: 3,
            retrieval_mode: retrievalMode,
            session_id: this.sessionId,
        };

        if (isStreaming) {
            try {
                const response = await fetch(`${API_BASE}/query/stream`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });

                if (!response.ok) {
                    throw new Error(`Streaming failed with HTTP ${response.status}`);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';
                let accumulatedText = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n\n');
                    buffer = lines.pop(); // keep last incomplete chunk

                    for (const rawEvent of lines) {
                        const trimmed = rawEvent.trim();
                        if (!trimmed.startsWith('data: ')) continue;
                        const dataStr = trimmed.substring(6);

                        try {
                            const event = JSON.parse(dataStr);
                            if (event.type === 'sources') {
                                if (event.session_id) {
                                    this.sessionId = event.session_id;
                                    localStorage.setItem('researchlens_session_id', this.sessionId);
                                    this.updateSessionDisplay(this.sessionId);
                                }
                                if (event.sources && event.sources.length > 0) {
                                    this.renderTurnSources(event.sources, sourcesContainer, sourcesCount, sourcesList);
                                }
                            } else if (event.type === 'token') {
                                accumulatedText += event.delta;
                                bodyDiv.innerHTML = this.formatMarkdown(accumulatedText) + '<span class="typing-cursor"></span>';
                            } else if (event.type === 'done') {
                                accumulatedText = event.answer || accumulatedText;
                                bodyDiv.innerHTML = this.formatMarkdown(accumulatedText);
                                if (event.processing_time_ms) {
                                    metaDiv.hidden = false;
                                    metaTime.textContent = `⏱ Processed in ${event.processing_time_ms} ms`;
                                }
                            } else if (event.type === 'error') {
                                bodyDiv.innerHTML = `<span style="color:var(--color-error)">Error: ${this.escapeHtml(event.error)}</span>`;
                            }
                        } catch (err) {
                            console.error('Failed to parse SSE event', err);
                        }
                    }
                    turnDiv.scrollIntoView({ behavior: 'smooth' });
                }

                // Refresh sessions to get updated title
                await this.loadSessions();
            } catch (err) {
                bodyDiv.innerHTML = `<span style="color:var(--color-error)">Streaming query failed: ${this.escapeHtml(err.message)}</span>`;
                this.showToast(`Query error: ${err.message}`, 'error');
            } finally {
                this.setProcessing(false);
            }
        } else {
            // Non-streaming fallback
            try {
                const response = await fetch(`${API_BASE}/query`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Query failed');
                }

                const data = await response.json();
                if (data.session_id) {
                    this.sessionId = data.session_id;
                    localStorage.setItem('researchlens_session_id', this.sessionId);
                    this.updateSessionDisplay(this.sessionId);
                }

                bodyDiv.innerHTML = this.formatMarkdown(data.answer);
                if (data.sources && data.sources.length > 0) {
                    this.renderTurnSources(data.sources, sourcesContainer, sourcesCount, sourcesList);
                }
                if (data.processing_time_ms) {
                    metaDiv.hidden = false;
                    metaTime.textContent = `⏱ Response time: ${data.processing_time_ms} ms`;
                }

                // Refresh sessions to get updated title
                await this.loadSessions();
            } catch (err) {
                bodyDiv.innerHTML = `<span style="color:var(--color-error)">Query failed: ${this.escapeHtml(err.message)}</span>`;
                this.showToast(`Query error: ${err.message}`, 'error');
            } finally {
                this.setProcessing(false);
            }
        }
    }

    renderTurnSources(sources, container, countSpan, listContainer) {
        container.hidden = false;
        countSpan.textContent = sources.length;
        listContainer.innerHTML = '';

        sources.forEach((src, idx) => {
            const card = document.createElement('div');
            card.className = 'source-card';

            const score = src.rerank_score ?? src.relevance_score;
            const scorePercent = Math.round(score * 100);
            const scoreClass =
                scorePercent >= 70 ? 'score-high' : scorePercent >= 40 ? 'score-med' : 'score-low';

            const methodBadge = src.rerank_score !== null && src.rerank_score !== undefined
                ? `<span class="source-channel-badge">⚡ Reranked (${(src.rerank_score).toFixed(2)})</span>`
                : src.retrieval_method
                ? `<span class="source-channel-badge">${this.escapeHtml(src.retrieval_method)}</span>`
                : '';

            card.innerHTML = `
                <div class="source-card-header">
                    <span class="source-tag">Source ${idx + 1}</span>
                    <span class="source-doc-name" title="${this.escapeHtml(src.filename)}">${this.escapeHtml(src.filename)}</span>
                    <span class="source-page">Page ${src.page_number}</span>
                    ${methodBadge}
                    <span class="source-score ${scoreClass}">${scorePercent}% match</span>
                </div>
                <div class="source-content">${this.escapeHtml(src.content)}</div>
            `;
            listContainer.appendChild(card);
        });
    }

    // ── Health & UI State ────────────────────────────────────────────────────

    async checkHealth() {
        try {
            const response = await fetch(`${API_BASE}/health`);
            if (response.ok) {
                const data = await response.json();
                this.elements.statusIndicator.className = 'status-indicator status-online';
                this.elements.statusText.textContent = `Online · v${data.version}`;
                this.updateDocCounts(data.documents_count);
            } else {
                throw new Error('Unhealthy');
            }
        } catch (_) {
            this.elements.statusIndicator.className = 'status-indicator status-offline';
            this.elements.statusText.textContent = 'Offline';
        }
    }

    setProcessing(processing) {
        this.isProcessing = processing;
        this.elements.askBtn.disabled = processing || this.elements.questionInput.value.trim().length < 3;
        // Strictly keep loadingState hidden unless needed for non-streaming
        if (this.elements.loadingState) {
            this.elements.loadingState.hidden = true;
        }
    }

    updateDocCounts(count) {
        const text = count === 1 ? '1 paper' : `${count} papers`;
        if (this.elements.docCount) this.elements.docCount.textContent = text;
        if (this.elements.docBadge) this.elements.docBadge.textContent = count;
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;

        this.elements.toastContainer.appendChild(toast);
        setTimeout(() => toast.classList.add('show'), 10);

        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }

    escapeHtml(str) {
        if (!str) return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    formatMarkdown(text) {
        if (!text) return '';
        let html = this.escapeHtml(text);

        // Bold
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Italic
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
        // Inline code
        html = html.replace(/`(.*?)`/g, '<code>$1</code>');
        // Source citations
        html = html.replace(
            /\[Source (\d+)\]/g,
            '<span class="source-citation" title="Citation: Source $1">[Source $1]</span>'
        );
        // Line breaks & paragraphs
        html = html.replace(/\n\n/g, '</p><p>');
        html = html.replace(/\n/g, '<br>');

        return `<p>${html}</p>`;
    }
}

// Instantiate on DOM load
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ResearchLensApp();
});
