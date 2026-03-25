/**
 * Shared predict modal and queue logic. Persists across page navigation via sessionStorage.
 * Must be loaded on all pages (base.html). Call GalaxyPredict.attachGrid(gridBody, saveGridToCache) when grid is present.
 */
(function() {
    'use strict';

    const PREDICT_STATE_KEY = 'galaxy-predict-state';
    const COMPLETED_PREDICTIONS_KEY = 'galaxy-completed-predictions';
    const MAX_CONCURRENT_PREDICT = 1;
    const RETRY_DELAY_MS = 2000;

    let predictDisplayIdCounter = 0;

    function getState() {
        try {
            const raw = sessionStorage.getItem(PREDICT_STATE_KEY);
            if (raw) {
                const data = JSON.parse(raw);
                return {
                    predictDisplayList: data.predictDisplayList || [],
                    predictQueue: (data.predictQueue || []).map(function(x) {
                        return { objId: x.objId, ra: x.ra, dec: x.dec };
                    }),
                    predictActiveCount: Math.min(data.predictActiveCount || 0, MAX_CONCURRENT_PREDICT),
                };
            }
        } catch (_) {}
        return { predictDisplayList: [], predictQueue: [], predictActiveCount: 0 };
    }

    function saveState(state) {
        try {
            const queueToSave = (state.predictQueue || []).map(function(x) {
                return { objId: x.objId, ra: x.ra, dec: x.dec };
            });
            sessionStorage.setItem(PREDICT_STATE_KEY, JSON.stringify({
                predictDisplayList: state.predictDisplayList,
                predictQueue: queueToSave,
                predictActiveCount: state.predictActiveCount,
            }));
        } catch (_) {}
    }

    function getCompletedPredictions() {
        try {
            const raw = sessionStorage.getItem(COMPLETED_PREDICTIONS_KEY);
            return raw ? JSON.parse(raw) : {};
        } catch (_) {}
        return {};
    }

    function setCompletedPrediction(objId, rowHtml) {
        const completed = getCompletedPredictions();
        completed[objId] = rowHtml;
        sessionStorage.setItem(COMPLETED_PREDICTIONS_KEY, JSON.stringify(completed));
    }

    function removeCompletedPrediction(objId) {
        const completed = getCompletedPredictions();
        delete completed[objId];
        sessionStorage.setItem(COMPLETED_PREDICTIONS_KEY, JSON.stringify(completed));
    }

    function clearCompletedPredictions() {
        sessionStorage.removeItem(COMPLETED_PREDICTIONS_KEY);
    }

    function getInProgressObjIds() {
        const state = getState();
        const ids = new Set();
        state.predictDisplayList.forEach(function(item) {
            if (['queued', 'predicting', 'retrying'].indexOf(item.status) >= 0) {
                ids.add(item.objId);
            }
        });
        return ids;
    }

    function parseSSE(buffer) {
        const events = [];
        let currentEvent = null;
        const lines = buffer.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            if (line.startsWith('event: ')) {
                if (currentEvent) events.push(currentEvent);
                currentEvent = { type: line.slice(7).trim(), data: [] };
            } else if (line.startsWith('data: ') && currentEvent) {
                currentEvent.data.push(line.slice(6));
            } else if (line === '' && currentEvent) {
                events.push(currentEvent);
                currentEvent = null;
            }
        }
        if (currentEvent) events.push(currentEvent);
        return events;
    }

    function isRateLimited(res, data) {
        if (res.status === 429) return true;
        const err = (data && data.error) || '';
        return /429|rate limit|too many requests/i.test(err);
    }

    function renderPredictModal() {
        const predictModal = document.getElementById('predict-modal');
        const predictModalList = document.getElementById('predict-modal-list');
        const predictModalCount = document.getElementById('predict-modal-count');
        if (!predictModal || !predictModalList) return;

        const state = getState();
        const total = state.predictDisplayList.length;
        if (total === 0) {
            predictModal.classList.remove('visible');
            predictModal.setAttribute('aria-hidden', 'true');
            return;
        }
        predictModal.classList.add('visible');
        predictModal.setAttribute('aria-hidden', 'false');
        if (predictModalCount) predictModalCount.textContent = total;

        predictModalList.innerHTML = '';
        state.predictDisplayList.forEach(function(item) {
            const li = document.createElement('li');
            li.className = 'predict-modal-item predict-modal-item-' + item.status;
            li.dataset.objId = item.objId;

            const label = document.createElement('span');
            label.className = 'predict-modal-item-label';
            label.textContent = 'Galaxy ' + item.objId;

            const status = document.createElement('span');
            status.className = 'predict-modal-item-status';
            var statusText = item.status === 'queued' ? 'Waiting…' :
                item.status === 'predicting' ? (item.progress >= 0 && item.progress < 100 ? 'Downloading… ' + Math.round(item.progress) + '%' : 'Predicting…') :
                item.status === 'retrying' ? 'Retrying…' :
                item.status === 'done' ? 'Done' :
                item.status === 'failed' ? (item.error || 'Failed') : '';
            status.textContent = statusText;

            const progressWrap = document.createElement('div');
            progressWrap.className = 'predict-modal-item-progress-wrap';
            const progressBar = document.createElement('div');
            progressBar.className = 'predict-modal-item-progress';
            var hasRealProgress = typeof item.progress === 'number' && item.progress >= 0 && item.progress <= 100;
            if ((item.status === 'predicting' || item.status === 'retrying') && !hasRealProgress) {
                progressBar.classList.add('indeterminate');
            } else {
                progressBar.style.width = (hasRealProgress ? item.progress : (item.progress || 0)) + '%';
            }
            progressWrap.appendChild(progressBar);

            li.appendChild(label);
            li.appendChild(status);
            li.appendChild(progressWrap);
            predictModalList.appendChild(li);
        });
    }

    function updatePredictDisplay(objId, status, progress, error) {
        const state = getState();
        const item = state.predictDisplayList.find(function(x) { return x.objId === objId; });
        if (item) {
            item.status = status;
            item.progress = progress;
            if (error) item.error = error;
            saveState(state);
            renderPredictModal();
        }
    }

    function removeFromPredictDisplay(objId, delay) {
        setTimeout(function() {
            const state = getState();
            const i = state.predictDisplayList.findIndex(function(x) { return x.objId === objId; });
            if (i >= 0) {
                state.predictDisplayList.splice(i, 1);
                saveState(state);
                renderPredictModal();
            }
        }, delay || 0);
    }

    async function runPredictRequest(item, gridBody, saveGridToCache) {
        const { objId, ra, dec, cell, originalContent, el } = item;
        let isRetrying = false;
        const state = getState();
        try {
            const fast = sessionStorage.getItem('fast_predict') === 'true';
            const res = await fetch('/api/retry_row_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ obj_id: objId, ra: ra, dec: dec, fast: fast }),
            });
            if (!res.ok) {
                const text = await res.text();
                throw new Error(text || 'Request failed');
            }
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let data = {};
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const parts = buffer.split('\n\n');
                buffer = parts.pop() || '';
                for (let p = 0; p < parts.length; p++) {
                    const events = parseSSE(parts[p] + '\n\n');
                    for (let e = 0; e < events.length; e++) {
                        const ev = events[e];
                        const payload = ev.data.join('\n');
                        if (ev.type === 'download_progress') {
                            try {
                                const prog = JSON.parse(payload);
                                const total = prog.total;
                                const downloaded = prog.downloaded || 0;
                                const pct = (total != null && total > 0)
                                    ? Math.min(100, Math.round((downloaded / total) * 100))
                                    : -1;
                                updatePredictDisplay(objId, 'predicting', pct);
                            } catch (_) {}
                        } else if (ev.type === 'row') {
                            try {
                                data = JSON.parse(payload);
                            } catch (_) {}
                        }
                    }
                }
            }
            if (buffer) {
                const events = parseSSE(buffer);
                for (let e = 0; e < events.length; e++) {
                    if (events[e].type === 'row') {
                        try {
                            data = JSON.parse(events[e].data.join('\n'));
                        } catch (_) {}
                    }
                }
            }

            if (isRateLimited(res, data)) {
                isRetrying = true;
                updatePredictDisplay(objId, 'retrying', -1);
                state.predictQueue.push(item);
                saveState(state);
                if (cell && cell.isConnected) {
                    cell.innerHTML = '<span class="spinner" aria-hidden="true"></span> <span class="retry-hint">Retrying…</span>';
                    cell.title = 'Rate limited, retrying automatically…';
                }
                setTimeout(function() {
                    const s = getState();
                    s.predictActiveCount = Math.max(0, s.predictActiveCount - 1);
                    saveState(s);
                    processPredictQueue(gridBody, saveGridToCache);
                }, RETRY_DELAY_MS);
                return;
            }

            var currentGridBody = document.getElementById('grid-body');
            var row = currentGridBody ? currentGridBody.querySelector('tr[data-obj-id="' + objId + '"]') : null;
            if (row && row.isConnected && data.html) {
                var temp = document.createElement('tbody');
                temp.innerHTML = data.html.trim();
                var newRow = temp.querySelector('tr');
                if (newRow) {
                    newRow.classList.add('row-fade-in');
                    row.replaceWith(newRow);
                    if (saveGridToCache) saveGridToCache();
                }
                updatePredictDisplay(objId, 'done', 100);
                removeFromPredictDisplay(objId, 500);
            } else if (data.html) {
                setCompletedPrediction(objId, data.html);
                updatePredictDisplay(objId, 'done', 100);
                removeFromPredictDisplay(objId, 500);
            } else {
                const errMsg = (data && data.error) || 'Request failed';
                updatePredictDisplay(objId, 'failed', 0, errMsg);
                removeFromPredictDisplay(objId, 3000);
                if (cell && cell.isConnected) {
                    cell.innerHTML = originalContent;
                    cell.classList.remove('view-cell-loading');
                    cell.removeAttribute('aria-busy');
                }
            }
        } catch (err) {
            isRetrying = false;
            const errMsg = err ? (err.message || String(err)) : 'Unknown error';
            updatePredictDisplay(objId, 'failed', 0, errMsg);
            removeFromPredictDisplay(objId, 3000);
            if (cell && cell.isConnected) {
                cell.innerHTML = originalContent;
                cell.classList.remove('view-cell-loading');
                cell.removeAttribute('aria-busy');
            }
        } finally {
            if (!isRetrying) {
                const s = getState();
                s.predictActiveCount = Math.max(0, s.predictActiveCount - 1);
                saveState(s);
                processPredictQueue(gridBody, saveGridToCache);
            }
        }
    }

    function processPredictQueue(gridBody, saveGridToCache) {
        const state = getState();
        if (state.predictQueue.length === 0 || state.predictActiveCount >= MAX_CONCURRENT_PREDICT) return;
        const queued = state.predictQueue.shift();
        state.predictActiveCount++;
        saveState(state);
        var cell = null, originalContent = '', el = null;
        if (gridBody) {
            var row = gridBody.querySelector('tr[data-obj-id="' + queued.objId + '"]');
            if (row) {
                var lastCell = row.querySelector('td:last-child');
                if (lastCell) {
                    cell = lastCell;
                    originalContent = lastCell.innerHTML;
                    el = lastCell.querySelector('.row-retry, .btn-predict') || lastCell;
                }
            }
        }
        var item = {
            objId: queued.objId,
            ra: queued.ra,
            dec: queued.dec,
            cell: cell,
            originalContent: originalContent,
            el: el,
        };
        updatePredictDisplay(queued.objId, 'predicting', -1);
        runPredictRequest(item, gridBody, saveGridToCache);
    }

    function showCellSpinner(cell, el) {
        if (cell) {
            cell.classList.add('view-cell-loading');
            cell.innerHTML = '<span class="spinner" aria-hidden="true"></span>';
            if (el) el.classList.add('retrying');
            cell.setAttribute('aria-busy', 'true');
        }
    }

    function handlePredictOrRetry(el, gridBody, saveGridToCache) {
        if (!el || el.classList.contains('retrying')) return;
        const objId = el.dataset.objId;
        const ra = el.dataset.ra;
        const dec = el.dataset.dec;
        if (!objId || !ra || !dec) return;
        if (getInProgressObjIds().has(objId)) return;
        const cell = el.closest('td');
        const originalContent = cell ? cell.innerHTML : '';
        showCellSpinner(cell, el);

        const state = getState();
        const existing = state.predictDisplayList.find(function(x) { return x.objId === objId; });
        if (!existing) {
            predictDisplayIdCounter++;
            state.predictDisplayList.push({ id: predictDisplayIdCounter, objId: objId, status: 'queued', progress: 0 });
            saveState(state);
            renderPredictModal();
        }
        const item = { objId, ra, dec, cell, originalContent, el };
        state.predictQueue.push(item);
        saveState(state);
        processPredictQueue(gridBody, saveGridToCache);
    }

    function clearPredictState() {
        sessionStorage.removeItem(PREDICT_STATE_KEY);
        clearCompletedPredictions();
        renderPredictModal();
    }

    function setupModalToggle() {
        var toggle = document.getElementById('predict-modal-toggle');
        var modal = document.getElementById('predict-modal');
        if (toggle && modal) {
            toggle.addEventListener('click', function() {
                modal.classList.toggle('collapsed');
                toggle.textContent = modal.classList.contains('collapsed') ? '+' : '−';
                toggle.setAttribute('aria-label', modal.classList.contains('collapsed') ? 'Expand' : 'Collapse');
            });
        }
    }

    function initOnPageLoad() {
        clearPredictState();
        setupModalToggle();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initOnPageLoad);
    } else {
        initOnPageLoad();
    }

    window.GalaxyPredict = {
        getState: getState,
        getInProgressObjIds: getInProgressObjIds,
        getCompletedPredictions: getCompletedPredictions,
        removeCompletedPrediction: removeCompletedPrediction,
        clearCompletedPredictions: clearCompletedPredictions,
        clearPredictState: clearPredictState,
        renderPredictModal: renderPredictModal,
        processPredictQueue: processPredictQueue,
        handlePredictOrRetry: handlePredictOrRetry,
        updatePredictDisplay: updatePredictDisplay,
    };
})();