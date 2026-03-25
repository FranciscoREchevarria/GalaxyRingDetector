/**
 * Predict page logic: 3D plot, Lupton controls, expand modal.
 * Loaded from base.html so it runs on initial page load and persists across htmx-boost navigation.
 * Without this, navigating to /predict via nav link would not execute the inline script (htmx
 * does not run scripts in swapped content), so the htmx:afterSwap handler would never be
 * registered and the 3D plot would not load on first prediction.
 */
(function() {
    'use strict';

    let plot3dData = null;
    let predict3dFirstLoad = true;
    let predict3dLoadingTimeout = null;
    let predict3dFetchId = 0;
    const PLOT3D_LOADING_DELAY_MS = 1000;

    const greenColorscale = [
        [0, 'rgb(200, 245, 200)'],
        [0.3, 'rgb(140, 200, 140)'],
        [0.6, 'rgb(80, 150, 80)'],
        [1, 'rgb(0, 80, 0)']
    ];
    const redColorscale = [
        [0, 'rgb(255, 220, 220)'],
        [0.15, 'rgb(255, 220, 220)'],
        [0.3, 'rgb(210, 160, 160)'],
        [0.6, 'rgb(150, 80, 80)'],
        [1, 'rgb(80, 0, 0)']
    ];
    const blueColorscale = [
        [0, 'rgb(220, 235, 255)'],
        [0.3, 'rgb(170, 195, 240)'],
        [0.6, 'rgb(120, 150, 215)'],
        [1, 'rgb(60, 80, 160)']
    ];

    function isValid3dData(data) {
        return data && Array.isArray(data.channels) && data.channels.length > 0 &&
            Array.isArray(data.x) && Array.isArray(data.y);
    }

    function buildTraces(data, opacity) {
        if (!isValid3dData(data)) return [];
        const op = opacity === 1 ? 1 : 0.75;
        return data.channels.map(ch => ({
            z: ch.z_data,
            x: data.x,
            y: data.y,
            type: 'surface',
            name: ch.name,
            colorscale: ch.name === 'g' ? greenColorscale : ch.name === 'r' ? redColorscale : blueColorscale,
            opacity: op
        }));
    }

    function buildLayout(margin, expanded) {
        const scene = {
            xaxis: { title: 'x', autorange: 'reversed' },
            yaxis: { title: 'y' },
            zaxis: { title: 'Flux' }
        };
        if (expanded) {
            scene.domain = { x: [0, 1], y: [0, 1] };
            scene.aspectmode = 'auto';
        }
        const layout = {
            margin: margin || { t: 40, r: 20, b: 40, l: 20 },
            scene: scene
        };
        if (expanded) layout.autosize = true;
        return layout;
    }

    const channelOrder = ['g', 'r', 'z'];

    function getVisibilityFromCheckboxes(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return [true, true, true];
        const inputs = container.querySelectorAll('input[name="channel"], input[name="channel-modal"]');
        return channelOrder.map(name => {
            const cb = Array.from(inputs).find(i => i.value === name);
            return cb ? cb.checked : true;
        });
    }

    function updatePlotVisibility(plotId, visibility) {
        const el = document.getElementById(plotId);
        if (!el) return;
        Plotly.restyle(el, { visible: visibility });
    }

    function setupChannelToggles(containerId, plotId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        container.querySelectorAll('input[name="channel"], input[name="channel-modal"]').forEach(input => {
            input.addEventListener('change', function() {
                const visibility = getVisibilityFromCheckboxes(containerId);
                updatePlotVisibility(plotId, visibility);
            });
        });
    }

    function updateZAxisWrap(plotId, wrap) {
        const el = document.getElementById(plotId);
        if (!el) return;
        if (wrap) {
            Plotly.relayout(el, { 'scene.zaxis.range': [0, 1], 'scene.zaxis.autorange': false });
        } else {
            Plotly.relayout(el, { 'scene.zaxis.autorange': true });
        }
    }

    function setupZAxisWrapToggle(checkboxId, plotId) {
        const cb = document.getElementById(checkboxId);
        if (!cb) return;
        cb.addEventListener('change', function() {
            updateZAxisWrap(plotId, cb.checked);
        });
    }

    function updatePlotOpacity(plotId, solid) {
        const el = document.getElementById(plotId);
        if (!el) return;
        const op = solid ? 1 : 0.75;
        Plotly.restyle(el, { opacity: [op, op, op] });
    }

    function setupSolidColorsToggle(checkboxId, plotId) {
        const cb = document.getElementById(checkboxId);
        if (!cb) return;
        cb.addEventListener('change', function() {
            updatePlotOpacity(plotId, cb.checked);
        });
    }

    function updatePlotData(data) {
        const el = document.getElementById('predict-3d-container');
        if (!el || !el.data || !isValid3dData(data)) return;
        plot3dData = data;
        const zArrays = data.channels.map(ch => ch.z_data);
        Plotly.restyle(el, { z: zArrays });
    }

    function updatePlotDataForElement(el, data) {
        if (!el || !el.data || !isValid3dData(data)) return;
        const zArrays = data.channels.map(ch => ch.z_data);
        Plotly.restyle(el, { z: zArrays });
    }

    function initPlotForElement(el, data, layout) {
        if (!el || !isValid3dData(data)) return;
        const traces = buildTraces(data);
        if (traces.length === 0) return;
        Plotly.newPlot(el, traces, layout || buildLayout(), { responsive: true });
    }

    function initPlot(data) {
        if (!isValid3dData(data)) return;
        plot3dData = data;
        const traces = buildTraces(data);
        if (traces.length === 0) return;
        Plotly.newPlot('predict-3d-container', traces, buildLayout(), { responsive: true });
        setupChannelToggles('predict-channel-toggles', 'predict-3d-container');
        setupZAxisWrapToggle('predict-zwrap-main', 'predict-3d-container');
        setupSolidColorsToggle('predict-solid-colors-main', 'predict-3d-container');
    }

    function setupExpandModal() {
        const modal = document.getElementById('predict-expand-modal');
        const modalTitle = document.getElementById('predict-expand-modal-title');
        const modalImage = document.getElementById('predict-expand-modal-image');
        const modal3dWrapper = document.getElementById('predict-expand-modal-3d-wrapper');
        const modal3d = document.getElementById('predict-expand-modal-3d');
        const closeBtn = document.getElementById('predict-close-expand-modal');
        const backdrop = modal ? modal.querySelector('.expand-modal-backdrop') : null;

        if (!modal || !closeBtn) return;

        function closeModal() {
            modal.setAttribute('hidden', '');
            document.body.style.overflow = '';
            modalImage.hidden = true;
            modal3dWrapper.hidden = true;
            modalImage.innerHTML = '';
            if (modal3d && modal3d.querySelector('.plotly')) {
                Plotly.purge(modal3d);
            }
        }

        function openModal() {
            document.body.style.overflow = 'hidden';
            modal.removeAttribute('hidden');
        }

        let rgbZoomScale = 1;

        function setupRgbZoom(imgEl, containerEl, innerEl, zoomIndicatorEl) {
            rgbZoomScale = 1;
            const w = imgEl.naturalWidth;
            const h = imgEl.naturalHeight;
            if (!w || !h) return;

            function applyZoom() {
                imgEl.style.width = w + 'px';
                imgEl.style.height = h + 'px';
                imgEl.style.transform = 'scale(' + rgbZoomScale + ')';
                imgEl.style.transformOrigin = '0 0';
                innerEl.style.width = (w * rgbZoomScale) + 'px';
                innerEl.style.height = (h * rgbZoomScale) + 'px';
                if (zoomIndicatorEl) zoomIndicatorEl.textContent = Math.round(rgbZoomScale * 100) + '%';
            }

            applyZoom();
            containerEl.scrollTop = 0;
            containerEl.scrollLeft = 0;

            containerEl.addEventListener('wheel', function onRgbWheel(e) {
                e.preventDefault();
                const rect = containerEl.getBoundingClientRect();
                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top;
                const scrollX = containerEl.scrollLeft;
                const scrollY = containerEl.scrollTop;

                const delta = e.deltaY > 0 ? -0.15 : 0.15;
                const newScale = Math.min(5, Math.max(0.5, rgbZoomScale + delta));
                if (newScale === rgbZoomScale) return;

                const ratio = newScale / rgbZoomScale;
                rgbZoomScale = newScale;
                applyZoom();

                containerEl.scrollLeft = (scrollX + mx) * ratio - mx;
                containerEl.scrollTop = (scrollY + my) * ratio - my;
            }, { passive: false });
        }

        const expandRgbBtn = document.getElementById('predict-expand-rgb');
        if (expandRgbBtn) {
            expandRgbBtn.onclick = function() {
                const img = document.getElementById('predict-galaxy-rgb-img');
                if (!img) return;
                modalTitle.textContent = 'RGB Image';
                modalImage.innerHTML = '<div class="expand-modal-image-zoom"><span class="rgb-zoom-percentage" aria-live="polite">100%</span><div class="expand-modal-image-inner"><img src="' + img.src + '" alt="Galaxy RGB cutout (expanded)" class="expand-modal-img"></div></div>';
                modalImage.hidden = false;
                modal3dWrapper.hidden = true;
                openModal();

                const zoomContainer = modalImage.querySelector('.expand-modal-image-zoom');
                const innerEl = modalImage.querySelector('.expand-modal-image-inner');
                const modalImg = modalImage.querySelector('.expand-modal-img');
                const zoomIndicator = modalImage.querySelector('.rgb-zoom-percentage');
                if (zoomContainer && innerEl && modalImg) {
                    if (modalImg.complete && modalImg.naturalWidth) {
                        setupRgbZoom(modalImg, zoomContainer, innerEl, zoomIndicator);
                    } else {
                        modalImg.addEventListener('load', function() {
                            setupRgbZoom(modalImg, zoomContainer, innerEl, zoomIndicator);
                        });
                    }
                }
            };
        }

        const expand3dBtn = document.getElementById('predict-expand-3d');
        if (expand3dBtn) {
            expand3dBtn.onclick = function() {
                if (!plot3dData) return;
                setupModalLuptonControls();
                syncModalLuptonFromMain();
                modalTitle.textContent = '3D Surface (g, r, z channels)';
                modalImage.hidden = true;
                modal3dWrapper.hidden = false;
                modal3d.innerHTML = '';
                ['g', 'r', 'z'].forEach(name => {
                    const cb = modal3dWrapper.querySelector('input[value="' + name + '"]');
                    if (cb) cb.checked = true;
                });
                const solidMain = document.getElementById('predict-solid-colors-main');
                const solidModal = document.getElementById('predict-solid-colors-modal');
                const opacity = solidMain && solidMain.checked ? 1 : 0.75;
                const traces = buildTraces(plot3dData, opacity);
                Plotly.newPlot(modal3d, traces, buildLayout({ t: 20, r: 20, b: 20, l: 0 }, true), { responsive: true });
                setupChannelToggles('predict-channel-toggles-modal', 'predict-expand-modal-3d');
                const zwrapModal = document.getElementById('predict-zwrap-modal');
                if (zwrapModal) zwrapModal.checked = document.getElementById('predict-zwrap-main') ? document.getElementById('predict-zwrap-main').checked : false;
                setupZAxisWrapToggle('predict-zwrap-modal', 'predict-expand-modal-3d');
                if (zwrapModal && zwrapModal.checked) updateZAxisWrap('predict-expand-modal-3d', true);
                if (solidModal) solidModal.checked = solidMain ? solidMain.checked : false;
                setupSolidColorsToggle('predict-solid-colors-modal', 'predict-expand-modal-3d');
                openModal();
                requestAnimationFrame(function() { Plotly.Plots.resize(modal3d); });
            };
        }

        closeBtn.addEventListener('click', closeModal);
        if (backdrop) backdrop.addEventListener('click', closeModal);
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && !modal.hasAttribute('hidden')) closeModal();
        });
    }

    let predict3dDebounce = null;

    function applyLuptonValues(stretch, q) {
        var stretchEl = document.getElementById('predict-lupton-stretch');
        var qEl = document.getElementById('predict-lupton-q');
        var stretchVal = document.getElementById('predict-lupton-stretch-value');
        var qVal = document.getElementById('predict-lupton-q-value');
        var stretchModal = document.getElementById('predict-lupton-stretch-modal');
        var qModal = document.getElementById('predict-lupton-q-modal');
        var stretchValModal = document.getElementById('predict-lupton-stretch-value-modal');
        var qValModal = document.getElementById('predict-lupton-q-value-modal');
        if (stretchEl) stretchEl.value = stretch;
        if (qEl) qEl.value = q;
        if (stretchVal) stretchVal.textContent = stretch;
        if (qVal) qVal.textContent = q;
        if (stretchModal) stretchModal.value = stretch;
        if (qModal) qModal.value = q;
        if (stretchValModal) stretchValModal.textContent = stretch;
        if (qValModal) qValModal.textContent = q;
        var img = document.getElementById('predict-galaxy-rgb-img');
        if (img && img.dataset.baseUrl) {
            var baseUrl = img.dataset.baseUrl;
            var sep = baseUrl.includes('?') ? '&' : '?';
            img.src = baseUrl + sep + 'stretch=' + encodeURIComponent(stretch) + '&q=' + encodeURIComponent(q);
        }
        if (predict3dDebounce) clearTimeout(predict3dDebounce);
        predict3dDebounce = setTimeout(function() {
            predict3dDebounce = null;
            fetchPredict3dData(stretch, q);
        }, 200);
    }

    function showPredict3dLoading(container) {
        if (!container) return;
        if (container.querySelector && container.querySelector('.plotly')) Plotly.purge(container);
        container.innerHTML = '<div class="plot3d-placeholder plot3d-loading">Loading...</div>';
    }

    function showPredict3dError(container, message, onRetry) {
        if (!container) return;
        if (container.querySelector && container.querySelector('.plotly')) Plotly.purge(container);
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn-retry';
        btn.textContent = 'Retry...';
        btn.addEventListener('click', onRetry);
        container.innerHTML = '';
        const wrap = document.createElement('div');
        wrap.className = 'plot3d-placeholder plot3d-error';
        const msgSpan = document.createElement('span');
        msgSpan.className = 'plot3d-error-msg';
        msgSpan.textContent = message || 'Failed to load 3D data';
        wrap.appendChild(msgSpan);
        wrap.appendChild(btn);
        container.appendChild(wrap);
    }

    function fetchPredict3dData(stretch, q) {
        const card = document.querySelector('#result-area .predict-result-card');
        if (!card) return;
        const ra = card.dataset.ra;
        const dec = card.dataset.dec;
        const fitsId = card.dataset.fitsId;
        const container = document.getElementById('predict-3d-container');
        if (!container) return;
        const url = fitsId
            ? '/api/galaxy_3d_data_fits?fits_id=' + encodeURIComponent(fitsId) + '&stretch=' + encodeURIComponent(stretch) + '&q=' + encodeURIComponent(q)
            : (ra && dec ? '/api/galaxy_3d_data?ra=' + encodeURIComponent(ra) + '&dec=' + encodeURIComponent(dec) + '&stretch=' + encodeURIComponent(stretch) + '&q=' + encodeURIComponent(q) : null);
        if (!url) return;

        var myFetchId = ++predict3dFetchId;
        if (predict3dLoadingTimeout) {
            clearTimeout(predict3dLoadingTimeout);
            predict3dLoadingTimeout = null;
        }
        if (predict3dFirstLoad) {
            predict3dFirstLoad = false;
            showPredict3dLoading(container);
        } else {
            predict3dLoadingTimeout = setTimeout(function() {
                predict3dLoadingTimeout = null;
                if (myFetchId === predict3dFetchId) {
                    var c = document.getElementById('predict-3d-container');
                    if (c) showPredict3dLoading(c);
                }
            }, PLOT3D_LOADING_DELAY_MS);
        }

        fetch(url)
            .then(function(r) {
                return r.json().then(function(data) {
                    if (!r.ok) {
                        const msg = (data && data.detail) ? data.detail : 'Request failed (' + r.status + ')';
                        throw new Error(msg);
                    }
                    return data;
                });
            })
            .then(function(data) {
                if (myFetchId === predict3dFetchId && predict3dLoadingTimeout) {
                    clearTimeout(predict3dLoadingTimeout);
                    predict3dLoadingTimeout = null;
                }
                if (!isValid3dData(data)) {
                    throw new Error('Invalid 3D data format');
                }
                const el = document.getElementById('predict-3d-container');
                if (el && el.data) {
                    updatePlotData(data);
                } else {
                    if (el) el.innerHTML = '';
                    initPlot(data);
                }
                var modal3dWrapper = document.getElementById('predict-expand-modal-3d-wrapper');
                var modal3d = document.getElementById('predict-expand-modal-3d');
                if (modal3dWrapper && !modal3dWrapper.hidden && modal3d && modal3d.querySelector('.plotly')) {
                    if (modal3d.data) {
                        updatePlotDataForElement(modal3d, data);
                    } else {
                        initPlotForElement(modal3d, data, buildLayout({ t: 20, r: 20, b: 20, l: 0 }, true));
                    }
                }
                setupExpandModal();
            })
            .catch(function(err) {
                if (myFetchId === predict3dFetchId && predict3dLoadingTimeout) {
                    clearTimeout(predict3dLoadingTimeout);
                    predict3dLoadingTimeout = null;
                }
                const msg = err && err.message ? err.message : String(err);
                showPredict3dError(container, 'Failed to load 3D data: ' + msg, function() {
                    fetchPredict3dData(stretch, q);
                });
                setupExpandModal();
            });
    }

    function updatePredict3dPlot() {
        var stretchEl = document.getElementById('predict-lupton-stretch');
        var qEl = document.getElementById('predict-lupton-q');
        if (!stretchEl || !qEl) return;
        applyLuptonValues(stretchEl.value, qEl.value);
    }

    function setupPredictLuptonControls() {
        var img = document.getElementById('predict-galaxy-rgb-img');
        var stretchEl = document.getElementById('predict-lupton-stretch');
        var qEl = document.getElementById('predict-lupton-q');
        if (!img || !stretchEl || !qEl) return;
        img.onerror = function() {
            img.alt = 'Failed to load galaxy image';
            img.classList.add('image-load-error');
        };
        img.onload = function() {
            img.classList.remove('image-load-error');
        };
        if (img.complete && img.naturalWidth === 0) {
            img.alt = 'Failed to load galaxy image';
            img.classList.add('image-load-error');
        }
        stretchEl.addEventListener('input', function() {
            applyLuptonValues(stretchEl.value, qEl.value);
        });
        qEl.addEventListener('input', function() {
            applyLuptonValues(stretchEl.value, qEl.value);
        });
    }

    var modalLuptonInitialized = false;

    function setupModalLuptonControls() {
        if (modalLuptonInitialized) return;
        var stretchModal = document.getElementById('predict-lupton-stretch-modal');
        var qModal = document.getElementById('predict-lupton-q-modal');
        if (!stretchModal || !qModal) return;
        modalLuptonInitialized = true;
        stretchModal.addEventListener('input', function() {
            applyLuptonValues(stretchModal.value, qModal.value);
        });
        qModal.addEventListener('input', function() {
            applyLuptonValues(stretchModal.value, qModal.value);
        });
    }

    function syncModalLuptonFromMain() {
        var stretchEl = document.getElementById('predict-lupton-stretch');
        var qEl = document.getElementById('predict-lupton-q');
        var stretchModal = document.getElementById('predict-lupton-stretch-modal');
        var qModal = document.getElementById('predict-lupton-q-modal');
        var stretchValModal = document.getElementById('predict-lupton-stretch-value-modal');
        var qValModal = document.getElementById('predict-lupton-q-value-modal');
        if (!stretchModal || !qModal) return;
        var stretch = stretchEl ? stretchEl.value : '0.5';
        var q = qEl ? qEl.value : '6';
        stretchModal.value = stretch;
        qModal.value = q;
        if (stretchValModal) stretchValModal.textContent = stretch;
        if (qValModal) qValModal.textContent = q;
    }

    function syncFastPrediction() {
        const cb = document.getElementById('fast-prediction');
        const urlInput = document.getElementById('fast-prediction-url');
        const fitsInput = document.getElementById('fast-prediction-fits');
        const val = cb && cb.checked ? 'true' : 'false';
        if (urlInput) urlInput.value = val;
        if (fitsInput) fitsInput.value = val;
    }

    function init() {
        const fastCb = document.getElementById('fast-prediction');
        if (fastCb) {
            fastCb.addEventListener('change', syncFastPrediction);
            syncFastPrediction();
        }

        document.addEventListener('change', function(e) {
            if (e.target && e.target.matches('input[name=fits_file]')) {
                const urlInput = document.getElementById('legacy_survey_url');
                if (urlInput) urlInput.value = '';
            }
        });

        htmx.on('htmx:beforeRequest', function(evt) {
            if (evt.detail.elt.closest('.predict-form')) {
                syncFastPrediction();
                const form = evt.detail.elt.closest('form');
                if (form && form.querySelector('input[name=legacy_survey_url]')) {
                    const fitsInput = document.getElementById('fits_file');
                    if (fitsInput) fitsInput.value = '';
                }
            }
        });

        htmx.on('htmx:afterSwap', function(evt) {
            if (evt.detail.target.id === 'result-area') {
                const card = evt.detail.target.querySelector('.predict-result-card');
                if (card) {
                    predict3dFirstLoad = true;
                    setupPredictLuptonControls();
                    const container = document.getElementById('predict-3d-container');
                    if (container) {
                        fetchPredict3dData('0.5', '6');
                    } else {
                        setupExpandModal();
                    }
                }
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
