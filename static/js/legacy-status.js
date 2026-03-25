/**
 * Ping /api/debug_legacy_survey and update the Server Status indicator in the nav.
 */
(function () {
    const statusEl = document.getElementById('legacy-survey-status');
    const textEl = document.getElementById('legacy-survey-status-text');
    if (!statusEl || !textEl) return;

    const dot = statusEl.querySelector('.status-dot');
    const endpoint = '/api/debug_legacy_survey';

    function setStatus(state, label) {
        if (dot) {
            dot.className = 'status-dot status-' + state;
        }
        textEl.textContent = label;
    }

    fetch(endpoint)
        .then(function (res) {
            return res.json().catch(function () {
                return { status: 'error', error: 'Invalid response' };
            });
        })
        .then(function (data) {
            if (data.status === 'ok') {
                setStatus('ok', 'online');
            } else {
                setStatus('error', 'offline');
                if (data.error && statusEl) {
                    statusEl.title = 'Legacy Survey API: ' + data.error;
                }
            }
        })
        .catch(function (err) {
            setStatus('error', 'offline');
            if (statusEl) {
                statusEl.title = 'Legacy Survey API: ' + (err.message || 'Request failed');
            }
        });
})();
