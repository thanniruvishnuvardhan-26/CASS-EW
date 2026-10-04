let isRunning = false;
let stepInterval = null;
let currentBands = 10;
let lastActionStr = "";
let prevAction = null;
let prevSpatialInfluenced = 0;
let prevEventsLength = 0;
let prevDetected = false;
let decisionHistory = []; // Phase D

// Analytics Tracking
let outcomeChart, speedChart, bandChart;
let outcomeCounts = { intercept: 0, miss: 0, fa: 0, tn: 0 };
let lastExecTime = performance.now();
let timeHistory = [];
let savedRuns = []; // Phase E
let runCounter = 1; // Phase E

const els = {
    btnStart: document.getElementById('btn-start'),
    btnPause: document.getElementById('btn-pause'),
    btnStep: document.getElementById('btn-step'),
    btnReset: document.getElementById('btn-reset'),
    btnDemo: document.getElementById('btn-demo'),
    seedInput: document.getElementById('seed-input'),
    speedInput: document.getElementById('speed-input'),
    timeVal: document.getElementById('time-val'),
    spectrum: document.getElementById('spectrum-container'),
    map: document.getElementById('map-container'),
    currentRx: document.getElementById('current-rx-val'),
    currentBand: document.getElementById('current-band-val'),
    nextAction: document.getElementById('next-action-val'),
    decisionState: document.getElementById('decision-state-val'),
    decRx: document.getElementById('dec-rx-val'),
    decBand: document.getElementById('dec-band-val'),
    decState: document.getElementById('dec-state-val'),
    decNext: document.getElementById('dec-next-val'),
    eventLog: document.getElementById('event-log'),
    benchBody: document.getElementById('bench-body'),
    systemStatusText: document.getElementById('system-status-text'),
    scenarioSelect: document.getElementById('scenario-select'),
    algorithmSelect: document.getElementById('algorithm-select'),
    btnApplyConfig: document.getElementById('btn-apply-config'),
    activeScenarioVal: document.getElementById('active-scenario-val'),
    activeAlgorithmVal: document.getElementById('active-algorithm-val'),
    activeBandsVal: document.getElementById('active-bands-val'),
    activeReceiversVal: document.getElementById('active-receivers-val'),
    sessionHistoryBody: document.getElementById('session-history-body'), // Phase E
    compareRun1: document.getElementById('compare-run-1'), // Phase E
    compareRun2: document.getElementById('compare-run-2'), // Phase E
    comparisonResults: document.getElementById('comparison-results'), // Phase E
    comparisonEmpty: document.getElementById('comparison-empty'), // Phase E
    comparisonBody: document.getElementById('comparison-body'), // Phase E
    compareWarning: document.getElementById('compare-warning'), // Phase E
    compareTh1: document.getElementById('compare-th-1'), // Phase E
    compareTh2: document.getElementById('compare-th-2'), // Phase E
    
    // Phase F
    btnPresentation: document.getElementById('btn-presentation'),
    wtOverlay: document.getElementById('walkthrough-overlay'),
    wtTitle: document.getElementById('wt-title'),
    wtProgress: document.getElementById('wt-progress'),
    wtDesc: document.getElementById('wt-desc'),
    btnWtClose: document.getElementById('btn-wt-close'),
    btnWtSkip: document.getElementById('btn-wt-skip'),
    btnWtNext: document.getElementById('btn-wt-next')
};

function initCharts() {
    if (typeof Chart === 'undefined') return;
    
    Chart.defaults.color = '#94A3B8';
    Chart.defaults.borderColor = '#2C3946';
    Chart.defaults.font.family = '"Inter", sans-serif';

    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
            x: { type: 'linear', position: 'bottom', title: { display: true, text: 'RF Time' } }
        },
        plugins: { legend: { position: 'top', labels: { boxWidth: 10, padding: 15 } } }
    };

    if (outcomeChart) outcomeChart.destroy();
    outcomeChart = new Chart(document.getElementById('outcomeChart'), {
        type: 'line',
        data: {
            datasets: [
                { label: 'Intercepted', data: [], borderColor: '#10B981', backgroundColor: '#10B981', pointRadius: 0, borderWidth: 2 },
                { label: 'True Negative', data: [], borderColor: '#0EA5E9', backgroundColor: '#0EA5E9', pointRadius: 0, borderWidth: 2 },
                { label: 'False Alarm', data: [], borderColor: '#F59E0B', backgroundColor: '#F59E0B', pointRadius: 0, borderWidth: 2 },
                { label: 'Miss', data: [], borderColor: '#EF4444', backgroundColor: '#EF4444', pointRadius: 0, borderWidth: 2 }
            ]
        },
        options: commonOptions
    });

    if (speedChart) speedChart.destroy();
    speedChart = new Chart(document.getElementById('speedChart'), {
        type: 'line',
        data: {
            datasets: [{ label: 'Dashboard Update Rate (Steps/sec)', data: [], borderColor: '#8B5CF6', backgroundColor: 'rgba(139,92,246,0.1)', fill: true, pointRadius: 0, borderWidth: 2, tension: 0.3 }]
        },
        options: Object.assign({}, commonOptions, {
            scales: { y: { beginAtZero: true, suggestedMax: 50 } }
        })
    });

    if (bandChart) bandChart.destroy();
    bandChart = new Chart(document.getElementById('bandChart'), {
        type: 'line',
        data: {
            datasets: [{ label: 'Tuned Band', data: [], borderColor: '#0EA5E9', backgroundColor: 'transparent', pointRadius: 2, stepped: true, borderWidth: 2 }]
        },
        options: Object.assign({}, commonOptions, {
            scales: {
                x: { type: 'linear', position: 'bottom', title: { display: true, text: 'RF Time' } },
                y: { min: -1, max: 10, ticks: { stepSize: 1 } }
            },
            plugins: { legend: { display: false } }
        })
    });
}

function setSystemState(stateStr) {
    if (els.systemStatusText) els.systemStatusText.innerText = stateStr;
}

function initSpectrum(num_bands) {
    els.spectrum.innerHTML = '';
    currentBands = num_bands;
    if (bandChart && bandChart.options.scales.y) {
        bandChart.options.scales.y.max = num_bands;
        bandChart.update();
    }
    for (let i = 0; i < num_bands; i++) {
        const d = document.createElement('div');
        d.className = 'band';
        d.id = `band-${i}`;
        d.innerHTML = `
            <div class="band-bar"><div class="band-fill" id="band-fill-${i}"></div></div>
            <div class="band-id">B${i}</div>
            <div class="band-status"><span class="status-indicator" id="band-status-${i}">○</span></div>
        `;
        els.spectrum.appendChild(d);
    }
}

async function resetSim(seed, scenario = null, algorithm = null) {
    const currentSteps = parseInt(els.timeVal.innerText) || 0;
    if (currentSteps > 0) {
        saveSnapshot(currentSteps);
    }

    const payload = { seed: seed };
    if (scenario) payload.scenario = scenario;
    if (algorithm) payload.algorithm = algorithm;

    const res = await fetch('/api/reset', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
    });
    const data = await res.json();
    initSpectrum(data.num_bands);
    els.timeVal.innerText = "0";
    if (els.currentScan) els.currentScan.innerText = "-";
    els.nextAction.innerText = "AWAITING START...";
    els.decisionState.innerText = "-";
    lastActionStr = "";
    prevAction = null;
    els.eventLog.innerHTML = "";
    updateMetrics({
        interception_rate: 0, total_obs: 0, total_det: 0, 
        exploration: 0, exploitation: 0, spatial_influenced: 0
    });
    prevSpatialInfluenced = 0;
    prevEventsLength = 0;
    prevDetected = false;
    decisionHistory = []; // Phase D
    const histBody = document.getElementById('history-body');
    if (histBody) histBody.innerHTML = '';
    
    // Reset Analytics
    outcomeCounts = { intercept: 0, miss: 0, fa: 0, tn: 0 };
    if (outcomeChart) {
        outcomeChart.data.datasets.forEach(ds => ds.data = []);
        outcomeChart.update();
    }
    if (speedChart) {
        speedChart.data.datasets[0].data = [];
        speedChart.update();
    }
    if (bandChart) {
        bandChart.data.datasets[0].data = [];
        bandChart.update();
    }
    
    if (els.activeScenarioVal) els.activeScenarioVal.innerText = data.active_scenario || "Mixed Environment";
    if (els.activeAlgorithmVal) els.activeAlgorithmVal.innerText = data.active_algorithm || "Phase 8 Spatial Integrated";
    if (els.activeBandsVal) els.activeBandsVal.innerText = data.num_bands;
    if (els.activeReceiversVal) els.activeReceiversVal.innerText = "3 (R1, R2, R3)";

    setSystemState("SYSTEM READY");
    updateIT({
        mean_intercept_time: null, median_intercept_time: null,
        eligible_episodes: 0, intercepted_episodes: 0, missed_episodes: 0
    });
    clearMap();
    const priorityBody = document.getElementById('priority-list-body');
    if (priorityBody) {
        priorityBody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-dark); padding: 16px;">Awaiting decision...</td></tr>';
    }
    document.body.classList.remove('is-running', 'is-paused');
}

async function stepSim() {
    const res = await fetch('/api/step', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({steps: 1})
    });
    const data = await res.json();
    if (data.error) return;
    
    updateUI(data);
}

function updateUI(data) {
    els.timeVal.innerText = data.time;
    
    // Analytics: Speed Tracking
    let now = performance.now();
    let elapsedSec = (now - lastExecTime) / 1000;
    lastExecTime = now;
    let speed = elapsedSec > 0 ? (1 / elapsedSec) : 0;
    
    // Update Spectrum
    let activeBeliefs = null;
    if (data.action && data.beliefs && data.beliefs[data.action.receiver]) {
        activeBeliefs = data.beliefs[data.action.receiver].belief;
    }

    for(let i=0; i<currentBands; i++) {
        const bandEl = document.getElementById(`band-${i}`);
        const statEl = document.getElementById(`band-status-${i}`);
        const fillEl = document.getElementById(`band-fill-${i}`);
        if (bandEl) bandEl.classList.remove('active', 'detected');
        if (statEl) statEl.innerText = "○";
        
        if (activeBeliefs && i < activeBeliefs.length) {
            let heightPct = Math.max(5, Math.min(100, activeBeliefs[i] * 100));
            if (fillEl) fillEl.style.height = `${heightPct}%`;
        } else {
            if (fillEl) fillEl.style.height = "5%";
        }
    }
    
    if (data.action) {
        let currentRxStr = "-";
        let currentBandStr = "-";
        
        if (prevAction) {
            currentRxStr = `${prevAction.receiver}`;
            currentBandStr = `BAND ${prevAction.band}`;
        } else {
            currentRxStr = `${data.action.receiver}`;
            currentBandStr = `BAND ${data.action.band}`;
        }
        
        const nextActionStr = `${data.action.receiver} → BAND ${data.action.band}`;
        
        if (nextActionStr !== lastActionStr) {
            lastActionStr = nextActionStr;
            highlightDecisionEvent();
        }
        
        els.currentRx.innerText = currentRxStr;
        els.currentBand.innerText = currentBandStr;
        els.nextAction.innerText = nextActionStr;
        
        if(els.decRx) els.decRx.innerText = currentRxStr;
        if(els.decBand) els.decBand.innerText = currentBandStr;
        if(els.decNext) els.decNext.innerText = nextActionStr;
        
        const bEl = document.getElementById(`band-${data.action.band}`);
        if (bEl) bEl.classList.add('active');
        const st = document.getElementById(`band-status-${data.action.band}`);
        
        if (data.action.detected) {
            if (bEl) {
                bEl.classList.add('detected');
                if (!prevDetected) {
                    // edge trigger detection event class
                    bEl.classList.add('detection-event');
                    setTimeout(() => bEl.classList.remove('detection-event'), 1000);
                }
            }
            if (st) st.innerText = "●";
        } else {
            if (st) st.innerText = "▌";
        }
        prevDetected = data.action.detected;
        
        // Explainability (Section 8 Standard)
        const rid = data.action.receiver;
        const b = data.action.band;
        
        updateValue('why-rx', rid);
        updateValue('why-band', 'BAND ' + b);
        
        const de = data.decision_explanation || {};
        const km = (data.knowledge_map && data.knowledge_map[rid] && data.knowledge_map[rid][b]) ? data.knowledge_map[rid][b] : null;

        let belVal = de.belief_contribution !== undefined ? parseFloat(de.belief_contribution).toFixed(3) : "N/A";
        let uncVal = de.uncertainty_contribution !== undefined ? parseFloat(de.uncertainty_contribution).toFixed(3) : (km ? parseFloat(km.uncertainty).toFixed(3) : "N/A");
        let tempVal = de.temporal_contribution !== undefined ? parseFloat(de.temporal_contribution).toFixed(3) : "N/A";
        let predVal = de.prediction_contribution !== undefined ? parseFloat(de.prediction_contribution).toFixed(3) : "N/A";
        let patVal = de.pattern_contribution !== undefined ? parseFloat(de.pattern_contribution).toFixed(3) : "N/A";
        let spatVal = de.spatial_contribution !== undefined ? parseFloat(de.spatial_contribution).toFixed(3) : "N/A";
        let staleVal = de.staleness_contribution !== undefined ? parseInt(de.staleness_contribution) : (km ? parseInt(km.staleness) : "N/A");
        let hopVal = de.hop_contribution !== undefined ? parseFloat(de.hop_contribution).toFixed(3) : (km ? parseFloat(km.hop_probability).toFixed(3) : "N/A");
        let predConfVal = km && km.prediction_confidence !== undefined ? parseFloat(km.prediction_confidence).toFixed(3) : "N/A";

        if (belVal === "N/A" && data.beliefs && data.beliefs[rid]) {
            const bp = data.beliefs[rid];
            belVal = bp.belief ? parseFloat(bp.belief[b]).toFixed(3) : "N/A";
            tempVal = bp.temporal ? parseFloat(bp.temporal[b]).toFixed(3) : "N/A";
            predVal = bp.prediction ? parseFloat(bp.prediction[b]).toFixed(3) : "N/A";
        }
        
        updateValue('why-belief', belVal);
        updateValue('why-uncertainty', uncVal);
        updateValue('why-temporal', tempVal);
        updateValue('why-pred', predVal);
        updateValue('why-pred-conf', predConfVal);
        updateValue('why-pattern', patVal);
        updateValue('why-spatial', spatVal !== "N/A" ? spatVal : (data.metrics.spatial_influenced > prevSpatialInfluenced ? "Active" : "0.000"));
        updateValue('why-staleness', staleVal);
        updateValue('why-hop', hopVal);
        
        const isExploration = de.is_exploration !== undefined ? de.is_exploration : (data.metrics.exploration > document.getElementById('perf-explor').innerText * 1);
        const isAntiStarvation = de.is_anti_starvation !== undefined ? de.is_anti_starvation : false;
        let stateStr = "EXPLOITATION";
        if (isAntiStarvation) stateStr = "ANTI-STARVATION";
        else if (isExploration) stateStr = "EXPLORATION";

        updateValue('why-state', stateStr);
        els.decisionState.innerText = stateStr;
        if(els.decState) els.decState.innerText = stateStr;
        
        const summaryEl = document.getElementById('why-summary');
        if (summaryEl) {
            let reasonMsg = de.reason || (isExploration ? "Epsilon exploration scan" : "Highest fused cognitive priority score");
            summaryEl.innerHTML = `Receiver <strong>${rid}</strong> commanded to <strong>BAND ${b}</strong>.<br><strong>Decision Reason</strong>: "${reasonMsg}".`;
        }

        // Real-Time Adaptive Priority List Update
        const priorityBody = document.getElementById('priority-list-body');
        if (priorityBody && de.priority_list && Array.isArray(de.priority_list) && de.priority_list.length > 0) {
            priorityBody.innerHTML = de.priority_list.map(cand => {
                const isSelected = cand.selected;
                const isExplore = cand.status === 'SELECTED (EXPLORE)';
                let rowClass = '';
                if (isSelected) {
                    rowClass = isExplore ? 'priority-explore-row' : 'priority-selected-row';
                }
                
                let statusHtml = '<span style="color: var(--text-dark);">-</span>';
                if (cand.status) {
                    const badgeClass = isExplore ? 'priority-status-explore' : 'priority-status-selected';
                    statusHtml = `<span class="priority-status-badge ${badgeClass}">${cand.status}</span>`;
                }

                const rxLabel = cand.receiver ? `${cand.receiver} : ` : '';
                const bandLabel = `${rxLabel}B${cand.band}`;
                const scoreStr = typeof cand.score === 'number' ? cand.score.toFixed(3) : cand.score;

                return `
                    <tr class="${rowClass}">
                        <td style="padding: 6px 10px; font-weight: 700; color: ${isSelected ? 'var(--accent-core)' : 'var(--text-muted)'};">${cand.rank}</td>
                        <td style="padding: 6px 10px; font-weight: 600; color: ${isSelected ? 'var(--text-main)' : 'var(--text-muted)'};">${bandLabel}</td>
                        <td class="priority-score-cell" style="padding: 6px 10px; color: ${isSelected ? 'var(--text-main)' : 'var(--text-dark)'};">${scoreStr}</td>
                        <td style="padding: 6px 10px;">${statusHtml}</td>
                    </tr>
                `;
            }).join('');
        }

        // Pipeline UI highlight
        const pipeStages = ['pipe-obs', 'pipe-rx', 'pipe-belief', 'pipe-temp', 'pipe-pat', 'pipe-dec', 'pipe-res', 'pipe-adapt'];
        pipeStages.forEach((id, idx) => {
            const el = document.getElementById(id);
            if (el) {
                setTimeout(() => {
                    el.classList.remove('active');
                    void el.offsetWidth;
                    el.classList.add('active');
                }, idx * 50); // Cascade effect
            }
        });
        
        // Analytics: Outcome Tracking
        let was_active = data.ground_truth.active_bands.includes(data.action.band);
        if (data.action.detected && was_active) outcomeCounts.intercept++;
        else if (!data.action.detected && !was_active) outcomeCounts.tn++;
        else if (data.action.detected && !was_active) outcomeCounts.fa++;
        else outcomeCounts.miss++;
        
        if (outcomeChart && bandChart && speedChart) {
            // Keep history window to last 200 items to prevent lag
            const maxPoints = 200;
            const pushData = (dataset, x, y) => {
                dataset.data.push({ x: x, y: y });
                if (dataset.data.length > maxPoints) dataset.data.shift();
            };
            
            pushData(outcomeChart.data.datasets[0], data.time, outcomeCounts.intercept);
            pushData(outcomeChart.data.datasets[1], data.time, outcomeCounts.tn);
            pushData(outcomeChart.data.datasets[2], data.time, outcomeCounts.fa);
            pushData(outcomeChart.data.datasets[3], data.time, outcomeCounts.miss);
            outcomeChart.update();
            
            pushData(bandChart.data.datasets[0], data.time, data.action.band);
            bandChart.update();
            
            // smooth out speed display somewhat
            let recentSpeed = speed;
            if (speedChart.data.datasets[0].data.length > 0) {
                recentSpeed = (speedChart.data.datasets[0].data[speedChart.data.datasets[0].data.length-1].y * 0.7) + (speed * 0.3);
            }
            pushData(speedChart.data.datasets[0], data.time, recentSpeed);
            speedChart.update();
        }
        
        prevAction = data.action;
    }
    
    // Check spatial influence
    let isSpatial = data.metrics.spatial_influenced > prevSpatialInfluenced;
    prevSpatialInfluenced = data.metrics.spatial_influenced;

    // Update Map
    renderMap(data.receivers, data.ground_truth.emitters, data.action ? data.action.receiver : null, isSpatial);
    
    // Update Metrics
    updateMetrics(data.metrics);
    updateIT(data.intercept_time);
    
    // Update Log
    const numNew = Math.max(0, data.events.length - prevEventsLength);
    els.eventLog.innerHTML = data.events.map((e, idx) => {
        const isNew = idx >= prevEventsLength;
        return `<div class="log-entry ${isNew ? 'value-update' : ''}">${e}</div>`;
    }).reverse().join('');
    prevEventsLength = data.events.length;
    
    // Update Phase C Analytics
    updatePhaseCAnalytics(data);

    // Update Phase D Decision History
    updateDecisionHistory(data);
}

function updateDecisionHistory(data) {
    if (!data.action) return;
    
    const histBody = document.getElementById('history-body');
    if (!histBody) return;

    const rx = data.action.receiver;
    const band = data.action.band;
    const detected = data.action.detected ? '<span style="color: #10B981">DETECTED</span>' : '<span style="color: #64748B">NO DETECT</span>';
    
    const bp = data.beliefs && data.beliefs[rx] ? data.beliefs[rx] : null;
    const formatMetric = (val) => val === null || val === undefined ? 'N/A' : parseFloat(val).toFixed(3);
    
    const belief = bp && bp.belief ? formatMetric(bp.belief[band]) : 'N/A';
    const temporal = bp && bp.temporal ? formatMetric(bp.temporal[band]) : 'N/A';
    const prediction = bp && bp.prediction ? formatMetric(bp.prediction[band]) : 'N/A';
    
    const entry = {
        time: data.time,
        rx: rx,
        band: band,
        result: detected,
        belief: belief,
        temporal: temporal,
        prediction: prediction
    };
    
    decisionHistory.unshift(entry);
    if (decisionHistory.length > 50) {
        decisionHistory.pop();
    }
    
    let html = '';
    decisionHistory.forEach(row => {
        html += `<tr>
            <td>${row.time}</td>
            <td>${row.rx}</td>
            <td>B${row.band}</td>
            <td>${row.result}</td>
            <td>${row.belief}</td>
            <td>${row.temporal}</td>
            <td>${row.prediction}</td>
        </tr>`;
    });
    histBody.innerHTML = html;
}

function updatePhaseCAnalytics(data) {
    const allBandsBody = document.getElementById('all-bands-body');
    const currentBandContent = document.getElementById('current-band-analysis-content');
    const nearestBandsBody = document.getElementById('nearest-bands-body');

    if (!allBandsBody || !currentBandContent || !nearestBandsBody) return;

    let selectedBand = null;
    let selectedRx = null;
    if (data.action) {
        selectedBand = data.action.band;
        selectedRx = data.action.receiver;
    }

    if (selectedRx === null || !data.beliefs || !data.beliefs[selectedRx]) {
        return; // No data yet
    }

    const bp = data.beliefs[selectedRx];
    
    // Build All Bands Table
    let allHtml = '';
    const formatMetric = (val) => val === null || val === undefined ? 'N/A' : parseFloat(val).toFixed(3);
    const formatInt = (val) => val === null || val === undefined ? 'N/A' : parseInt(val);

    const kmData = (data.knowledge_map && data.knowledge_map[selectedRx]) ? data.knowledge_map[selectedRx] : null;

    for (let i = 0; i < currentBands; i++) {
        const isSelected = i === selectedBand;
        const rowClass = isSelected ? 'style="background-color: rgba(14, 165, 233, 0.1);"' : '';
        const badge = isSelected ? '<span class="badge" style="background-color: #0EA5E9; margin-left: 8px;">SELECTED</span>' : '';
        
        const kmEntry = kmData && kmData[i] ? kmData[i] : null;

        const belief = kmEntry ? formatMetric(kmEntry.belief) : (bp.belief ? formatMetric(bp.belief[i]) : 'N/A');
        const unc = kmEntry ? formatMetric(kmEntry.uncertainty) : 'N/A';
        const stale = kmEntry ? formatInt(kmEntry.staleness) : 'N/A';
        const temp = kmEntry ? formatMetric(kmEntry.temporal_score) : (bp.temporal ? formatMetric(bp.temporal[i]) : 'N/A');
        const pred = kmEntry ? formatMetric(kmEntry.prediction_score) : (bp.prediction ? formatMetric(bp.prediction[i]) : 'N/A');
        const predConf = kmEntry ? formatMetric(kmEntry.prediction_confidence) : 'N/A';
        const hopProb = kmEntry ? formatMetric(kmEntry.hop_probability) : 'N/A';
        const obs = bp.observations ? formatInt(bp.observations[i]) : 'N/A';
        const det = bp.detections ? formatInt(bp.detections[i]) : 'N/A';

        allHtml += `<tr ${rowClass}>
            <td>B${i} ${badge}</td>
            <td>${belief}</td>
            <td>${unc}</td>
            <td>${stale}</td>
            <td>${temp}</td>
            <td>${pred}</td>
            <td>${predConf}</td>
            <td>${hopProb}</td>
            <td>${obs}</td>
            <td>${det}</td>
        </tr>`;
    }
    allBandsBody.innerHTML = allHtml;

    // Current Band Analysis
    if (selectedBand !== null) {
        const belief = bp.belief ? formatMetric(bp.belief[selectedBand]) : 'N/A';
        const obs = bp.observations ? formatInt(bp.observations[selectedBand]) : 'N/A';
        const det = bp.detections ? formatInt(bp.detections[selectedBand]) : 'N/A';
        const lastObs = bp.last_observed ? (bp.last_observed[selectedBand] >= 0 ? formatInt(bp.last_observed[selectedBand]) : 'N/A') : 'N/A';
        const temp = bp.temporal ? formatMetric(bp.temporal[selectedBand]) : 'N/A';
        const pred = bp.prediction ? formatMetric(bp.prediction[selectedBand]) : 'N/A';
        const freshness = (lastObs !== 'N/A' && data.time !== undefined) ? (data.time - parseInt(lastObs)) : 'N/A';
        
        currentBandContent.innerHTML = `
            <div style="font-size: 16px; font-weight: bold; color: #fff; margin-bottom: 8px;">CURRENT BAND: B${selectedBand}</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px;">
                <span style="color: var(--text-muted);">BELIEF</span><span class="highlight-blue">${belief}</span>
                <span style="color: var(--text-muted);">OBSERVATIONS</span><span class="highlight-blue">${obs}</span>
                <span style="color: var(--text-muted);">DETECTIONS</span><span class="highlight-blue">${det}</span>
                <span style="color: var(--text-muted);">LAST OBSERVED</span><span class="highlight-blue">${lastObs}</span>
                <span style="color: var(--text-muted);">FRESHNESS</span><span class="highlight-blue">${freshness}</span>
                <span style="color: var(--text-muted);">TEMPORAL EVIDENCE</span><span class="highlight-blue">${temp}</span>
                <span style="color: var(--text-muted);">PREDICTION EVIDENCE</span><span class="highlight-blue">${pred}</span>
            </div>
        `;
    }

    // Nearest-5 Band Comparison
    if (selectedBand !== null) {
        let nearestBands = [];
        // Calculate deterministic index proximity
        for (let i = 0; i < currentBands; i++) {
            if (i !== selectedBand) {
                nearestBands.push({ band: i, dist: Math.abs(i - selectedBand) });
            }
        }
        // Sort by distance, then by band index for deterministic tie-breaking
        nearestBands.sort((a, b) => {
            if (a.dist !== b.dist) return a.dist - b.dist;
            return a.band - b.band;
        });
        
        const top5 = nearestBands.slice(0, 5).map(x => x.band);
        let compareBands = [selectedBand, ...top5];
        // Sort bands nicely for display (optional, or display selected first)
        
        let nearHtml = '';
        compareBands.forEach(i => {
            const isSelected = i === selectedBand;
            const rowClass = isSelected ? 'style="background-color: rgba(14, 165, 233, 0.1);"' : '';
            const badge = isSelected ? '<span class="badge" style="background-color: #0EA5E9; margin-left: 8px;">SELECTED</span>' : '';
            
            const belief = bp.belief ? formatMetric(bp.belief[i]) : 'N/A';
            const obs = bp.observations ? formatInt(bp.observations[i]) : 'N/A';
            const det = bp.detections ? formatInt(bp.detections[i]) : 'N/A';

            nearHtml += `<tr ${rowClass}>
                <td>B${i} ${badge}</td>
                <td>${belief}</td>
                <td>${obs}</td>
                <td>${det}</td>
            </tr>`;
        });
        nearestBandsBody.innerHTML = nearHtml;
    }
}

function updateValue(id, newValue) {
    const el = document.getElementById(id);
    if (!el) return;
    const current = el.innerText;
    if (current !== String(newValue)) {
        el.innerText = newValue;
        el.classList.remove('value-update');
        void el.offsetWidth; // trigger reflow
        el.classList.add('value-update');
    }
}

function highlightDecisionEvent() {
    const decisionBox = document.querySelector('.primary-decision');
    const explainBox = document.querySelector('.decision-explainability');
    if (decisionBox) {
        decisionBox.classList.remove('panel-flash');
        void decisionBox.offsetWidth;
        decisionBox.classList.add('panel-flash');
    }
    if (explainBox) {
        explainBox.classList.remove('panel-flash');
        void explainBox.offsetWidth;
        explainBox.classList.add('panel-flash');
    }
}

function updateMetrics(m) {
    updateValue('perf-ir', (m.interception_rate * 100).toFixed(2) + '%');
    updateValue('perf-obs', m.total_obs);
    updateValue('perf-det', m.total_det);
    updateValue('perf-explor', m.exploration);
    updateValue('perf-exploit', m.exploitation);
    updateValue('perf-spatial', m.spatial_influenced);
}

function updateIT(it) {
    updateValue('it-mean', it.mean_intercept_time !== null ? it.mean_intercept_time.toFixed(2) : "N/A");
    updateValue('it-median', it.median_intercept_time !== null ? it.median_intercept_time.toFixed(2) : "N/A");
    updateValue('it-eligible', it.eligible_episodes);
    updateValue('it-intercepted', it.intercepted_episodes);
    updateValue('it-missed', it.missed_episodes);
}

function renderMap(rx, emitters, activeRxId = null, isSpatial = false) {
    els.map.innerHTML = '';
    // Scale coords to map box (assume 0-15 space mapped to 10%-90%)
    const sX = x => 10 + (x / 15) * 80;
    const sY = y => 10 + (y / 15) * 80;
    
    const collisions = new Set();
    emitters.forEach(e => {
        rx.forEach(r => {
            const dx = e.position[0] - r.position[0];
            const dy = e.position[1] - r.position[1];
            const dist = Math.sqrt(dx*dx + dy*dy);
            if (dist < 1.0) { // Threshold in physical space
                collisions.add(e.name);
                collisions.add(r.id);
            }
        });
    });
    
    emitters.forEach(e => {
        const d = document.createElement('div');
        let classes = 'map-entity emitter-point' + (e.active ? ' active' : '');
        if (collisions.has(e.name)) classes += ' collision-offset';
        d.className = classes;
        d.style.left = sX(e.position[0]) + '%';
        d.style.top = sY(e.position[1]) + '%';
        d.innerHTML = `<span class="marker">◆</span><span class="entity-label">${e.name}</span>`;
        els.map.appendChild(d);
    });
    
    rx.forEach(r => {
        const d = document.createElement('div');
        const isActive = (r.id === activeRxId);
        let classes = 'map-entity receiver-point' + (isActive ? ' scanning' : '');
        if (collisions.has(r.id)) classes += ' collision-offset';
        d.className = classes;
        d.style.left = sX(r.position[0]) + '%';
        d.style.top = sY(r.position[1]) + '%';
        
        let labelText = r.id;
        if (isActive && isSpatial) {
            labelText += ' ★';
        }
        
        d.innerHTML = `<span class="marker">◉</span><span class="entity-label">${labelText}</span>`;
        els.map.appendChild(d);
    });
}

function clearMap() {
    els.map.innerHTML = '';
}

async function loadBenchmark() {
    const res = await fetch('/api/benchmark');
    const data = await res.json();
    els.benchBody.innerHTML = '';
    for (const [alg, vals] of Object.entries(data)) {
        els.benchBody.innerHTML += `
            <tr>
                <td>${alg}</td>
                <td>${vals.ir}</td>
                <td>${vals.miss}</td>
                <td>${vals.fa}</td>
                <td>${vals.eff}</td>
            </tr>
        `;
    }
}

function play() {
    if (isRunning) return;
    isRunning = true;
    document.body.classList.remove('is-paused');
    document.body.classList.add('is-running');
    setSystemState("ACTIVE AND LISTENING");
    lastExecTime = performance.now(); // Reset speed tracking so it doesn't drop after pause
    stepInterval = setInterval(stepSim, parseInt(els.speedInput.value));
}

function pause() {
    if (!isRunning && stepInterval === null) return;
    isRunning = false;
    document.body.classList.remove('is-running');
    document.body.classList.add('is-paused');
    setSystemState("SYSTEM PAUSED");
    clearInterval(stepInterval);
}

els.btnStart.addEventListener('click', play);
els.btnPause.addEventListener('click', pause);
els.btnStep.addEventListener('click', () => { pause(); stepSim(); });
els.btnReset.addEventListener('click', () => {
    pause();
    resetSim(parseInt(els.seedInput.value), els.scenarioSelect.value, els.algorithmSelect.value);
});
els.btnDemo.addEventListener('click', async () => {
    pause();
    els.seedInput.value = 42;
    await resetSim(42, els.scenarioSelect.value, els.algorithmSelect.value);
    play();
});
els.speedInput.addEventListener('input', () => {
    if (isRunning) {
        pause();
        play();
    }
});
if (els.btnApplyConfig) {
    els.btnApplyConfig.addEventListener('click', () => {
        pause();
        resetSim(parseInt(els.seedInput.value), els.scenarioSelect.value, els.algorithmSelect.value);
    });
}

// Phase E: Session Comparison
function saveSnapshot(steps) {
    const runId = `Run ${runCounter++}`;
    const ir = els.btnStart ? document.getElementById('perf-ir').innerText : '0.00%';
    const snapshot = {
        id: runId,
        scenario: els.activeScenarioVal ? els.activeScenarioVal.innerText : 'Unknown',
        algorithm: els.activeAlgorithmVal ? els.activeAlgorithmVal.innerText : 'Unknown',
        steps: steps,
        ir: ir,
        metrics: {
            'Interception Rate': ir,
            'Total Observations': document.getElementById('perf-obs').innerText,
            'Total Detections': document.getElementById('perf-det').innerText,
            'Intercepts (Hits)': outcomeCounts.intercept,
            'Misses': outcomeCounts.miss,
            'False Alarms': outcomeCounts.fa,
            'True Negatives': outcomeCounts.tn,
            'Exploration Steps': document.getElementById('perf-explor').innerText,
            'Exploitation Steps': document.getElementById('perf-exploit').innerText,
            'Spatial Influenced': document.getElementById('perf-spatial').innerText,
            'Mean Latency': document.getElementById('it-mean').innerText,
            'Median Latency': document.getElementById('it-median').innerText,
            'Eligible Episodes': document.getElementById('it-eligible').innerText,
            'Intercepted Episodes': document.getElementById('it-intercepted').innerText,
            'Missed Episodes': document.getElementById('it-missed').innerText
        }
    };

    savedRuns.unshift(snapshot);
    if (savedRuns.length > 5) {
        savedRuns.pop();
    }

    updateSessionHistoryUI();
}

function updateSessionHistoryUI() {
    if (!els.sessionHistoryBody) return;

    if (savedRuns.length === 0) {
        els.sessionHistoryBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No completed runs yet. Reset to save a run.</td></tr>`;
        return;
    }

    els.sessionHistoryBody.innerHTML = savedRuns.map(run => `
        <tr>
            <td>${run.id}</td>
            <td>${run.scenario}</td>
            <td>${run.algorithm}</td>
            <td>${run.steps}</td>
            <td>${run.ir}</td>
        </tr>
    `).join('');

    // Update Dropdowns
    const updateSelect = (selectEl, selectedId) => {
        if (!selectEl) return;
        selectEl.innerHTML = '<option value="">Select Run...</option>' + savedRuns.map(run => 
            `<option value="${run.id}" ${run.id === selectedId ? 'selected' : ''}>${run.id} (${run.algorithm})</option>`
        ).join('');
    };

    const sel1 = els.compareRun1.value;
    const sel2 = els.compareRun2.value;
    updateSelect(els.compareRun1, sel1);
    updateSelect(els.compareRun2, sel2);
    
    // Auto-select first two if available and nothing selected
    if (savedRuns.length >= 2 && !els.compareRun1.value && !els.compareRun2.value) {
        els.compareRun1.value = savedRuns[0].id;
        els.compareRun2.value = savedRuns[1].id;
        renderComparison();
    }
}

function renderComparison() {
    if (!els.compareRun1 || !els.compareRun2) return;

    const id1 = els.compareRun1.value;
    const id2 = els.compareRun2.value;

    if (!id1 || !id2) {
        els.comparisonResults.style.display = 'none';
        els.comparisonEmpty.style.display = 'block';
        return;
    }

    const run1 = savedRuns.find(r => r.id === id1);
    const run2 = savedRuns.find(r => r.id === id2);

    if (!run1 || !run2) return;

    els.comparisonEmpty.style.display = 'none';
    els.comparisonResults.style.display = 'block';

    els.compareTh1.innerText = `${run1.id} (${run1.algorithm})`;
    els.compareTh2.innerText = `${run2.id} (${run2.algorithm})`;

    if (run1.scenario !== run2.scenario || run1.steps !== run2.steps) {
        els.compareWarning.style.display = 'block';
    } else {
        els.compareWarning.style.display = 'none';
    }

    const metricsKeys = Object.keys(run1.metrics);
    els.comparisonBody.innerHTML = metricsKeys.map(key => {
        return `
            <tr>
                <td>${key}</td>
                <td>${run1.metrics[key]}</td>
                <td>${run2.metrics[key]}</td>
            </tr>
        `;
    }).join('');
}

if (els.compareRun1) els.compareRun1.addEventListener('change', renderComparison);
if (els.compareRun2) els.compareRun2.addEventListener('change', renderComparison);

// Phase F: Walkthrough State & Logic
let walkthroughStage = 0;
let wasRunningBeforeWt = false;

const wtStages = [
    {
        title: "STAGE 1: OBSERVE",
        desc: "The system receives observable scan results from the RF environment. Notice the true environment state in the Spatial Map.",
        highlight: '.map-panel'
    },
    {
        title: "STAGE 2: UPDATE BELIEF",
        desc: "The receiver updates its internal belief of the signal environment based on the latest scan.",
        highlight: '#pipe-belief'
    },
    {
        title: "STAGE 3: REASON OVER TIME",
        desc: "The system evaluates temporal patterns (how often and when the signal appears).",
        highlight: '#pipe-temp'
    },
    {
        title: "STAGE 4: CHOOSE ACTION",
        desc: "Based on all evidence, the system makes a decision on which receiver should scan which band next.",
        highlight: '.primary-decision'
    },
    {
        title: "STAGE 5: OBSERVE RESULT",
        desc: "The chosen receiver executes the scan, resulting in a DETECT or NO DETECT.",
        highlight: '.spectrum-panel'
    },
    {
        title: "STAGE 6: EXPLAINABILITY",
        desc: "The decision logic is exposed here to show EXACTLY why the system made this choice.",
        highlight: '.decision-explainability'
    }
];

function startWalkthrough() {
    walkthroughStage = 0;
    if (els.wtOverlay) els.wtOverlay.style.display = 'flex';
    wasRunningBeforeWt = isRunning;
    pause(); // Pause sim for the walkthrough
    updateWalkthroughUI();
}

function closeWalkthrough() {
    if (els.wtOverlay) els.wtOverlay.style.display = 'none';
    clearWalkthroughHighlights();
    if (wasRunningBeforeWt) {
        play();
    }
}

function updateWalkthroughUI() {
    clearWalkthroughHighlights();
    if (walkthroughStage >= wtStages.length) {
        closeWalkthrough();
        return;
    }
    
    const stage = wtStages[walkthroughStage];
    if (els.wtTitle) els.wtTitle.innerText = stage.title;
    if (els.wtProgress) els.wtProgress.innerText = `STEP ${walkthroughStage + 1} / ${wtStages.length}`;
    if (els.wtDesc) els.wtDesc.innerText = stage.desc;
    
    if (stage.highlight) {
        const target = document.querySelector(stage.highlight);
        if (target) {
            target.classList.add('walkthrough-highlight');
            target.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }
    
    if (els.btnWtNext) els.btnWtNext.innerText = walkthroughStage === wtStages.length - 1 ? "Finish" : "Next";
}

function clearWalkthroughHighlights() {
    document.querySelectorAll('.walkthrough-highlight').forEach(el => {
        el.classList.remove('walkthrough-highlight');
    });
}

if (els.btnWtClose) els.btnWtClose.addEventListener('click', closeWalkthrough);
if (els.btnWtSkip) els.btnWtSkip.addEventListener('click', closeWalkthrough);
if (els.btnWtNext) els.btnWtNext.addEventListener('click', () => {
    walkthroughStage++;
    updateWalkthroughUI();
});

if (els.btnPresentation) {
    els.btnPresentation.addEventListener('click', () => {
        const isPres = document.body.classList.toggle('presentation-mode');
        els.btnPresentation.innerText = isPres ? "EXIT PRESENTATION VIEW" : "PRESENTATION VIEW";
        
        if (isPres) {
            startWalkthrough();
        } else {
            closeWalkthrough();
        }
    });
}

// ==========================================
// TSRD SYNTHETIC DATASET INTERACTION LOGIC
// ==========================================
const btnTsrdValidate = document.getElementById('btn-tsrd-validate');
const btnTsrdPreview = document.getElementById('btn-tsrd-preview');
const btnTsrdBenchmark = document.getElementById('btn-tsrd-benchmark');
const tsrdFileSelect = document.getElementById('tsrd-file-select');
const tsrdStatusArea = document.getElementById('tsrd-status-area');
const tsrdResultsContainer = document.getElementById('tsrd-results-table-container');
const tsrdResultsBody = document.getElementById('tsrd-results-body');

if (btnTsrdValidate) {
    btnTsrdValidate.addEventListener('click', async () => {
        const filePath = tsrdFileSelect.value;
        tsrdStatusArea.style.display = 'block';
        tsrdStatusArea.innerText = `[VALIDATING] Loading and auditing schema for: ${filePath}...`;
        try {
            const res = await fetch('/api/tsrd/validate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_path: filePath })
            });
            const data = await res.json();
            if (data.status === 'success') {
                const rep = data.report;
                tsrdStatusArea.innerText = 
                    `=== TSRD DATASET AUDIT REPORT ===\n` +
                    `Status: ${rep.is_valid ? 'VALIDATED (CONFORMS TO TSRD SCHEMA)' : 'INVALID'}\n` +
                    `Source: ${rep.source_path}\n` +
                    `Format: ${rep.format.toUpperCase()} | Scan Mode: ${rep.scan_mode}\n` +
                    `Causal Suitability: ${rep.causal_suitability}\n` +
                    `Records: ${rep.record_count} | Unique Emitter IDs: ${rep.label_count}\n` +
                    `Time Coverage: ${rep.time_range_s[0].toFixed(4)}s to ${rep.time_range_s[1].toFixed(4)}s (${(rep.time_range_us[1] - rep.time_range_us[0]).toFixed(0)} us)\n` +
                    `Frequency Span: ${rep.frequency_range_mhz[0].toFixed(2)} MHz to ${rep.frequency_range_mhz[1].toFixed(2)} MHz\n` +
                    `Pulse Width Span: ${rep.pulse_width_range_us[0].toFixed(3)} us to ${rep.pulse_width_range_us[1].toFixed(3)} us\n` +
                    `Amplitude Span: ${rep.amplitude_range_dbm[0].toFixed(1)} dBm to ${rep.amplitude_range_dbm[1].toFixed(1)} dBm\n` +
                    `Quarantine Integrity: Observable PDW stream quarantined from evaluator labels.\n` +
                    (rep.warnings.length ? `Warnings: ${rep.warnings.join(' | ')}\n` : '');
            } else {
                tsrdStatusArea.innerText = `[ERROR] Validation failed: ${data.message}`;
            }
        } catch (e) {
            tsrdStatusArea.innerText = `[NETWORK ERROR] Could not validate: ${e.message}`;
        }
    });
}

if (btnTsrdPreview) {
    btnTsrdPreview.addEventListener('click', async () => {
        const filePath = tsrdFileSelect.value;
        tsrdStatusArea.style.display = 'block';
        tsrdStatusArea.innerText = `[PREVIEW] Fetching first 5 scheduler-observable PDW records...`;
        try {
            const res = await fetch('/api/tsrd/preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_path: filePath, limit: 5 })
            });
            const data = await res.json();
            if (data.status === 'success') {
                let previewText = `=== SCHEDULER-OBSERVABLE PDW STREAM (FIRST 5 RECORDS) ===\n`;
                previewText += `NOTE: Ground-truth emitter labels strictly hidden.\n\n`;
                data.preview.forEach(p => {
                    previewText += `[PDW #${p.index}] ToA: ${p.timestamp_us.toFixed(1)} us (${p.timestamp_s.toFixed(6)}s) | Freq: ${p.frequency_mhz.toFixed(2)} MHz | PW: ${p.pulse_width_us.toFixed(3)} us | AoA: ${p.angle_of_arrival_deg.toFixed(1)} deg | Amp: ${p.amplitude_dbm.toFixed(1)} dBm\n`;
                });
                tsrdStatusArea.innerText = previewText;
            } else {
                tsrdStatusArea.innerText = `[ERROR] Preview failed: ${data.message}`;
            }
        } catch (e) {
            tsrdStatusArea.innerText = `[NETWORK ERROR] Could not preview: ${e.message}`;
        }
    });
}

if (btnTsrdBenchmark) {
    btnTsrdBenchmark.addEventListener('click', async () => {
        const filePath = tsrdFileSelect.value;
        tsrdStatusArea.style.display = 'block';
        tsrdStatusArea.innerText = `[EXECUTING] Running Common-Trace Benchmark (100 steps) across all baseline algorithms on identical RF environment trace...`;
        tsrdResultsContainer.style.display = 'none';
        tsrdResultsBody.innerHTML = '';
        try {
            const res = await fetch('/api/tsrd/benchmark', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_path: filePath, steps: 100, seed: 42, include_rl: true })
            });
            const data = await res.json();
            if (data.status === 'success') {
                tsrdStatusArea.innerText = `[COMPLETE] Common-Trace benchmark finished on ${data.dataset} (${data.scan_mode} | ${data.causal_suitability}). Results rendered below.`;
                tsrdResultsContainer.style.display = 'block';
                for (const [schedName, r] of Object.entries(data.results)) {
                    const row = document.createElement('tr');
                    const isCass = schedName.includes('Cognitive');
                    row.style.background = isCass ? 'rgba(16, 185, 129, 0.1)' : 'transparent';
                    if (isCass) row.style.fontWeight = '600';
                    row.innerHTML = `
                        <td style="color: ${isCass ? '#10B981' : 'inherit'};">${schedName}</td>
                        <td>${r.total_scans}</td>
                        <td>${r.total_hits} / ${r.total_opportunities}</td>
                        <td>${(r.interception_rate * 100).toFixed(1)}%</td>
                        <td>${(r.false_alarm_rate * 100).toFixed(1)}%</td>
                        <td>${(r.scan_efficiency * 100).toFixed(1)}%</td>
                        <td>${r.mean_intercept_time_s ? r.mean_intercept_time_s.toFixed(4) + 's' : 'N/A'}</td>
                    `;
                    tsrdResultsBody.appendChild(row);
                }
            } else {
                tsrdStatusArea.innerText = `[ERROR] Benchmark failed: ${data.message}`;
            }
        } catch (e) {
            tsrdStatusArea.innerText = `[NETWORK ERROR] Could not run benchmark: ${e.message}`;
        }
    });
}

// =====================================================================
// PHASE 6: USER DATASET IMPORT, VALIDATION & REPLAY LOGIC
// =====================================================================

let activeDatasetFilePath = null;
let activeDatasetFormat = null;
let activeDatasetKeys = [];
let activeDatasetValidationReport = null;
let activeDatasetRunResult = null;

const dsEls = {
    fileSelect: document.getElementById('dataset-file-select'),
    btnInspectSelected: document.getElementById('btn-inspect-selected'),
    fileInput: document.getElementById('dataset-file-input'),
    btnChooseFile: document.getElementById('btn-choose-file'),
    btnUploadFile: document.getElementById('btn-upload-file'),
    mappingPanel: document.getElementById('dataset-mapping-panel'),
    sourceIndicator: document.getElementById('dataset-source-indicator'),
    
    // Mapping selects
    mapTimestamp: document.getElementById('map-timestamp'),
    unitTimestamp: document.getElementById('unit-timestamp'),
    mapFrequency: document.getElementById('map-frequency'),
    unitFrequency: document.getElementById('unit-frequency'),
    mapPW: document.getElementById('map-pw'),
    unitPW: document.getElementById('unit-pw'),
    mapAoA: document.getElementById('map-aoa'),
    unitAoA: document.getElementById('unit-aoa'),
    mapAmplitude: document.getElementById('map-amplitude'),
    unitAmplitude: document.getElementById('unit-amplitude'),
    mapEmitter: document.getElementById('map-emitter'),
    btnValidate: document.getElementById('btn-validate-dataset'),

    // Validation
    validationArea: document.getElementById('dataset-validation-area'),
    validationBadge: document.getElementById('validation-badge'),
    validationLog: document.getElementById('validation-text-log'),

    // Run Config
    runConfigPanel: document.getElementById('dataset-run-config-panel'),
    runSeed: document.getElementById('ds-run-seed'),
    runSteps: document.getElementById('ds-run-steps'),
    runDwell: document.getElementById('ds-run-dwell'),
    runBand: document.getElementById('ds-run-band'),
    btnRunDataset: document.getElementById('btn-run-dataset'),

    // Results
    resultPanel: document.getElementById('dataset-result-panel'),
    resOpp: document.getElementById('ds-res-opp'),
    resCounts: document.getElementById('ds-res-counts'),
    resPd: document.getElementById('ds-res-pd'),
    resPfa: document.getElementById('ds-res-pfa'),
    resEff: document.getElementById('ds-res-eff'),
    resIr: document.getElementById('ds-res-ir'),
    resBandDist: document.getElementById('ds-res-band-distribution'),
    resWarnings: document.getElementById('ds-res-warnings'),
    btnExportJson: document.getElementById('btn-export-json'),
    modeBadge: document.getElementById('workflow-mode-badge')
};

function populateSelectOptions(selectEl, keys, defaultMatch = "", isOptional = true) {
    if (!selectEl) return;
    selectEl.innerHTML = '';
    if (isOptional) {
        const optNone = document.createElement('option');
        optNone.value = "";
        optNone.innerText = "-- None / Default --";
        selectEl.appendChild(optNone);
    }
    
    let matched = false;
    keys.forEach(k => {
        const opt = document.createElement('option');
        opt.value = k;
        opt.innerText = k;
        if (!matched && defaultMatch && k.toLowerCase() === defaultMatch.toLowerCase()) {
            opt.selected = true;
            matched = true;
        }
        selectEl.appendChild(opt);
    });
}

function displayInspection(data) {
    activeDatasetFilePath = data.file_path;
    activeDatasetFormat = data.format;
    activeDatasetKeys = data.raw_keys || [];
    
    if (dsEls.sourceIndicator) {
        dsEls.sourceIndicator.innerText = `File: ${data.filename} (${data.format.toUpperCase()}) | Detected Keys: [${activeDatasetKeys.join(', ')}]`;
    }

    // Auto-match common aliases
    const aliases = {
        timestamp: ['timestamp', 'toa', 'time', 'timestamp_s', 'timestamp_us', 'time_s'],
        frequency: ['frequency', 'freq', 'frequency_mhz', 'freq_mhz', 'center_freq', 'carrier_freq'],
        pulse_width: ['pulse_width', 'pw', 'pulse_width_us', 'pw_us', 'duration'],
        aoa: ['aoa', 'angle', 'angle_of_arrival', 'aoa_deg'],
        amplitude: ['amplitude', 'power', 'amplitude_dbm', 'power_dbm', 'amp'],
        emitter_label: ['emitter_label', 'label', 'emitter_id', 'emitter', 'tx_id']
    };

    function findMatch(candidates) {
        for (const c of candidates) {
            const found = activeDatasetKeys.find(k => k.toLowerCase() === c.toLowerCase());
            if (found) return found;
        }
        return "";
    }

    const matchedTimestampCol = findMatch(aliases.timestamp);
    populateSelectOptions(dsEls.mapTimestamp, activeDatasetKeys, matchedTimestampCol, false);
    populateSelectOptions(dsEls.mapFrequency, activeDatasetKeys, findMatch(aliases.frequency), false);
    populateSelectOptions(dsEls.mapPW, activeDatasetKeys, findMatch(aliases.pulse_width), true);
    populateSelectOptions(dsEls.mapAoA, activeDatasetKeys, findMatch(aliases.aoa), true);
    populateSelectOptions(dsEls.mapAmplitude, activeDatasetKeys, findMatch(aliases.amplitude), true);
    populateSelectOptions(dsEls.mapEmitter, activeDatasetKeys, findMatch(aliases.emitter_label), true);

    // Set intelligent unit defaults based on file format and matched column name
    if (dsEls.unitTimestamp) {
        const tsLower = (matchedTimestampCol || '').toLowerCase();
        if (tsLower.includes('_us') || tsLower.includes('micro')) {
            dsEls.unitTimestamp.value = 'us';
        } else if (tsLower.includes('_ms') || tsLower.includes('milli')) {
            dsEls.unitTimestamp.value = 'ms';
        } else if (tsLower.includes('_ns') || tsLower.includes('nano')) {
            dsEls.unitTimestamp.value = 'ns';
        } else if (tsLower.includes('_s') || tsLower.includes('sec')) {
            dsEls.unitTimestamp.value = 's';
        } else if (data.format === 'hdf5') {
            dsEls.unitTimestamp.value = 'us';
        } else {
            // Canonical CSV/JSON standard default is seconds
            dsEls.unitTimestamp.value = 's';
        }
    }

    if (dsEls.mappingPanel) dsEls.mappingPanel.style.display = 'block';
    if (dsEls.validationArea) dsEls.validationArea.style.display = 'none';
    if (dsEls.runConfigPanel) dsEls.runConfigPanel.style.display = 'none';
    if (dsEls.resultPanel) dsEls.resultPanel.style.display = 'none';
}

if (dsEls.btnChooseFile && dsEls.fileInput) {
    dsEls.btnChooseFile.addEventListener('click', () => dsEls.fileInput.click());
    dsEls.fileInput.addEventListener('change', () => {
        if (dsEls.fileInput.files.length > 0) {
            dsEls.btnChooseFile.innerText = dsEls.fileInput.files[0].name;
        }
    });
}

if (dsEls.btnInspectSelected) {
    dsEls.btnInspectSelected.addEventListener('click', async () => {
        const filePath = dsEls.fileSelect.value;
        try {
            const res = await fetch('/api/dataset/inspect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_path: filePath })
            });
            const data = await res.json();
            if (data.status === 'success') {
                displayInspection(data);
            } else {
                alert(`Inspection failed: ${data.message}`);
            }
        } catch (e) {
            alert(`Network error during inspection: ${e.message}`);
        }
    });
}

if (dsEls.btnUploadFile) {
    dsEls.btnUploadFile.addEventListener('click', async () => {
        if (!dsEls.fileInput.files || dsEls.fileInput.files.length === 0) {
            alert('Please select a file to upload first.');
            return;
        }
        const formData = new FormData();
        formData.append('file', dsEls.fileInput.files[0]);

        try {
            const res = await fetch('/api/dataset/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.status === 'success') {
                // Add to dropdown and select
                const opt = document.createElement('option');
                opt.value = data.file_path;
                opt.innerText = `${data.filename} (Uploaded)`;
                dsEls.fileSelect.appendChild(opt);
                dsEls.fileSelect.value = data.file_path;
                displayInspection(data);
            } else {
                alert(`Upload failed: ${data.message}`);
            }
        } catch (e) {
            alert(`Network error during upload: ${e.message}`);
        }
    });
}

if (dsEls.btnValidate) {
    dsEls.btnValidate.addEventListener('click', async () => {
        if (!activeDatasetFilePath) return;

        const fieldMapping = {
            timestamp: dsEls.mapTimestamp.value,
            frequency: dsEls.mapFrequency.value,
            pulse_width: dsEls.mapPW.value,
            aoa: dsEls.mapAoA.value,
            amplitude: dsEls.mapAmplitude.value,
            emitter_label: dsEls.mapEmitter.value
        };

        const units = {
            timestamp: dsEls.unitTimestamp.value,
            frequency: dsEls.unitFrequency.value,
            pulse_width: dsEls.unitPW.value,
            aoa: dsEls.unitAoA.value,
            amplitude: dsEls.unitAmplitude.value
        };

        try {
            const res = await fetch('/api/dataset/validate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    file_path: activeDatasetFilePath,
                    field_mapping: fieldMapping,
                    units: units
                })
            });
            const data = await res.json();
            if (data.status === 'success') {
                const rep = data.report;
                activeDatasetValidationReport = rep;
                dsEls.validationArea.style.display = 'block';

                if (rep.is_valid) {
                    dsEls.validationBadge.className = 'badge';
                    dsEls.validationBadge.style.background = '#10B981';
                    dsEls.validationBadge.style.color = '#000';
                    dsEls.validationBadge.innerText = 'VALIDATED (PASS)';
                    if (dsEls.runConfigPanel) dsEls.runConfigPanel.style.display = 'block';
                } else {
                    dsEls.validationBadge.className = 'badge danger';
                    dsEls.validationBadge.style.background = '#EF4444';
                    dsEls.validationBadge.style.color = '#fff';
                    dsEls.validationBadge.innerText = 'VALIDATION FAILED';
                    if (dsEls.runConfigPanel) dsEls.runConfigPanel.style.display = 'none';
                }

                let textLog = `=== DATASET VALIDATION REPORT ===\n`;
                textLog += `Source: ${rep.source_path}\n`;
                textLog += `Format: ${rep.format.toUpperCase()} | Valid Records: ${rep.valid_rows} / ${rep.total_rows}\n`;
                textLog += `Time Range: ${rep.time_range_s[0].toFixed(4)}s to ${rep.time_range_s[1].toFixed(4)}s\n`;
                textLog += `Frequency Range: ${rep.frequency_range_mhz[0].toFixed(2)} MHz to ${rep.frequency_range_mhz[1].toFixed(2)} MHz\n`;
                textLog += `Resolved Field Mapping: ${JSON.stringify(rep.resolved_field_mapping)}\n`;
                textLog += `Explicit Units: ${JSON.stringify(rep.conversion_summary)}\n`;
                if (rep.warnings && rep.warnings.length) {
                    textLog += `Warnings:\n  - ${rep.warnings.join('\n  - ')}\n`;
                }
                if (rep.validation_errors && rep.validation_errors.length) {
                    textLog += `Errors:\n  - ${rep.validation_errors.join('\n  - ')}\n`;
                }
                dsEls.validationLog.innerText = textLog;
            } else {
                alert(`Validation request error: ${data.message}`);
            }
        } catch (e) {
            alert(`Network error during validation: ${e.message}`);
        }
    });
}

if (dsEls.btnRunDataset) {
    dsEls.btnRunDataset.addEventListener('click', async () => {
        if (!activeDatasetFilePath || !activeDatasetValidationReport || !activeDatasetValidationReport.is_valid) {
            alert('Dataset must be validated successfully before running CASS-EW.');
            return;
        }

        const fieldMapping = {
            timestamp: dsEls.mapTimestamp.value,
            frequency: dsEls.mapFrequency.value,
            pulse_width: dsEls.mapPW.value,
            aoa: dsEls.mapAoA.value,
            amplitude: dsEls.mapAmplitude.value,
            emitter_label: dsEls.mapEmitter.value
        };

        const units = {
            timestamp: dsEls.unitTimestamp.value,
            frequency: dsEls.unitFrequency.value,
            pulse_width: dsEls.unitPW.value,
            aoa: dsEls.unitAoA.value,
            amplitude: dsEls.unitAmplitude.value
        };

        const steps = parseInt(dsEls.runSteps.value) || 100;
        const dwell_time_s = parseFloat(dsEls.runDwell.value) || 0.005;
        const seed = parseInt(dsEls.runSeed.value) || 42;
        const initial_band = parseInt(dsEls.runBand.value) || 0;

        dsEls.btnRunDataset.disabled = true;
        dsEls.btnRunDataset.innerText = 'REPLAYING THROUGH CASS-EW...';

        try {
            const res = await fetch('/api/dataset/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    file_path: activeDatasetFilePath,
                    field_mapping: fieldMapping,
                    units: units,
                    steps: steps,
                    dwell_time_s: dwell_time_s,
                    seed: seed,
                    initial_band: initial_band
                })
            });
            const data = await res.json();
            if (data.status === 'success') {
                const r = data.result;
                activeDatasetRunResult = r;
                displayDatasetRunResult(r);
            } else {
                alert(`Replay run failed: ${data.message}`);
            }
        } catch (e) {
            alert(`Network error during replay run: ${e.message}`);
        } finally {
            dsEls.btnRunDataset.disabled = false;
            dsEls.btnRunDataset.innerText = 'RUN CASS-EW ON DATASET';
        }
    });
}

function displayDatasetRunResult(r) {
    if (!dsEls.resultPanel) return;
    dsEls.resultPanel.style.display = 'block';

    // Update Live Cognitive Status bar to reflect Dataset Replay
    const opModeEls = document.querySelectorAll('.mode-val');
    opModeEls.forEach(el => el.innerText = "DATASET REPLAY");

    if (dsEls.modeBadge) {
        dsEls.modeBadge.innerText = "MODE: DATASET REPLAY ACTIVE";
        dsEls.modeBadge.style.color = "#10B981";
        dsEls.modeBadge.style.borderColor = "#10B981";
    }

    // Populate Audited Metrics
    dsEls.resOpp.innerText = `${r.observation_opportunities} opportunities`;
    dsEls.resCounts.innerText = `${r.hits} / ${r.misses} / ${r.false_alarms}`;
    
    dsEls.resPd.innerText = (r.receiver_pd !== null && r.receiver_pd !== undefined) 
        ? `${(r.receiver_pd * 100).toFixed(1)}%` 
        : "N/A (0 Opportunities)";
        
    dsEls.resPfa.innerText = (r.receiver_pfa !== null && r.receiver_pfa !== undefined) 
        ? `${(r.receiver_pfa * 100).toFixed(1)}%` 
        : "N/A (0 Quiet Scans)";
        
    dsEls.resEff.innerText = (r.scan_efficiency !== null && r.scan_efficiency !== undefined) 
        ? `${(r.scan_efficiency * 100).toFixed(1)}%` 
        : "N/A";

    if (r.global_emitter_interception_rate !== null && r.global_emitter_interception_rate !== undefined) {
        dsEls.resIr.innerText = `${r.intercepted_emitters} / ${r.total_ground_truth_emitters} (${(r.global_emitter_interception_rate * 100).toFixed(1)}%)`;
    } else {
        dsEls.resIr.innerText = "N/A — NO GROUND TRUTH";
    }

    // Band distribution
    let distStr = `Executed Scans: ${r.executed_scans} | Simulated Duration: ${r.simulation_duration_s.toFixed(4)}s\n`;
    distStr += `Per-Band Scans: [${r.per_band_scan_counts.join(', ')}]\n`;
    distStr += `Per-Band Hits:  [${r.per_band_hit_counts.join(', ')}]`;
    dsEls.resBandDist.innerText = distStr;

    // Warnings
    const allWarnings = (r.warnings || []).concat(r.validation_warnings || []);
    if (allWarnings.length > 0) {
        dsEls.resWarnings.style.display = 'block';
        dsEls.resWarnings.innerHTML = `<strong>OPERATIONAL WARNINGS:</strong><br>• ${allWarnings.join('<br>• ')}`;
    } else {
        dsEls.resWarnings.style.display = 'none';
    }

    // Update main spectrum visualization with latest band decisions
    if (r.band_decisions && r.band_decisions.length > 0) {
        const lastBand = r.band_decisions[r.band_decisions.length - 1];
        if (els.currentBand) els.currentBand.innerText = `BAND ${lastBand}`;
        if (els.nextAction) els.nextAction.innerText = `REPLAY COMPLETE (${r.executed_scans} SCANS)`;
    }
}

if (dsEls.btnExportJson) {
    dsEls.btnExportJson.addEventListener('click', () => {
        if (!activeDatasetRunResult) {
            alert('No dataset run result available to export.');
            return;
        }
        const blob = new Blob([JSON.stringify(activeDatasetRunResult, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const fname = activeDatasetRunResult.dataset_path 
            ? activeDatasetRunResult.dataset_path.split(/[\\/]/).pop().replace(/\.[^/.]+$/, "") 
            : "dataset";
        a.download = `${fname}_cass_ew_run_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });
}

// Init
initCharts();
loadBenchmark();
resetSim(42);



