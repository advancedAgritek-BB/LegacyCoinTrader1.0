// LegacyCoinTrader Frontend JavaScript

let lastWalletSummary = null;
let lastWalletSummaryTimestamp = 0;
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initApp();
    // Initialize theme system
    initThemeSystem();
});

function initApp() {
    // Initialize sidebar functionality
    initSidebar();
    
    // Initialize navigation highlighting
    initNavigation();
    
    // Initialize execution mode controls
    initModeSwitcher();

    // Initialize any charts or data visualizations
    initCharts();
    
    // Initialize real-time updates (single interval)
    initRealTimeUpdates();
}

// Sidebar functionality with accessibility
function initSidebar() {
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    const navLinks = document.querySelectorAll('.nav-link');

    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            toggleSidebar();
        });

        // Keyboard support for sidebar toggle
        sidebarToggle.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                toggleSidebar();
            }
        });
    }

    function toggleSidebar() {
        const isOpen = sidebar.classList.contains('show');
        sidebar.classList.toggle('show');
        mainContent.classList.toggle('sidebar-collapsed');

        // Update ARIA attributes
        sidebarToggle.setAttribute('aria-expanded', !isOpen);
        sidebar.setAttribute('aria-hidden', isOpen);
    }

    // Close sidebar on mobile when clicking outside
    document.addEventListener('click', function(e) {
        if (window.innerWidth <= 1024) {
            if (!sidebar.contains(e.target) && !sidebarToggle.contains(e.target)) {
                sidebar.classList.remove('show');
                mainContent.classList.remove('sidebar-collapsed');
                sidebarToggle.setAttribute('aria-expanded', 'false');
                sidebar.setAttribute('aria-hidden', 'true');
            }
        }
    });

    // Handle window resize
    window.addEventListener('resize', function() {
        if (window.innerWidth > 1024) {
            sidebar.classList.remove('show');
            mainContent.classList.remove('sidebar-collapsed');
            sidebarToggle.setAttribute('aria-expanded', 'false');
            sidebar.setAttribute('aria-hidden', 'false');
        }
    });

    // Keyboard navigation for sidebar menu
    let currentFocusIndex = -1;

    document.addEventListener('keydown', function(e) {
        if (sidebar.classList.contains('show')) {
            switch(e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    currentFocusIndex = Math.min(currentFocusIndex + 1, navLinks.length - 1);
                    navLinks[currentFocusIndex].focus();
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    currentFocusIndex = Math.max(currentFocusIndex - 1, 0);
                    navLinks[currentFocusIndex].focus();
                    break;
                case 'Escape':
                    e.preventDefault();
                    toggleSidebar();
                    sidebarToggle.focus();
                    break;
            }
        }
    });

    // Focus management
    sidebar.addEventListener('focusin', function() {
        currentFocusIndex = Array.from(navLinks).indexOf(document.activeElement);
    });
}

function initModeSwitcher() {
    const container = document.querySelector('.mode-switcher');
    const toggleBtn = document.getElementById('modeToggleBtn');
    const menu = document.getElementById('modeMenu');
    const statusEl = document.getElementById('modeMenuStatus');
    const labelEl = document.getElementById('headerModeLabel');

    if (!container || !toggleBtn || !menu || !statusEl || !labelEl) {
        return;
    }

    let currentMode = normalizeExecutionMode(container.dataset.mode || 'dry_run');

    function setStatus(message, variant) {
        statusEl.textContent = message;
        statusEl.classList.remove('success', 'error', 'pending');
        if (variant) {
            statusEl.classList.add(variant);
        }
    }

    function refreshMenuState() {
        const resetItem = menu.querySelector('[data-action="reset-paper"]');
        const switchLiveItem = menu.querySelector('[data-action="switch-live"]');
        const switchPaperItem = menu.querySelector('[data-action="switch-paper"]');

        if (resetItem) {
            resetItem.disabled = currentMode !== 'dry_run';
        }
        if (switchLiveItem) {
            switchLiveItem.disabled = currentMode === 'live';
        }
        if (switchPaperItem) {
            switchPaperItem.disabled = currentMode === 'dry_run';
        }
    }

    function updateLabel(mode) {
        currentMode = normalizeExecutionMode(mode);
        container.dataset.mode = currentMode;
        labelEl.textContent = formatModeLabel(currentMode);
        if (currentMode === 'dry_run') {
            setStatus('Paper mode tools available while running in dry run.', null);
        } else {
            setStatus('Switch back to paper mode to access reset tooling.', null);
        }
        refreshMenuState();
    }

    function closeMenu() {
        menu.classList.remove('open');
        toggleBtn.setAttribute('aria-expanded', 'false');
    }

    function openMenu() {
        menu.classList.add('open');
        toggleBtn.setAttribute('aria-expanded', 'true');
    }

    toggleBtn.addEventListener('click', function(event) {
        event.stopPropagation();
        if (menu.classList.contains('open')) {
            closeMenu();
        } else {
            openMenu();
        }
    });

    document.addEventListener('click', function(event) {
        if (!menu.contains(event.target) && event.target !== toggleBtn) {
            closeMenu();
        }
    });

    menu.addEventListener('click', function(event) {
        const item = event.target.closest('.mode-menu-item');
        if (!item || item.disabled) {
            return;
        }
        event.preventDefault();
        event.stopPropagation();
        handleModeAction(item.dataset.action);
    });

    async function handleModeAction(action) {
        if (!action) {
            return;
        }

        try {
            toggleBtn.disabled = true;
            setStatus('Applying changes…', 'pending');

            let response;
            if (action === 'reset-paper') {
                response = await fetch('/api/paper/reset', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
            } else if (action === 'switch-live') {
                response = await fetch('/api/execution-mode', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode: 'live' })
                });
            } else if (action === 'switch-paper') {
                response = await fetch('/api/execution-mode', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode: 'dry_run' })
                });
            } else {
                setStatus('Unknown action requested.', 'error');
                return;
            }

            let data = {};
            if (!response.ok) {
                try {
                    data = await response.json();
                } catch (err) {
                    data = {};
                }
                const message = data.error || `Request failed (${response.status})`;
                setStatus(message, 'error');
                return;
            }

            try {
                data = await response.json();
            } catch (err) {
                data = {};
            }

            if (action === 'reset-paper') {
                updateLabel('dry_run');
                if (typeof data.balance === 'number') {
                    const balanceEl = document.getElementById('headerWalletBalance');
                    if (balanceEl) {
                        balanceEl.textContent = formatCurrency(data.balance);
                    }
                }

                if (Array.isArray(data.errors) && data.errors.length > 0) {
                    setStatus(data.errors.join('; '), 'error');
                } else {
                    setStatus('Paper trading environment reset. Updating display...', 'success');
                }

                // Check if we should force empty positions display
                if (data.force_empty_positions) {
                    console.log('Reset detected, forcing empty positions display');
                    // Clear position cards immediately
                    const positionsContainer = document.querySelector('.positions-grid');
                    if (positionsContainer) {
                        const existingCards = positionsContainer.querySelectorAll('.position-card');
                        existingCards.forEach(card => card.remove());

                        // Show "no positions" message
                        const noPositionsMsg = positionsContainer.querySelector('.no-positions-message');
                        if (!noPositionsMsg) {
                            const msg = document.createElement('div');
                            msg.className = 'no-positions-message';
                            msg.innerHTML = '<p>No open positions</p>';
                            positionsContainer.appendChild(msg);
                        }

                        // Update position count
                        const positionCountElement = document.querySelector('.position-count');
                        if (positionCountElement) {
                            positionCountElement.textContent = '0 positions';
                        }

                        const metricValueElement = document.querySelector('.metric-value.ai-gradient');
                        if (metricValueElement) {
                            metricValueElement.textContent = '0';
                        }
                    }
                }

                // Update dashboard data
                if (typeof updateAllData === 'function') {
                    updateAllData();
                }
                updateHeaderFinance();
            } else if (action === 'switch-live') {
                updateLabel('live');
                setStatus('Switched to live trading mode. Restart the engine if required.', 'success');
                updateHeaderFinance();
            } else if (action === 'switch-paper') {
                updateLabel('dry_run');
                setStatus('Switched to paper trading mode.', 'success');
                updateHeaderFinance();
            }
        } catch (error) {
            console.error('Mode action failed', error);
            setStatus('Unable to complete request.', 'error');
        } finally {
            toggleBtn.disabled = false;
        }
    }

    updateLabel(currentMode);
}

// Navigation highlighting
function initNavigation() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
}

// Chart.js implementations removed to prevent conflicts with custom canvas charts
function initCharts() {
    console.log('Charts initialization skipped - using custom canvas implementations');
}

// Real-time updates - SINGLE interval to prevent conflicts
function initRealTimeUpdates() {
    // Single update interval every 10 seconds to prevent conflicts
    setInterval(updateAllData, 10000);
    // Header-specific refresh every 60 seconds
    setInterval(updateHeaderFinance, 60000);
    
    // Initial update
    updateAllData();
    updateHeaderFinance();
}

// Update all data in one function to prevent conflicts
function updateAllData() {
    // Update bot status
    updateBotStatus();
    
    // Update dashboard metrics
    updateDashboardMetrics();
    
    // Update live signals
    updateLiveSignals();
    
    // Update open positions (throttled to prevent excessive updates)
    if (typeof window.updateOpenPositions === 'function') {
        updateOpenPositionsThrottled();
    } else {
        console.log('updateOpenPositions function not available yet, skipping update');
    }
}

// Update header wallet balance and unrealized PnL
function updateHeaderFinance(summary) {
    const hasHeaderTargets = document.getElementById('headerWalletBalance') || document.getElementById('headerUnrealizedPnl');
    if (!hasHeaderTargets) return;

    if (summary) {
        applyWalletSummary(summary);
        return;
    }

    if (lastWalletSummary && (Date.now() - lastWalletSummaryTimestamp) < 5000) {
        applyWalletSummary(lastWalletSummary);
        return;
    }

    fetch('/api/wallet-pnl')
        .then(r => (r.ok ? r.json() : null))
        .then(data => {
            if (data) {
                applyWalletSummary(data);
            }
        })
        .catch(() => {});
}

function applyWalletSummary(summary) {
    if (!summary || typeof summary !== 'object') {
        return;
    }

    lastWalletSummary = summary;
    lastWalletSummaryTimestamp = Date.now();

    const balanceEl = document.getElementById('headerWalletBalance');
    const unrealEl = document.getElementById('headerUnrealizedPnl');

    if (balanceEl && summary.balance !== undefined && summary.balance !== null) {
        balanceEl.textContent = formatCurrency(summary.balance);
    }

    if (unrealEl && summary.unrealized_pnl !== undefined && summary.unrealized_pnl !== null) {
        const sign = summary.unrealized_pnl >= 0 ? '+' : '-';
        const formatted = formatCurrency(Math.abs(summary.unrealized_pnl));
        unrealEl.textContent = `${sign}${formatted}`;
        unrealEl.classList.toggle('text-success', summary.unrealized_pnl >= 0);
        unrealEl.classList.toggle('text-danger', summary.unrealized_pnl < 0);
    }
}

// Update bot status
function updateBotStatus() {
    fetch('/api/bot-status')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            const statusIndicator = document.querySelector('.status-indicator');
            const statusText = document.querySelector('.bot-status span');
            
            if (statusIndicator && statusText) {
                if (data.data && data.data.bot_running) {
                    statusIndicator.classList.add('online');
                    statusIndicator.classList.remove('offline');
                    statusText.textContent = 'Bot Online';
                } else {
                    statusIndicator.classList.remove('online');
                    statusIndicator.classList.add('offline');
                    statusText.textContent = 'Bot Offline';
                }
            }
            
            // Update uptime display
            const uptimeElement = document.getElementById('uptime');
            if (uptimeElement && data.uptime) {
                uptimeElement.textContent = data.uptime;
            }
        })
        .catch(error => console.error('Error updating bot status:', error));
}

// Update metrics function removed to prevent layout conflicts

// Update dashboard metrics
function updateDashboardMetrics() {
    fetch('/api/dashboard-metrics')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                console.error('Error updating dashboard metrics:', data.error);
                return;
            }

            const walletSummary = data.wallet_summary || null;

            if (data.performance) {
                updatePerformanceMetrics(data.performance, walletSummary);
            }

            if (walletSummary) {
                updateHeaderFinance(walletSummary);
            }

            if (data.allocation) {
                updateAllocationDisplay(data.allocation);
            }

            if (data.recent_trades) {
                updateRecentTrades(data.recent_trades);
            }

            // Chart updates removed to prevent conflicts with custom canvas charts
        })
        .catch(error => console.error('Error updating dashboard metrics:', error));
}

// Update live signals
function updateLiveSignals() {
    fetch('/api/live-signals')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                console.error('Error updating live signals:', data.error);
                return;
            }
            
            // Update asset scores if available
            if (data && Object.keys(data).length > 0) {
                updateAssetScores(data);
            }
        })
        .catch(error => console.error('Error updating live signals:', error));
}

// Throttled version of updateOpenPositions to prevent excessive updates
let lastOpenPositionsUpdate = 0;
const OPEN_POSITIONS_UPDATE_INTERVAL = 10000; // 10 seconds minimum between updates

function updateOpenPositionsThrottled() {
    const now = Date.now();
    if (now - lastOpenPositionsUpdate < OPEN_POSITIONS_UPDATE_INTERVAL) {
        console.log('Skipping open positions update - too soon since last update');
        return;
    }
    
    lastOpenPositionsUpdate = now;
    updateOpenPositions();
}

// Update open positions function
function updateOpenPositions() {
    console.log('updateOpenPositions called from app.js');
    // For dashboard, positions are loaded statically when page renders
    // If we need to update positions, we should reload the page
    console.log('Dashboard positions are loaded statically - no dynamic update needed');
}

// Update performance metrics
function updatePerformanceMetrics(performance, walletSummary) {
    const summary = walletSummary && typeof walletSummary === 'object' ? walletSummary : {};
    const metrics = performance || {};

    if (walletSummary) {
        lastWalletSummary = walletSummary;
        lastWalletSummaryTimestamp = Date.now();
    }

    const totalPnlValue = summary.total_pnl ?? metrics.total_pnl ?? 0;
    const totalPnl = document.getElementById('totalPnl');
    if (totalPnl) {
        totalPnl.textContent = formatCurrency(totalPnlValue);
        totalPnl.classList.remove('text-success', 'text-danger', 'text-muted');
        totalPnl.classList.add(totalPnlValue >= 0 ? 'text-success' : 'text-danger');
    }

    const pnlPct = summary.pnl_percentage ?? metrics.pnl_percentage;
    const totalPnlPercentage = document.getElementById('totalPnlPercentage');
    if (totalPnlPercentage && pnlPct !== undefined && pnlPct !== null) {
        const sign = pnlPct >= 0 ? '+' : '';
        totalPnlPercentage.textContent = `${sign}${Number(pnlPct).toFixed(2)}%`;
        const metricChange = totalPnlPercentage.parentElement;
        if (metricChange) {
            metricChange.classList.remove('positive', 'negative');
            metricChange.classList.add(pnlPct >= 0 ? 'positive' : 'negative');
        }
    }

    const walletPnl = document.getElementById('walletPnl');
    if (walletPnl) {
        walletPnl.textContent = formatCurrency(totalPnlValue);
        walletPnl.classList.remove('text-success', 'text-danger');
        walletPnl.classList.add(totalPnlValue >= 0 ? 'text-success' : 'text-danger');
    }

    const balanceValue = summary.balance ?? summary.current_balance ?? metrics.balance;
    const currentBalance = document.getElementById('currentWalletBalance');
    if (currentBalance && balanceValue !== undefined && balanceValue !== null) {
        currentBalance.textContent = formatCurrency(balanceValue);
    }

    const initialValue = summary.initial_balance ?? metrics.initial_balance;
    const initialBalance = document.getElementById('initialBalance');
    if (initialBalance && initialValue !== undefined && initialValue !== null) {
        initialBalance.textContent = formatCurrency(initialValue);
    }

    const totalTrades = document.getElementById('totalTrades');
    if (totalTrades && metrics.trades_today !== undefined) {
        totalTrades.textContent = metrics.trades_today;
    }

    const tradesToday = document.getElementById('tradesToday');
    if (tradesToday && metrics.total_trades !== undefined && metrics.trades_today !== undefined) {
        const totalToday = metrics.trades_today;
        const change = totalToday > 0 ? `+${totalToday} today` : '+0 today';
        tradesToday.textContent = change;
    }

    const winRate = document.getElementById('winRate');
    if (winRate && metrics.win_rate !== undefined && metrics.win_rate !== null) {
        winRate.textContent = formatPercentage(metrics.win_rate * 100);
    }
}

// Update allocation display
function updateAllocationDisplay(allocation) {
    // This would update the strategy allocation visualization
    // Implementation depends on your specific UI structure
    console.log('Allocation updated:', allocation);
}

// Update recent trades
function updateRecentTrades(recentTrades) {
    // This would update the recent trades display
    // Implementation depends on your specific UI structure
    console.log('Recent trades updated:', recentTrades);
}

// Update asset scores
function updateAssetScores(assetScores) {
    // This would update the asset scores display
    // Implementation depends on your specific UI structure
    console.log('Asset scores updated:', assetScores);
}

// Chart.js update functions removed to prevent conflicts

// Bot control functions
function startBot() {
    fetch('/start_bot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            mode: 'dry_run'
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Bot start response:', data);

        if (data.status === 'started' || data.status === 'already_running') {
            if (data.status === 'started') {
                showToast('Bot started successfully!', 'success');
            } else {
                showToast('Bot is already running', 'info');
            }
            // Update status without page reload
            setTimeout(() => updateAllData(), 1000);
        } else if (data.status && data.status.startsWith('error:')) {
            showToast('Error starting bot: ' + data.status.substring(6), 'error');
        } else {
            showToast('Unexpected response: ' + data.status, 'warning');
        }
    })
    .catch(error => {
        console.error('Error starting bot:', error);
        showToast('Error starting bot: ' + error.message, 'error');
    });
}

function stopBot() {
    fetch('/stop', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        showToast('Bot stopped successfully!', 'success');
        // Update status without page reload
        setTimeout(() => updateAllData(), 1000);
    })
    .catch(error => {
        console.error('Error stopping bot:', error);
        showToast('Error stopping bot', 'error');
    });
}

function stopConflicts() {
    fetch('/stop_conflicts', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        showToast('Conflicting processes stopped', 'success');
        // Update status without page reload
        setTimeout(() => updateAllData(), 1000);
    })
    .catch(error => {
        console.error('Error stopping conflicts:', error);
        showToast('Error stopping conflicts', 'error');
    });
}

// Utility functions
function formatCurrency(amount, currency = 'USD') {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: currency
    }).format(amount);
}

function formatModeLabel(mode) {
    if (!mode) {
        return '';
    }
    const normalised = String(mode).toLowerCase();
    if (normalised === 'dry_run' || normalised === 'paper' || normalised === 'paper_trading' || normalised === 'dry') {
        return 'Paper';
    }
    if (normalised === 'live') {
        return 'Live';
    }
    return normalised.replace(/_/g, ' ');
}

function normalizeExecutionMode(mode) {
    const normalised = String(mode || '').trim().toLowerCase();
    if (!normalised) {
        return 'dry_run';
    }
    if (['dry_run', 'dry', 'paper', 'paper_trading', 'simulation', 'sim'].includes(normalised)) {
        return 'dry_run';
    }
    if (['live', 'production', 'real'].includes(normalised)) {
        return 'live';
    }
    return normalised;
}

function formatPercentage(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value / 100);
}

function formatNumber(value) {
    return new Intl.NumberFormat('en-US').format(value);
}

// Toast notifications
function showToast(message, type = 'info', duration = 4000) {
    // Remove existing toasts to prevent spam
    const existingToasts = document.querySelectorAll('.toast');
    if (existingToasts.length >= 3) {
        existingToasts[0].remove();
    }

    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            pointer-events: none;
            max-width: 400px;
        `;
        document.body.appendChild(toastContainer);
    }

    const toast = document.createElement('div');
    toast.className = `toast toast-${type} animate-bounce-in`;
    toast.style.cssText = `
        background: var(--glass-primary);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-${type === 'success' ? 'success' : type === 'error' ? 'danger' : type === 'warning' ? 'warning' : 'info'});
        color: var(--text-primary);
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: var(--shadow-lg);
        pointer-events: auto;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        transition: all var(--transition-normal);
    `;

    // Create toast content
    const toastContent = document.createElement('div');
    toastContent.className = 'toast-content';
    toastContent.style.cssText = `
        display: flex;
        align-items: center;
        gap: 12px;
    `;

    const icon = document.createElement('i');
    icon.className = `fas fa-${getToastIcon(type)}`;
    icon.style.cssText = `
        font-size: 18px;
        color: var(--${type === 'success' ? 'success' : type === 'error' ? 'danger' : type === 'warning' ? 'warning' : 'info'});
        flex-shrink: 0;
    `;

    const span = document.createElement('span');
    span.textContent = message;
    span.style.cssText = `
        flex: 1;
        line-height: 1.4;
        font-weight: 500;
    `;

    // Add close button
    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '×';
    closeBtn.style.cssText = `
        background: none;
        border: none;
        color: var(--text-muted);
        font-size: 20px;
        cursor: pointer;
        padding: 0;
        margin-left: 12px;
        flex-shrink: 0;
        transition: color var(--transition-fast);
    `;
    closeBtn.onmouseover = () => closeBtn.style.color = 'var(--text-primary)';
    closeBtn.onmouseout = () => closeBtn.style.color = 'var(--text-muted)';
    closeBtn.onclick = (e) => {
        e.stopPropagation();
        removeToast(toast);
    };

    toastContent.appendChild(icon);
    toastContent.appendChild(span);
    toastContent.appendChild(closeBtn);
    toast.appendChild(toastContent);

    // Add progress bar
    const progressBar = document.createElement('div');
    progressBar.className = 'toast-progress';
    progressBar.style.cssText = `
        position: absolute;
        bottom: 0;
        left: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--${type === 'success' ? 'success' : type === 'error' ? 'danger' : type === 'warning' ? 'warning' : 'info'}), transparent);
        animation: toastProgress ${duration}ms linear forwards;
    `;
    toast.appendChild(progressBar);

    toastContainer.appendChild(toast);

    // Enhanced hover effects
    toast.onmouseenter = () => {
        toast.style.transform = 'translateX(-4px) scale(1.02)';
        progressBar.style.animationPlayState = 'paused';
    };
    toast.onmouseleave = () => {
        toast.style.transform = 'translateX(0) scale(1)';
        progressBar.style.animationPlayState = 'running';
    };

    // Auto remove with animation
    setTimeout(() => {
        if (toast.parentNode) {
            removeToast(toast);
        }
    }, duration);
}

function removeToast(toast) {
    toast.style.animation = 'slideOutRight 0.4s ease-in forwards';
    setTimeout(() => {
        if (toast.parentNode) {
            toast.remove();
        }
    }, 400);
}

function getToastIcon(type) {
    const icons = {
        success: 'check-circle',
        error: 'exclamation-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Toast styles are now in CSS file
// No need to add styles dynamically

// ===== THEME SYSTEM =====

// Theme management functions with accessibility
function initThemeSystem() {
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const themeMenu = document.getElementById('themeMenu');
    const themeOptions = document.querySelectorAll('.theme-option');

    // Load saved theme or default to midnight
    const savedTheme = localStorage.getItem('legacycointrader-theme') || 'midnight';
    setTheme(savedTheme);

    // Set up ARIA attributes
    if (themeToggleBtn) {
        themeToggleBtn.setAttribute('aria-haspopup', 'true');
        themeToggleBtn.setAttribute('aria-expanded', 'false');
        themeToggleBtn.setAttribute('aria-label', 'Change theme');
    }

    if (themeMenu) {
        themeMenu.setAttribute('role', 'menu');
        themeMenu.setAttribute('aria-label', 'Theme selection menu');
    }

    themeOptions.forEach((option, index) => {
        option.setAttribute('role', 'menuitem');
        option.setAttribute('tabindex', '0');
        option.setAttribute('aria-label', `Select ${option.querySelector('span').textContent} theme`);
    });

    // Toggle theme menu
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            toggleThemeMenu();
        });

        // Keyboard support
        themeToggleBtn.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                toggleThemeMenu();
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                if (!themeMenu.classList.contains('show')) {
                    toggleThemeMenu();
                }
                // Focus first theme option
                const firstOption = themeOptions[0];
                if (firstOption) firstOption.focus();
            }
        });
    }

    function toggleThemeMenu() {
        const isOpen = themeMenu.classList.contains('show');
        themeMenu.classList.toggle('show');
        themeToggleBtn.setAttribute('aria-expanded', !isOpen);

        if (!isOpen) {
            // Focus management when opening
            themeMenu.classList.add('focus-trap');
        } else {
            themeMenu.classList.remove('focus-trap');
        }
    }

    // Handle theme selection with keyboard support
    themeOptions.forEach(option => {
        option.addEventListener('click', function() {
            selectTheme(this);
        });

        option.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                selectTheme(this);
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                const nextOption = this.nextElementSibling;
                if (nextOption) nextOption.focus();
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                const prevOption = this.previousElementSibling;
                if (prevOption) prevOption.focus();
                else themeToggleBtn.focus();
            } else if (e.key === 'Escape') {
                e.preventDefault();
                toggleThemeMenu();
                themeToggleBtn.focus();
            }
        });
    });

    function selectTheme(optionElement) {
        const selectedTheme = optionElement.getAttribute('data-theme');
        setTheme(selectedTheme);
        toggleThemeMenu();
        themeToggleBtn.focus();
    }

    // Close theme menu when clicking outside
    document.addEventListener('click', function(e) {
        if (!themeToggleBtn.contains(e.target) && !themeMenu.contains(e.target)) {
            themeMenu.classList.remove('show');
            themeToggleBtn.setAttribute('aria-expanded', 'false');
            themeMenu.classList.remove('focus-trap');
        }
    });

    // Close theme menu on escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && themeMenu.classList.contains('show')) {
            e.preventDefault();
            toggleThemeMenu();
            themeToggleBtn.focus();
        }
    });
}

function setTheme(themeName) {
    // Remove all existing theme classes
    const body = document.body;
    body.classList.remove('theme-midnight', 'theme-neon', 'theme-cyberpunk', 'theme-minimal');

    // Add the selected theme class
    if (themeName !== 'midnight') {
        body.classList.add(`theme-${themeName}`);
    }

    // Update active theme option in menu
    const themeOptions = document.querySelectorAll('.theme-option');
    themeOptions.forEach(option => {
        option.classList.remove('active');
        if (option.getAttribute('data-theme') === themeName) {
            option.classList.add('active');
        }
    });

    // Save theme preference
    localStorage.setItem('legacycointrader-theme', themeName);

    // Show toast notification
    if (window.LegacyCoinTrader && window.LegacyCoinTrader.showToast) {
        const themeNames = {
            'midnight': 'Midnight',
            'neon': 'Neon',
            'cyberpunk': 'Cyberpunk',
            'minimal': 'Minimal'
        };
        window.LegacyCoinTrader.showToast(`Theme changed to ${themeNames[themeName]}`, 'success');
    }

    console.log(`Theme changed to: ${themeName}`);
}

// Get current theme
function getCurrentTheme() {
    const savedTheme = localStorage.getItem('legacycointrader-theme');
    return savedTheme || 'midnight';
}

// Export theme functions
window.ThemeSystem = {
    setTheme,
    getCurrentTheme,
    initThemeSystem
};

// ===== COMPONENT SYSTEM =====

// Base Component Class
class BaseComponent {
    constructor(options = {}) {
        this.options = { ...this.getDefaultOptions(), ...options };
        this.element = null;
        this.isRendered = false;
        this.eventListeners = [];
    }

    getDefaultOptions() {
        return {
            classes: [],
            attributes: {},
            animations: true
        };
    }

    create() {
        this.element = this.render();
        this.bindEvents();
        if (this.options.animations) {
            this.addAnimations();
        }
        this.isRendered = true;
        return this.element;
    }

    render() {
        // Override in subclasses
        throw new Error('render() must be implemented by subclass');
    }

    bindEvents() {
        // Override in subclasses
    }

    addAnimations() {
        if (this.element) {
            this.element.classList.add('animate-fade-in');
        }
    }

    update(data) {
        // Override in subclasses for data updates
    }

    destroy() {
        // Clean up event listeners
        this.eventListeners.forEach(({ element, event, handler }) => {
            element.removeEventListener(event, handler);
        });
        this.eventListeners = [];

        if (this.element && this.element.parentNode) {
            this.element.parentNode.removeChild(this.element);
        }
        this.isRendered = false;
    }

    addEventListener(element, event, handler) {
        element.addEventListener(event, handler);
        this.eventListeners.push({ element, event, handler });
    }
}

// Metric Card Component
class MetricCard extends BaseComponent {
    constructor(options = {}) {
        super(options);
        this.data = options.data || {};
    }

    getDefaultOptions() {
        return {
            ...super.getDefaultOptions(),
            title: '',
            value: 0,
            change: 0,
            changeLabel: '',
            icon: 'fas fa-chart-line',
            type: 'default', // success, warning, danger, info
            format: 'number' // number, currency, percentage
        };
    }

    render() {
        const card = document.createElement('div');
        card.className = `metric-card ai-card ${this.options.classes.join(' ')}`;
        card.setAttribute('data-component', 'metric-card');

        // Apply any additional attributes
        Object.keys(this.options.attributes).forEach(key => {
            card.setAttribute(key, this.options.attributes[key]);
        });

        card.innerHTML = `
            <div class="metric-icon">
                <i class="${this.options.icon}"></i>
            </div>
            <div class="metric-content">
                <h3>${this.options.title}</h3>
                <div class="metric-value ai-gradient ${this.getValueClass()}">
                    ${this.formatValue(this.options.value)}
                </div>
                <div class="metric-change ${this.getChangeClass()}">
                    <i class="${this.getChangeIcon()}"></i>
                    ${this.formatChange(this.options.change)} ${this.options.changeLabel}
                </div>
            </div>
        `;

        return card;
    }

    getValueClass() {
        if (this.options.type === 'success') return 'positive';
        if (this.options.type === 'danger') return 'negative';
        return '';
    }

    getChangeClass() {
        if (this.options.change > 0) return 'positive';
        if (this.options.change < 0) return 'negative';
        return '';
    }

    getChangeIcon() {
        if (this.options.change > 0) return 'fas fa-arrow-up';
        if (this.options.change < 0) return 'fas fa-arrow-down';
        return 'fas fa-minus';
    }

    formatValue(value) {
        switch (this.options.format) {
            case 'currency':
                return formatCurrency(value);
            case 'percentage':
                return formatPercentage(value);
            default:
                return formatNumber(value);
        }
    }

    formatChange(change) {
        if (this.options.format === 'percentage') {
            return `${change > 0 ? '+' : ''}${change.toFixed(2)}%`;
        }
        return `${change > 0 ? '+' : ''}${this.formatValue(change)}`;
    }

    update(newData) {
        this.data = { ...this.data, ...newData };
        if (this.element) {
            // Update the card content with new data
            const valueEl = this.element.querySelector('.metric-value');
            const changeEl = this.element.querySelector('.metric-change');

            if (valueEl) {
                valueEl.textContent = this.formatValue(newData.value || this.options.value);
                valueEl.className = `metric-value ai-gradient ${this.getValueClass()}`;
            }

            if (changeEl && newData.change !== undefined) {
                const changeIcon = changeEl.querySelector('i');
                changeEl.className = `metric-change ${this.getChangeClass()}`;
                if (changeIcon) {
                    changeIcon.className = this.getChangeIcon();
                }
                changeEl.innerHTML = `<i class="${this.getChangeIcon()}"></i> ${this.formatChange(newData.change)} ${newData.changeLabel || this.options.changeLabel}`;
            }

            // Add update animation
            this.element.classList.add('animate-pulse');
            setTimeout(() => {
                this.element.classList.remove('animate-pulse');
            }, 500);
        }
    }
}

// Position Card Component
class PositionCard extends BaseComponent {
    constructor(options = {}) {
        super(options);
        this.position = options.position || {};
    }

    getDefaultOptions() {
        return {
            ...super.getDefaultOptions(),
            showChart: true,
            showActions: true,
            compact: false
        };
    }

    render() {
        const card = document.createElement('div');
        card.className = `position-card ${this.getPositionClass()} ${this.options.classes.join(' ')}`;
        card.setAttribute('data-symbol', this.position.symbol);
        card.setAttribute('data-component', 'position-card');

        // Apply any additional attributes
        Object.keys(this.options.attributes).forEach(key => {
            card.setAttribute(key, this.options.attributes[key]);
        });

        const pnlSign = this.position.pnl >= 0 ? '+' : '';
        const pnlValueSign = this.position.pnl_value >= 0 ? '+' : '';

        card.innerHTML = `
            <!-- Position Header -->
            <div class="position-header">
                <div class="position-info">
                    <div class="position-symbol ${this.getPositionClass()}">
                        ${this.position.symbol}
                    </div>
                    <div class="position-type ${this.position.side || 'long'}">
                        ${this.position.side?.toUpperCase() || 'LONG'}
                    </div>
                </div>
                <div class="position-pnl">
                    <div class="pnl-percentage ${this.getPositionClass()}">
                        ${pnlSign}${this.position.pnl?.toFixed(2) || '0.00'}%
                    </div>
                    <div class="pnl-value ${this.getPositionClass()}">
                        ${pnlValueSign}$${this.position.pnl_value?.toFixed(2) || '0.00'}
                    </div>
                </div>
            </div>

            <!-- Position Metrics Grid -->
            <div class="position-metrics">
                <div class="metrics-row">
                    <div class="metric-column">
                        <div class="metric-item">
                            <div class="metric-label">ENTRY PRICE</div>
                            <div class="metric-value ${this.getPositionClass()}">$${this.position.entry_price?.toFixed(2) || '0.00'}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">POSITION SIZE</div>
                            <div class="metric-value ${this.getPositionClass()}">${this.position.size?.toFixed(2) || '0.00'}</div>
                        </div>
                    </div>
                    <div class="metric-column">
                        <div class="metric-item">
                            <div class="metric-label">CURRENT PRICE</div>
                            <div class="metric-value ${this.getPositionClass()}">$${this.position.current_price?.toFixed(2) || '0.00'}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">CURRENT VALUE</div>
                            <div class="metric-value ${this.getPositionClass()}">$${this.position.current_value?.toFixed(2) || '0.00'}</div>
                        </div>
                    </div>
                </div>
            </div>

            ${this.options.showChart ? `
            <!-- Chart Section -->
            <div class="position-chart">
                <div class="chart-container">
                    <div class="chart-loading" id="loading-${this.position.symbol?.replace('/', '-') || 'unknown'}">
                        <div class="loading-spinner"></div>
                        <span>Loading chart...</span>
                    </div>
                    <canvas id="chart-${this.position.symbol?.replace('/', '-') || 'unknown'}" width="300" height="80" style="display: none;"></canvas>
                </div>
                <div class="chart-info">
                    <div class="trend-indicator ${this.getPositionClass()}">
                        ${this.position.pnl >= 0 ? 'UPWARD' : 'DOWNWARD'}
                    </div>
                    <div class="r-squared">R²: ${this.position.r_squared?.toFixed(1) || '0.0'}%</div>
                </div>
            </div>
            ` : ''}

            ${this.options.showActions ? `
            <!-- Position Actions -->
            <div class="position-actions">
                <button class="btn btn-danger" data-action="market-sell" data-symbol="${this.position.symbol}" data-amount="${this.position.size}" onclick="sellPosition('${this.position.symbol}', ${this.position.size})">
                    Market Sell
                </button>
            </div>
            ` : ''}
        `;

        return card;
    }

    getPositionClass() {
        return (this.position.pnl >= 0) ? 'positive' : 'negative';
    }

    update(newPosition) {
        this.position = { ...this.position, ...newPosition };
        if (this.element) {
            // Update all the position data in the DOM
            this.updatePositionData();
            // Add update animation
            this.element.classList.add('animate-pulse');
            setTimeout(() => {
                this.element.classList.remove('animate-pulse');
            }, 500);
        }
    }

    updatePositionData() {
        if (!this.element) return;

        const pnlSign = this.position.pnl >= 0 ? '+' : '';
        const pnlValueSign = this.position.pnl_value >= 0 ? '+' : '';

        // Update P&L percentage
        const pnlPercentageEl = this.element.querySelector('.pnl-percentage');
        if (pnlPercentageEl) {
            pnlPercentageEl.textContent = `${pnlSign}${this.position.pnl?.toFixed(2) || '0.00'}%`;
            pnlPercentageEl.className = `pnl-percentage ${this.getPositionClass()}`;
        }

        // Update P&L value
        const pnlValueEl = this.element.querySelector('.pnl-value');
        if (pnlValueEl) {
            pnlValueEl.textContent = `${pnlValueSign}$${this.position.pnl_value?.toFixed(2) || '0.00'}`;
            pnlValueEl.className = `pnl-value ${this.getPositionClass()}`;
        }

        // Update current price
        const currentPriceEl = this.element.querySelector('.metric-value');
        if (currentPriceEl) {
            currentPriceEl.textContent = `$${this.position.current_price?.toFixed(2) || '0.00'}`;
            currentPriceEl.className = `metric-value ${this.getPositionClass()}`;
        }

        // Update current value
        const currentValueEl = this.element.querySelectorAll('.metric-value')[1];
        if (currentValueEl) {
            currentValueEl.textContent = `$${this.position.current_value?.toFixed(2) || '0.00'}`;
            currentValueEl.className = `metric-value ${this.getPositionClass()}`;
        }

        // Update trend indicator
        const trendEl = this.element.querySelector('.trend-indicator');
        if (trendEl) {
            trendEl.textContent = this.position.pnl >= 0 ? 'UPWARD' : 'DOWNWARD';
            trendEl.className = `trend-indicator ${this.getPositionClass()}`;
        }

        // Update R-squared
        const rsquaredEl = this.element.querySelector('.r-squared');
        if (rsquaredEl) {
            rsquaredEl.textContent = `R²: ${this.position.r_squared?.toFixed(1) || '0.0'}%`;
        }

        // Update overall card styling
        this.element.className = `position-card ${this.getPositionClass()} ${this.options.classes.join(' ')}`;
    }
}

// Component Factory
const ComponentFactory = {
    create: function(type, options = {}) {
        switch (type) {
            case 'metric-card':
                return new MetricCard(options);
            case 'position-card':
                return new PositionCard(options);
            default:
                throw new Error(`Unknown component type: ${type}`);
        }
    },

    // Batch create multiple components
    createMultiple: function(components) {
        return components.map(comp => this.create(comp.type, comp.options));
    },

    // Render components to a container
    renderTo: function(container, components) {
        if (typeof container === 'string') {
            container = document.querySelector(container);
        }

        if (!container) {
            throw new Error('Container not found');
        }

        components.forEach(component => {
            if (!component.isRendered) {
                container.appendChild(component.create());
            }
        });
    }
};

// Export component system
window.ComponentSystem = {
    BaseComponent,
    MetricCard,
    PositionCard,
    ComponentFactory
};

// Export functions for use in other scripts
window.LegacyCoinTrader = {
    showToast,
    formatCurrency,
    formatPercentage,
    formatNumber
};

// Log successful initialization
console.log('LegacyCoinTrader initialized successfully');
