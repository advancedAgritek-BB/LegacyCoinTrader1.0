// LegacyCoinTrader Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initApp();
});

function initApp() {
    // Initialize sidebar functionality
    initSidebar();
    
    // Initialize navigation highlighting
    initNavigation();
    
    // Initialize any charts or data visualizations
    initCharts();
    
    // Initialize real-time updates (single interval)
    initRealTimeUpdates();
}

// Sidebar functionality
function initSidebar() {
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            sidebar.classList.toggle('show');
            mainContent.classList.toggle('sidebar-collapsed');
        });
    }
    
    // Close sidebar on mobile when clicking outside
    document.addEventListener('click', function(e) {
        if (window.innerWidth <= 1024) {
            if (!sidebar.contains(e.target) && !sidebarToggle.contains(e.target)) {
                sidebar.classList.remove('show');
                mainContent.classList.remove('sidebar-collapsed');
            }
        }
    });
    
    // Handle window resize
    window.addEventListener('resize', function() {
        if (window.innerWidth > 1024) {
            sidebar.classList.remove('show');
            mainContent.classList.remove('sidebar-collapsed');
        }
    });
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

// Initialize charts
function initCharts() {
    // Performance Chart
    const performanceCtx = document.getElementById('performanceChart');
    if (performanceCtx) {
        new Chart(performanceCtx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    label: 'Portfolio Value',
                    data: [10000, 10500, 11200, 10800, 12000, 11800],
                    borderColor: '#00d4aa',
                    backgroundColor: 'rgba(0, 212, 170, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#f8fafc'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#cbd5e1'
                        },
                        grid: {
                            color: '#334155'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#cbd5e1'
                        },
                        grid: {
                            color: '#334155'
                        }
                    }
                }
            }
        });
    }
    
    // Trading Volume Chart
    const volumeCtx = document.getElementById('volumeChart');
    if (volumeCtx) {
        new Chart(volumeCtx, {
            type: 'bar',
            data: {
                labels: ['BTC', 'ETH', 'SOL', 'MATIC', 'ADA'],
                datasets: [{
                    label: '24h Volume',
                    data: [1200000, 800000, 500000, 300000, 200000],
                    backgroundColor: [
                        '#00d4aa',
                        '#6366f1',
                        '#f59e0b',
                        '#10b981',
                        '#ef4444'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#f8fafc'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#cbd5e1'
                        },
                        grid: {
                            color: '#334155'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#cbd5e1'
                        },
                        grid: {
                            color: '#334155'
                        }
                    }
                }
            }
        });
    }
}

// Real-time updates - SINGLE interval to prevent conflicts
function initRealTimeUpdates() {
    // Single update interval every 10 seconds to prevent conflicts
    setInterval(updateAllData, 10000);
    
    // Initial update
    updateAllData();
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

// Update bot status
function updateBotStatus() {
    fetch('/api/bot-status')
        .then(response => response.json())
        .then(data => {
            const statusIndicator = document.querySelector('.status-indicator');
            const statusText = document.querySelector('.bot-status span');
            
            if (statusIndicator && statusText) {
                if (data.running) {
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

// Update metrics
function updateMetrics() {
    // This would fetch real-time metrics from your backend
    // For now, we'll just simulate some updates
    const metricValues = document.querySelectorAll('.metric-value');
    metricValues.forEach(metric => {
        // Add a subtle animation to show the metric is updating
        metric.style.transform = 'scale(1.05)';
        setTimeout(() => {
            metric.style.transform = 'scale(1)';
        }, 200);
    });
}

// Update dashboard metrics
function updateDashboardMetrics() {
    fetch('/api/dashboard-metrics')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error updating dashboard metrics:', data.error);
                return;
            }
            
            // Update performance metrics
            if (data.performance) {
                updatePerformanceMetrics(data.performance);
            }
            
            // Update allocation display
            if (data.allocation) {
                updateAllocationDisplay(data.allocation);
            }
            
            // Update recent trades
            if (data.recent_trades) {
                updateRecentTrades(data.recent_trades);
            }
        })
        .catch(error => console.error('Error updating dashboard metrics:', error));
}

// Update live signals
function updateLiveSignals() {
    fetch('/api/live-signals')
        .then(response => response.json())
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
    // This function is defined in the HTML template
    // We just need to ensure it exists before calling it
    if (typeof window.updateOpenPositions === 'function') {
        window.updateOpenPositions();
    } else {
        console.log('updateOpenPositions function not available yet');
    }
}

// Update performance metrics
function updatePerformanceMetrics(performance) {
    // Update P&L
    const totalPnl = document.getElementById('totalPnl');
    if (totalPnl && performance.total_pnl !== undefined) {
        totalPnl.textContent = formatCurrency(performance.total_pnl);
    }
    
    // Update total trades
    const totalTrades = document.getElementById('totalTrades');
    if (totalTrades && performance.total_trades !== undefined) {
        totalTrades.textContent = performance.total_trades;
    }
    
    // Update win rate
    const winRate = document.getElementById('winRate');
    if (winRate && performance.win_rate !== undefined) {
        winRate.textContent = formatPercentage(performance.win_rate * 100);
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
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type} fade-in`;
    toast.innerHTML = `
        <div class="toast-content">
            <i class="fas fa-${getToastIcon(type)}"></i>
            <span>${message}</span>
        </div>
    `;
    
    document.body.appendChild(toast);
    
    // Remove toast after 3 seconds
    setTimeout(() => {
        toast.remove();
    }, 3000);
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

// Export functions for use in other scripts
window.LegacyCoinTrader = {
    showToast,
    formatCurrency,
    formatPercentage,
    formatNumber
};

// Log successful initialization
console.log('LegacyCoinTrader initialized successfully');
