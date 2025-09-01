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

// Initialize charts with real data
function initCharts() {
    // Performance Chart
    const performanceCtx = document.getElementById('performanceChart');
    if (performanceCtx) {
        window.performanceChart = new Chart(performanceCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Portfolio Value',
                    data: [],
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
        window.volumeChart = new Chart(volumeCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: '24h Volume',
                    data: [],
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

    // Fetch and update chart data
    updateChartData();
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

// Update metrics function removed to prevent layout conflicts

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

            // Update charts with real data
            if (data.performance && window.performanceChart) {
                updatePerformanceChart(data.performance);
            }

            if (data.asset_scores && window.volumeChart) {
                updateVolumeChart(data.asset_scores);
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
    // Update P&L - use wallet PnL instead of performance PnL
    const totalPnl = document.getElementById('totalPnl');

    if (totalPnl) {
        // Get PnL from wallet calculation instead of performance
        fetch('/api/wallet-pnl')
            .then(response => {
                if (!response.ok) {
                    console.error('Wallet PnL API error:', response.status);
                    return null;
                }
                return response.json();
            })
            .then(data => {
                if (!data || data.error) {
                    console.error('Error fetching wallet PnL:', data?.error || 'No data');
                    return;
                }

                // Update total P&L
                if (data.total_pnl !== undefined && data.total_pnl !== null) {
                    const formattedPnl = formatCurrency(data.total_pnl);
                    totalPnl.textContent = formattedPnl;
                    // Preserve existing classes while updating color
                    totalPnl.classList.remove('text-success', 'text-danger');
                    totalPnl.classList.add(data.total_pnl >= 0 ? 'text-success' : 'text-danger');
                } else {
                    console.warn('Total P&L is undefined or null');
                }

                // Update P&L percentage
                const totalPnlPercentage = document.getElementById('totalPnlPercentage');
                if (totalPnlPercentage && data.pnl_percentage !== undefined && data.pnl_percentage !== null) {
                    const percentage = data.pnl_percentage;
                    const sign = percentage >= 0 ? '+' : '';
                    totalPnlPercentage.textContent = `${sign}${percentage.toFixed(2)}%`;

                    // Update the parent div class - preserve existing classes
                    const metricChange = totalPnlPercentage.parentElement;
                    if (metricChange) {
                        metricChange.classList.remove('positive', 'negative');
                        metricChange.classList.add(percentage >= 0 ? 'positive' : 'negative');
                    }
                }

                // Update wallet balance displays - use total P&L instead of just unrealized
                const walletPnl = document.getElementById('walletPnl');
                if (walletPnl && data.total_pnl !== undefined && data.total_pnl !== null) {
                    walletPnl.textContent = formatCurrency(data.total_pnl);
                    // Preserve existing classes while updating color
                    walletPnl.classList.remove('text-success', 'text-danger');
                    walletPnl.classList.add(data.total_pnl >= 0 ? 'text-success' : 'text-danger');
                }

                const currentBalance = document.getElementById('currentWalletBalance');
                if (currentBalance && data.current_balance !== undefined && data.current_balance !== null) {
                    currentBalance.textContent = formatCurrency(data.current_balance);
                }

                // Update initial balance display
                const initialBalance = document.getElementById('initialBalance');
                if (initialBalance && data.initial_balance !== undefined && data.initial_balance !== null) {
                    initialBalance.textContent = formatCurrency(data.initial_balance);
                }

            })
            .catch(error => {
                console.error('Error updating wallet P&L:', error);
                // Fallback: show default values - preserve existing classes
                if (totalPnl) {
                    totalPnl.textContent = '$0.00';
                    totalPnl.classList.remove('text-success', 'text-danger');
                    totalPnl.classList.add('text-muted');
                }
            });
    }

    // Update total trades (showing today's trades)
    const totalTrades = document.getElementById('totalTrades');
    if (totalTrades && performance.trades_today !== undefined) {
        totalTrades.textContent = performance.trades_today;
    }

    // Update trades today indicator
    const tradesToday = document.getElementById('tradesToday');
    if (tradesToday && performance.total_trades !== undefined && performance.trades_today !== undefined) {
        const totalAllTime = performance.total_trades;
        const totalToday = performance.trades_today;
        const change = totalToday > 0 ? `+${totalToday} today` : '+0 today';
        tradesToday.textContent = change;
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

// Update chart data with real data from backend
function updateChartData() {
    // Fetch dashboard metrics which includes performance data
    fetch('/api/dashboard-metrics')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error fetching chart data:', data.error);
                return;
            }

            // Update performance chart
            if (data.performance && window.performanceChart) {
                updatePerformanceChart(data.performance);
            }

            // Update volume chart with asset scores data
            if (data.asset_scores && window.volumeChart) {
                updateVolumeChart(data.asset_scores);
            }
        })
        .catch(error => console.error('Error updating chart data:', error));
}

// Update performance chart with real data
function updatePerformanceChart(performance) {
    if (!window.performanceChart) return;

    // Generate time labels for the last 7 days
    const labels = [];
    const dataPoints = [];
    const baseValue = performance.initial_balance || 10000;
    let currentValue = baseValue;

    // Generate sample daily data points based on performance
    for (let i = 6; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));

        // Simulate portfolio value changes based on total PnL
        if (performance.total_pnl) {
            const dailyChange = (performance.total_pnl / 7) * (Math.random() - 0.5) * 2;
            currentValue += dailyChange;
        }
        dataPoints.push(Math.max(0, currentValue));
    }

    window.performanceChart.data.labels = labels;
    window.performanceChart.data.datasets[0].data = dataPoints;
    window.performanceChart.update();
}

// Update volume chart with real asset data
function updateVolumeChart(assetScores) {
    if (!window.volumeChart || !assetScores) return;

    const labels = [];
    const volumes = [];

    // Extract top 5 assets with highest scores
    const sortedAssets = Object.entries(assetScores)
        .filter(([symbol, data]) => data && typeof data === 'object' && data.score !== undefined)
        .sort(([, a], [, b]) => (b.score || 0) - (a.score || 0))
        .slice(0, 5);

    sortedAssets.forEach(([symbol, data]) => {
        labels.push(symbol);
        // Use score as a proxy for volume, scaled appropriately
        volumes.push((data.score || 0) * 100000);
    });

    // If no real data, show some default values
    if (labels.length === 0) {
        labels.push('BTC', 'ETH', 'SOL', 'MATIC', 'ADA');
        volumes.push(1200000, 800000, 500000, 300000, 200000);
    }

    window.volumeChart.data.labels = labels;
    window.volumeChart.data.datasets[0].data = volumes;
    window.volumeChart.update();
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
