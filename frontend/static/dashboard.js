// Dashboard page JS (moved from inline script to comply with CSP)

function showManualPriceModal() {
  const modal = new bootstrap.Modal(document.getElementById('manualPriceModal'));
  modal.show();
}

function saveManualPrice() {
  const symbol = document.getElementById('manualSymbol').value.trim().toUpperCase();
  const price = parseFloat(document.getElementById('manualPrice').value);
  const validityHours = parseInt(document.getElementById('validityHours').value);

  if (!symbol || !price || price <= 0) {
    window.LegacyCoinTrader.showToast('Please enter valid symbol and price', 'error');
    return;
  }

  fetch('/api/manual-prices', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol, price, validity_hours: validityHours })
  })
  .then(r => r.json())
  .then(data => {
    if (data.status === 'success') {
      window.LegacyCoinTrader.showToast(`Manual price set for ${symbol}`, 'success');
      bootstrap.Modal.getInstance(document.getElementById('manualPriceModal')).hide();
      loadManualPrices();
      document.getElementById('manualSymbol').value = '';
      document.getElementById('manualPrice').value = '';
      document.getElementById('validityHours').value = '24';
    } else {
      window.LegacyCoinTrader.showToast(data.error || 'Failed to set manual price', 'error');
    }
  })
  .catch(() => window.LegacyCoinTrader.showToast('Error setting manual price', 'error'));
}

function startBot() {
  const modal = new bootstrap.Modal(document.getElementById('botControlModal'));
  modal.show();
}

function stopBot() {
  if (!confirm('Are you sure you want to stop the trading bot?')) return;
  fetch('/stop_bot', { method: 'POST', headers: { 'Content-Type': 'application/json' }})
    .then(r => r.json())
    .then(data => {
      if (data.status === 'stopped') {
        window.LegacyCoinTrader.showToast('Bot stopped successfully', 'success');
        setTimeout(() => window.location.reload(), 1000);
      }
    })
    .catch(() => window.LegacyCoinTrader.showToast('Error stopping bot', 'error'));
}

function pauseBot() {
  if (!confirm('Are you sure you want to pause the trading bot? It can be resumed later.')) return;
  fetch('/pause_bot', { method: 'POST', headers: { 'Content-Type': 'application/json' }})
    .then(r => r.json())
    .then(data => {
      if (data.status === 'paused') {
        window.LegacyCoinTrader.showToast('Bot paused successfully', 'success');
        setTimeout(() => window.location.reload(), 1000);
      }
    })
    .catch(() => window.LegacyCoinTrader.showToast('Error pausing bot', 'error'));
}

function confirmBotAction() {
  const mode = document.getElementById('executionMode').value;
  const walletBalance = document.getElementById('walletBalance').value;
  const modal = document.getElementById('botControlModal');
  const confirmBtn = document.getElementById('btnConfirmBotAction');
  const originalBtnText = confirmBtn.innerHTML;
  const debugStatus = document.getElementById('debugStatus');
  const debugMessage = document.getElementById('debugMessage');

  confirmBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting Bot...';
  confirmBtn.disabled = true;
  debugStatus.classList.remove('d-none');
  debugMessage.textContent = 'Preparing to start bot...';

  const start = () => startBotWithMode(mode, modal, confirmBtn, originalBtnText, debugMessage);

  if (mode === 'dry_run' && walletBalance) {
    debugMessage.textContent = 'Saving wallet balance...';
    fetch('/api/paper-wallet-balance', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ balance: parseFloat(walletBalance) })
    })
    .then(r => r.json())
    .then(data => {
      if (data.success) {
        debugMessage.textContent = 'Wallet balance saved, starting bot...';
        start();
      } else {
        debugMessage.textContent = 'Error saving wallet balance';
        window.LegacyCoinTrader.showToast('Error saving wallet balance', 'error');
        confirmBtn.innerHTML = originalBtnText;
        confirmBtn.disabled = false;
        debugStatus.classList.add('d-none');
      }
    })
    .catch(err => {
      debugMessage.textContent = 'Error saving wallet balance: ' + err.message;
      window.LegacyCoinTrader.showToast('Error saving wallet balance', 'error');
      confirmBtn.innerHTML = originalBtnText;
      confirmBtn.disabled = false;
      debugStatus.classList.add('d-none');
    });
  } else {
    debugMessage.textContent = 'Starting bot...';
    start();
  }
}

function startBotWithMode(mode, modal, confirmBtn, originalBtnText, debugMessageElement) {
  debugMessageElement.textContent = 'Sending start request to server...';
  fetch('/start_bot', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode })
  })
  .then(r => r.json())
  .then(data => {
    if (data.status === 'started' || data.status === 'running' || data.status === 'already_running') {
      debugMessageElement.textContent = 'Bot started successfully!';
      window.LegacyCoinTrader.showToast(`Bot started in ${mode} mode`, 'success');
      setTimeout(() => { bootstrap.Modal.getInstance(modal).hide(); window.location.reload(); }, 2000);
    } else {
      debugMessageElement.textContent = `Failed to start bot: ${data.status}`;
      window.LegacyCoinTrader.showToast(`Failed to start bot: ${data.status}`, 'error');
      confirmBtn.innerHTML = originalBtnText;
      confirmBtn.disabled = false;
      setTimeout(() => document.getElementById('debugStatus').classList.add('d-none'), 5000);
    }
  })
  .catch(err => {
    debugMessageElement.textContent = 'Error starting bot: ' + err.message;
    window.LegacyCoinTrader.showToast('Error starting bot: ' + err.message, 'error');
    confirmBtn.innerHTML = originalBtnText;
    confirmBtn.disabled = false;
    setTimeout(() => document.getElementById('debugStatus').classList.add('d-none'), 5000);
  });
}

function toggleWalletBalanceField() {
  const isDryRun = document.getElementById('executionMode').value === 'dry_run';
  const section = document.getElementById('walletBalanceSection');
  section.classList.toggle('d-none', !isDryRun);
}

function updateWalletBalance() {
  const newBalance = prompt('Enter new wallet balance (USD):', '10000');
  if (!newBalance || isNaN(newBalance)) return;
  fetch('/api/paper-wallet-balance', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ balance: parseFloat(newBalance) })
  })
  .then(r => r.json())
  .then(data => {
    if (data.success) {
      window.LegacyCoinTrader.showToast('Wallet balance updated successfully', 'success');
      setTimeout(() => window.location.reload(), 1000);
    } else {
      window.LegacyCoinTrader.showToast('Error updating wallet balance', 'error');
    }
  })
  .catch(() => window.LegacyCoinTrader.showToast('Error updating wallet balance', 'error'));
}

function startLiveUpdates() { console.log('Live updates initialized'); }

function updateDashboard() {
  fetch('/api/live-updates')
    .then(r => r.json())
    .then(data => {
      if (data.error) { console.error('Error updating dashboard:', data.error); return; }
      if (data.bot_status) { updateBotStatusDisplay(data.bot_status); }
      if (data.paper_wallet_balance !== undefined) { updatePaperWalletBalance(data.paper_wallet_balance); }
    })
    .catch(err => console.error('Error fetching live updates:', err));

  fetch('/api/wallet-balance')
    .then(r => r.json())
    .then(data => { if (data.success && data.balance !== undefined) updatePaperWalletBalance(data.balance); })
    .catch(err => console.error('Error fetching wallet balance:', err));

  updateOpenPositions();
}

function updateBotStatusDisplay(botStatus) {
  const uptimeElement = document.getElementById('uptime');
  if (uptimeElement && botStatus.uptime) uptimeElement.textContent = botStatus.uptime;
  const startBtn = document.getElementById('btnStartBot');
  const stopBtn = document.getElementById('btnStopBot');
  if (botStatus.running && startBtn) startBtn.classList.add('d-none');
  else if (!botStatus.running && stopBtn) stopBtn.classList.add('d-none');
}

function updatePaperWalletBalance(balance) {
  const walletBalanceDisplay = document.querySelector('.paper-wallet-balance');
  if (walletBalanceDisplay) walletBalanceDisplay.textContent = `$${balance.toFixed(2)}`;
  const topWalletBalance = document.getElementById('topWalletBalance');
  if (topWalletBalance) topWalletBalance.textContent = balance.toFixed(2);
}

function updateOpenPositions() {
  fetch('/api/open-positions')
    .then(r => r.json())
    .then(() => { updateWalletPnl?.(); })
    .catch(err => console.error('Error updating open positions:', err));
}

function refreshOpenPositions() {
  const refreshBtn = document.getElementById('btnRefreshOpenPositions');
  if (refreshBtn) { refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>'; refreshBtn.disabled = true; }
  fetch('/api/open-positions')
    .then(r => r.json())
    .then(() => {
      if (refreshBtn) { refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i>'; refreshBtn.disabled = false; }
    })
    .catch(() => {
      if (refreshBtn) { refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i>'; refreshBtn.disabled = false; }
    });
}

function clearOldPositions() {
  if (!confirm('Are you sure you want to clear old position entries? This will remove positions older than 24 hours.')) return;
  const clearBtn = document.getElementById('btnClearOldPositions');
  if (clearBtn) { clearBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Clearing...'; clearBtn.disabled = true; }
  fetch('/api/clear-old-positions', { method: 'POST', headers: { 'Content-Type': 'application/json' }})
    .then(r => r.json())
    .then(data => {
      if (data.success) {
        window.LegacyCoinTrader.showToast('Old positions cleared successfully', 'success');
        setTimeout(() => refreshOpenPositions(), 1000);
      } else {
        window.LegacyCoinTrader.showToast(`Error: ${data.error}`, 'error');
      }
    })
    .catch(() => window.LegacyCoinTrader.showToast('Error clearing old positions', 'error'))
    .finally(() => {
      if (clearBtn) { clearBtn.innerHTML = '<i class="fas fa-trash"></i> Clear Old'; clearBtn.disabled = false; }
    });
}

function loadManualPrices() {
  fetch('/api/manual-prices')
    .then(r => r.json())
    .then(data => {
      const listDiv = document.getElementById('manualPricesList');
      const noPricesDiv = document.getElementById('noManualPrices');
      if (data.manual_prices && Object.keys(data.manual_prices).length > 0) {
        let html = '<div class="table-responsive"><table class="table table-sm"><thead><tr><th>Symbol</th><th>Price</th><th>Expires</th><th>Actions</th></tr></thead><tbody>';
        for (const [symbol, priceData] of Object.entries(data.manual_prices)) {
          const expiryDate = new Date(priceData.timestamp * 1000 + priceData.validity_hours * 3600000);
          const timeLeft = Math.max(0, Math.floor((expiryDate - new Date()) / (1000 * 60 * 60)));
          html += `<tr>
            <td><strong>${symbol}</strong></td>
            <td>$${priceData.price.toFixed(6)}</td>
            <td>${timeLeft}h left</td>
            <td>
              <button class="btn btn-sm btn-outline-danger" data-symbol="${symbol}" data-action="delete-manual-price">
                <i class="fas fa-trash"></i>
              </button>
            </td>
          </tr>`;
        }
        html += '</tbody></table></div>';
        listDiv.innerHTML = html;
        noPricesDiv.style.display = 'none';
      } else {
        listDiv.innerHTML = '';
        noPricesDiv.style.display = 'block';
      }
    })
    .catch(err => console.error('Error loading manual prices:', err));
}

function loadPriceSourceHealth() {
  fetch('/api/price-source-health')
    .then(r => r.json())
    .then(data => {
      const healthDiv = document.getElementById('priceSourceHealth');
      if (data.price_sources) {
        let html = '<div class="table-responsive"><table class="table table-sm"><thead><tr><th>Source</th><th>Health Score</th><th>Success Rate</th><th>Attempts</th><th>Status</th></tr></thead><tbody>';
        for (const [source, health] of Object.entries(data.price_sources)) {
          const healthScore = health.health_score;
          let statusBadge = '';
          if (healthScore >= 0.8) statusBadge = '<span class="badge bg-success">Excellent</span>';
          else if (healthScore >= 0.6) statusBadge = '<span class="badge bg-warning">Good</span>';
          else if (healthScore >= 0.3) statusBadge = '<span class="badge bg-warning">Fair</span>';
          else statusBadge = '<span class="badge bg-danger">Poor</span>';
          html += `<tr>
            <td><strong>${source}</strong></td>
            <td>${(healthScore * 100).toFixed(1)}%</td>
            <td>${(health.success_rate * 100).toFixed(1)}%</td>
            <td>${health.total_attempts}</td>
            <td>${statusBadge}</td>
          </tr>`;
        }
        html += '</tbody></table></div>';
        if (data.best_sources && data.best_sources.length > 0) {
          html += '<div class="mt-3"><strong>Top Performing Sources:</strong> ' +
            data.best_sources.map(source => `<span class=\"badge bg-info me-1\">${source}</span>`).join('') +
            '</div>';
        }
        healthDiv.innerHTML = html;
      }
    })
    .catch(err => console.error('Error loading price source health:', err));
}

document.addEventListener('DOMContentLoaded', function() {
  const executionMode = document.getElementById('executionMode');
  if (executionMode) executionMode.addEventListener('change', toggleWalletBalanceField);
  toggleWalletBalanceField();

  document.getElementById('btnStartBot')?.addEventListener('click', startBot);
  document.getElementById('btnPauseBot')?.addEventListener('click', pauseBot);
  document.getElementById('btnStopBot')?.addEventListener('click', stopBot);
  document.getElementById('btnStopConflicts')?.addEventListener('click', () => {
    fetch('/stop_conflicts', { method: 'POST' })
      .then(r => r.json())
      .then(() => { window.LegacyCoinTrader.showToast('Conflicting processes stopped', 'success'); setTimeout(() => window.location.reload(), 1000); })
      .catch(() => window.LegacyCoinTrader.showToast('Error stopping conflicts', 'error'));
  });
  document.getElementById('btnForceRefresh')?.addEventListener('click', () => {
    const currentUrl = window.location.href;
    window.location.href = currentUrl + '?refresh=' + Date.now();
  });
  document.getElementById('btnClearOldPositions')?.addEventListener('click', clearOldPositions);
  document.getElementById('btnRefreshOpenPositions')?.addEventListener('click', refreshOpenPositions);
  document.getElementById('btnShowManualPriceModal')?.addEventListener('click', showManualPriceModal);
  document.getElementById('btnSaveManualPrice')?.addEventListener('click', saveManualPrice);
  document.getElementById('btnConfirmBotAction')?.addEventListener('click', confirmBotAction);
  document.getElementById('btnUpdateWalletBalance')?.addEventListener('click', updateWalletBalance);

  document.getElementById('manualPricesList')?.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-action="delete-manual-price"]');
    if (!btn) return;
    const symbol = btn.getAttribute('data-symbol');
    if (!confirm(`Delete manual price for ${symbol}?`)) return;
    fetch(`/api/manual-prices/${symbol}`, { method: 'DELETE' })
      .then(r => r.json())
      .then(data => {
        if (data.status === 'deleted') {
          window.LegacyCoinTrader.showToast(`Manual price deleted for ${symbol}`, 'success');
          loadManualPrices();
        } else {
          window.LegacyCoinTrader.showToast(data.error || 'Failed to delete manual price', 'error');
        }
      })
      .catch(() => window.LegacyCoinTrader.showToast('Error deleting manual price', 'error'))
  });

  startLiveUpdates();
  loadManualPrices();
  loadPriceSourceHealth();
  setInterval(() => { updateOpenPositions(); loadManualPrices(); loadPriceSourceHealth(); }, 30000);
});


