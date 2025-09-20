/* Position card mini-charts */

const CHART_LOADING_ATTR = 'chartLoading';
const CHART_LOADED_ATTR = 'chartLoaded';
const CHART_OBSERVED_ATTR = 'chartObserved';
const CHART_ERROR_ATTR = 'chartError';

let chartObserver = null;
let gridMutationObserver = null;
let batchTimer = null;
let chartsInitialized = false;

const BATCH_SIZE = 6;
const BATCH_DELAY_MS = 80;
const pendingSymbolQueue = new Map(); // symbol -> { cards: Set<Element>, force: boolean }
const inFlightSymbols = new Set();
let lastBatchIssuedAt = 0;

document.addEventListener('DOMContentLoaded', () => {
  waitForPositionCards();
});

function waitForPositionCards() {
  // Check every 500ms for position cards, up to 10 seconds
  let attempts = 0;
  const maxAttempts = 20;

  const checkInterval = setInterval(() => {
    const positionCards = document.querySelectorAll('.position-card[data-symbol]');

    if (positionCards.length > 0) {
      clearInterval(checkInterval);
      loadAllCharts();
    } else if (attempts >= maxAttempts) {
      clearInterval(checkInterval);
    }

    attempts++;
  }, 500);
}

function setupChartObserver() {
  if (chartObserver || !('IntersectionObserver' in window)) {
    return chartObserver;
  }

  chartObserver = new IntersectionObserver(handleCardVisibility, {
    root: null,
    rootMargin: '200px 0px',
    threshold: 0.1
  });

  return chartObserver;
}

function setupGridMutationObserver() {
  if (gridMutationObserver) {
    return;
  }

  const grid = document.querySelector('.positions-grid');
  if (!grid || !('MutationObserver' in window)) {
    return;
  }

  gridMutationObserver = new MutationObserver(mutations => {
    mutations.forEach(mutation => {
      mutation.addedNodes.forEach(node => {
        if (!(node instanceof Element)) {
          return;
        }

        if (node.matches && node.matches('.position-card[data-symbol]')) {
          registerPositionCard(node);
        } else {
          node.querySelectorAll?.('.position-card[data-symbol]').forEach(registerPositionCard);
        }
      });
    });
  });

  gridMutationObserver.observe(grid, { childList: true });
}

function registerPositionCard(card) {
  if (!card || !(card instanceof Element)) {
    return;
  }

  const symbol = card.getAttribute('data-symbol');
  if (!symbol) {
    return;
  }

  if (card.dataset[CHART_LOADED_ATTR] === 'true' || card.dataset[CHART_LOADING_ATTR] === 'true') {
    return;
  }

  if (!('IntersectionObserver' in window)) {
    // Fallback for older browsers – load immediately
    loadChartForSymbol(symbol, card);
    return;
  }

  setupChartObserver();

  if (card.dataset[CHART_OBSERVED_ATTR] === 'true') {
    return;
  }

  card.dataset[CHART_OBSERVED_ATTR] = 'true';
  chartObserver.observe(card);
}

function handleCardVisibility(entries) {
  entries.forEach(entry => {
    if (!entry.isIntersecting) {
      return;
    }

    const card = entry.target;
    const symbol = card.getAttribute('data-symbol');
    if (!symbol) {
      chartObserver.unobserve(card);
      return;
    }

    chartObserver.unobserve(card);
    loadChartForSymbol(symbol, card);
  });
}

function registerAllPositionCards() {
  const positionCards = document.querySelectorAll('.position-card[data-symbol]');
  if (!positionCards.length) {
    return;
  }

  positionCards.forEach(card => registerPositionCard(card));
}

function loadAllCharts() {
  if (chartsInitialized) {
    console.log('Position charts already initialized; skipping duplicate run.');
    return;
  }
  const positionCards = document.querySelectorAll('.position-card[data-symbol]');
  if (!positionCards.length) {
    console.warn('loadAllCharts called before position cards were ready; skipping for now.');
    return;
  }

  chartsInitialized = true;

  if (!('IntersectionObserver' in window)) {
    console.warn('IntersectionObserver not supported. Position charts will load immediately.');
  }

  setupGridMutationObserver();
  registerAllPositionCards();

  // Load charts immediately for visible cards, lazy load others
  positionCards.forEach((card, index) => {
    const symbol = card.getAttribute('data-symbol');
    if (!symbol) {
      return;
    }

    // Check if card is visible in viewport
    const rect = card.getBoundingClientRect();
    const isVisible = rect.top < window.innerHeight && rect.bottom > 0;

    if (isVisible || index < 3) { // Load first 3 immediately, plus any visible ones
      try {
        loadChartForSymbol(symbol, card, { force: true });
        if (chartObserver) {
          chartObserver.unobserve(card);
        }
      } catch (err) {
        console.warn(`Immediate chart load failed for ${symbol}`, err);
      }
    }
  });
}

function loadChartForSymbol(symbol, cardElement, options = {}) {
  if (!cardElement) {
    console.warn(`Missing card element for ${symbol}`);
    return;
  }

  const force = Boolean(options.force);
  if (!force) {
    if (cardElement.dataset[CHART_LOADED_ATTR] === 'true') {
      console.log(`Chart already loaded for ${symbol}, skipping.`);
      return;
    }
    if (cardElement.dataset[CHART_LOADING_ATTR] === 'true') {
      console.log(`Chart already loading for ${symbol}, skipping.`);
      return;
    }
  }

  cardElement.dataset[CHART_LOADING_ATTR] = 'true';
  delete cardElement.dataset[CHART_ERROR_ATTR];
  showChartLoadingState(cardElement);

  queueChartFetch(symbol, cardElement, { force });
}

function queueChartFetch(symbol, cardElement, { force = false } = {}) {
  let entry = pendingSymbolQueue.get(symbol);
  if (!entry) {
    entry = { cards: new Set(), force: false };
    pendingSymbolQueue.set(symbol, entry);
  }
  entry.cards.add(cardElement);
  entry.force = entry.force || force;

  if (force) {
    flushSymbolQueue(true);
    return;
  }

  if (pendingSymbolQueue.size >= BATCH_SIZE) {
    flushSymbolQueue();
    return;
  }

  if (batchTimer) {
    return;
  }

  const elapsed = Date.now() - lastBatchIssuedAt;
  const delay = Math.max(0, elapsed < BATCH_DELAY_MS ? BATCH_DELAY_MS - elapsed : BATCH_DELAY_MS);
  batchTimer = setTimeout(() => flushSymbolQueue(), delay);
}

function flushSymbolQueue(force = false) {
  if (batchTimer) {
    clearTimeout(batchTimer);
    batchTimer = null;
  }

  if (!pendingSymbolQueue.size) {
    return;
  }

  const symbolsToFetch = [];
  const cardsBySymbol = new Map();

  for (const [symbol, entry] of pendingSymbolQueue) {
    if (inFlightSymbols.has(symbol)) {
      continue;
    }
    if (!force && symbolsToFetch.length >= BATCH_SIZE && !entry.force) {
      continue;
    }
    pendingSymbolQueue.delete(symbol);
    inFlightSymbols.add(symbol);
    symbolsToFetch.push(symbol);
    cardsBySymbol.set(symbol, Array.from(entry.cards));
  }

  if (!symbolsToFetch.length) {
    return;
  }

  lastBatchIssuedAt = Date.now();

  fetch(`/api/v1/market-data/batch-candles`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      symbols: symbolsToFetch,
      limit: 100,
      timeframe: '5m',
      exchange: 'kraken',
      force_fresh: true,  // Force fresh data for charts
    }),
  })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      handleBatchResponse(data, cardsBySymbol);
    })
    .catch(error => {
      for (const [symbol, cards] of cardsBySymbol.entries()) {
        cards.forEach(card => handleChartFailure(symbol, card, `API Error: ${error.message}`));
      }
    })
    .finally(() => {
      symbolsToFetch.forEach(symbol => inFlightSymbols.delete(symbol));
      if (pendingSymbolQueue.size) {
        flushSymbolQueue();
      }
    });
}

function handleBatchResponse(data, cardsBySymbol) {
  const results = data && data.results ? data.results : {};

  for (const [symbol, cards] of cardsBySymbol.entries()) {
    const payload = results[symbol];
    if (!payload || payload.error) {
      const message = payload && payload.error ? payload.error : 'No chart data available';
      cards.forEach(card => handleChartFailure(symbol, card, message));
      continue;
    }

    cards.forEach(card => renderChartPayload(symbol, card, payload));
  }
}

function renderChartPayload(symbol, cardElement, chartData) {
  if (!cardElement || !cardElement.isConnected) {
    return;
  }

  const candles = Array.isArray(chartData.candles) ? chartData.candles : [];

  if (!candles.length) {
    handleChartFailure(symbol, cardElement, 'No chart data available');
    return;
  }

  const chartContainer = cardElement.querySelector('.chart-container');
  const loadingElement = cardElement.querySelector('.chart-loading');
  const canvasElement = cardElement.querySelector('canvas');
  if (!chartContainer || !loadingElement || !canvasElement) {
    handleChartFailure(symbol, cardElement, 'Chart container missing');
    return;
  }

  const rect = chartContainer.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvasElement.width = Math.max(1, rect.width * dpr);
  canvasElement.height = Math.max(1, rect.height * dpr);
  canvasElement.style.width = `${rect.width}px`;
  canvasElement.style.height = `${rect.height}px`;

  const ctx = canvasElement.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);

  loadingElement.style.display = 'none';
  canvasElement.style.display = 'block';

  const entryAttr = parseFloat(cardElement.dataset.entryPrice || '');
  const stopAttr = parseFloat(cardElement.dataset.stopLoss || '');
  const entryLine = Number.isFinite(entryAttr) ? entryAttr : null;
  const stopLine = Number.isFinite(stopAttr) ? stopAttr : null;

  renderEnhancedChart(canvasElement, candles, '5m', {
    entryPrice: entryLine,
    stopLossPrice: stopLine,
    symbol: symbol,
  });

  cardElement.dataset[CHART_LOADED_ATTR] = 'true';
  delete cardElement.dataset[CHART_LOADING_ATTR];
  delete cardElement.dataset[CHART_ERROR_ATTR];

  if (chartData.source === 'fallback' || chartData.note) {
    addFallbackIndicator(cardElement, canvasElement);
  }
}

function addFallbackIndicator(cardElement, canvasElement) {
  const parent = canvasElement.parentNode;
  if (!parent || parent.querySelector('.fallback-indicator')) {
    return;
  }
  parent.style.position = 'relative';
  const indicator = document.createElement('div');
  indicator.className = 'fallback-indicator';
  indicator.textContent = 'DEMO DATA';
  indicator.style.cssText = `
    position: absolute;
    top: 5px;
    right: 5px;
    background: rgba(255, 193, 7, 0.9);
    color: #000;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: bold;
    z-index: 10;
  `;
  parent.appendChild(indicator);
}

function handleChartFailure(symbol, cardElement, message) {
  if (!cardElement || !cardElement.isConnected) {
    return;
  }

  delete cardElement.dataset[CHART_LOADING_ATTR];
  delete cardElement.dataset[CHART_OBSERVED_ATTR];
  cardElement.dataset[CHART_ERROR_ATTR] = message;
  showChartError(cardElement, message);

  setTimeout(() => {
    if (cardElement.isConnected && cardElement.dataset[CHART_LOADED_ATTR] !== 'true') {
      registerPositionCard(cardElement);
    }
  }, 15000);
}

window.registerPositionCardForLazyCharts = function(cardOrSymbol, options = {}) {
  let card = cardOrSymbol;
  if (typeof cardOrSymbol === 'string') {
    card = document.querySelector(`.position-card[data-symbol="${cardOrSymbol}"]`);
  }

  if (!card) {
    return;
  }

  if (options.force) {
    delete card.dataset[CHART_LOADED_ATTR];
    delete card.dataset[CHART_OBSERVED_ATTR];
    delete card.dataset[CHART_ERROR_ATTR];
  }

  registerPositionCard(card);
};

function showChartLoadingState(card) {
  const loadingElement = card.querySelector('.chart-loading');
  const canvasElement = card.querySelector('canvas');
  if (!loadingElement) {
    return;
  }

  const spinner = loadingElement.querySelector('.loading-spinner');
  if (spinner) {
    spinner.style.display = '';
  }

  const loadingText = loadingElement.querySelector('span');
  if (loadingText) {
    loadingText.textContent = 'Loading chart...';
  }

  const errorMessage = loadingElement.querySelector('.chart-error-message');
  if (errorMessage) {
    errorMessage.remove();
  }

  loadingElement.style.display = 'flex';
  if (canvasElement) {
    canvasElement.style.display = 'none';
  }
}

function showChartError(card, message) {
  const loadingElement = card.querySelector('.chart-loading');
  const canvasElement = card.querySelector('canvas');
  if (!loadingElement) {
    return;
  }

  const spinner = loadingElement.querySelector('.loading-spinner');
  if (spinner) {
    spinner.style.display = 'none';
  }

  const loadingText = loadingElement.querySelector('span');
  if (loadingText) {
    loadingText.textContent = 'Chart unavailable';
  }

  let errorMessage = loadingElement.querySelector('.chart-error-message');
  if (!errorMessage) {
    errorMessage = document.createElement('div');
    errorMessage.className = 'chart-error-message';
    errorMessage.style.marginTop = '6px';
    errorMessage.style.fontSize = '11px';
    errorMessage.style.color = '#f97316';
    errorMessage.style.textAlign = 'center';
    loadingElement.appendChild(errorMessage);
  }

  errorMessage.textContent = message || 'Unable to load chart data';
  loadingElement.style.display = 'flex';
  if (canvasElement) {
    canvasElement.style.display = 'none';
  }
}

function renderEnhancedChart(canvas, candles, timeframe = '5m', options = {}) {
  try {
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.width / dpr;
    const height = canvas.height / dpr;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    if (!Array.isArray(candles) || candles.length === 0) {
      return;
    }

    const closePrices = candles
      .map(candle => {
        if (Array.isArray(candle)) {
          return parseFloat(candle[4]);
        }
        if (candle && typeof candle === 'object') {
          return parseFloat(candle.close ?? candle[4]);
        }
        return NaN;
      })
      .filter(price => Number.isFinite(price));

    if (!closePrices.length) {
      return;
    }

    const entryPrice = Number.isFinite(options.entryPrice) ? Number(options.entryPrice) : null;
    const stopLossPrice = Number.isFinite(options.stopLossPrice) ? Number(options.stopLossPrice) : null;

    let minPrice = Math.min(...closePrices);
    let maxPrice = Math.max(...closePrices);

    if (Number.isFinite(entryPrice)) {
      minPrice = Math.min(minPrice, entryPrice);
      maxPrice = Math.max(maxPrice, entryPrice);
    }

    if (Number.isFinite(stopLossPrice)) {
      minPrice = Math.min(minPrice, stopLossPrice);
      maxPrice = Math.max(maxPrice, stopLossPrice);
    }

    if (!Number.isFinite(minPrice) || !Number.isFinite(maxPrice)) {
      return;
    }

    if (maxPrice === minPrice) {
      const pad = maxPrice === 0 ? 0.5 : Math.abs(maxPrice) * 0.005;
      minPrice -= pad;
      maxPrice += pad;
    }

    const priceRange = maxPrice - minPrice;
    const paddingX = Math.max(8, width * 0.04);
    const paddingTop = 10;
    const paddingBottom = 16;
    const drawableWidth = Math.max(1, width - paddingX * 2);
    const drawableHeight = Math.max(1, height - (paddingTop + paddingBottom));

    const priceToY = (price) => {
      if (!Number.isFinite(price)) {
        return null;
      }
      const clamped = Math.min(Math.max(price, minPrice), maxPrice);
      const relative = (clamped - minPrice) / priceRange;
      return paddingTop + (1 - relative) * drawableHeight;
    };

    const firstPrice = closePrices[0];
    const lastPrice = closePrices[closePrices.length - 1];
    const isUptrend = lastPrice >= firstPrice;
    const trendColor = isUptrend ? '#22c55e' : '#ef4444';

    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.strokeStyle = trendColor;
    ctx.beginPath();

    closePrices.forEach((price, index) => {
      const x = paddingX + (index / Math.max(closePrices.length - 1, 1)) * drawableWidth;
      const y = priceToY(price);
      if (y === null) {
        return;
      }
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    const drawReferenceLine = (price, color, label, position = 'above') => {
      if (!Number.isFinite(price)) {
        return;
      }
      const y = priceToY(price);
      if (y === null) {
        return;
      }

      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(paddingX, y);
      ctx.lineTo(width - paddingX, y);
      ctx.stroke();
      ctx.setLineDash([]);

      const labelText = `${label} $${price.toFixed(2)}`;
      ctx.fillStyle = color;
      ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      ctx.textAlign = 'right';

      if (position === 'below') {
        ctx.textBaseline = 'top';
        const labelY = Math.min(height - paddingBottom, y + 4);
        ctx.fillText(labelText, width - paddingX - 4, labelY);
      } else {
        ctx.textBaseline = 'bottom';
        const labelY = Math.max(paddingTop + 6, y - 4);
        ctx.fillText(labelText, width - paddingX - 4, labelY);
      }
      ctx.restore();
    };

    if (Number.isFinite(entryPrice)) {
      drawReferenceLine(entryPrice, '#60a5fa', 'Entry', 'above');
    }

    if (Number.isFinite(stopLossPrice)) {
      drawReferenceLine(stopLossPrice, '#f97316', 'Stop', 'below');
    }

    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.font = '9px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(timeframe.toUpperCase(), paddingX, height - 4);

    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    const priceLabel = `$${lastPrice.toFixed(2)}`;
    ctx.fillText(priceLabel, width - paddingX, paddingTop - 2);
  } catch (error) {
    console.error('Error rendering chart:', error);
  }
}
