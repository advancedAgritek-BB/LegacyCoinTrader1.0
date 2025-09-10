/* Position card mini-charts (CSP-compliant, no inline scripts) */

(function () {
  function parseNumberWithCommas(str) {
    if (!str) return 0;
    return parseFloat(String(str).replace(/,/g, '')) || 0;
  }

  function getBackendPositionData() {
    if (window.positionData && Object.keys(window.positionData).length) {
      return window.positionData;
    }
    // Fallback: try to build from DOM metrics
    const map = {};
    document.querySelectorAll('.position-card').forEach(card => {
      const symbolEl = card.querySelector('.position-symbol');
      if (!symbolEl) return;
      const symbol = symbolEl.textContent.trim();
      const metricValues = card.querySelectorAll('.metric-value');
      const entryPrice = parseNumberWithCommas((metricValues[0]?.textContent || '').replace('$', ''));
      const size = parseNumberWithCommas(metricValues[1]?.textContent || '0');
      const currentPrice = parseNumberWithCommas((metricValues[2]?.textContent || '').replace('$', ''));
      const pnlText = (card.querySelector('.pnl-percentage')?.textContent || '0%').replace('%', '');
      const pnl = parseFloat(pnlText) || 0;
      map[symbol] = {
        symbol,
        entry_price: entryPrice,
        current_price: currentPrice,
        size,
        pnl
      };
    });
    return map;
  }

  function getCachedChartData(symbol) {
    try {
      const cached = localStorage.getItem('chart_' + symbol);
      if (cached) {
        const parsed = JSON.parse(cached);
        const now = Date.now();
        // Cache for 5 minutes
        if (now - parsed.timestamp < 300000) {
          return parsed.data;
        } else {
          localStorage.removeItem('chart_' + symbol);
        }
      }
    } catch (e) {
      console.warn('Error reading chart cache for', symbol, e);
    }
    return null;
  }

  function setCachedChartData(symbol, data) {
    try {
      const cacheData = {
        data: data,
        timestamp: Date.now()
      };
      localStorage.setItem('chart_' + symbol, JSON.stringify(cacheData));
    } catch (e) {
      console.warn('Error caching chart data for', symbol, e);
    }
  }

  async function fetchRealChartData(symbol) {
    console.log('fetchRealChartData called for symbol:', symbol);
    
    // Check cache first
    const cached = getCachedChartData(symbol);
    if (cached && cached.length > 0) {
      console.log('Using cached chart data for', symbol);
      return cached;
    }

    const tryFetch = async (sym) => {
      const controller = new AbortController();
      const tid = setTimeout(() => controller.abort(), 10000);
      const res = await fetch('/api/candle-data?symbol=' + encodeURIComponent(sym) + '&limit=50&interval=5m', { signal: controller.signal });
      clearTimeout(tid);
      const data = await res.json();
      return { res, data };
    };

    try {
      console.log('Fetching fresh data for symbol:', symbol);
      let { res, data } = await tryFetch(symbol);
      console.log('API response for', symbol, ':', data, 'Status:', res.status);

      // If BTC symbol returned no candles, try Kraken XBT alias
      if ((!res.ok || !data || !Array.isArray(data.candles) || data.candles.length === 0) && typeof symbol === 'string' && symbol.includes('BTC/')) {
        const alias = symbol.replace('BTC/', 'XBT/');
        console.log('Retrying with Kraken alias', alias, 'for', symbol);
        try {
          const r2 = await tryFetch(alias);
          res = r2.res;
          data = r2.data;
          console.log('Alias API response for', alias, ':', data, 'Status:', res.status);
        } catch (aliasErr) {
          console.warn('Alias fetch failed for', alias, aliasErr);
        }
      }

      // Check if the API returned an error (either HTTP error or success: false)
      if (!res.ok || (data && data.success === false)) {
        const errorMsg = data && data.error ? data.error : 'HTTP ' + res.status;
        console.warn('API error for', symbol, ':', errorMsg);
        return [];
      }

      if (data && Array.isArray(data.candles)) {
        const prices = data.candles.map(c => parseFloat(c.close));
        if (prices && prices.length > 0) {
          setCachedChartData(symbol, prices);
          return prices;
        }
      }
      return [];
    } catch (e) {
      console.warn('fetchRealChartData failed for', symbol, e);
      return [];
    }
  }

  function drawEnhancedChart(ctx, canvas, prices, entryPrice, isPositive, stopPrice) {
    const width = canvas.width;
    const height = canvas.height;
    const padding = 15;

    // Include stop price in bounds if provided
    const hasStop = typeof stopPrice === 'number' && isFinite(stopPrice) && stopPrice > 0;
    const minPrice = hasStop ? Math.min(...prices, entryPrice, stopPrice) : Math.min(...prices, entryPrice);
    const maxPrice = hasStop ? Math.max(...prices, entryPrice, stopPrice) : Math.max(...prices, entryPrice);
    const priceRange = maxPrice - minPrice;
    if (priceRange <= 0) {
      ctx.strokeStyle = isPositive ? '#00ff88' : '#ff4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(padding, height / 2);
      ctx.lineTo(width - padding, height / 2);
      ctx.stroke();
      return;
    }

    ctx.strokeStyle = isPositive ? '#00ff88' : '#ff4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < prices.length; i++) {
      const x = padding + (i / (prices.length - 1)) * (width - 2 * padding);
      const y = height - padding - ((prices[i] - minPrice) / priceRange) * (height - 2 * padding);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // entry line
    const entryY = height - padding - ((entryPrice - minPrice) / priceRange) * (height - 2 * padding);
    ctx.strokeStyle = '#9966ff';
    ctx.setLineDash([5, 5]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, entryY);
    ctx.lineTo(width - padding, entryY);
    ctx.stroke();
    ctx.setLineDash([]);

    // entry price label
    ctx.fillStyle = '#9966ff';
    ctx.font = 'bold 11px Arial';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    const labelText = '$' + entryPrice.toFixed(2);
    const labelX = width - padding - 5;
    const labelY = entryY - 3;
    ctx.fillText(labelText, labelX, labelY);

    // stop line (stop loss or trailing stop)
    if (hasStop) {
      const stopY = height - padding - ((stopPrice - minPrice) / priceRange) * (height - 2 * padding);
      ctx.strokeStyle = '#ffcc00';
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding, stopY);
      ctx.lineTo(width - padding, stopY);
      ctx.stroke();
      ctx.setLineDash([]);

      // stop price label
      ctx.fillStyle = '#ffcc00';
      ctx.font = 'bold 11px Arial';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      const stopLabel = 'SL $' + Number(stopPrice).toFixed(2);
      const stopLabelX = width - padding - 5;
      const stopLabelY = stopY + 3;
      ctx.fillText(stopLabel, stopLabelX, stopLabelY);
    }
  }

  function drawErrorChart(ctx, canvas, message, symbol = null) {
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(239, 68, 68, 0.08)';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 1;
    ctx.strokeRect(1, 1, w - 2, h - 2);
    ctx.fillStyle = '#ef4444';
    ctx.font = '11px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(message, w / 2, h / 2 - 5);
    // Add a small hint for BTC symbols
    if (message === 'No data' && symbol && symbol.includes('BTC/')) {
      ctx.font = '9px Arial';
      ctx.fillStyle = '#888';
      ctx.fillText('(Check Kraken API)', w / 2, h / 2 + 8);
    }
  }

  async function createPositionChart(canvasId, position) {
    try {
      const canvas = document.getElementById(canvasId);
      const loadingDiv = document.getElementById('loading-' + canvasId.replace('chart-', ''));

      if (!canvas) {
        console.warn('Canvas not found:', canvasId);
        return;
      }

    // Show loading state
    if (loadingDiv) {
      loadingDiv.style.display = 'flex';
    }
    canvas.style.display = 'none';

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      // Hide loading and show error
      if (loadingDiv) loadingDiv.style.display = 'none';
      canvas.style.display = 'block';
      return;
    }

    canvas.width = 300;
    canvas.height = 80;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    console.log('Creating chart for', position.symbol);

    // Add timeout to prevent hanging
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error('Chart fetch timeout')), 15000)
    );

    let prices;
    try {
      prices = await Promise.race([fetchRealChartData(position.symbol), timeoutPromise]);
    } catch (error) {
      console.warn('Chart fetch failed or timed out for', position.symbol, ':', error.message);
      prices = [];
    }

    console.log('Fetched prices for', position.symbol, ':', prices ? prices.length : 0, 'items');

    if (prices && prices.length) {
      console.log('Drawing chart with real data for', position.symbol);
      drawEnhancedChart(ctx, canvas, prices, Number(position.entry_price) || 0, (position.pnl || 0) >= 0, Number(position.stop_price) || undefined);
    } else {
      console.log('Drawing error chart for', position.symbol);
      drawErrorChart(ctx, canvas, 'No data', position.symbol);
    }

    // Hide loading and show canvas
    console.log('Chart creation complete for', position.symbol, '- hiding loading, showing canvas');
    if (loadingDiv) {
      loadingDiv.style.display = 'none';
      console.log('Loading div hidden for', position.symbol);
    }
      canvas.style.display = 'block';
      console.log('Canvas displayed for', position.symbol);
    } catch (error) {
      console.error('Error creating chart for', position.symbol, ':', error);
      // Ensure loading is hidden and show error state
      const canvas = document.getElementById(canvasId);
      const loadingDiv = document.getElementById('loading-' + canvasId.replace('chart-', ''));

      if (loadingDiv) loadingDiv.style.display = 'none';
      if (canvas) {
        canvas.style.display = 'block';
        try {
          const ctx = canvas.getContext('2d');
          if (ctx) {
            drawErrorChart(ctx, canvas, 'Error', position.symbol);
          }
        } catch (drawError) {
          console.error('Failed to draw error chart:', drawError);
        }
      }
    }
  }

  let chartQueue = [];
  let isProcessing = false;
  function queueChart(canvas, position) {
    chartQueue.push({ canvas, position });
    processQueue();
  }
  function processQueue() {
    if (isProcessing || chartQueue.length === 0) return;
    isProcessing = true;
    const { canvas, position } = chartQueue.shift();
    console.log('Processing chart for', position.symbol);

    createPositionChart(canvas.id, position).then(() => {
      console.log('Chart processed successfully for', position.symbol);
    }).catch((error) => {
      console.error('Chart processing failed for', position.symbol, ':', error);
    }).finally(() => {
      isProcessing = false;
      console.log('Chart processing complete, moving to next in queue');
      setTimeout(processQueue, 100);
    });
  }

  // Intersection Observer for lazy loading
  let observer = null;

  function initLazyLoading() {
    const canvases = document.querySelectorAll('canvas[id^="chart-"]');
    if (!canvases.length) return;

    const posMap = getBackendPositionData();

    // Create intersection observer
    observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const canvas = entry.target;
          const symbol = canvas.id.replace('chart-', '').replace(/-/g, '/');
          const position = posMap[symbol];

          if (position && !canvas.hasAttribute('data-loaded')) {
            canvas.setAttribute('data-loaded', 'true');
            queueChart(canvas, position);
            observer.unobserve(canvas); // Stop observing once loaded
          } else if (!position) {
            // DOM fallback already handled in getBackendPositionData
            const fallback = getBackendPositionData()[symbol];
            if (fallback && !canvas.hasAttribute('data-loaded')) {
              canvas.setAttribute('data-loaded', 'true');
              queueChart(canvas, fallback);
              observer.unobserve(canvas);
            }
          }
        }
      });
    }, {
      rootMargin: '50px' // Start loading 50px before chart comes into view
    });

    // Observe all canvases
    canvases.forEach(canvas => {
      observer.observe(canvas);
    });

    // Load first 3 charts immediately for better perceived performance
    const immediateLoad = Array.from(canvases).slice(0, 3);
    immediateLoad.forEach(canvas => {
      const symbol = canvas.id.replace('chart-', '').replace(/-/g, '/');
      const position = posMap[symbol];
      if (position && !canvas.hasAttribute('data-loaded')) {
        canvas.setAttribute('data-loaded', 'true');
        queueChart(canvas, position);
        observer.unobserve(canvas);
      }
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    initLazyLoading();
  });
})();

