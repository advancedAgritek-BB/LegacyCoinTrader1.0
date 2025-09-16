// Test script to verify chart rendering fixes
// This simulates the chart coordinate calculations with the updated logic

function testChartScaling() {
    console.log("🧪 Testing Chart Scaling Fixes\n");

    // Test case 1: Long position with positive P&L
    console.log("📈 Test Case 1: Long Position with Positive P&L");
    const entryPrice1 = 50000;
    const currentPrice1 = 51000;
    const stopLoss1 = 49000;
    const prices1 = [49500, 49800, 50200, 50500, 50800, 51000];
    const trendPrices1 = [49600, 49900, 50200, 50500, 50800, 51100];

    testPositionChart("Long-Profit", entryPrice1, currentPrice1, stopLoss1, prices1, trendPrices1);

    // Test case 2: Short position with positive P&L
    console.log("\n📉 Test Case 2: Short Position with Positive P&L");
    const entryPrice2 = 50000;
    const currentPrice2 = 49000;
    const stopLoss2 = 51000;
    const prices2 = [50500, 50200, 49900, 49600, 49300, 49000];
    const trendPrices2 = [50400, 50100, 49800, 49500, 49200, 48900];

    testPositionChart("Short-Profit", entryPrice2, currentPrice2, stopLoss2, prices2, trendPrices2);

    // Test case 3: Flat price range (edge case)
    console.log("\n📊 Test Case 3: Flat Price Range");
    const entryPrice3 = 50000;
    const currentPrice3 = 50000;
    const stopLoss3 = 50000;
    const prices3 = [50000, 50000, 50000, 50000, 50000, 50000];
    const trendPrices3 = [50000, 50000, 50000, 50000, 50000, 50000];

    testPositionChart("Flat-Range", entryPrice3, currentPrice3, stopLoss3, prices3, trendPrices3);
}

function testPositionChart(testName, entryPrice, currentPrice, stopLossPrice, prices, trendPrices) {
    // Simulate the updated chart scaling logic
    const allPrices = [...prices, ...trendPrices, entryPrice];
    if (stopLossPrice && stopLossPrice > 0) {
        allPrices.push(stopLossPrice);
    }

    const minPrice = Math.min(...allPrices);
    const maxPrice = Math.max(...allPrices);
    let priceRange = maxPrice - minPrice;

    // Apply the fix for zero price range
    if (priceRange === 0) {
        priceRange = Math.max(maxPrice * 0.001, 0.01);
    }

    const canvasHeight = 120;

    // Calculate Y coordinates
    const entryY = canvasHeight - ((entryPrice - minPrice) / priceRange) * canvasHeight;
    const currentY = canvasHeight - ((currentPrice - minPrice) / priceRange) * canvasHeight;
    const stopLossY = stopLossPrice ? canvasHeight - ((stopLossPrice - minPrice) / priceRange) * canvasHeight : null;

    console.log(`   Price Range: $${minPrice.toFixed(2)} - $${maxPrice.toFixed(2)} ($${priceRange.toFixed(2)})`);
    console.log(`   Entry Price: $${entryPrice} → Y: ${entryY.toFixed(1)}px`);
    console.log(`   Current Price: $${currentPrice} → Y: ${currentY.toFixed(1)}px`);
    if (stopLossY !== null) {
        console.log(`   Stop Loss: $${stopLossPrice} → Y: ${stopLossY.toFixed(1)}px`);
    }

    // Validation
    const entryInRange = entryY >= 0 && entryY <= canvasHeight;
    const currentInRange = currentY >= 0 && currentY <= canvasHeight;
    const stopLossInRange = stopLossY === null || (stopLossY >= 0 && stopLossY <= canvasHeight);

    console.log(`   ✅ Entry in range: ${entryInRange}`);
    console.log(`   ✅ Current in range: ${currentInRange}`);
    console.log(`   ✅ Stop loss in range: ${stopLossInRange}`);

    // Check positioning logic
    const pnl = currentPrice > entryPrice ? (currentPrice - entryPrice) / entryPrice * 100 : (entryPrice - currentPrice) / entryPrice * -100;
    console.log(`   P&L: ${pnl > 0 ? '+' : ''}${pnl.toFixed(2)}%`);

    if (pnl > 0) {
        if (currentPrice > entryPrice) {
            // Long position profit
            const correctPositioning = currentY < entryY; // Current price should be above entry price
            console.log(`   🎯 Long Profit: ${correctPositioning ? '✅' : '❌'} Current price ${correctPositioning ? 'above' : 'below'} entry price`);
        } else {
            // Short position profit
            const correctPositioning = currentY > entryY; // Current price should be below entry price
            console.log(`   🎯 Short Profit: ${correctPositioning ? '✅' : '❌'} Current price ${correctPositioning ? 'below' : 'above'} entry price`);
        }
    }
}

// Run the tests
testChartScaling();

console.log("\n✅ Chart scaling tests completed!");
console.log("\n💡 If all tests show ✅ for positioning, the chart rendering fixes should work correctly.");
console.log("   The entry price line should now be properly positioned relative to the price data.");
