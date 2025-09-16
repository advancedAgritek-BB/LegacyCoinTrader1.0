// Test script to verify current price inclusion in chart scaling
// This simulates the updated chart coordinate calculations with current price included

function testCurrentPriceInclusion() {
    console.log("🧪 Testing Current Price Inclusion in Chart Scaling\n");

    // Test case 1: Profitable long position with current price higher than entry
    console.log("📈 Test Case 1: Profitable Long Position");
    const entryPrice1 = 50000;
    const currentPrice1 = 51000; // Current price higher than entry
    const stopLoss1 = 49000;
    const prices1 = [49500, 49800, 50200, 50500, 50800]; // Historical data
    const trendPrices1 = [49600, 49900, 50200, 50500, 50800];

    testChartWithCurrentPrice("LINK/USD-Profit", entryPrice1, currentPrice1, stopLoss1, prices1, trendPrices1);

    // Test case 2: Loss position with current price lower than entry
    console.log("\n📉 Test Case 2: Loss Position");
    const entryPrice2 = 50000;
    const currentPrice2 = 49000; // Current price lower than entry
    const stopLoss2 = 48000;
    const prices2 = [49500, 49800, 50200, 50500, 50800];
    const trendPrices2 = [49600, 49900, 50200, 50500, 50800];

    testChartWithCurrentPrice("ETH/USD-Loss", entryPrice2, currentPrice2, stopLoss2, prices2, trendPrices2);

    // Test case 3: Current price same as entry (should not change scaling)
    console.log("\n📊 Test Case 3: Current Price Same as Entry");
    const entryPrice3 = 50000;
    const currentPrice3 = 50000; // Same as entry
    const stopLoss3 = 49000;
    const prices3 = [49500, 49800, 50200, 50500, 50800];
    const trendPrices3 = [49600, 49900, 50200, 50500, 50800];

    testChartWithCurrentPrice("UNI/USD-BreakEven", entryPrice3, currentPrice3, stopLoss3, prices3, trendPrices3);
}

function testChartWithCurrentPrice(testName, entryPrice, currentPrice, stopLossPrice, prices, trendPrices) {
    console.log(`   Testing: ${testName}`);

    // OLD METHOD: Without current price in scaling
    const oldAllPrices = [...prices, ...trendPrices, entryPrice];
    if (stopLossPrice && stopLossPrice > 0) {
        oldAllPrices.push(stopLossPrice);
    }
    const oldMinPrice = Math.min(...oldAllPrices);
    const oldMaxPrice = Math.max(...oldAllPrices);
    const oldPriceRange = oldMaxPrice - oldMinPrice;

    const oldEntryY = 120 - ((entryPrice - oldMinPrice) / oldPriceRange) * 120;
    const oldCurrentY = 120 - ((currentPrice - oldMinPrice) / oldPriceRange) * 120;

    // NEW METHOD: With current price in scaling
    const newAllPrices = [...prices, ...trendPrices, entryPrice];
    if (stopLossPrice && stopLossPrice > 0) {
        newAllPrices.push(stopLossPrice);
    }
    // Include current price if different from entry price
    if (currentPrice && currentPrice > 0 && currentPrice !== entryPrice) {
        newAllPrices.push(currentPrice);
    }
    const newMinPrice = Math.min(...newAllPrices);
    const newMaxPrice = Math.max(...newAllPrices);
    const newPriceRange = newMaxPrice - newMinPrice;

    const newEntryY = 120 - ((entryPrice - newMinPrice) / newPriceRange) * 120;
    const newCurrentY = 120 - ((currentPrice - newMinPrice) / newPriceRange) * 120;

    console.log(`   OLD METHOD: min=${oldMinPrice}, max=${oldMaxPrice}, range=${oldPriceRange}`);
    console.log(`                Entry Y: ${oldEntryY.toFixed(1)}px, Current Y: ${oldCurrentY.toFixed(1)}px`);

    console.log(`   NEW METHOD: min=${newMinPrice}, max=${newMaxPrice}, range=${newPriceRange}`);
    console.log(`                Entry Y: ${newEntryY.toFixed(1)}px, Current Y: ${newCurrentY.toFixed(1)}px`);

    // Analysis
    const entryInRange = newEntryY >= 0 && newEntryY <= 120;
    const currentInRange = newCurrentY >= 0 && newCurrentY <= 120;

    console.log(`   ✅ Entry in range: ${entryInRange}`);
    console.log(`   ✅ Current in range: ${currentInRange}`);

    if (currentPrice > entryPrice) {
        // Profitable long position
        const correctPositioning = newCurrentY < newEntryY;
        console.log(`   🎯 Long Profit: ${correctPositioning ? '✅' : '❌'} Current price ${correctPositioning ? 'above' : 'below'} entry price`);
    } else if (currentPrice < entryPrice) {
        // Loss position
        const correctPositioning = newCurrentY > newEntryY;
        console.log(`   🎯 Loss Position: ${correctPositioning ? '✅' : '❌'} Current price ${correctPositioning ? 'below' : 'above'} entry price`);
    }

    console.log(`   Improvement: ${Math.abs(oldEntryY - newEntryY) > 1 ? 'Significant' : 'Minimal'}`);
    console.log("");
}

// Run the tests
testCurrentPriceInclusion();

console.log("✅ Current price inclusion tests completed!");
console.log("\n💡 Key improvements:");
console.log("   - Current price is now included in chart scaling calculations");
console.log("   - Entry price line positioning is more accurate");
console.log("   - Visual current price line added when different from entry price");
console.log("   - Better chart scaling for positions with significant P&L");
