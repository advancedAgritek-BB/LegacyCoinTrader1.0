// Test script to manually trigger refresh functionality
// Copy and paste this into the browser console when viewing the dashboard

console.log('Testing manual refresh functionality...');

// Function to simulate refreshPrices
async function testRefreshPrices() {
  console.log('Simulating refreshPrices() call...');

  try {
    // Fetch positions
    const positionsResponse = await fetch('/api/open-positions');
    const positions = await positionsResponse.json();
    console.log('Positions fetched:', positions.length, 'positions');

    // Fetch P&L data
    const pnlResponse = await fetch('/api/portfolio/pnl');
    const pnlData = await pnlResponse.json();
    console.log('P&L data fetched:', pnlData);

    // Simulate updating position cards
    positions.forEach(position => {
      console.log(`Position ${position.symbol}: P&L ${position.pnl}% ($${position.pnl_value})`);
    });

    console.log('Refresh simulation completed successfully');
    return { positions, pnlData };

  } catch (error) {
    console.error('Refresh simulation failed:', error);
    return null;
  }
}

// Run the test
testRefreshPrices();
