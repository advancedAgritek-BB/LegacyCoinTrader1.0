#!/usr/bin/env python3
"""
Frontend Migration Status Checker

This script analyzes the frontend to identify which endpoints still need
migration to use the microservice architecture.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple


def analyze_frontend_migration():
    """Analyze frontend migration status."""

    frontend_file = Path("frontend/app.py")

    if not frontend_file.exists():
        print("❌ frontend/app.py not found")
        return

    with open(frontend_file, 'r') as f:
        content = f.read()

    # Find all crypto_bot imports
    crypto_bot_imports = re.findall(r'from crypto_bot\.(.*?) import', content)
    crypto_bot_lines = []
    for i, line in enumerate(content.split('\n'), 1):
        if 'from crypto_bot' in line:
            crypto_bot_lines.append((i, line.strip()))

    # Find all API routes
    route_pattern = r'@app\.route\(["\']([^"\']+)["\']'
    routes = re.findall(route_pattern, content)

    # Find routes that still use crypto_bot imports
    routes_needing_migration = []
    migrated_routes = []

    for route in routes:
        # Find the function that handles this route
        route_start = content.find(f'@app.route("{route}"')
        if route_start == -1:
            route_start = content.find(f"@app.route('{route}'")

        if route_start != -1:
            # Find the function definition after the route decorator
            func_match = re.search(r'def (\w+)\(', content[route_start:])
            if func_match:
                func_name = func_match.group(1)
                func_start = route_start + content[route_start:].find(f'def {func_name}')
                func_end = func_start + content[func_start:].find('\n\n@app.route') if '\n\n@app.route' in content[func_start:] else len(content)

                func_content = content[func_start:func_end]

                if any(f'from crypto_bot.{imp}' in func_content for imp in crypto_bot_imports):
                    routes_needing_migration.append(route)
                else:
                    migrated_routes.append(route)

    # Analyze gateway usage
    gateway_usage = len(re.findall(r'get_gateway_json\(|post_gateway_json\(', content))

    print("🔍 Frontend Migration Analysis")
    print("=" * 50)

    print(f"\n📊 Overall Status:")
    print(f"  • Total crypto_bot imports: {len(crypto_bot_imports)}")
    print(f"  • Gateway API calls: {gateway_usage}")
    print(f"  • Total API routes: {len(routes)}")
    print(f"  • Routes migrated: {len(migrated_routes)}")
    print(f"  • Routes needing migration: {len(routes_needing_migration)}")

    migration_percentage = (len(migrated_routes) / len(routes)) * 100 if routes else 0
    print(f"  • Migration progress: {migration_percentage:.1f}%")

    if migration_percentage >= 80:
        print("🎉 Excellent progress! Frontend is mostly migrated.")
    elif migration_percentage >= 50:
        print("👍 Good progress! Halfway there.")
    else:
        print("⚠️  More work needed on frontend migration.")

    print(f"\n🔧 Routes Needing Migration ({len(routes_needing_migration)}):")
    for route in routes_needing_migration[:10]:  # Show first 10
        print(f"  • {route}")
    if len(routes_needing_migration) > 10:
        print(f"  ... and {len(routes_needing_migration) - 10} more")

    print(f"\n✅ Already Migrated Routes ({len(migrated_routes)}):")
    for route in migrated_routes[:10]:  # Show first 10
        print(f"  • {route}")
    if len(migrated_routes) > 10:
        print(f"  ... and {len(migrated_routes) - 10} more")

    print(f"\n📦 Crypto Bot Imports Still Used ({len(crypto_bot_imports)}):")
    for imp in crypto_bot_imports[:10]:  # Show first 10
        print(f"  • crypto_bot.{imp}")
    if len(crypto_bot_imports) > 10:
        print(f"  ... and {len(crypto_bot_imports) - 10} more")

    print(f"\n💡 Migration Strategy:")
    print("1. Replace crypto_bot imports with gateway calls")
    print("2. Update route handlers to use microservice APIs")
    print("3. Add error handling for service unavailability")
    print("4. Test each migrated endpoint thoroughly")

    print(f"\n🚀 Next Steps:")
    if routes_needing_migration:
        print(f"Focus on migrating: {routes_needing_migration[0]}")
    print("Run: python3 frontend_migration_status.py (to check progress)")


if __name__ == "__main__":
    analyze_frontend_migration()
