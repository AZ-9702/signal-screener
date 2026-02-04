#!/usr/bin/env python3
"""Quick script to update sector/industry for all cached results."""
import os
import json
import glob
import time
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "results")

def update_sectors():
    """Update sector/industry for all cached results."""
    files = glob.glob(os.path.join(CACHE_DIR, "*.json"))
    total = len(files)
    updated = 0
    skipped = 0
    errors = 0

    print(f"Found {total} cached results", flush=True)

    for i, filepath in enumerate(files):
        ticker = os.path.basename(filepath).replace(".json", "")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Skip if already has sector data
            if data.get("sector") and data.get("industry"):
                skipped += 1
                if (i + 1) % 50 == 0:
                    print(f"Progress: {i+1}/{total} (updated: {updated}, skipped: {skipped}, no data: {errors})", flush=True)
                continue

            # Fetch from yfinance with retry
            for attempt in range(3):
                try:
                    info = yf.Ticker(ticker).info or {}
                    break
                except Exception:
                    time.sleep(1)
                    info = {}

            sector = info.get("sector", "")
            industry = info.get("industry", "")

            if sector or industry:
                data["sector"] = sector
                data["industry"] = industry
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
                updated += 1
            else:
                errors += 1

            # Progress
            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{total} (updated: {updated}, skipped: {skipped}, no data: {errors})")

            # Rate limit - 0.3 second between requests
            time.sleep(0.3)

        except Exception as e:
            errors += 1

    print(f"\nDone! Updated: {updated}, Skipped: {skipped}, Errors: {errors}", flush=True)

if __name__ == "__main__":
    update_sectors()
