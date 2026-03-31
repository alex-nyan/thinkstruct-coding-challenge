"""
Concurrent User Load Test
=========================
Simulates N users (default 100) hitting the patent search API simultaneously
and measures how parallelism affects response times.

Requires the Flask app to be running on localhost:8080.

Usage:
    python load_test.py              # 100 concurrent users
    python load_test.py --users 50   # 50 concurrent users
"""

import time, statistics, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request
from urllib.parse import urlencode, quote
import json

BASE_URL = "http://localhost:8080"

QUERIES = [
    "non-pneumatic tire",
    "wheel assembly with metallic tread",
    "vehicle suspension system",
    "brake rotor cooling",
    "electric vehicle battery mounting",
    "tire noise reduction",
    "steering column assembly",
    "hub bearing unit",
    "rim with reinforced spoke",
    "autonomous vehicle sensor mount",
    "airless tire with deformable structure",
    "carbon fiber wheel construction",
    "anti-lock braking system",
    "regenerative braking mechanism",
    "adaptive cruise control",
    "vehicle frame structure",
    "exhaust heat recovery",
    "differential gear assembly",
    "pneumatic tire tread pattern",
    "wheel balancing apparatus",
]


def single_search(user_id: int) -> dict:
    """Simulate one user performing a search."""
    query = QUERIES[user_id % len(QUERIES)]
    params = urlencode({"q": query, "level": "patent", "method": "combined", "top_k": "10"})
    url = f"{BASE_URL}/api/search?{params}"

    t0 = time.perf_counter()
    try:
        req = Request(url)
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "user_id": user_id,
            "query": query,
            "status": "ok",
            "response_ms": round(elapsed_ms, 2),
            "results_count": len(data.get("results", [])),
            "server_ms": data.get("timing", {}).get("total_ms", 0),
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "user_id": user_id,
            "query": query,
            "status": "error",
            "response_ms": round(elapsed_ms, 2),
            "error": str(e),
        }


def run_load_test(num_users: int):
    print(f"\n{'='*60}")
    print(f"LOAD TEST — {num_users} CONCURRENT USERS")
    print(f"{'='*60}")

    # Warm-up: single request to ensure engine is ready
    print("\n[0] Warm-up request...")
    warmup = single_search(0)
    if warmup["status"] != "ok":
        print(f"  ERROR: Server not responding — {warmup.get('error', 'unknown')}")
        print(f"  Make sure the Flask app is running: python app.py")
        return
    print(f"  Warm-up: {warmup['response_ms']:.1f}ms")

    # Baseline: sequential single-user timing
    print("\n[1] Baseline — 10 sequential requests (single user)...")
    baseline_times = []
    for i in range(10):
        result = single_search(i)
        if result["status"] == "ok":
            baseline_times.append(result["response_ms"])
    baseline_avg = statistics.mean(baseline_times)
    baseline_p50 = statistics.median(baseline_times)
    print(f"  Avg: {baseline_avg:.1f}ms | Median: {baseline_p50:.1f}ms | "
          f"Min: {min(baseline_times):.1f}ms | Max: {max(baseline_times):.1f}ms")

    # Concurrent test
    print(f"\n[2] Concurrent — {num_users} simultaneous requests...")
    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=num_users) as pool:
        futures = {pool.submit(single_search, i): i for i in range(num_users)}
        for future in as_completed(futures):
            results.append(future.result())
    wall_clock = (time.perf_counter() - t0) * 1000

    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] != "ok"]
    response_times = [r["response_ms"] for r in ok]
    server_times = [r["server_ms"] for r in ok]

    print(f"\n{'='*60}")
    print(f"RESULTS — {num_users} CONCURRENT USERS")
    print(f"{'='*60}")
    print(f"\n  Successful requests:  {len(ok)}/{num_users}")
    print(f"  Failed requests:      {len(errors)}")
    print(f"  Wall-clock time:      {wall_clock:.0f}ms")
    print(f"  Throughput:           {len(ok) / (wall_clock/1000):.1f} req/sec")

    if response_times:
        print(f"\n  Response Time (client-side, includes network):")
        print(f"    Avg:    {statistics.mean(response_times):.1f}ms")
        print(f"    Median: {statistics.median(response_times):.1f}ms")
        print(f"    P95:    {sorted(response_times)[int(len(response_times)*0.95)]:.1f}ms")
        print(f"    Min:    {min(response_times):.1f}ms")
        print(f"    Max:    {max(response_times):.1f}ms")

    if server_times:
        print(f"\n  Server Processing Time (search engine only):")
        print(f"    Avg:    {statistics.mean(server_times):.1f}ms")
        print(f"    Median: {statistics.median(server_times):.1f}ms")
        print(f"    Min:    {min(server_times):.1f}ms")
        print(f"    Max:    {max(server_times):.1f}ms")

    # Compare baseline vs concurrent
    if response_times and baseline_times:
        concurrent_avg = statistics.mean(response_times)
        slowdown = concurrent_avg / baseline_avg
        print(f"\n  Performance Impact:")
        print(f"    Baseline (1 user):    {baseline_avg:.1f}ms avg")
        print(f"    Concurrent ({num_users} users): {concurrent_avg:.1f}ms avg")
        print(f"    Slowdown factor:      {slowdown:.2f}x")

    if errors:
        print(f"\n  Errors:")
        for e in errors[:5]:
            print(f"    User {e['user_id']}: {e.get('error', 'unknown')}")

    # Scaling test: 1, 10, 25, 50, 100 users
    test_levels = [n for n in [1, 10, 25, 50, 100] if n <= num_users]
    if len(test_levels) > 1:
        print(f"\n{'='*60}")
        print("SCALING ANALYSIS")
        print(f"{'='*60}")
        print(f"  {'Users':>6} | {'Avg (ms)':>10} | {'P95 (ms)':>10} | {'Throughput':>12} | {'Slowdown':>10}")
        print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")

        for n in test_levels:
            batch_results = []
            with ThreadPoolExecutor(max_workers=n) as pool:
                futures = {pool.submit(single_search, i): i for i in range(n)}
                for future in as_completed(futures):
                    batch_results.append(future.result())

            times = [r["response_ms"] for r in batch_results if r["status"] == "ok"]
            if times:
                avg = statistics.mean(times)
                p95 = sorted(times)[int(len(times) * 0.95)]
                throughput = len(times) / (max(times) / 1000)
                slow = avg / baseline_avg
                print(f"  {n:>6} | {avg:>10.1f} | {p95:>10.1f} | {throughput:>9.1f}/s  | {slow:>9.2f}x")

    print(f"\n{'='*60}")
    print("Load test complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patent Search Load Test")
    parser.add_argument("--users", type=int, default=100, help="Number of concurrent users (default: 100)")
    args = parser.parse_args()
    run_load_test(args.users)
