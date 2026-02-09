import numpy as np
import itertools
import time
from polygon import Polygon


def create_regular_polygon(n, radius=1.0, center=(0, 0)):
    """Create a regular n-sided polygon."""
    angles = np.linspace(0, 2 * np.pi, n + 1)[:-1] + np.pi / 2
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    vertices = np.column_stack([x, y])
    return Polygon(vertices.tolist())


def test_greedy_optimality(polygon):
    """
    Test if greedy algorithm finds the optimal solution for a polygon.

    Returns:
    dict with test results
    """
    greedyDiagonals, greedyWeight = polygon.greedy_triangulation()
    allTriangulations = polygon.all_triangulation()

    optimalWeight = float('inf')
    for triangulation in allTriangulations:
        weight = sum(polygon.calcDistance(i, j) for i, j in triangulation)
        if weight < optimalWeight:
            optimalWeight = weight

    isOptimal = abs(greedyWeight - optimalWeight) < 1e-9
    difference = greedyWeight - optimalWeight
    percentWorse = (difference / optimalWeight * 100) if optimalWeight > 0 else 0

    return {
        'is_optimal': isOptimal,
        'greedy_weight': greedyWeight,
        'optimal_weight': optimalWeight,
        'difference': difference,
        'percent_worse': percentWorse,
        'vertices': polygon.verticies.copy()
    }


def generate_systematic_convex_polygons(n, num_radii_steps=10, radius_range=(0.5, 2.5),
                                        angle_perturbation_steps=0):
    """
    Systematically generate all convex polygons by varying parameters.

    Yields:
    Polygon objects
    """
    radius_values = np.linspace(radius_range[0], radius_range[1], num_radii_steps)
    base_angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]

    if angle_perturbation_steps == 0:
        total_combinations = num_radii_steps ** n
        print(f"Total combinations: {total_combinations:,}")
        print(f"(varying {n} radii, each with {num_radii_steps} values)")

        for radii_combo in itertools.product(radius_values, repeat=n):
            radii = np.array(radii_combo)
            angles = base_angles

            x = radii * np.cos(angles)
            y = radii * np.sin(angles)
            vertices = np.column_stack([x, y])

            yield Polygon(vertices.tolist(), validateConvex=False)

    else:
        angle_perturb_values = np.linspace(-0.2, 0.2, angle_perturbation_steps)
        total_combinations = (num_radii_steps ** n) * (angle_perturbation_steps ** n)
        print(f"Total combinations: {total_combinations:,}")
        print(f"(varying {n} radii × {n} angles)")

        for radii_combo in itertools.product(radius_values, repeat=n):
            for angle_perturbs in itertools.product(angle_perturb_values, repeat=n):
                radii = np.array(radii_combo)
                angles = base_angles + np.array(angle_perturbs)
                angles = np.sort(angles)

                x = radii * np.cos(angles)
                y = radii * np.sin(angles)
                vertices = np.column_stack([x, y])

                try:
                    poly = Polygon(vertices.tolist(), validateConvex=True)
                    yield poly
                except ValueError:
                    continue


def generate_extreme_polygons(n, num_samples=1000):
    """
    Generate "extreme" polygons with very different radii to maximize chances of failure.

    Yields:
    Polygon objects
    """
    print(f"Generating {num_samples} extreme polygons with large radius variations")

    base_angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]

    for _ in range(num_samples):
        # Create very uneven radius distributions
        radii = np.random.uniform(0.1, 3.0, n)

        # Try different patterns
        pattern = np.random.choice(['alternating', 'gradient', 'spike', 'random'])

        if pattern == 'alternating':
            # Alternate between small and large
            for i in range(n):
                radii[i] = 0.2 if i % 2 == 0 else 2.5
        elif pattern == 'gradient':
            # Gradual change
            radii = np.linspace(0.2, 2.5, n)
        elif pattern == 'spike':
            # One or two very different vertices
            radii = np.ones(n)
            radii[0] = 3.0
            radii[n // 2] = 0.2
        # else: keep random

        angles = base_angles
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        vertices = np.column_stack([x, y])

        try:
            yield Polygon(vertices.tolist(), validateConvex=True)
        except ValueError:
            continue


def generate_stretched_polygons(n, num_samples=1000):
    """
    Generate stretched/elongated polygons along one axis.

    Yields:
    Polygon objects
    """
    print(f"Generating {num_samples} stretched/elongated polygons")

    base_angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]

    for _ in range(num_samples):
        radii = np.random.uniform(0.5, 1.5, n)
        angles = base_angles

        x = radii * np.cos(angles)
        y = radii * np.sin(angles)

        # Stretch along an axis
        stretch_x = np.random.uniform(0.3, 3.0)
        stretch_y = np.random.uniform(0.3, 3.0)

        x *= stretch_x
        y *= stretch_y

        vertices = np.column_stack([x, y])

        try:
            yield Polygon(vertices.tolist(), validateConvex=True)
        except ValueError:
            continue


def generate_clustered_vertices(n, num_samples=1000):
    """
    Generate polygons where some vertices are clustered together.

    Yields:
    Polygon objects
    """
    print(f"Generating {num_samples} polygons with clustered vertices")

    for _ in range(num_samples):
        # Create angle clusters
        angles = []
        current_angle = 0

        for i in range(n):
            if np.random.random() < 0.3:  # 30% chance of cluster
                angles.append(current_angle + np.random.uniform(-0.1, 0.1))
            else:
                angles.append(current_angle)
            current_angle += 2 * np.pi / n

        angles = np.sort(angles) % (2 * np.pi)
        radii = np.random.uniform(0.5, 1.5, n)

        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        vertices = np.column_stack([x, y])

        try:
            yield Polygon(vertices.tolist(), validateConvex=True)
        except ValueError:
            continue


def exhaustive_test_greedy(n_vertices=5, num_radii_steps=10, save_failures=True,
                           max_polygons=None, angle_perturbation_steps=0):
    """
    Exhaustively test greedy algorithm on ALL possible polygons.

    Returns:
    dict with statistics and failure cases
    """
    print(f"{'=' * 70}")
    print(f"EXHAUSTIVE SEARCH FOR GREEDY ALGORITHM FAILURES")
    print(f"{'=' * 70}")
    print(f"Testing {n_vertices}-gons")
    print(f"Radius steps: {num_radii_steps} per vertex")
    if angle_perturbation_steps > 0:
        print(f"Angle perturbation steps: {angle_perturbation_steps}")
    print(f"{'=' * 70}\n")

    stats = {
        'total': 0,
        'optimal': 0,
        'suboptimal': 0,
        'failures': [],
        'differences': [],
        'percent_worse_values': []
    }

    start_time = time.time()
    last_print_time = start_time

    generator = generate_systematic_convex_polygons(
        n_vertices,
        num_radii_steps=num_radii_steps,
        angle_perturbation_steps=angle_perturbation_steps
    )

    for polygon in generator:
        if max_polygons and stats['total'] >= max_polygons:
            print(f"\nReached maximum of {max_polygons:,} polygons. Stopping.")
            break

        current_time = time.time()
        if current_time - last_print_time >= 2.0:
            elapsed = current_time - start_time
            rate = stats['total'] / elapsed if elapsed > 0 else 0
            print(f"Progress: {stats['total']:,} polygons tested | "
                  f"Rate: {rate:.1f} poly/sec | "
                  f"Failures: {stats['suboptimal']} | "
                  f"Elapsed: {elapsed:.1f}s")
            last_print_time = current_time

        try:
            result = test_greedy_optimality(polygon)

            stats['total'] += 1

            if result['is_optimal']:
                stats['optimal'] += 1
            else:
                stats['suboptimal'] += 1
                stats['differences'].append(result['difference'])
                stats['percent_worse_values'].append(result['percent_worse'])

                if save_failures:
                    stats['failures'].append({
                        'polygon_id': stats['total'],
                        'vertices': result['vertices'],
                        'greedy_weight': result['greedy_weight'],
                        'optimal_weight': result['optimal_weight'],
                        'difference': result['difference'],
                        'percent_worse': result['percent_worse']
                    })

                    print(f"\n{'!' * 70}")
                    print(f"FOUND FAILURE #{stats['suboptimal']} at polygon #{stats['total']}")
                    print(f"{'!' * 70}")
                    print(f"Greedy weight:  {result['greedy_weight']:.8f}")
                    print(f"Optimal weight: {result['optimal_weight']:.8f}")
                    print(f"Difference:     {result['difference']:.8f} ({result['percent_worse']:.6f}% worse)")
                    print(f"Vertices:\n{result['vertices']}")
                    print(f"{'!' * 70}\n")

        except Exception as e:
            print(f"\nError on polygon {stats['total']}: {e}")
            continue

    elapsed_time = time.time() - start_time

    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Total polygons tested: {stats['total']:,}")
    print(f"Greedy found optimal:  {stats['optimal']:,} ({stats['optimal'] / stats['total'] * 100:.2f}%)")
    print(f"Greedy suboptimal:     {stats['suboptimal']:,} ({stats['suboptimal'] / stats['total'] * 100:.2f}%)")
    print(f"\nTime elapsed: {elapsed_time:.2f}s")
    print(f"Average rate: {stats['total'] / elapsed_time:.1f} polygons/second")

    if stats['suboptimal'] > 0:
        print(f"\n{'=' * 70}")
        print("FAILURE STATISTICS")
        print(f"{'=' * 70}")
        print(f"Average difference: {np.mean(stats['differences']):.8f}")
        print(f"Max difference:     {np.max(stats['differences']):.8f}")
        print(f"Min difference:     {np.min(stats['differences']):.8f}")
        print(f"Average % worse:    {np.mean(stats['percent_worse_values']):.6f}%")
        print(f"Max % worse:        {np.max(stats['percent_worse_values']):.6f}%")

        print(f"\n{'=' * 70}")
        print("TOP 10 WORST CASES")
        print(f"{'=' * 70}")
        sorted_failures = sorted(stats['failures'], key=lambda x: x['difference'], reverse=True)
        for i, failure in enumerate(sorted_failures[:10], 1):
            print(f"\n{i}. Polygon #{failure['polygon_id']}")
            print(f"   Greedy weight:  {failure['greedy_weight']:.8f}")
            print(f"   Optimal weight: {failure['optimal_weight']:.8f}")
            print(f"   Difference:     {failure['difference']:.8f} ({failure['percent_worse']:.6f}% worse)")
            print(f"   Vertices:")
            for vi, v in enumerate(failure['vertices']):
                print(f"      v{vi}: ({v[0]:.6f}, {v[1]:.6f})")
    else:
        print(f"\n{'=' * 70}")
        print("🎉 NO FAILURES FOUND! 🎉")
        print("Greedy algorithm found optimal solution for ALL tested polygons!")
        print(f"{'=' * 70}")

    return stats


def run_generator_test(generator, test_name):
    """
    Test greedy algorithm on polygons from a generator.

    Returns:
    dict with statistics
    """
    print(f"\n{'=' * 70}")
    print(f"TEST: {test_name}")
    print(f"{'=' * 70}\n")

    stats = {
        'total': 0,
        'optimal': 0,
        'suboptimal': 0,
        'failures': [],
        'differences': [],
        'percent_worse_values': []
    }

    start_time = time.time()
    worst_case = None
    worst_percent = 0

    for polygon in generator:
        try:
            result = test_greedy_optimality(polygon)

            stats['total'] += 1

            if result['is_optimal']:
                stats['optimal'] += 1
            else:
                stats['suboptimal'] += 1
                stats['differences'].append(result['difference'])
                stats['percent_worse_values'].append(result['percent_worse'])

                stats['failures'].append({
                    'polygon_id': stats['total'],
                    'vertices': result['vertices'],
                    'greedy_weight': result['greedy_weight'],
                    'optimal_weight': result['optimal_weight'],
                    'difference': result['difference'],
                    'percent_worse': result['percent_worse']
                })

                if result['percent_worse'] > worst_percent:
                    worst_percent = result['percent_worse']
                    worst_case = result

                    print(f"🔥 NEW WORST CASE at polygon #{stats['total']}: {worst_percent:.4f}% worse")

        except Exception as e:
            continue

        if stats['total'] % 500 == 0:
            print(f"Progress: {stats['total']:,} tested | Failures: {stats['suboptimal']} | "
                  f"Worst so far: {worst_percent:.4f}%")

    elapsed_time = time.time() - start_time

    print(f"\n{'=' * 70}")
    print(f"RESULTS FOR: {test_name}")
    print(f"{'=' * 70}")
    print(f"Total tested: {stats['total']:,}")
    print(f"Failures: {stats['suboptimal']:,} ({stats['suboptimal'] / stats['total'] * 100:.2f}%)")

    if stats['suboptimal'] > 0:
        print(f"Max % worse: {np.max(stats['percent_worse_values']):.6f}%")
        print(f"Avg % worse: {np.mean(stats['percent_worse_values']):.6f}%")

        if worst_case:
            print(f"\n🏆 WORST CASE FROM THIS TEST:")
            print(f"   Greedy:  {worst_case['greedy_weight']:.8f}")
            print(f"   Optimal: {worst_case['optimal_weight']:.8f}")
            print(f"   Worse by: {worst_case['percent_worse']:.6f}%")
            print(f"   Vertices:\n{worst_case['vertices']}")

    print(f"Time: {elapsed_time:.2f}s")

    return stats


def comprehensive_test_suite(n_vertices=5):
    """
    Run a comprehensive suite of tests to find the worst greedy failures.

    Returns:
    dict with all results
    """
    print(f"\n{'#' * 70}")
    print(f"{'#' * 70}")
    print(f"  COMPREHENSIVE GREEDY ALGORITHM TEST SUITE")
    print(f"  Target: Find cases where greedy is 20%+ worse than optimal")
    print(f"{'#' * 70}")
    print(f"{'#' * 70}\n")

    all_stats = []
    global_worst = None
    global_worst_percent = 0

    # Test 1: Standard systematic with moderate granularity
    print("\n" + "=" * 70)
    print("TEST 1: Systematic scan (moderate granularity)")
    stats1 = exhaustive_test_greedy(n_vertices=n_vertices, num_radii_steps=15)
    all_stats.append(('Systematic (15 steps)', stats1))

    # Test 2: Wide radius range
    print("\n" + "=" * 70)
    print("TEST 2: Systematic scan (wide radius range)")
    stats2 = exhaustive_test_greedy(n_vertices=n_vertices, num_radii_steps=12,
                                    max_polygons=50000)
    # Modify to use wider range - need to update generator call
    all_stats.append(('Wide range', stats2))

    # Test 3: Extreme polygons
    gen3 = generate_extreme_polygons(n_vertices, num_samples=10000)
    stats3 = run_generator_test(gen3, "Extreme radius variations")
    all_stats.append(('Extreme', stats3))

    # Test 4: Stretched polygons
    gen4 = generate_stretched_polygons(n_vertices, num_samples=10000)
    stats4 = run_generator_test(gen4, "Stretched/elongated polygons")
    all_stats.append(('Stretched', stats4))

    # Test 5: Clustered vertices
    gen5 = generate_clustered_vertices(n_vertices, num_samples=10000)
    stats5 = run_generator_test(gen5, "Clustered vertices")
    all_stats.append(('Clustered', stats5))

    # Test 6: High granularity systematic (if time permits)
    print("\n" + "=" * 70)
    print("TEST 6: High granularity systematic scan")
    stats6 = exhaustive_test_greedy(n_vertices=n_vertices, num_radii_steps=20,
                                    max_polygons=100000)
    all_stats.append(('High granularity', stats6))

    # Test 7: Hexagons (more triangulations = more room for error)
    if n_vertices <= 5:  # Only if we started with pentagons
        print("\n" + "=" * 70)
        print("TEST 7: Hexagons (14 possible triangulations)")
        stats7 = exhaustive_test_greedy(n_vertices=6, num_radii_steps=8,
                                        max_polygons=50000)
        all_stats.append(('Hexagons', stats7))

    # Find global worst
    print(f"\n\n{'#' * 70}")
    print(f"{'#' * 70}")
    print("  GLOBAL SUMMARY - ALL TESTS")
    print(f"{'#' * 70}")
    print(f"{'#' * 70}\n")

    for test_name, stats in all_stats:
        if stats['failures']:
            worst_in_test = max(stats['failures'], key=lambda x: x['percent_worse'])
            print(f"\n{test_name}:")
            print(f"  Total tested: {stats['total']:,}")
            print(f"  Failures: {stats['suboptimal']:,}")
            print(f"  Worst case: {worst_in_test['percent_worse']:.6f}% worse")

            if worst_in_test['percent_worse'] > global_worst_percent:
                global_worst_percent = worst_in_test['percent_worse']
                global_worst = worst_in_test

    if global_worst:
        print(f"\n{'=' * 70}")
        print("GLOBAL WORST CASE FOUND")
        print(f"{'=' * 70}")
        print(f"Greedy weight:  {global_worst['greedy_weight']:.8f}")
        print(f"Optimal weight: {global_worst['optimal_weight']:.8f}")
        print(f"WORSE BY:       {global_worst['percent_worse']:.6f}%")
        print(f"\nVertices:")
        for vi, v in enumerate(global_worst['vertices']):
            print(f"  v{vi}: ({v[0]:.8f}, {v[1]:.8f})")

        if global_worst['percent_worse'] >= 20:
            print(f"\nTARGET ACHIEVED! Found case ≥20% worse!")
        else:
            print(f"\n📊 Best we found: {global_worst['percent_worse']:.4f}%")
            print(f"   Still need: {20 - global_worst['percent_worse']:.4f}% more to reach 20%")

    return all_stats, global_worst