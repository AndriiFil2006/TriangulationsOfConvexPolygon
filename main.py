from polygon import Polygon
from test_greedy import (exhaustive_test_greedy, create_regular_polygon,
                         comprehensive_test_suite)


'''
#Optimal Solution
testPol = Polygon([[0.20704211, -1.42460432], [-1.49795428, -2.43316306], [-1.54028276, 2.50191822], [0.26414350, 1.81750453], [0.65106250, 0]])
print(testPol.compare_triangulations())
'''


if __name__ == "__main__":
    # Run comprehensive test suite
    print("Starting comprehensive test suite...")
    print("This will run multiple test strategies to find the worst greedy failures.\n")

    all_results, worst_case = comprehensive_test_suite(n_vertices=5)

    # If we found a really bad case, analyze it in detail
    if worst_case and worst_case['percent_worse'] >= 15:
        print("\n\n" + "=" * 70)
        print("DETAILED ANALYSIS OF WORST CASE")
        print("=" * 70)
        poly = Polygon(worst_case['vertices'].tolist())
        poly.compare_triangulations()