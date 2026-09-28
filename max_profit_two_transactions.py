"""
Best Time to Buy and Sell Stock III
Maximum profit with at most 2 transactions
"""

class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        """
        Find the maximum profit with at most 2 transactions.
        
        A transaction consists of buying and then selling.
        You cannot engage in multiple transactions at the same time 
        (i.e., you must sell the stock before you buy again).
        
        Args:
            prices: List of integers representing stock prices on each day
            
        Returns:
            Maximum profit that can be achieved with at most 2 transactions
            
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Example:
            Input: prices = [3,3,5,0,0,3,1,4]
            Output: 6
            Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), 
                        profit = 3-0 = 3.
                        Then buy on day 7 (price = 1) and sell on day 8 (price = 4), 
                        profit = 4-1 = 3.
                        Total profit = 3 + 3 = 6.
        """
        if not prices or len(prices) < 2:
            return 0
        
        n = len(prices)
        
        # left[i] = max profit with at most 1 transaction from prices[0...i]
        left = [0] * n
        min_price = prices[0]
        
        for i in range(1, n):
            min_price = min(min_price, prices[i])
            left[i] = max(left[i-1], prices[i] - min_price)
        
        # right[i] = max profit with at most 1 transaction from prices[i...n-1]
        right = [0] * n
        max_price = prices[n-1]
        
        for i in range(n-2, -1, -1):
            max_price = max(max_price, prices[i])
            right[i] = max(right[i+1], max_price - prices[i])
        
        # Find max profit with at most 2 transactions
        max_profit = 0
        for i in range(n):
            max_profit = max(max_profit, left[i] + right[i])
        
        return max_profit


class SolutionSpaceOptimized:
    def maxProfit(self, prices: list[int]) -> int:
        """
        Space-optimized solution using state machine approach.
        
        States:
        - buy1: Max profit after first buy
        - sell1: Max profit after first sell
        - buy2: Max profit after second buy
        - sell2: Max profit after second sell
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            prices: List of integers representing stock prices on each day
            
        Returns:
            Maximum profit that can be achieved with at most 2 transactions
        """
        if not prices or len(prices) < 2:
            return 0
        
        buy1 = float('-inf')   # Max profit after first buy
        sell1 = 0              # Max profit after first sell
        buy2 = float('-inf')   # Max profit after second buy
        sell2 = 0              # Max profit after second sell
        
        for price in prices:
            buy1 = max(buy1, -price)           # Buy or skip
            sell1 = max(sell1, buy1 + price)   # Sell or skip
            buy2 = max(buy2, sell1 - price)    # Buy second or skip
            sell2 = max(sell2, buy2 + price)   # Sell second or skip
        
        return sell2


# Test cases
if __name__ == "__main__":
    test_cases = [
        ([3, 3, 5, 0, 0, 3, 1, 4], 6),
        ([1, 2, 3, 4, 5], 4),
        ([7, 6, 4, 3, 1], 0),
        ([1], 0),
        ([2, 1], 0),
        ([3, 2, 6, 5, 0, 3], 7),
        ([2, 4, 1], 2),
    ]
    
    solution = Solution()
    solution_optimized = SolutionSpaceOptimized()
    
    print("Testing Solution (Dynamic Programming):")
    for prices, expected in test_cases:
        result = solution.maxProfit(prices)
        status = "✓" if result == expected else "✗"
        print(f"{status} prices={prices}, expected={expected}, got={result}")
    
    print("\nTesting SolutionSpaceOptimized (State Machine):")
    for prices, expected in test_cases:
        result = solution_optimized.maxProfit(prices)
        status = "✓" if result == expected else "✗"
        print(f"{status} prices={prices}, expected={expected}, got={result}")
