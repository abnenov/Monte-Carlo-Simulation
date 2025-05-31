# Monte Carlo Simulation
import numpy as np
import matplotlib.pyplot as plt
import time

# =============================================================================
# PART 1: THEORETICAL INTRODUCTION
# =============================================================================

print("=== MONTE CARLO SIMULATION ===")
print("What is a simulation?")
print("- Simulation is a technique for modeling real systems using computer programs")
print("- Used in science to study complex systems that are difficult to solve analytically")
print("- Useful because it allows testing different scenarios without real experiments")
print()

print("How are statistics useful in simulation?")
print("- We use random numbers to model uncertain processes")
print("- We apply probability theory to analyze results")
print("- Through multiple iterations we obtain statistically valid results")
print()

print("What is a Monte Carlo simulation?")
print("- Method for solving deterministic problems through random sampling")
print("- Named after the gambling district in Monaco")
print("- Especially useful for multidimensional integrals and complex probability problems")
print()

# =============================================================================
# PART 2: NUMERICAL INTEGRATION WITH MONTE CARLO
# =============================================================================

def monte_carlo_integration(func, a, b, n_samples=100000):
    """
    Computes definite integral using Monte Carlo method
    
    Parameters:
    func: function to integrate
    a, b: integration bounds
    n_samples: number of random points
    """
    # Generate random points in interval [a, b]
    x_random = np.random.uniform(a, b, n_samples)
    
    # Calculate function values
    y_values = func(x_random)
    
    # Monte Carlo approximation of integral
    integral_approx = (b - a) * np.mean(y_values)
    
    return integral_approx

def trapezoidal_rule(func, a, b, n_intervals=10000):
    """
    Computes definite integral using trapezoidal rule
    """
    x = np.linspace(a, b, n_intervals + 1)
    y = func(x)
    h = (b - a) / n_intervals
    integral_approx = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return integral_approx

# Test functions
def test_function_1(x):
    """f(x) = x^2, integral from 0 to 1 = 1/3"""
    return x**2

def test_function_2(x):
    """f(x) = sin(x), integral from 0 to π = 2"""
    return np.sin(x)

def test_function_3(x):
    """f(x) = e^(-x^2), Gaussian function"""
    return np.exp(-x**2)

print("=== TESTING NUMERICAL INTEGRATION ===")

# Test 1: f(x) = x^2 from 0 to 1
print("Test 1: ∫₀¹ x² dx = 1/3 ≈ 0.3333")
exact_1 = 1/3

start_time = time.time()
mc_result_1 = monte_carlo_integration(test_function_1, 0, 1, 100000)
mc_time_1 = time.time() - start_time

start_time = time.time()
trap_result_1 = trapezoidal_rule(test_function_1, 0, 1, 10000)
trap_time_1 = time.time() - start_time

print(f"Monte Carlo: {mc_result_1:.6f} (error: {abs(mc_result_1 - exact_1):.6f}, time: {mc_time_1:.6f}s)")
print(f"Trapezoidal: {trap_result_1:.6f} (error: {abs(trap_result_1 - exact_1):.6f}, time: {trap_time_1:.6f}s)")
print()

# Test 2: f(x) = sin(x) from 0 to π
print("Test 2: ∫₀^π sin(x) dx = 2")
exact_2 = 2

mc_result_2 = monte_carlo_integration(test_function_2, 0, np.pi, 100000)
trap_result_2 = trapezoidal_rule(test_function_2, 0, np.pi, 10000)

print(f"Monte Carlo: {mc_result_2:.6f} (error: {abs(mc_result_2 - exact_2):.6f})")
print(f"Trapezoidal: {trap_result_2:.6f} (error: {abs(trap_result_2 - exact_2):.6f})")
print()

# =============================================================================
# PART 3: VISUALIZATION OF MONTE CARLO METHOD
# =============================================================================

def visualize_monte_carlo_integration(func, a, b, n_samples=1000):
    """Visualizes Monte Carlo integration"""
    
    # Generate random points
    x_random = np.random.uniform(a, b, n_samples)
    y_random = np.random.uniform(0, max(func(np.linspace(a, b, 1000))), n_samples)
    
    # Determine which points are under the curve
    under_curve = y_random <= func(x_random)
    
    # Calculate approximation
    area_rectangle = (b - a) * max(func(np.linspace(a, b, 1000)))
    ratio_under = np.sum(under_curve) / n_samples
    integral_approx = area_rectangle * ratio_under
    
    # Visualization
    plt.figure(figsize=(10, 6))
    
    # The function
    x_smooth = np.linspace(a, b, 1000)
    plt.plot(x_smooth, func(x_smooth), 'b-', linewidth=2, label='f(x) = x²')
    
    # Points under the curve (green)
    plt.scatter(x_random[under_curve], y_random[under_curve], 
                c='green', alpha=0.6, s=1, label='Points under curve')
    
    # Points above the curve (red)
    plt.scatter(x_random[~under_curve], y_random[~under_curve], 
                c='red', alpha=0.6, s=1, label='Points above curve')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Monte Carlo Integration: ∫₀¹ x² dx ≈ {integral_approx:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return integral_approx

print("=== VISUALIZATION OF MONTE CARLO METHOD ===")
mc_visual_result = visualize_monte_carlo_integration(test_function_1, 0, 1, 5000)

# =============================================================================
# PART 4: REAL-LIFE APPLICATION - ESTIMATING π
# =============================================================================

def estimate_pi_monte_carlo(n_samples=100000):
    """
    Estimates π using Monte Carlo method
    Based on area ratio: circle/square = π/4
    """
    # Generate random points in square [-1, 1] x [-1, 1]
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    
    # Check which points are inside unit circle
    inside_circle = (x**2 + y**2) <= 1
    
    # Approximation of π
    pi_estimate = 4 * np.sum(inside_circle) / n_samples
    
    return pi_estimate, x, y, inside_circle

print("=== ESTIMATING π WITH MONTE CARLO ===")

# Testing with different number of points
sample_sizes = [1000, 10000, 100000, 1000000]
pi_estimates = []

for n in sample_sizes:
    pi_est, _, _, _ = estimate_pi_monte_carlo(n)
    pi_estimates.append(pi_est)
    error = abs(pi_est - np.pi)
    print(f"n = {n:>7}: π ≈ {pi_est:.6f} (error: {error:.6f})")

print(f"True value: π = {np.pi:.6f}")
print()

# Visualization of π calculation
pi_est, x, y, inside = estimate_pi_monte_carlo(5000)

plt.figure(figsize=(8, 8))
plt.scatter(x[inside], y[inside], c='red', alpha=0.6, s=1, label=f'Inside circle')
plt.scatter(x[~inside], y[~inside], c='blue', alpha=0.6, s=1, label='Outside circle')

# Draw the circle
theta = np.linspace(0, 2*np.pi, 1000)
plt.plot(np.cos(theta), np.sin(theta), 'black', linewidth=2)

# Draw the square
plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'black', linewidth=2)

plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.gca().set_aspect('equal')
plt.title(f'Monte Carlo estimation of π ≈ {pi_est:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# PART 5: CONVERGENCE ANALYSIS
# =============================================================================

def analyze_convergence():
    """Analyzes how accuracy improves with increasing number of points"""
    
    max_samples = 50000
    step = 1000
    sample_counts = range(step, max_samples + 1, step)
    
    pi_estimates = []
    errors = []
    
    # Generate many points at once for efficiency
    x_all = np.random.uniform(-1, 1, max_samples)
    y_all = np.random.uniform(-1, 1, max_samples)
    inside_all = (x_all**2 + y_all**2) <= 1
    
    cumulative_inside = np.cumsum(inside_all)
    
    for i, n in enumerate(sample_counts):
        pi_est = 4 * cumulative_inside[n-1] / n
        error = abs(pi_est - np.pi)
        
        pi_estimates.append(pi_est)
        errors.append(error)
    
    # Visualization of convergence
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(sample_counts, pi_estimates, 'b-', alpha=0.7)
    plt.axhline(y=np.pi, color='r', linestyle='--', label=f'π = {np.pi:.6f}')
    plt.xlabel('Number of points')
    plt.ylabel('π approximation')
    plt.title('Convergence to π')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(sample_counts, errors, 'g-', alpha=0.7)
    plt.xlabel('Number of points')
    plt.ylabel('Absolute error')
    plt.title('Error reduction')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print("=== CONVERGENCE ANALYSIS ===")
analyze_convergence()

# =============================================================================
# PART 6: PERFORMANCE COMPARISON
# =============================================================================

def performance_comparison():
    """Compares Monte Carlo performance with analytical methods"""
    
    print("=== PERFORMANCE COMPARISON ===")
    print("Method         | Accuracy | Time    | Advantages        | Disadvantages")
    print("-" * 75)
    print("Monte Carlo    | Medium   | Fast    | Universal, simple | Stochastic")
    print("Trapezoidal    | High     | Medium  | Deterministic     | 1D only, complex")
    print("Analytical     | Exact    | Fastest | Exact result      | Not always possible")
    print()
    
    print("Monte Carlo is especially useful when:")
    print("- We have multidimensional integrals")
    print("- Integration domain is complex")
    print("- Analytical solution is difficult or impossible")
    print("- We seek approximation rather than exact result")

performance_comparison()

# =============================================================================
# PART 7: ADDITIONAL MONTE CARLO APPLICATION - OPTION PRICING
# =============================================================================

def black_scholes_monte_carlo(S0, K, T, r, sigma, n_simulations=100000):
    """
    Monte Carlo simulation for European call option pricing
    
    Parameters:
    S0: Initial stock price
    K: Strike price
    T: Time to maturity
    r: Risk-free rate
    sigma: Volatility
    n_simulations: Number of Monte Carlo simulations
    """
    
    # Generate random price paths
    dt = T
    Z = np.random.standard_normal(n_simulations)
    
    # Stock price at maturity using geometric Brownian motion
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Payoff for call option
    payoffs = np.maximum(ST - K, 0)
    
    # Present value of expected payoff
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    return option_price, ST, payoffs

def analytical_black_scholes(S0, K, T, r, sigma):
    """Analytical Black-Scholes formula for comparison"""
    from scipy.stats import norm
    
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    call_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call_price

print("\n=== ADDITIONAL APPLICATION: OPTION PRICING ===")

# Option parameters
S0 = 100    # Current stock price
K = 105     # Strike price
T = 1       # Time to maturity (1 year)
r = 0.05    # Risk-free rate
sigma = 0.2 # Volatility

# Monte Carlo simulation
mc_price, stock_prices, payoffs = black_scholes_monte_carlo(S0, K, T, r, sigma)

# Analytical solution
try:
    analytical_price = analytical_black_scholes(S0, K, T, r, sigma)
    print(f"Monte Carlo option price: ${mc_price:.4f}")
    print(f"Analytical option price:  ${analytical_price:.4f}")
    print(f"Error: ${abs(mc_price - analytical_price):.4f}")
except ImportError:
    print(f"Monte Carlo option price: ${mc_price:.4f}")
    print("(scipy not available for analytical comparison)")

# Visualize stock price distribution at maturity
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(stock_prices, bins=50, alpha=0.7, density=True)
plt.axvline(S0, color='red', linestyle='--', label=f'Initial price: ${S0}')
plt.axvline(K, color='green', linestyle='--', label=f'Strike price: ${K}')
plt.xlabel('Stock price at maturity')
plt.ylabel('Density')
plt.title('Distribution of Stock Prices at Maturity')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(payoffs, bins=50, alpha=0.7, density=True)
plt.xlabel('Option payoff')
plt.ylabel('Density')
plt.title('Distribution of Option Payoffs')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# PART 8: MONTE CARLO FOR COMPLEX GEOMETRY - AREA ESTIMATION
# =============================================================================

def estimate_complex_area():
    """Estimates area of complex shape using Monte Carlo"""
    
    def is_inside_shape(x, y):
        """Define a complex shape: intersection of circle and sine wave"""
        circle_condition = x**2 + y**2 <= 1
        sine_condition = y <= 0.3 * np.sin(5*x) + 0.5
        return circle_condition & sine_condition
    
    n_samples = 100000
    
    # Generate random points in bounding box
    x_random = np.random.uniform(-1, 1, n_samples)
    y_random = np.random.uniform(-1, 1, n_samples)
    
    # Check which points are inside shape
    inside = is_inside_shape(x_random, y_random)
    
    # Estimate area
    bounding_box_area = 4  # 2x2 square
    estimated_area = bounding_box_area * np.sum(inside) / n_samples
    
    # Visualization
    plt.figure(figsize=(10, 8))
    
    # Sample of points for visualization
    n_vis = 5000
    x_vis = x_random[:n_vis]
    y_vis = y_random[:n_vis]
    inside_vis = inside[:n_vis]
    
    plt.scatter(x_vis[inside_vis], y_vis[inside_vis], c='red', alpha=0.6, s=1, label='Inside shape')
    plt.scatter(x_vis[~inside_vis], y_vis[~inside_vis], c='blue', alpha=0.6, s=1, label='Outside shape')
    
    # Draw shape boundary
    x_boundary = np.linspace(-1, 1, 1000)
    y_sine = 0.3 * np.sin(5*x_boundary) + 0.5
    plt.plot(x_boundary, y_sine, 'black', linewidth=2, label='Sine boundary')
    
    theta = np.linspace(0, 2*np.pi, 1000)
    plt.plot(np.cos(theta), np.sin(theta), 'black', linewidth=2, label='Circle boundary')
    
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal')
    plt.title(f'Complex Shape Area Estimation ≈ {estimated_area:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return estimated_area

print("\n=== COMPLEX GEOMETRY APPLICATION ===")
complex_area = estimate_complex_area()

# =============================================================================
# CONCLUSION
# =============================================================================

print("\n=== CONCLUSION ===")
print("Monte Carlo simulation is a powerful technique that:")
print("1. Uses randomness to solve deterministic problems")
print("2. Is especially useful for complex, multidimensional tasks")
print("3. Offers a compromise between accuracy and implementation simplicity")
print("4. Finds wide application in various fields of science")
print("5. Demonstrates the importance of statistics in computer science")
print()
print("Key applications demonstrated:")
print("- Numerical integration")
print("- Mathematical constant estimation (π)")
print("- Financial modeling (option pricing)")
print("- Geometric area calculation")
print()
print("Advantages:")
print("- Easy to implement and understand")
print("- Scales well to high dimensions")
print("- Parallelizable")
print("- Handles complex geometries")
print()
print("Disadvantages:")
print("- Convergence can be slow (O(1/√n))")
print("- Results have inherent randomness")
print("- May require many samples for high precision")
