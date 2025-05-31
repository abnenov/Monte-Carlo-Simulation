# Monte Carlo Simulation

import numpy as np
import matplotlib.pyplot as plt
import time

# =============================================================================
# ЧАСТ 1: ТЕОРЕТИЧНО ВЪВЕДЕНИЕ
# =============================================================================

print("=== MONTE CARLO SIMULATION ===")
print("Какво е симулация?")
print("- Симулацията е техника за моделиране на реални системи чрез компютърни програми")
print("- Използва се в науката за изследване на сложни системи, които са трудни за аналитично решаване")
print("- Полезна е защото позволява тестване на различни сценарии без реални експерименти")
print()

print("Как статистиката помага в симулацията?")
print("- Използваме случайни числа за моделиране на неопределени процеси")
print("- Прилагаме теория на вероятностите за анализ на резултатите")
print("- Чрез многократни итерации получаваме статистически валидни резултати")
print()

print("Какво е Monte Carlo симулация?")
print("- Метод за решаване на детерминирани проблеми чрез случайно семплиране")
print("- Наречен на хазартния квартал в Монако")
print("- Особено полезен за многомерни интеграли и сложни вероятностни проблеми")
print()

# =============================================================================
# ЧАСТ 2: ЧИСЛЕНO ИНТЕГРИРАНЕ С MONTE CARLO
# =============================================================================

def monte_carlo_integration(func, a, b, n_samples=100000):
    """
    Изчислява определен интеграл използвайки Monte Carlo метод
    
    Parameters:
    func: функция за интегриране
    a, b: граници на интегриране
    n_samples: брой случайни точки
    """
    # Генериране на случайни точки в интервала [a, b]
    x_random = np.random.uniform(a, b, n_samples)
    
    # Изчисляване на стойностите на функцията
    y_values = func(x_random)
    
    # Monte Carlo приближение на интеграла
    integral_approx = (b - a) * np.mean(y_values)
    
    return integral_approx

def trapezoidal_rule(func, a, b, n_intervals=10000):
    """
    Изчислява определен интеграл използвайки трапецовидното правило
    """
    x = np.linspace(a, b, n_intervals + 1)
    y = func(x)
    h = (b - a) / n_intervals
    integral_approx = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return integral_approx

# Тестови функции
def test_function_1(x):
    """f(x) = x^2, интеграл от 0 до 1 = 1/3"""
    return x**2

def test_function_2(x):
    """f(x) = sin(x), интеграл от 0 до π = 2"""
    return np.sin(x)

def test_function_3(x):
    """f(x) = e^(-x^2), Gaussian функция"""
    return np.exp(-x**2)

print("=== ТЕСТВАНЕ НА NUMERICAL INTEGRATION ===")

# Тест 1: f(x) = x^2 от 0 до 1
print("Тест 1: ∫₀¹ x² dx = 1/3 ≈ 0.3333")
exact_1 = 1/3

start_time = time.time()
mc_result_1 = monte_carlo_integration(test_function_1, 0, 1, 100000)
mc_time_1 = time.time() - start_time

start_time = time.time()
trap_result_1 = trapezoidal_rule(test_function_1, 0, 1, 10000)
trap_time_1 = time.time() - start_time

print(f"Monte Carlo: {mc_result_1:.6f} (грешка: {abs(mc_result_1 - exact_1):.6f}, време: {mc_time_1:.6f}s)")
print(f"Трапецовидно: {trap_result_1:.6f} (грешка: {abs(trap_result_1 - exact_1):.6f}, време: {trap_time_1:.6f}s)")
print()

# Тест 2: f(x) = sin(x) от 0 до π
print("Тест 2: ∫₀^π sin(x) dx = 2")
exact_2 = 2

mc_result_2 = monte_carlo_integration(test_function_2, 0, np.pi, 100000)
trap_result_2 = trapezoidal_rule(test_function_2, 0, np.pi, 10000)

print(f"Monte Carlo: {mc_result_2:.6f} (грешка: {abs(mc_result_2 - exact_2):.6f})")
print(f"Трапецовидно: {trap_result_2:.6f} (грешка: {abs(trap_result_2 - exact_2):.6f})")
print()

# =============================================================================
# ЧАСТ 3: ВИЗУАЛИЗАЦИЯ НА MONTE CARLO МЕТОДА
# =============================================================================

def visualize_monte_carlo_integration(func, a, b, n_samples=1000):
    """Визуализира Monte Carlo интегриране"""
    
    # Генериране на случайни точки
    x_random = np.random.uniform(a, b, n_samples)
    y_random = np.random.uniform(0, max(func(np.linspace(a, b, 1000))), n_samples)
    
    # Определяне кои точки са под кривата
    under_curve = y_random <= func(x_random)
    
    # Изчисляване на приближението
    area_rectangle = (b - a) * max(func(np.linspace(a, b, 1000)))
    ratio_under = np.sum(under_curve) / n_samples
    integral_approx = area_rectangle * ratio_under
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    
    # Функцията
    x_smooth = np.linspace(a, b, 1000)
    plt.plot(x_smooth, func(x_smooth), 'b-', linewidth=2, label='f(x) = x²')
    
    # Точки под кривата (зелени)
    plt.scatter(x_random[under_curve], y_random[under_curve], 
                c='green', alpha=0.6, s=1, label='Точки под кривата')
    
    # Точки над кривата (червени)
    plt.scatter(x_random[~under_curve], y_random[~under_curve], 
                c='red', alpha=0.6, s=1, label='Точки над кривата')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Monte Carlo интегриране: ∫₀¹ x² dx ≈ {integral_approx:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return integral_approx

print("=== ВИЗУАЛИЗАЦИЯ НА MONTE CARLO МЕТОДА ===")
mc_visual_result = visualize_monte_carlo_integration(test_function_1, 0, 1, 5000)

# =============================================================================
# ЧАСТ 4: ПРИЛОЖЕНИЕ В РЕАЛЕН ПРОБЛЕМ - ИЗЧИСЛЯВАНЕ НА π
# =============================================================================

def estimate_pi_monte_carlo(n_samples=100000):
    """
    Изчислява π използвайки Monte Carlo метод
    Базира се на съотношението на площите: кръг/квадрат = π/4
    """
    # Генериране на случайни точки в квадрат [-1, 1] x [-1, 1]
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    
    # Проверка кои точки са в единичния кръг
    inside_circle = (x**2 + y**2) <= 1
    
    # Приближение на π
    pi_estimate = 4 * np.sum(inside_circle) / n_samples
    
    return pi_estimate, x, y, inside_circle

print("=== ИЗЧИСЛЯВАНЕ НА π С MONTE CARLO ===")

# Тестване с различен брой точки
sample_sizes = [1000, 10000, 100000, 1000000]
pi_estimates = []

for n in sample_sizes:
    pi_est, _, _, _ = estimate_pi_monte_carlo(n)
    pi_estimates.append(pi_est)
    error = abs(pi_est - np.pi)
    print(f"n = {n:>7}: π ≈ {pi_est:.6f} (грешка: {error:.6f})")

print(f"Истинската стойност: π = {np.pi:.6f}")
print()

# Визуализация на изчислението на π
pi_est, x, y, inside = estimate_pi_monte_carlo(5000)

plt.figure(figsize=(8, 8))
plt.scatter(x[inside], y[inside], c='red', alpha=0.6, s=1, label=f'Вътре в кръга')
plt.scatter(x[~inside], y[~inside], c='blue', alpha=0.6, s=1, label='Извън кръга')

# Рисуване на кръга
theta = np.linspace(0, 2*np.pi, 1000)
plt.plot(np.cos(theta), np.sin(theta), 'black', linewidth=2)

# Рисуване на квадрата
plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'black', linewidth=2)

plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.gca().set_aspect('equal')
plt.title(f'Monte Carlo изчисляване на π ≈ {pi_est:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# ЧАСТ 5: АНАЛИЗ НА КОНВЕРГЕНЦИЯТА
# =============================================================================

def analyze_convergence():
    """Анализира как се подобрява точността с увеличаване на броя точки"""
    
    max_samples = 50000
    step = 1000
    sample_counts = range(step, max_samples + 1, step)
    
    pi_estimates = []
    errors = []
    
    # Генериране на много точки наведнъж за ефективност
    x_all = np.random.uniform(-1, 1, max_samples)
    y_all = np.random.uniform(-1, 1, max_samples)
    inside_all = (x_all**2 + y_all**2) <= 1
    
    cumulative_inside = np.cumsum(inside_all)
    
    for i, n in enumerate(sample_counts):
        pi_est = 4 * cumulative_inside[n-1] / n
        error = abs(pi_est - np.pi)
        
        pi_estimates.append(pi_est)
        errors.append(error)
    
    # Визуализация на конвергенцията
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(sample_counts, pi_estimates, 'b-', alpha=0.7)
    plt.axhline(y=np.pi, color='r', linestyle='--', label=f'π = {np.pi:.6f}')
    plt.xlabel('Брой точки')
    plt.ylabel('Приближение на π')
    plt.title('Конвергенция към π')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(sample_counts, errors, 'g-', alpha=0.7)
    plt.xlabel('Брой точки')
    plt.ylabel('Абсолютна грешка')
    plt.title('Намаляване на грешката')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print("=== АНАЛИЗ НА КОНВЕРГЕНЦИЯТА ===")
analyze_convergence()

# =============================================================================
# ЧАСТ 6: СРАВНЕНИЕ НА ПРОИЗВОДИТЕЛНОСТТА
# =============================================================================

def performance_comparison():
    """Сравнява производителността на Monte Carlo с аналитичните методи"""
    
    print("=== СРАВНЕНИЕ НА ПРОИЗВОДИТЕЛНОСТТА ===")
    print("Метод          | Точност | Време   | Предимства | Недостатъци")
    print("-" * 70)
    print("Monte Carlo    | Средна  | Бързо   | Универсален, прост | Стохастичен")
    print("Трапецовидно   | Висока  | Средно  | Детерминиран | Само 1D, сложност")
    print("Аналитично     | Точно   | Най-бързо | Точен резултат | Не винаги възможно")
    print()
    
    print("Monte Carlo е особено полезен когато:")
    print("- Имаме многомерни интеграли")
    print("- Областта на интегриране е сложна")
    print("- Аналитичното решение е трудно или невъзможно")
    print("- Търсим приближение, а не точен резултат")

performance_comparison()

# =============================================================================
# ЗАКЛЮЧЕНИЕ
# =============================================================================

print("\n=== ЗАКЛЮЧЕНИЕ ===")
print("Monte Carlo симулацията е мощна техника, която:")
print("1. Използва случайност за решаване на детерминирани проблеми")
print("2. Е особено полезна за сложни, многомерни задачи")
print("3. Предлага компромис между точност и простота на имплементация")
print("4. Намира широко приложение в различни области на науката")
print("5. Демонстрира важността на статистиката в компютърните науки")
