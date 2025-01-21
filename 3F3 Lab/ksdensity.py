import numpy as np
import matplotlib.pyplot as plt
import scipy

def ksdensity(data, width=0.3):
    """Returns kernel smoothing function from data points in data"""
    def ksd(x_axis):
        def n_pdf(x, mu=0., sigma=1.):  # normal pdf
            u = (x - mu) / abs(sigma)
            y = (1 / (np.sqrt(2 * np.pi) * abs(sigma)))
            y *= np.exp(-u * u / 2)
            return y
        prob = [n_pdf(x_i, data, width) for x_i in x_axis]
        pdf = [np.average(pr) for pr in prob]  # each row is one x value
        return np.array(pdf)
    return ksd

def inverse_cdf(n, exp_lambda):
    x = np.random.rand(n)
    samples = - np.log(1 - x) / exp_lambda
    return samples

# # Plot normal distribution
# fig, ax = plt.subplots(2)
# x = np.random.randn(1000)
# ax[0].hist(x, bins=30) # number of bins

# ks_density = ksdensity(x, width=0.4)
# # np.linspace(start, stop, number of steps)
# x_values = np.linspace(-5., 5., 100)
# ax[1].plot(x_values, ks_density(x_values))

# # Plot uniform distribution
# fig2, ax2 = plt.subplots(2)
# x = np.random.rand(10000)
# count, edges = np.histogram(x, bins=20)
# print(count)
# ax2[0].hist(x, bins=20)

# ks_density = ksdensity(x, width=0.2)
# x_values = np.linspace(-1., 2., 100)
# ax2[1].plot(x_values, ks_density(x_values))
# plt.show()

def q_1_1():
    x = np.random.randn(1000) # 1000 Gaussian
    count, bins, _ = plt.hist(x, bins=30, label='Histogram Count Data')

    bin_width = bins[1] - bins[0]

    mean = 0
    std_dev = 1
    x_values = np.linspace(-4, 4, 1000)
    pdf = scipy.stats.norm.pdf(x_values, mean, std_dev) * 1000 * bin_width

    plt.plot(x_values, pdf, 'r-', lw=2, label='Exact Normal PDF')

    plt.legend()
    plt.show()

def q_1_2():
    x = np.random.rand(1000) # 1000 Uniform
    count, bins, _ = plt.hist(x, bins=30, label = 'Histogram Count Data')

    bin_width = bins[1] - bins[0]
    
    a, b = 0, 1
    x_values = np.linspace(a, b, 1000)
    pdf = [(1 / (b - a)) * 1000 * bin_width] * 1000
    
    plt.plot(x_values, pdf, 'r-', lw=2, label='Exact Uniform PDF')
    plt.legend()
    plt.show()

def q_1_3():
    x = np.random.randn(1000) # 1000 Gaussian
    
    ks_density = ksdensity(x, width=0.4)
    x_values = np.linspace(-4., 4., 100)
    plt.plot(x_values, ks_density(x_values), label='Kernel Density Estimate')

    mean = 0
    std_dev = 1
    pdf = scipy.stats.norm.pdf(x_values, mean, std_dev)

    plt.plot(x_values, pdf, 'r-', lw=2, label='Exact Normal PDF')

    plt.legend()
    plt.show()

def q_1_4():
    x = np.random.rand(1000) # 1000 Uniform
    
    a, b = -2., 2.
    x_values = np.linspace(a, b, 1000)

    ks_density = ksdensity(x, width=0.135) # Note kernel width for Normal
    plt.plot(x_values, ks_density(x_values), label='Kernel Density Estimate')

    pdf = [(1 / (b - a))] * 1000
    
    plt.plot(x_values, pdf, 'r-', lw=2, label='Exact Uniform PDF')
    plt.legend()
    plt.show()

def q_1_5(n):
    x = np.random.rand(n) # n Uniform
    count, bins, _ = plt.hist(x, bins=10, color='gray', edgecolor='black', label = 'Histogram Count Data')

    bin_width = bins[1] - bins[0]
    
    a, b = 0., 1.
    x_values = np.linspace(a, b, n)
    pdf = [(1 / (b - a)) * n * bin_width] * n
    
    
    plt.plot(x_values, pdf, 'r-', lw=2, label='Exact Uniform PDF')

    mean = count.mean()
    std_dev = count.std()
    print(mean, std_dev)
    plt.axhline(mean - 3 * std_dev, color='cyan', linestyle='--', label='-3 Std Dev')
    plt.axhline(mean + 3 * std_dev, color='orange', linestyle='--', label='+3 Std Dev') 
    plt.legend()
    plt.show()

def q_1_6(n):
    x = np.random.randn(n)
    count, bins, _ = plt.hist(x, bins=10, color='gray', edgecolor='black', label = 'Histogram Count Data')

    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    mean = 0
    std_dev = 1
    x_values = np.linspace(-4, 4, n)
    pdf = scipy.stats.norm.pdf(x_values, mean, std_dev) * n * bin_width

    bin_means = []
    bin_std_devs = []
    for i in range(10): # Bin count
        p_i = scipy.stats.norm.cdf(bins[i + 1], mean, std_dev) - scipy.stats.norm.cdf(bins[i], mean, std_dev)
        bin_means.append(n * p_i)

        sigma_i=  np.sqrt(p_i * (1 - p_i))
        bin_std_devs.append(np.sqrt(n) * sigma_i)
    
    for i in range(10):
        plt.hlines(bin_means[i], bins[i], bins[i+1], colors='red', linestyles='--', linewidth=1.5)
        plt.hlines(bin_means[i] - 3 * bin_std_devs[i], bins[i], bins[i+1], colors='cyan', linestyles='--', linewidth=1.5)
        plt.hlines(bin_means[i] + 3 * bin_std_devs[i], bins[i], bins[i+1], colors='orange', linestyles='--', linewidth=1.5)

    plt.legend()
    plt.show()


def q_2_1(a, b):
    x = np.random.randn(10000)
    y = a * x + b
    plt.hist(x, bins=20, label='p(x)')
    count, bins, _ = plt.hist(y, bins=20, label='p(y)')

    bin_width = bins[1] - bins[0]

    mean = b
    std_dev = a
    x_values = np.linspace(-4., 4., 10000)
    pdf = scipy.stats.norm.pdf(x_values, mean, std_dev) * 10000 * bin_width

    plt.plot(x_values, pdf, 'r-', lw=2, label='Jacobian Predicted p(y)')

    plt.legend()
    plt.show()

def q_2_2():
    x = np.random.randn(10000)
    y = x * x
    plt.hist(x, bins=20, label='p(x)')
    count, bins, _ = plt.hist(y, bins=20, label='p(y)')

    bin_width = bins[1] - bins[0]

    x_values = np.linspace(0.1, 4., 10000)
    pdf = np.exp(- x_values / 2) * 10000 * bin_width / (np.sqrt(2 * np.pi * x_values))
    plt.plot(x_values, pdf, 'r-', lw=2, label='Jacobian Predicted p(y)')

    plt.legend()
    plt.show()

def q_2_3():
    x = np.random.uniform(low=0, high=2 * np.pi, size=10000)
    y = np.sin(x)
    plt.hist(x, bins=20, label='p(x)')
    count, bins, _ = plt.hist(y, bins=20, label='p(y)')

    bin_width = bins[1] - bins[0]

    x_values = np.linspace(0.1, 0.995, 10000)
    pdf = 1 / (2 * np.pi * np.cos(np.arcsin(x_values))) * 10000 * bin_width
    plt.plot(x_values, pdf, 'r-', lw=2, label='Jacobian Predicted p(y)')

    plt.legend()
    plt.show()

def q_2_3_1():
    x = np.random.uniform(low=0, high=2 * np.pi, size=100000)
    y = []
    for value in x:
        y.append(min(np.sin(value), 0.7))
    # plt.hist(x, bins=20, label='p(x)')
    count, bins, _ = plt.hist(y, bins=40, label='sampled p(y)')
    plt.axvline(0.7, color='red', linestyle='--', label='y = 0.7')

    bin_width = bins[1] - bins[0]

    x_values = np.linspace(-0.99, 0.6999, 1000)
    pdf = 1 / (2 * np.pi * np.cos(np.arcsin(x_values))) * 200000 * bin_width
    plt.plot(x_values, pdf, 'r-', lw=2, label='Jacobian Predicted p(y)')

    plt.legend()
    plt.show()

def q_3_1(n):
    x = np.random.rand(n)
    y = inverse_cdf(n, 1)
    # count, bins, _ = plt.hist(y, bins=20, label='Sampled Exp. Dist.') 
    

    # bin_width = bins[1] - bins[0]

    ks_density = ksdensity(y, width=0.1)
    x_values = np.linspace(0.1, 6., n)
    pdf = np.exp(- x_values)
    plt.plot(x_values, pdf, label='Exact Exp. Density')
    plt.plot(x_values, ks_density(x_values), label='KDE')

    mean = sum(pdf)/n
    variance = (sum(pdf ** 2 - mean ** 2))/n
    print(mean)
    print(variance)

    plt.legend()
    plt.show()

def q_3_2(n):
    def monte_carlo(n):
        x = np.random.rand(n)
        y = inverse_cdf(n, 1)

        mean = sum(y) / n
        var = sum(y ** 2 - mean ** 2) / n
        
        print(- mean + 1)
        print(- var + 1)

        return mean, var

    sizes = np.arange(0, 300, 1)
    error = []
    for i in sizes:
        mean, var = monte_carlo(i)
        error.append((mean - 1) ** 2)

    ref = 1/sizes
    plt.plot(sizes,ref, label='reference y=1/x')
    
    plt.plot(sizes, error, label='Monte Carlo estimate')
    plt.xlabel("Monte Carlo sample size")
    plt.ylabel("Squred mean error")

    plt.legend()
    plt.show()

def q_4_1(n, alpha, beta):
    b = np.arctan(beta * np.tan(np.pi * alpha / 2)) / alpha
    s = (1 + (beta ** 2) * (np.tan(np.pi * alpha / 2) ** 2)) ** (1 / (2 * alpha))

    u = np.random.uniform(low=(-np.pi/2), high=(np.pi/2), size=n)
    v = np.random.exponential(scale=1, size=n)

    x = s * np.sin(alpha * (u + b)) * ((np.cos(u - alpha * (u + b)) / v) ** ((1 - alpha)/alpha)) / (np.cos(u) ** (1 / alpha))

    bin_width = 0.5
    bins = np.arange(-10, 10 + bin_width, bin_width)
    
    plt.hist(x, bins=bins, align='mid')
    plt.title(f'alpha = {alpha}, beta = {beta}')

    plt.show()

def q_4_2(n, alpha, beta, t):
    b = np.arctan(beta * np.tan(np.pi * alpha / 2)) / alpha
    s = (1 + (beta ** 2) * (np.tan(np.pi * alpha / 2) ** 2)) ** (1 / (2 * alpha))

    u = np.random.uniform(low=(-np.pi/2), high=(np.pi/2), size=n)
    v = np.random.exponential(scale=1, size=n)

    x = s * np.sin(alpha * (u + b)) * ((np.cos(u - alpha * (u + b)) / v) ** ((1 - alpha)/alpha)) / (np.cos(u) ** (1 / alpha))

    bin_width = 0.5
    bins = np.arange(-12, 12 + bin_width, bin_width)
    
    plt.hist(x, bins=bins, align='mid')
    if t != 0:
        plt.axvline(x=t, color='red', linestyle='--', label=f'x={t}')
        plt.axvline(x=-t, color='red', linestyle='--', label=f'x={-t}')
    else:
        plt.axvline(x=t, color='red', linestyle='--', label=f'x={t}')
    plt.title(f'alpha = {alpha}, beta = {beta}, t = {t}')

    tail_probability = (np.sum(x > t) + np.sum(x < -t)) / len(x)
    print(tail_probability)

    mean = 0
    std_dev = 1
    gaussian_tail_prob = 2 * (1 - scipy.stats.norm.cdf(t, mean, std_dev))
    print(gaussian_tail_prob)

    plt.legend()
    plt.show()

def q_4_2_2(n, alpha, beta, t):
    b = np.arctan(beta * np.tan(np.pi * alpha / 2)) / alpha
    s = (1 + (beta ** 2) * (np.tan(np.pi * alpha / 2) ** 2)) ** (1 / (2 * alpha))

    u = np.random.uniform(low=(-np.pi/2), high=(np.pi/2), size=n)
    v = np.random.exponential(scale=1, size=n)

    x = s * np.sin(alpha * (u + b)) * ((np.cos(u - alpha * (u + b)) / v) ** ((1 - alpha)/alpha)) / (np.cos(u) ** (1 / alpha))

    # bin_width = 0.5
    # bins = np.arange(-12, 12 + bin_width, bin_width)
    
    # plt.hist(x, bins=bins, align='mid')

    x_values = np.linspace(np.e, 100., n)
    ks_density = ksdensity(x, width = 0.9)
    plt.loglog(x_values, ks_density(x_values), label="KDE")

    log_x_values = np.log(x_values)
    log_kd = np.log(ks_density(x_values))
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(log_x_values, log_kd)
    print(f"Slope: {slope} Intercept: {intercept}")
    plt.loglog(x_values, np.e ** (intercept + slope * log_x_values), label=f'Fit: y = {np.e**intercept:.2f} * x^{slope:.2f}', color='red')

    plt.title(f'alpha = {alpha}, beta = {beta}')

    plt.legend()
    plt.show()

def q_4_3(n, alpha, beta, t):
    x_values = np.linspace(-4., 4., n)
    for a in alpha:
        b = np.arctan(beta * np.tan(np.pi * a / 2)) / a
        s = (1 + (beta ** 2) * (np.tan(np.pi * a / 2) ** 2)) ** (1 / (2 * a))

        u = np.random.uniform(low=(-np.pi/2), high=(np.pi/2), size=n)
        v = np.random.exponential(scale=1, size=n)

        x = s * np.sin(a * (u + b)) * ((np.cos(u - a * (u + b)) / v) ** ((1 - a)/a)) / (np.cos(u) ** (1 / a))

        # bin_width = 0.5
        # bins = np.arange(-10, 10 + bin_width, bin_width)
        
        # plt.hist(x, bins=bins, align='mid')

        ks_density = ksdensity(x, width = 0.5)
        plt.plot(x_values, ks_density(x_values), label=f'α={a}')

    mean = 0
    std_dev = np.sqrt(2)
    pdf = scipy.stats.norm.pdf(x_values, mean, std_dev)
    plt.plot(x_values, pdf, linestyle='--', label='Standard Gaussian')

    plt.legend()
    plt.show()
if __name__ == "__main__":
    q_1_4()
