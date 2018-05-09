import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

#read in dataset
data = sp.genfromtxt("datasets/web_traffic.tsv", delimiter="\t")

#preprocessing/cleaning the data
#split dimensions into 2 vectors
x = data[:,0]
y = data[:,1]

#remove nan values
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

#visualise the data
def plot_graph(x, y, models, fname, mx=None, ymax=None, xmin=None):
    ''' plot input data '''

    plt.figure(num=None, figsize=(8, 6))
    plt.clf()
    plt.scatter(x, y, s=10)
    plt.title("Web traffic over the last month")
    plt.xlabel("Time")
    plt.ylabel("Hits/hour")
    plt.xticks(
        [w * 7 * 24 for w in range(10)], ['week %i' % w for w in range(10)])

    if models:
        if mx is None:
            mx = sp.linspace(0, x[-1], 1000)
        for model, style, color in zip(models, linestyles, colors):
            # print "Model:",model
            # print "Coeffs:",model.coeffs
            plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)

        plt.legend(["d=%i" % m.order for m in models], loc="upper left")

    plt.autoscale(tight=True)
    plt.ylim(ymin=0)
    if ymax:
        plt.ylim(ymax=ymax)
    if xmin:
        plt.xlim(xmin=xmin)
    plt.grid(True, linestyle='-', color='0.75')
    plt.savefig(fname)

def error(f, x, y):
    return sp.sum((f(x) - y) ** 2)

#linear regression model
fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)

#create a function from the model parameters (f1)
f1 = sp.poly1d(fp1)

print("Linear Regression Model parameters: %s" %  fp1)
print("Error of the Linear Regression model: %s" % error(f1, x, y))
#plot_graph(x, y, f1)

fp2, res2, rank2, sv2, rcond2 = sp.polyfit(x, y, 2, full=True)
print("Model parameters of fp2: %s" % fp2)
print("Error of the model of fp2:", res2)

f2 = sp.poly1d(fp2)
f3 = sp.poly1d(sp.polyfit(x, y, 3))
f10 = sp.poly1d(sp.polyfit(x, y, 10))
f100 = sp.poly1d(sp.polyfit(x, y, 100))

# fit and plot a model using the knowledge about inflection point
inflection = 3.5 * 7 * 24
inflection = int(inflection)
xa = x[:inflection]
ya = y[:inflection]
xb = x[inflection:]
yb = y[inflection:]

fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))

#plot_graph(x, y, [fa, fb], os.path.join(CHART_DIR, "1400_01_05.png"))
print("Errors for the complete data set:")
for f in [f1, f2, f3, f10, f100]:
    print("Error d=%i: %f" % (f.order, error(f, x, y)))

print("Errors for only the time after inflection point")
for f in [f1, f2, f3, f10, f100]:
    print("Error d=%i: %f" % (f.order, error(f, xb, yb)))

print("Error inflection=%f" % (error(fa, xa, ya) + error(fb, xb, yb)))

# extrapolating into the future
#plot_graph(
#    x, y, [f1, f2, f3, f10, f100],
#    os.path.join(CHART_DIR, "1400_01_06.png"),
#    mx=sp.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
#    ymax=10000, xmin=0 * 7 * 24)

print("Trained only on data after inflection point")
fb1 = fb
fb2 = sp.poly1d(sp.polyfit(xb, yb, 2))
fb3 = sp.poly1d(sp.polyfit(xb, yb, 3))
fb10 = sp.poly1d(sp.polyfit(xb, yb, 10))
fb100 = sp.poly1d(sp.polyfit(xb, yb, 100))

print("Errors for only the time after inflection point")
for f in [fb1, fb2, fb3, fb10, fb100]:
    print("Error d=%i: %f" % (f.order, error(f, xb, yb)))

#plot_graph(
#    x, y, [fb1, fb2, fb3, fb10, fb100],
#    os.path.join(CHART_DIR, "1400_01_07.png"),
#    mx=sp.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
#    ymax=10000, xmin=0 * 7 * 24)

# separating training from testing data
frac = 0.3
split_idx = int(frac * len(xb))
shuffled = sp.random.permutation(list(range(len(xb))))
test = sorted(shuffled[:split_idx])
train = sorted(shuffled[split_idx:])
fbt1 = sp.poly1d(sp.polyfit(xb[train], yb[train], 1))
fbt2 = sp.poly1d(sp.polyfit(xb[train], yb[train], 2))
print("fbt2(x)= \n%s" % fbt2)
print("fbt2(x)-100,000= \n%s" % (fbt2-100000))
fbt3 = sp.poly1d(sp.polyfit(xb[train], yb[train], 3))
fbt10 = sp.poly1d(sp.polyfit(xb[train], yb[train], 10))
fbt100 = sp.poly1d(sp.polyfit(xb[train], yb[train], 100))

print("Test errors for only the time after inflection point")
for f in [fbt1, fbt2, fbt3, fbt10, fbt100]:
    print("Error d=%i: %f" % (f.order, error(f, xb[test], yb[test])))

#plot_graph(
#    x, y, [fbt1, fbt2, fbt3, fbt10, fbt100],
#    os.path.join(CHART_DIR, "1400_01_08.png"),
#    mx=sp.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
#    ymax=10000, xmin=0 * 7 * 24)

print(fbt2)
print(fbt2 - 100000)
reached_max = fsolve(fbt2 - 100000, x0=800) / (7 * 24)
print("100,000 hits/hour expected at week %f" % reached_max[0])
