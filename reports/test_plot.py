import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Test Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.savefig("test_plot.png")
plt.close()

print("Test plot saved as test_plot.png")