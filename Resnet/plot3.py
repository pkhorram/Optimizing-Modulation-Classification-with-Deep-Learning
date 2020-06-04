import pickle
from sklearn.preprocessing import normalize
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


with open("Accuracy resutls", "rb") as fp:
    accuracy_results_paper_model = pickle.load(fp)

y = []
x = []

for i in range(len(accuracy_results_paper_model)):
    x.append(accuracy_results_paper_model[i][1])
x = sorted(x)


for i in x:
    for j in range(len(accuracy_results_paper_model)):
        if accuracy_results_paper_model[j][1]  == i:
            y.append(accuracy_results_paper_model[j][0])


plt.plot(x, y, marker='d', label='Paper Model')
plt.title('Model performance vs. SNR')
plt.xlabel('SNR')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()
plt.savefig("./Plots/accuracy_vs_snr.png")
plt.show()
