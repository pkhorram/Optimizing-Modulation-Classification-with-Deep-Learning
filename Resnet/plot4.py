import pickle
from sklearn.preprocessing import normalize
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


with open("confusion_matrix_results", "rb") as fp:
    confusion_matrix = pickle.load(fp)

# Extracting Confusion Matrix for SNR 18dB (Use different index for different SNR value)
# and Normalizing Confusion Matrix Values
normalized_confusion_matrix = normalize(confusion_matrix[18][0], norm='l1')

label_name = ["BPSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
df_cm = pd.DataFrame(normalized_confusion_matrix, label_name, label_name)
sn.set(font_scale=0.9)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, cmap="Blues")
plt.xticks(ticks=[ 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5], rotation=45)
plt.title('Confusion Matrix for SNR = 18 dB')
plt.savefig("./Plots/confusion_matrix_18dB.png")
plt.show()
