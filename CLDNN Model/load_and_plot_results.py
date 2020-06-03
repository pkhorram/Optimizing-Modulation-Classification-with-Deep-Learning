import pickle
from sklearn.preprocessing import normalize
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# Validation, Training Loss and Accuracy Plotting Code
with open("./results_fifth_run_dropout0.3_300epochs/history/cldnn/SNR_2_history", "rb") as fp:
    history = pickle.load(fp)

plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss vs Epochs for SNR = 2 dB')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("./Plots/loss_vs_epochs.png")
plt.show()

plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model performance vs. Epoch for SNR = 2 dB')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("./Plots/accuracy_vs_epochs.png")
plt.show()

# Accuracy vs SNR Plotting Code
with open("./results_third_run_dropouts/cldnn/accuracy_results", "rb") as fp:
    accuracy_results_paper_model = pickle.load(fp)
with open("./results_fifth_run_dropout0.3_300epochs/accuracy_results", "rb") as fp:
    accuracy_results_final = pickle.load(fp)

y = []
x = []

for i in range(len(accuracy_results_final)):
    x.append(accuracy_results_final[i][1])
x = sorted(x)


for i in x:
    for j in range(len(accuracy_results_final)):
        if accuracy_results_final[j][1]  == i:
            y.append(accuracy_results_final[j][0])

plt.plot(x, y, marker='*', label='Final Model with Maxpool')

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

# Confusion Matrix Plotting Code
with open("./final_model_results/confusion_matrix_results", "rb") as fp:
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



