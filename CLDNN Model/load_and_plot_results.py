import pickle
from sklearn.preprocessing import normalize
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# This code picks up the files from the specified folders to plot the results
# # Validation, Training Loss and Accuracy Plotting Code
with open("./big_data_results/model_history.pkl", "rb") as fp:
    history = pickle.load(fp)

plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("./Plots/loss_vs_epochs.png")
plt.show()

plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model performance vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("./Plots/accuracy_vs_epochs.png")
plt.show()

# Accuracy vs SNR Plotting Code
with open("./smaller_dataset_actual_model_results/accuracy_results", "rb") as fp:
    acc1 = pickle.load(fp)
with open("./bigger_dataset_actual_model_results/accuracy_results", "rb") as fp:
    acc2 = pickle.load(fp)
with open("./complete_SNR_results/accuracy_results", "rb") as fp:
    acc3 = pickle.load(fp)
with open("./big_data_results/accuracy_results", "rb") as fp:
    acc4 = pickle.load(fp)

# This commented code plots accuracy results with 20 models trained separately on different SNR
# with open("../final_model_results/accuracy_results", "rb") as fp:
#     acc5 = pickle.load(fp)
#
# y = []
# x = []
#
# for i in range(len(acc5)):
#     x.append(acc5[i][1])
# x = sorted(x)
#
#
# for i in x:
#     for j in range(len(acc5)):
#         if acc5[j][1] == i:
#             y.append(acc5[j][0])
#
# plt.plot(x, y, marker='v', label='2016.10A Modified Model with 20 models')

y = []
x = []

for i in range(len(acc4)):
    x.append(acc4[i][1])
x = sorted(x)


for i in x:
    for j in range(len(acc4)):
        if acc4[j][1] == i:
            y.append(acc4[j][0])

plt.plot(x, y, marker='v', label='2016.10B Modified Model')

y = []
x = []

for i in range(len(acc3)):
    x.append(acc3[i][1])
x = sorted(x)


for i in x:
    for j in range(len(acc3)):
        if acc3[j][1] == i:
            y.append(acc3[j][0])

plt.plot(x, y, marker='*', label='2016.10A Modified Model')

y = []
x = []

for i in range(len(acc2)):
    x.append(acc2[i][1])
x = sorted(x)


for i in x:
    for j in range(len(acc2)):
        if acc2[j][1]  == i:
            y.append(acc2[j][0])

plt.plot(x, y, marker='o', label='2016.10B Paper Model')

y = []
x = []

for i in range(len(acc1)):
    x.append(acc1[i][1])
x = sorted(x)


for i in x:
    for j in range(len(acc1)):
        if acc1[j][1]  == i:
            y.append(acc1[j][0])


plt.plot(x, y, marker='d', label='2016.10A Paper Model')

plt.title('Model performance vs. SNR')
plt.xlabel('SNR')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()
plt.savefig("./Plots/accuracy_vs_snr.png")
plt.show()

# Confusion Matrix Plotting Code
with open("./big_data_results/confusion_matrix_results", "rb") as fp:
    confusion_matrix = pickle.load(fp)

# Extracting Confusion Matrix for SNR 18dB (Use different index for different SNR value)
# and Normalizing Confusion Matrix Values
normalized_confusion_matrix = normalize(confusion_matrix[18][0], norm='l1')

label_name = ["8PSK", "AM-DSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
df_cm = pd.DataFrame(normalized_confusion_matrix, label_name, label_name)
sn.set(font_scale=0.9)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, cmap="Blues")
plt.xticks(ticks=[ 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], rotation=45)
plt.title('Confusion Matrix for SNR = 16 dB')
plt.savefig("./Plots/confusion_matrix_18dB.png")
plt.show()
