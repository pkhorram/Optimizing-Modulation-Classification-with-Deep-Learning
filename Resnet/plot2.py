import pickle
from sklearn.preprocessing import normalize
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


with open("./history/SNR_2_history", "rb") as fp:
    history = pickle.load(fp)

plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model performance vs. Epoch for SNR = 2 dB')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("./Plots/accuracy_vs_epochs.png")
plt.show()
