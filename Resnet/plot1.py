import pickle
from sklearn.preprocessing import normalize
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


with open("./history/SNR_2_history", "rb") as fp:
    history = pickle.load(fp)

plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss vs Epochs for SNR = 2 dB')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("./Plots/loss_vs_epochs.png")
plt.show()