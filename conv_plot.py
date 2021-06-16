import matplotlib
import matplotlib.pyplot as plt
import pandas as pd  
from matplotlib.pyplot import figure
import numpy as np

# convergence plot 

figure(figsize=(6, 4), dpi=80)

font = {'size' : 12} 
matplotlib.rc('font', **font)

#plot all models side by side 
linestyles = ["--", ":", (0, (1, 10)),"-.", (0, (3, 5, 1, 5, 1, 5))]
markers = ["o", "s", "d", "o", "s", "d"]

topos = ["random", "fc", "ring"]
mixing_steps = [1,3,5,7,10]

df = pd.read_csv("out_topos.csv", header=None)

for i, top in enumerate(topos):
    top_dat = df.loc[df.iloc[:,0] == top] 
    for j, mix_steps in enumerate(top_dat.loc[:,4].unique()):
        loss = df.loc[df.iloc[:,0] == top].loc[df.iloc[:, 4] == mix_steps].iloc[:,-1].to_numpy().astype(float)
        if top == "fc":
            t = np.arange(0, loss.shape[0])
            plt.plot(t, loss, marker="None", color='black', markerfacecolor="white", label=top + " - 1 mix")
        else:
            t = np.arange(0, loss.shape[0])
            plt.plot(t, loss, linestyle=linestyles[j], marker="None", color='black', markerfacecolor="white", label=top + " - " + str(mix_steps) + " mixes")

plt.yscale('log')
plt.gcf().subplots_adjust(bottom=0.15)
plt.legend()
plt.grid(True, which="major")
plt.xlabel("Iteration")
plt.ylabel("f(x) - f(x*)") 
plt.show()

