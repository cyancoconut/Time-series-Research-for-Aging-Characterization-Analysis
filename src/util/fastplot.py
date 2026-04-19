#fastplot.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_ID(df,ID):
    df.loc[df["ID"]==ID].plot(y="Current",title=ID)

def plot_target_type(df,target):
    mask = df.loc[df["target"]==target,"ID"].unique()
    ID_List = mask
    plt.close()
    for id in ID_List[0:5]:
        df.loc[df["ID"]==id].plot(y=["Current"],title=id)

def plot_values(df,column):
    mask = df[df[column].notna()]
    plt.close()
    df[mask].plot(y=[column],title=id)
