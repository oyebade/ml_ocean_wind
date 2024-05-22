import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def plt_act_pred(X:np.ndarray, y:np.ndarray, y_pred:np.ndarray, 
            vers:str, save_plt_pth:str=None, plot_caption:str=None,
            rmse:float= None, r2:float=None,
            ylabel:str=None, xlabel:str=None):
    """plots the actual output data and predicted output data

    Args:
        X (np.ndarray): input data
        y (np.ndarray): actual output data
        y_pred (np.ndarray): predicted output data
        save_plt_pth (str, optional): directory to save data. Defaults to None.
        plot_caption (str, optional): plot caption. Defaults to None.
        rmse (float, optional): rmse. Defaults to None.
        r2 (float, optional): coefficient of determination. Defaults to None.
        ylabel (str, optional): plot y axis label. Defaults to None.
        xlabel (str, optional): plot x axis label. Defaults to None.
    """    
    
    plt_act = plt.scatter(X.to_numpy()[:,0], y.to_numpy())
    plt_pred = plt.scatter(X.to_numpy()[:,0], y_pred)
    plt.legend((plt_act, plt_pred), ("Actual", "Predicted"),
            scatterpoints=1, loc="upper right", ncol=1, fontsize=12)
    full_plot_caption = plot_caption + "; rmse: " + str(rmse) + " ; r2: " + str(r2)
    plt.title(full_plot_caption, fontsize=18)
    plt.ylabel(ylabel, fontsize = 16)
    plt.xlabel(xlabel, fontsize=16)
    plt.yticks(fontsize = 14) 
    plt.xticks(fontsize = 14)
    
    if save_plt_pth is not None:
        fullFileName = os.path.join(save_plt_pth, plot_caption + "_" + vers + ".png")
        plt.savefig(fullFileName, bbox_inches='tight')
    plt.clf()
    plt.close()
        
        
def plt_feat_imp(model:object, X:pd.DataFrame, feat_import_capt:str, vers:str, save_plt_pth:str=None):
    """plot feature importance for model

    Args:
        model (object): xgboost ML model
        X (pd.DataFrame): pandas input data
        feat_import_capt (str): feature importance plot caption
        save_plt_pth (str, optional): directory to save plot. Defaults to None.
    """    
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(15, 12))
    fig, ax = plt.subplots()
    plt.title(feat_import_capt, fontsize=20)
    plt.yticks(fontsize = 18) 
    plt.xticks(fontsize = 18) 
    ax.barh(range(len(importances)), importances[indices])
    ax.set_yticks(range(len(importances)))
    _ = ax.set_yticklabels(np.array(X.columns)[indices])
    
    if save_plt_pth is not None:
        fullFileName = os.path.join(save_plt_pth, feat_import_capt + "_" + vers + ".png")
        plt.savefig(fullFileName, bbox_inches='tight')
    plt.show()
