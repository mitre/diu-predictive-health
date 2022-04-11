import os
import math
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image
import openslide

def ROC(fpr, tpr, roc_auc):
    '''
    Plot ROC curve, given list of FPR, list of TPR and AUC.
    '''
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def thresh_plot(fpr, tpr, thresh):
    '''
    For a subset of data which is either all negative or all positive, plot the false positive or false positive rate as a function of the threshold.
    '''
    if not math.isnan(tpr[0]):
        plt.plot(thresh, 1 - tpr)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('False Negative Rate (1 - Sensitivity)')
        plt.xlabel('Cancer Threshold')
        plt.show()
    if not math.isnan(fpr[0]):
        plt.plot(thresh, fpr)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('False Positive Rate (1 - Specificity)')
        plt.xlabel('Cancer Threshold')
        plt.show()
    
def draw_single_patch(fov_path, patch):
    '''
    Draw a label around a single ROI within an FOV.
    '''
    plt.figure(figsize = (15,15))
    plt.imshow(Image.open(fov_path))
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False) 
    
    rect = Rectangle((patch['points']['x'], patch['points']['y']),128,128,linewidth=3,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

def draw_heatmap(patches, save=False, path=None):
    '''
    Draw a heatmap of the ROI predictions for a given FOV. 
    '''
    test = [[0]*14 for i in range(14)]

    for p in patches:
        i = int(p['id'])
        d, r = divmod(i, 14)
        test[r][d] = p['conf'][1]
        
    fig = go.Figure(data=go.Heatmap(z=test, zmin=0, zmax=1, xgap=1, ygap=1, colorscale="Viridis_r"))
    margin = 10
    fig.update_layout(height=600,width=600, margin = {'l':margin,'r':margin,'t':margin,'b':margin})
    fig.update_yaxes(visible=False)
    fig.update_xaxes(visible=False)
    fig.show()
    if save:
        fig.write_image(path)

def plot_fov(fov_path, patches=[], save=False, path=None, size=20, threshold=0.5, subclass=None, OOD=None):
    '''
    Plot an FOV, with ROI labels based on given results.
    '''
    fig = plt.figure(figsize = (size,size))
    plt.imshow(Image.open(fov_path))
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)   
    if OOD:
        for i in patches:
            if i['conf'][1] > threshold:
                ax.add_patch(Circle((i['points'][0]['x'] + 64, i['points'][0]['y'] + 64), 62, linewidth=1.5, edgecolor='r', facecolor='none'))
    elif subclass:
        for i in patches:
            if i['short_name'] == subclass:
                if i['object_class'] == 'Benign':
                    color = 'b'
                    if i['conf'][1] > threshold:
                        ax.add_patch(Circle((i['points'][0]['x'] + 64, i['points'][0]['y'] + 64), 62, linewidth=1.5, edgecolor='lime', facecolor='none'))
                else:
                    color = 'r'
                    if i['conf'][0] >= threshold:
                        ax.add_patch(Circle((i['points'][0]['x'] + 64, i['points'][0]['y'] + 64), 62, linewidth=1.5, edgecolor='lime', facecolor='none'))
                ax.add_patch(Rectangle((i['points'][0]['x'], i['points'][0]['y']), 125,125, linewidth=1.5, edgecolor=color, facecolor='none'))
    else:
        for i in patches:
            if i['object_class'] == 'Benign':
                if i['conf'][1] > threshold:
                    ax.add_patch(Circle((i['points'][0]['x'] + 64, i['points'][0]['y'] + 64), 62, linewidth=1.5, edgecolor='lime', facecolor='none'))
                ax.add_patch(Rectangle((i['points'][0]['x'], i['points'][0]['y']), 125, 125, linewidth=1.5, edgecolor='b', facecolor='none'))
            else:
                if i['conf'][0] >= threshold:
                    ax.add_patch(Circle((i['points'][0]['x'] + 64, i['points'][0]['y'] + 64), 62, linewidth=1.5, edgecolor='lime', facecolor='none'))
                ax.add_patch(Rectangle((i['points'][0]['x'], i['points'][0]['y']), 125, 125, linewidth=1.5, edgecolor='r', facecolor='none'))
    if save:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        fig.savefig(path, bbox_inches='tight')
        plt.cla()
        plt.clf()
    else:
        plt.show()
        plt.clf()