'''====================================================================
-------------------------- USER INTERFACE -----------------------------
===================================================================='''
import numpy as np

import sys
sys.path.insert(0,'C:\\Users\\Honza\\Documents\\GitHub\\LesionAdjuster')

import lesionAdjuster.GUI as g

ROOT = 'C:/Users/Honza/Documents/GitHub/LesionAdjuster/'

#LOAD IMAGES
pth_to_preds = ROOT + 'data/img_input.npz'
imgs = np.load(pth_to_preds)
x_img = imgs['x_img'] #864(rows) x 1232 (cols) -> 864/1232=0.7
y_form = imgs['true_form']
pred = imgs['pred_form']

#RUN GUI
LA = g.LesionAdjuster(root_pth=ROOT)
LA.showGUI(pth_to_img='data/33.png', 
           y_form=y_form,
           pred=pred)

