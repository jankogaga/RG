!gdown --id 1580YcFpDL1SxEoPNtnF8ibGKH76oCWaY
%mv Rg-export.pkl export.pkl
!pip install fastai --upgrade

from ipywidgets import *
from fastai.data.all import *
from fastai.vision.all import *
from fastai.vision.widgets import *
 
btn_upload = widgets.FileUpload()
learn_inf = load_learner('export.pkl')
out_pl = widgets.Output()
lbl_pred = widgets.Label()
 
def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
 
btn_run = widgets.Button(description='Classify')
btn_run.on_click(on_click_classify)
 
VBox([widgets.Label('Select your Rg image:'), 
      btn_upload, btn_run, out_pl, lbl_pred])