import os
import cv2
import zipfile
from fastapi import FastAPI, File
from fastapi.responses import FileResponse
from api.ml.model import setup_cfg, VisualizationDemo


app = FastAPI()

@app.post("/predict")
def predict(img: bytes = File(...), split: int = 0):
    return None
    # Clear old files and save input image
    os.system('rm response.zip && rm -rf output && mkdir output')
    input_img = open(f'output/input.png', "wb")
    input_img.write(img)
    input_img.close()
    
    
    cfg = setup_cfg(split)
    demo = VisualizationDemo(cfg, vis_highest_scoring=False, output_dir='output')
    
    input_img = cv2.imread('output/input.png')
    predictions = demo.run_on_image(input_img)

    with zipfile.ZipFile('response.zip', 'w') as myzip:
        for i in os.listdir('output'):
            myzip.write(f'output/{i}')

    response = FileResponse(path='response.zip', filename='response.zip')
    return response

    
