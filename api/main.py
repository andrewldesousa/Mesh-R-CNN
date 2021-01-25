import os
import cv2
import zipfile
from fastapi import FastAPI, File
from fastapi.responses import FileResponse
from api.ml.model import setup_cfg, MeshRCNNModel


app = FastAPI()


@app.post("/predict")
def predict(img: bytes = File(...), split: int = 0):
    # Clear old files and save input image
    os.system('rm response.zip && rm -rf output && mkdir output')
    input_img = open(f'output/input.png', "wb")
    input_img.write(img)
    input_img.close()

    cfg = setup_cfg(split)
    model = MeshRCNNModel(cfg, vis_highest_scoring=False, output_dir='output')
    
    input_img = cv2.imread('output/input.png')
    model.run_on_image(input_img, focal_length=20.0)

    with zipfile.ZipFile('response.zip', 'w') as zip_file:
        for i in os.listdir('output'):
            zip_file.write(f'output/{i}')

    response = FileResponse(path='response.zip', filename='response.zip')
    return response
