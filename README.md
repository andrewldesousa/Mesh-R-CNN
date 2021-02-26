# Mesh R-CNN Playground - Backend

This repository contains a web application for interacting with Mesh R-CNN and our additions to the model. The API has been developed using the FastAPI library.


![Mesh R-CNN Archiecture](mesh_arch.png)


## Usage

For running the app with hot reload functionality.

`uvicorn api.main:app --reload --host 0.0.0.0`

If you don't need hot reload functionality, then running

`uvicorn api.main:app --host 0.0.0.0`


## Functionality

