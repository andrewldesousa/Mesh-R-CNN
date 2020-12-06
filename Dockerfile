FROM continuumio/anaconda3

RUN conda create -n env python=3.8
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

COPY ./api /api/api
COPY requirements.txt /requirements.txt

RUN apt-get update && \
    apt-get upgrade -y
RUN apt-get install -y git
RUN apt install -y build-essential

RUN git clone https://github.com/facebookresearch/meshrcnn.git
RUN conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
RUN conda install pip
RUN pip install -r requirements.txt


RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.7/index.html
RUN pip install -e ./meshrcnn/
RUN apt install -y libgl1-mesa-glx

RUN apt-get install -y freeglut3 freeglut3-dev libxi-dev libxmu-dev
RUN apt-get install -y software-properties-common
ENV PYTHONPATH=/api


WORKDIR /api

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["api.main:app", "--reload", "--host", "0.0.0.0"]
