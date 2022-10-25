
FROM python:3.8-bullseye

COPY requirements.txt requirements.txt

#install dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    libgl1 \
    libc6 \
    libeigen3-dev\
    cmake \
    libboost-all-dev \
    libgomp1 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

#install TEASER according to github instructions
RUN apt install -y cmake libeigen3-dev libboost-all-dev 

RUN git clone https://github.com/MIT-SPARK/TEASER-plusplus.git TEASER-plusplus 

RUN cd TEASER-plusplus 

RUN mkdir build && cd build

RUN cmake TEASER-plusplus/ && make

RUN cmake -DTEASERPP_PYTHON_VERSION=3.8 TEASER-plusplus/ && make teaserpp_python

RUN cd .. && cd python

RUN pip install python/

RUN cd ../../..

RUN pip install -r requirements.txt

COPY ./applet ./applet

CMD [ "python3.8", "./applet/idreg.py" ]