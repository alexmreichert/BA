FROM python:3.9-bullseye

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install --no-install-recommends -y \
    libgl1 \
    libgomp1 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

COPY ./app ./app

CMD [ "python", "./app/generation.py" ]