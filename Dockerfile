FROM resin/rpi-raspbian:jessie

# put your app in this directory
# don't forget to add a file named requirements.txt
ADD app /app

# Install Python and dependencies to build pillow and tensorflow
RUN apt-get update \
               && apt-get install -y \
                              python3 \
                              python3-pip \
                              build-essential \
                              python3-dev \
                              zlib1g-dev \
                              libjpeg-dev \
                              wget \
               && pip3 install --upgrade pip \
               && pip install --upgrade setuptools \
               && pip install -r /app/requirements.txt \
               && pip install http://ci.tensorflow.org/view/Nightly/job/nightly-pi-python3/122/artifact/output-artifacts/tensorflow-1.5.0-cp34-none-any.whl

# Expose the port
EXPOSE 80

# Set the working directory
WORKDIR /app

# Run the flask server for the endpoints
CMD ["python3","app.py"]