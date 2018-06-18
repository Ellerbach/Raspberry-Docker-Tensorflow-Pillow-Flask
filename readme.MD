This project will allow you to create a Docker image on Raspberry Pi and run prediction from ML/AI models using Tensorflow, Pillow and Flask from any Machine Learning (ML) or Artificial Intelligence (AI) model.

# Step 0: Lazy? ok, here is the docker image for you

If you don't want to fully build the image, just use the prebuild image ```docker pull ellerbach/tensor_pillow_flask``` you will find details on [Docker Hub](https://hub.docker.com/r/ellerbach/tensor_pillow_flask/)

You'll just need to make sure you have docker installed, the model and the labels in the app directory, and use the Dockerfile ```existing.Dockerfile``` to create and install the image.

# Step 1: Preparing the Raspberry Pi

First, we'll need to install Docker. As usual, better to run an update:

```sudo apt-get update```

Then we'll need to install Docker:

```curl -sSL https://get.docker.com |sh```

This command will run on the back the full installation, it is using the apt-get command and a fully automated deployment. It's taking a while, so be patient!

Once, done, I recommend to run the follwing command line which nwill allow the default ```pi``` user to run docker containers whithout having to use the ```sudo``` command.

```sudo usermod -aG docker pi``` where ```pi``` is the default user. If you are running using another user, just change it.

# Step 2: Exporting your prediction model

Depending what you are using to generate your prediction model, you will have to export your model in an as compact as possible model compatible with tensor flow.
For example, if you are using [CustomVision.ai](https://www.customvision.ai), select the compact versions, train your model, then click on export and select any of the docker export for example. Details [in the docs here](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/export-your-model).

The file will be called ```model.pb```.

You will need as well to create a file called ```labels.txt``` containing the class you want to predict. It has of course to be in the same order as in your model and containing the same number of classes!

In this exmaple, I'm classifing pizza and hotdogs. So the model has been trained to classify pizza and hotdogs. My ```labels.txt``` file will contains 3 lines, one "pizza", one "hotdog" and one empty one.

Make sure you'll replace in the ```app```directory the model and the label file. You don't have to touch the rest of the code if you're using the same names.

Please note that the app which is given as an example is coming from [CustomVision.ai](https://www.customvision.ai). It is based from the docker export. The ```app.py``` file have been sligtly modify to support ```png``` files. Files need to be open in RGB mode like that:
```PY
img = Image.open(imageData).convert('RGB')
```

# Step 3: Creating the Raspberry Pi image

This is where the complicated things are starting. To run the prediction on the RPI, we will need the following elements:
* Python 3 and Pip - well, that's the basic :-)
* tensorflow - this is the core engine to make the prediction
* Pillow - we'll need this librarie for the image manipulation, resizing, normalizing. Other libraries can be installed to manipylate the images but this is a very popular and simple to use. Unfortunately, as wel will see later on, it does require quite some dependencies and it will be tricky on the RPI
* numpy - the must have to pass arrays to tensorflow.
* flask - to easilly handle a REST API call

If you'll try to install them just like that, you'll fail. They need to be install in a specific order and especially Pillow which require jpeg manipulation libraries.

The right order is to first install those components:
* python3
* python3-pip
* build-essential
* python3-dev
* zlib1g-dev
* libjpeg-dev
* wget

Once done, we can install the advanced Python libraries thru a classical ```pip install```:
* pillow
* numpy
* flask
You'll find those libraries in the file called ```requirements.txt``` in the app directory.

And finally, we willbe able to install tensorflow. The trick to install tensorflow is to use the gear file and install it with pip as well:

```pip install http://ci.tensorflow.org/view/Nightly/job/nightly-pi-python3/122/artifact/output-artifacts/tensorflow-1.5.0-cp34-none-any.whl```

So all up, our Dockerfile will look like:
```CMD
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
```

Now you are ready to build the image. For this, use the following command in the directory where you have the Dockerfile:

```sudo docker build -t <name_image> .```

```<name_image>``` is a unique name you will need to give to your image. It will make your life easier when you'll want to start your container.

# Step 4: Patience... and troubleshooting

As you will see, you will have to be very very very very patient while the container will be built for your Raspberry. 

## Option 1: If all goes right

You'll be ready to run the container and test it!

## Option 2: In case of error

In some cases with a weak internet, the download of source code can be corrupted or incomplete, in this case, the build will stop.

Solution: slive the problem!!
Create intermediate docker containers and use them incrementally. For example, create a first container before installing tensorflow. So create a first Dockerfile like that:

```CMD
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
               && pip install -r /app/requirements.txt 
```

Run the command: ```sudo docker build -t myinterim .```

If all goes right, **replace** your Dockerfile by the next one:

```CMD
FROM myinterim

RUN pip install http://ci.tensorflow.org/view/Nightly/job/nightly-pi-python3/122/artifact/output-artifacts/tensorflow-1.5.0-cp34-none-any.whl

# Expose the port
EXPOSE 80

# Set the working directory
WORKDIR /app

# Run the flask server for the endpoints
CMD ["python3","app.py"]
```

And run the command: ```sudo docker build -t <name_image> .``` where ```<name_image>``` is a unique name you will need to give to your image

If it didn't work with the first file, slice the problem a bit more, create another intermediate image, and just build on top of your images. That should help you understanding from where the problem is coming.

# Step 5: Runing the container and making the prediction

## Launching the container

To run a container, just use the command: 

```docker run -p 127.0.0.1:80:80 -d <name_image>``` where ```<name_image>``` is a unique name you gave to your image.

this will take couple of second, you'll get back the prompt. 

**PATIENCE**, wait couple of more seconds, the container need to start, flask needs to start, the model needs to start. 

## Checking the container is running

You can check that the container is running by using the command:

```docker ps```

You should be able to see your container running like this:
```CMD
CONTAINER ID        IMAGE                           COMMAND                  CREATED             STATUS              PORTS                  NAMES
62ba5a90ae4f        ellerbach/tensor_pillow_flask   "/usr/bin/entry.sh p…"   17 minutes ago      Up 16 minutes       127.0.0.1:80->80/tcp   nervous_kapitsa
```

## Troubleshouting

Use the command ```docker logs <name_image>``` to check what is happening. Remember, ```<name_image>``` is a unique name you gave to your image.

Read the logs, they'll tell you what is wrong. Most common mistakes are names of the models, invalid model formats, empty labels file.

## Testing with a real image

From the RPI, just download and save an image you want to check and run the following command from the directory where you saved the image:

```curl -X POST http://127.0.0.1/image -F imageData=@imagetest.jpg``` where ```imagetest.jpg``` is the name of the picture you want to test.

You should get a result like this:
```JSON
{"created":"2018-06-18T06:45:01.546493","id":"","iteration":"","predictions":[{"boundingBox":null,"probability":1.700000069604357e-07,"tagId":"","tagName":"hotdog"},{"boundingBox":null,"probability":1.0,"tagId":"","tagName":"pizza"}],"project":""}
```
In this case, I tested with a pizza image and it found out that it is a pizza :-)

If you are testing from another computer, use the same but change the IP address:

```curl -X POST http://raspberry_ip_address/image -F imageData=@imagetest.jpg``` where ```raspberry_ip_address``` is the ip address or the name of your Raspberry and ```imagetest.jpg``` is the name of the picture you want to test.

You can use Postman or any other tool or code to post the picture you want to analyze.

And you will of course get the same results.

# Step 6: Use case

One of the best use case is when you couple this docker file with a webcam, taking pictures and recognizing objects on the fly.

You can as well deploy it using Azure IoT Edge. You can of course enrich the example, add more code.

Feedbacks welcome!