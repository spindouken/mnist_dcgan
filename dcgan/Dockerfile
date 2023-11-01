#   docker run -it --rm --gpus all -p 8888:8888 -v $(pwd):/usr/src/app dcgan

# Use an official TensorFlow runtime as a parent image
FROM tensorflow/tensorflow:2.6.0-gpu

# Set the working directory to /app
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /app
COPY . /usr/src/app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 8888 available to the world outside this container
EXPOSE 8888

