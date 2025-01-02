# Description: Dockerfile for building the image for the application
FROM python:3.12

# Set the working directory in the container
WORKDIR /application

# Copy the current directory contents into the container at /application
COPY . /application

# Install AWS CLI
RUN apt update -y && apt install awscli -y 

# Install Python dependencies
RUN pip install -r requirements.txt

# Run the application
CMD [ "Python3","application.py" ]
