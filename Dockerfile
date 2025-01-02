# Use a Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /application

# Install system dependencies (including git)
RUN apt-get update && apt-get install -y --no-install-recommends git

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clean up APT when done
RUN apt-get remove -y git && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Command to run the application
CMD ["python", "application.py"]
