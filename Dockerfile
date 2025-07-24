# Use Miniconda3 as base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment.yml and project files
COPY environment.yml ./
COPY requirements.txt ./
COPY . /app

# Create the conda environment
RUN conda env create -f environment.yml

# Activate the environment and install pip dependencies (if any)
RUN /bin/bash -c "source activate HF && pip install --no-cache-dir -r requirements.txt"

# Make sure conda environment is activated by default
SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/envs/HF/bin:$PATH

# Set environment variable for OpenAI API key (user should pass at runtime)
ENV OPENAI_API_KEY=""

# Default command
CMD ["python", "pipeline.py"] 