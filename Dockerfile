# Use official Python image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app


# Copy requirements and install dependencies
COPY requirements.txt ./
# Install build tools for scikit-surprise
RUN apt-get update \
	&& apt-get install -y --no-install-recommends build-essential python3-dev \
	&& pip install --no-cache-dir -r requirements.txt \
	&& apt-get remove -y build-essential python3-dev \
	&& apt-get autoremove -y \
	&& rm -rf /var/lib/apt/lists/*


# Copy the rest of the application code
COPY . .

# Explicitly copy model files (pickle files)
COPY svd_full_model.pkl ./
COPY svdpp_full_model.pkl ./
COPY optimized_svdpp.pkl ./

# Expose port (change if your app uses a different port)
EXPOSE 8888

# Set the default command to run your main application
CMD ["python", "main.py"]
