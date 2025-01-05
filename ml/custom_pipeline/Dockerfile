FROM python:3.12.0

WORKDIR /app

# Copy the necessary dependencies to the container
COPY requirements.txt ./

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all to the container
COPY . .

# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]