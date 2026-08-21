FROM node:20-slim

WORKDIR /app

# Install node modules
COPY package.json package-lock.json* ./
RUN npm install

# Copy application code
COPY . .

# Expose frontend dev port
EXPOSE 5173
