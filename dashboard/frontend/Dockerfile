# Use an official Node.js runtime as a parent image
# Choose a version that matches your project's dependencies
FROM node:22-bullseye-slim

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json (or yarn.lock)
COPY package*.json ./

# Install dependencies
# Use npm ci for cleaner installs in CI/Docker environments
RUN npm ci

# Copy the rest of the application code
COPY . .

# Build the application for production
RUN npm run build

# Expose the port the app runs on
# Vite's preview server defaults to port 4173
EXPOSE 8831

# Command to run the application in production
# 'npm run preview' uses Vite's preview server to serve the built files
CMD ["npm", "run", "preview"]
