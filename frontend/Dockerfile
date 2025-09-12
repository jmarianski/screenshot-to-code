FROM node:22-bullseye-slim

# Set the working directory in the container
WORKDIR /app

# Copy package.json and yarn.lock
COPY package.json yarn.lock /app/

# Set the environment variable to skip Puppeteer download
ENV PUPPETEER_SKIP_DOWNLOAD=true

# Install dependencies
RUN yarn install


# Copy the current directory contents into the container at /app
COPY ./ /app/

# Expose port 5173 to access the server
EXPOSE 5173

# Create entrypoint script that installs dependencies and starts dev server
RUN echo '#!/bin/bash\necho "Installing dependencies..."\nyarn install\necho "Starting development server..."\nyarn run dev --host 0.0.0.0' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Use entrypoint script instead of direct CMD
CMD ["/app/entrypoint.sh"]
