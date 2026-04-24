FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine

# Copy build files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx template
COPY nginx.conf /etc/nginx/templates/default.conf.template

# Nginx will automatically substitute $PORT in templates!
EXPOSE 8080

# Start nginx (it auto-processes templates)
CMD ["nginx", "-g", "daemon off;"]