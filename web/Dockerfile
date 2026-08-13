# Multi-stage build: dev is Vite with HMR, prod is static files behind nginx.
# The target is selected in compose (build.target); the default is prod.

FROM node:22-alpine AS deps
WORKDIR /app
# npm ci installs strictly from package-lock.json and fails if it drifted from
# package.json — unlike npm install, which silently installed the wrong thing
COPY package.json package-lock.json ./
RUN npm ci

# --- dev: the Vite dev server, sources are mounted in compose ---
FROM deps AS dev
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# --- static build ---
FROM deps AS builder
COPY . .
RUN npm run build

# --- prod: the built bundle behind nginx, which also proxies /api ---
FROM nginx:1.27-alpine AS prod
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
