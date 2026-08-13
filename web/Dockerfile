# Многоступенчатая сборка: dev — Vite с HMR, prod — статика за nginx.
# Цель выбирается в compose (build.target), дефолт — prod.

FROM node:22-alpine AS deps
WORKDIR /app
# npm ci ставит строго по package-lock.json и падает, если он разошёлся с
# package.json — в отличие от npm install, который молча ставил не то
COPY package.json package-lock.json ./
RUN npm ci

# --- разработка: dev-сервер Vite, исходники монтируются в compose ---
FROM deps AS dev
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# --- сборка статики ---
FROM deps AS builder
COPY . .
RUN npm run build

# --- прод: собранный бандл за nginx, который же проксирует /api ---
FROM nginx:1.27-alpine AS prod
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
