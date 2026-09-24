FROM python:3.10-slim-bullseye
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/
COPY --from=eclipse-temurin:17-jre-focal /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="/opt/java/openjdk/bin:$PATH"
ENV UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY . .
RUN uv sync --locked
CMD ["uv", "run", "--locked", "pytest", "-v"]
