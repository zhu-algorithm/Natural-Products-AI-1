FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir '.[train]'
ENTRYPOINT ["npfm"]
