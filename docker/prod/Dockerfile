FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
  FLIT_ROOT_INSTALL=1 \
  STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
  STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \ 
  PORT=8551

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["sh", "-c", "sh setup.sh && streamlit run app.py"]

