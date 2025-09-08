
FROM python:3.10-slim

EXPOSE 8080

WORKDIR /home/runner

RUN useradd --uid 10000 runner

COPY requirements.txt requirements.txt
COPY app/app.py app.py
COPY app/model.py model.py
COPY deployment-config.yaml deployment-config.yaml
COPY startup.sh startup.sh
RUN chmod +x ./startup.sh \
    && chown -R runner:runner /home/runner

USER runner

ENV PATH="/home/runner/.local/bin:/home/runner:${PATH}"

RUN pip install --no-cache-dir -r requirements.txt

CMD ["/home/runner/startup.sh"]