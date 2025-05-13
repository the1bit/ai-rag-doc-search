FROM ollama/ollama

# Install curl
RUN apt-get update && apt-get install -y curl

COPY start.sh /start.sh
RUN chmod +x /start.sh

ENTRYPOINT ["/start.sh"]
