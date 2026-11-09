FROM python:3.12-slim AS wheel-builder

WORKDIR /build
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml hatch_build.py LICENSE ./
COPY src/ src/
# Fail the image build rather than silently shipping without the measured
# native sampler. Keep compilers/headers out of the final runtime image.
RUN LIBREWXR_REQUIRE_NATIVE=1 pip wheel --no-deps --no-cache-dir --wheel-dir /wheels .

FROM python:3.12-slim
WORKDIR /app
COPY --from=wheel-builder /wheels /wheels
# Install the built wheel with MCP enabled, without copying compiler tools.
RUN set -- /wheels/librewxr-*.whl \
    && pip install --no-cache-dir "${1}[mcp]" \
    && rm -rf /wheels

LABEL org.opencontainers.image.title="LibreWXR" \
      org.opencontainers.image.description="Self-hostable Rain Viewer API replacement - radar composites from public sources, served as a Rain Viewer-compatible JSON/tile API." \
      org.opencontainers.image.source="https://github.com/JoshuaKimsey/LibreWXR" \
      org.opencontainers.image.url="https://github.com/JoshuaKimsey/LibreWXR" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later"

EXPOSE 8080

CMD ["python", "-m", "librewxr.main"]
