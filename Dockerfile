FROM mambaorg/micromamba:1.5.8

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all --yes

WORKDIR /workspace
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace
RUN python -m pip install --no-deps -e .

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["python", "-c", "import hyperspatial; print(hyperspatial.__version__)"]

