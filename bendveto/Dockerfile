# lf-013: the lf-009 image with the bend veto wrapped around the driver. No file of the base image changes.
# BENDVETO_ARM = off (identical to lf-009) | shadow (log only) | active
FROM py123d-garage-ltfv6:lf-009
COPY bendveto.py bendveto_hook.py /opt/bendveto/
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 BENDVETO_ARM=active PYTHONPATH=/opt/bendveto
ENTRYPOINT ["python", "/opt/bendveto/bendveto_hook.py"]
