# From Docker image built with Dockerfile.fenicsx-mixed-dim
FROM ceciledc/fenicsx_mixed_dim:latest

# Upgrade setuptools
USER root
RUN rm -rf /usr/lib/python3/dist-packages/setuptools*

RUN pip3 install setuptools
RUN pip3 install networkx
RUN pip3 install matplotlib
RUN pip3 install scipy
RUN pip3 install pandas

# Clone and install networks-fenicsx
RUN git clone https://github.com/cdaversin/networks_fenicsx.git && \
    cd networks_fenicsx && \
    python3 -m pip install .
