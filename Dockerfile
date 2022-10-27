# We choose ubuntu 22.04 as our base docker image
FROM ubuntu:22.04

# Install pip and git with apt
RUN apt-get update && \
    apt-get install -y python3-pip git

# We upgrade pip and setuptools
# We remove the version of setuptools install via apt, as it is outdated
RUN python3 -m pip install pip setuptools --upgrade
RUN apt-get purge -y python3-setuptools


# We set the working directory to install docker dependencies
WORKDIR /tmp/

# We remove the contents of the temporary directory to minimize the size of the image
RUN rm -rf /tmp

# Create user with a home directory
ARG NB_USER
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

# Copy current directory
WORKDIR ${HOME}
COPY . ${HOME}

# Install the Python-module
RUN python3 -m pip install ${HOME}

# Change ownership of home directory
USER root
RUN chown -R ${NB_UID} ${HOME}

USER ${NB_USER}
ENTRYPOINT []
