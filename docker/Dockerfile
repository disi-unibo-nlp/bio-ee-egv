#https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_20-12.html#rel_20-12
FROM nvcr.io/nvidia/pytorch:20.09-py3
LABEL maintainer="DISI NLU Research Group"

# Zero interaction (default answers to all questions)
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /event-graph-verbalization/

# Add requirements file
ADD requirements.txt /event-graph-verbalization/

# Install dependencies
RUN apt-get update -y && \
    apt-get install -y curl \
                       git \
                       bash \
                       nano \
		       ssmtp \
                       subversion && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install wrapt --upgrade --ignore-installed
RUN pip install -r requirements.txt
RUN pip install --upgrade jax jaxlib==0.1.57+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip install jax==0.2.8
RUN pip install tensorboard==2.6.0
RUN pip install gdown
RUN pip install rouge_score
RUN apt-get update && apt-get install cron && apt-get install ssmtp && apt-get install lsof
RUN conda install gh --channel conda-forge
ENTRYPOINT gh auth login

# Back to default frontend
ENV DEBIAN_FRONTEND=dialog
