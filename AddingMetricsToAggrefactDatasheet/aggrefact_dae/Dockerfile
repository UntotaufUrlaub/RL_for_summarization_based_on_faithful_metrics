FROM python:3.6

# fetch the corenlp server - version 3.9.1.
RUN curl -O -L http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip
RUN apt-get update && apt-get upgrade -y && apt-get install -y unzip default-jre-headless
RUN unzip stanford-corenlp-full-2018-02-27.zip

# checkout the project.
RUN git clone https://github.com/tagoyal/factuality-datasets.git
# switch device to cpu, as there seems to be a missmatch between the torch, cuda and gpu.
RUN sed -i '108c\    device = torch.device("cpu")' factuality-datasets/evaluate_generated_outputs.py

# Inject the pretrained model, which needs to be downloaded manually first (and in some cases unziped), into the docker.
COPY factuality-datasets-main/DAE_xsum_human_best_ckpt /DAE_xsum_human_best_ckpt

# install the requirements.txt
RUN pip install -r factuality-datasets/requirements.txt
RUN pip install pandas

CMD ./run.sh

# docker build -t dae:batched .
# docker run --rm --gpus all -v /home/sturmja/aggrefact_dae/create_input_file.py:/create_input_file.py -v /home/sturmja/aggrefact_dae/run.sh:/run.sh -v /home/sturmja/aggre_fact_final.csv:/aggre_fact_final.csv -v /home/sturmja/aggrefact_dae/add_score_dae.py:/add_score_dae.py dae:batched
