FROM python:3.9

RUN apt-get update && apt-get install git -y --no-install-recommends && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone -b compare_two_texts --single-branch https://github.com/UntotaufUrlaub/QuestEval_Eval1Summary_WithinDocker.git QuestEval_CompareTwoTexts_WithinDocker
WORKDIR QuestEval_CompareTwoTexts_WithinDocker
RUN pip install -e .

CMD python /add_score_questeval24.py

# docker build -f Dockerfile -t aggrefact_questeval24:0.1 .
# docker run --rm --gpus all -v /home/jakob/Desktop/Storage/uni/22SoSe/MA/Code/metriken/QuestEval/aggrefact_questeval24/add_score_questeval24.py:/add_score_questeval24.py -v /home/jakob/Desktop/Storage/uni/22SoSe/MA/Code/human_alignment/AggreFact-main/data/:/data/ -v /home/jakob/Desktop/Storage/uni/22SoSe/MA/Code/human_alignment/AggreFact-main/add_score.py:/add_score.py aggrefact_questeval24:0.1