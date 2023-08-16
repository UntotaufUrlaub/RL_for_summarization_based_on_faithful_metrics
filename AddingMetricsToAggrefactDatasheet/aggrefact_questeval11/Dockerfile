FROM python:3.7

RUN apt-get update && apt-get install git -y --no-install-recommends && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/UntotaufUrlaub/QuestEval_Eval1Summary_WithinDocker.git
WORKDIR QuestEval_Eval1Summary_WithinDocker
RUN pip install -e .

CMD python /add_score_questeval.py

# docker build -f Dockerfile -t aggrefact_questeval11:0.1 .
# docker run --rm --gpus all -v /home/sturmja/aggrefact_questeval11/add_score_questeval.py:/add_score_questeval.py -v /home/sturmja/aggre_fact_final.csv:/aggre_fact_final.csv -v /home/sturmja/aggre_fact_timebenchmark_wip.csv:/aggre_fact_timebenchmark_wip.csv -v /home/sturmja/add_score.py:/add_score.py aggrefact_questeval11:0.1