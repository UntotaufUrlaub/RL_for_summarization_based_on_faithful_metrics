FROM python:3.9

RUN pip install summac
RUN python -c "import nltk; nltk.download('punkt')"

CMD python /add_score_summac.py

# docker build -t summac:docker_batch .
# docker run --rm --gpus all -v /home/sturmja/aggrefact_summac/add_score_summac.py:/add_score_summac.py -v /home/sturmja/aggre_fact_final.csv:/data/aggre_fact_final.csv -v /home/sturmja/aggre_fact_timebenchmark_wip.csv:/data/aggre_fact_timebenchmark_wip.csv -v /home/sturmja/add_score_batched.py:/add_score.py summac:docker_batch