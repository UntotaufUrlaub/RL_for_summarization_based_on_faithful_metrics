#!/bin/sh
cd stanford-corenlp-full-2018-02-27
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000 > /dev/null 2>&1 &

python -c "import time; print('start:', time.time())"

cd ..
python create_input_file.py

cd ../factuality-datasets
python evaluate_generated_outputs.py --model_type electra_dae --model_dir ../DAE_xsum_human_best_ckpt --input_file ../sample_cache.txt | grep 'Sent-level pred:' > ../cache.txt

cd ..
python add_score_dae.py

python -c "import time; print('end:  ', time.time())"