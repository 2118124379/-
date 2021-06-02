CUDA_LAUNCH_BLOCKING=1 
python main.py --train electronic --save_dir ./data
#python main.py --test electronic --save_dir ./data --review_model ./data/model/2_512_64/56_review_model.tar --sketch_model ../sketch/data/electronic/model/2_512_64/6_sketch_model.tar --topic_model ../topic/data/electronic/model/2_512_1024/64_topic_model.tar
