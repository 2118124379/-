CUDA_LAUNCH_BLOCKING=1 
python main.py --train electronic --save_dir ./data/electronic
#python main.py --test electronic --save_dir ./data/electronic --sketch_model ./data/electronic/model/2_512_64/4_sketch_model.tar --topic_model ../topic/data/electronic/model/2_512_1024/63_topic_model.tar
