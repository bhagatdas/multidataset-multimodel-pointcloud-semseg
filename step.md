python preprocess_datasets.py --dataset Toronot-3D --data_dir data/
python preprocess_datasets.py --dataset Paris-Lille-3D --data_dir data/

python preprocess_datasets.py --dataset both --data_dir data/ --num_segments 16 --train_segments 14 --val_segments 1 --test_segments 1

# set your value in config
python train.py


