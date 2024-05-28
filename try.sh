python main.py \
--dataset_dir ./datasets \
--batch_size 128 \
--epochs 15 \
--lr 0.5 --wd 0 \
--seed 0 \
--fig_name lr=0.5-lr_sche-wd=0-mixup.png

python main.py --dataset_dir ./datasets --batch_size 128 --epochs 15 --lr 0.5 --wd 0 --seed 0 --fig_name lr=0.5-lr_sche-wd=0.png
python main.py --dataset_dir ./datasets --batch_size 128 --epochs 15 --lr 0.5 --wd 0 --seed 0 --lr_scheduler --fig_name lr=0.05-lr_sche.png