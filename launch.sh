ENV='CartPole-v1'
TRAIN_EPS=100
LR=0.002
DF=0.98
SEEDS=(1 2 3)
DIR=out/$ENV/EPS$TRAIN_EPS-LR$LR-DF$DF

for seed in ${SEEDS[@]}; do
    ./venv/bin/python3 run.py --train_eps $TRAIN_EPS --n_layers 3 --seed $seed --test_eps 30 --lr $LR --batch_size 256 --warm_up_steps 500 --epsilon_hlife 1500 --save_dir $DIR/$seed --discount_factor $DF 
done
