# Robust 

## Commands

### prepare data (MLM)
``` shell
python main.py --do_prepare --task cola --mlm_task
```

### train data (MLM)
``` shell
python main.py --do_train --task cola --mlm_task --batch_size 16 --lr 5e-4
```