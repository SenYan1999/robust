import argparse

parser = argparse.ArgumentParser()

# main mode
parser.add_argument('--do_prepare', action='store_true')
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--random_init', action='store_true')
parser.add_argument('--adv_train', action='store_true')
parser.add_argument('--adv_fix_embedding', action='store_true')
parser.add_argument('--random_train', action='store_true')
parser.add_argument('--fp16', action='store_true')

# resume training
parser.add_argument('--resume', action='store_true', help='resume training')
parser.add_argument('--saved_model', type=str, default='')

# distributed
parser.add_argument('--multiprocessing_distributed', action='store_true')
parser.add_argument('--ngpus_per_node', type=int, default=2)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:12345')
parser.add_argument('--world_size', type=int, default=1)

# data prepare
parser.add_argument('--data_dir', type=str, default='glue_data')
parser.add_argument('--task', type=str, default='CoLA')
parser.add_argument('--max_len', type=int, default=150)
parser.add_argument('--sample_rate', type=float, default=1.0)
parser.add_argument('--clip_dataset', action='store_true', help='clip the total dataset to reduce the training time')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--pad_idx', type=int, default=0)

# mlm task
parser.add_argument('--mlm_task', action='store_true')
parser.add_argument('--mask_token_id', type=int, default=103)
parser.add_argument('--replace_prob', type=float, default=0.8)
parser.add_argument('--replace_random_token_prob', type=float, default=0.1)
parser.add_argument('--mask_prob', type=float, default=0.15)

# model
parser.add_argument('--use_mlm_pretrained_model', action='store_true')
parser.add_argument('--bert_name', type=str, default='bert-base-uncased')
parser.add_argument('--num_hidden_layers', type=int, default=4)
parser.add_argument('--num_attention_heads', type=int, default=4)
parser.add_argument('--hidden_size', type=int, default=768)
parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
parser.add_argument('--drop_p', type=float, default=0.1)
parser.add_argument('--num_class', type=int, default=3)

# lstm model
parser.add_argument('--lstm_d_embed', type=int, default=256)
parser.add_argument('--lstm_d_hidden', type=int, default=512)

# adv
parser.add_argument('--adv_k', default=1, type=int)
parser.add_argument('--adv_epsilon', default=1e-6, type=float)
parser.add_argument('--adv_eta', default=1e-3, type=float)
parser.add_argument('--adv_alpha', default=1, type=float)
parser.add_argument('--adv_step_size', default=1e-3, type=float)
parser.add_argument('--adv_noise_gamma', default=1e-6, type=float)
parser.add_argument('--adv_project_type', default='inf', type=str)

# optimizer
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--adam_epsilon', type=float, default=1e-6)
parser.add_argument('--adam_beta', type=tuple, default=(0.9, 0.98))
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--warmup_proportion', type=float, default=0.06)
parser.add_argument('--drop_out', type=float, default=0.1)
parser.add_argument('--print_interval', type=int, default=20)

# save & log
parser.add_argument('--log_path', type=str, default='log/log.log')
parser.add_argument('--log_dir', type=str, default='log/')
parser.add_argument('--save_dir', type=str, default='save_model/')
parser.add_argument('--tensorboard_path', type=str, default='log/tensorboard')

# parse args
args = parser.parse_args()
