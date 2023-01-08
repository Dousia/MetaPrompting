import os
import argparse
import random

import torch
import numpy as np
# import datetime

import dataloader as loader
from model import MetaTransformerModelWrapper
from utils import tprint

def parse_args():
    parser = argparse.ArgumentParser(
        description="MetaPrompting")

    # data configuration
    parser.add_argument("--data_path", type=str, default="data/",
                        help="path to dataset")
    parser.add_argument("--dataset", type=str, default="huffpost",
                        help="name of the dataset. "
                             "Options: [20newsgroup, amazon, huffpost, reuters]")
    parser.add_argument("--stored_episodes_dir", default="./episode_data/",
                        help="path to stored episode, "
                             "if given, will directly load the episode from the path, "
                             "otherwise new episodes will be built from training data.")
    parser.add_argument("--output_dir", default="./output_dir/",
                        type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written")

    # model configuration
    parser.add_argument("--pretrained_model", default="bert",
                        help="use PLM embedding (only available for sent-level datasets: huffpost, fewrel")
    parser.add_argument("--model_type", default="bert-base-uncased")
    parser.add_argument("--pretrained_cache_dir", default="./pretrained_models/",
                        type=str, help="path to the cache_dir of transformers")
    parser.add_argument("--embed_size", default=768, type=int,
                        help="Prompt tokens embedding size")

    # task configuration
    parser.add_argument("--n_way", type=int, default=5, help="the number of classes for each task")
    parser.add_argument("--k_shot", type=int, default=5,
                        help="the number of support examples for each class for each task")
    parser.add_argument("--l_query", type=int, default=15,
                        help="the number of query examples for each class for each task")

    # train/test configuration
    parser.add_argument("--train_epochs", type=int, default=300,
                        help="max num of training epochs")
    parser.add_argument("--val_episodes", type=int, default=100,
                        help="episodes sampled during each validation")
    parser.add_argument("--test_episodes", type=int, default=1000,
                        help="episodes sampled during each test")

    # training options
    parser.add_argument("--eval_every_step", type=int, default=100, help="eval_every_step")
    parser.add_argument("--seed", type=int, default=1999, help="seed")
    parser.add_argument("--patience", type=int, default=5, help="patience")
    parser.add_argument("--clip_grad", type=float, default=1., help="gradient clipping")
    parser.add_argument("--prompt_template", type=int, default=0)
    parser.add_argument("--n_adapt_epochs", type=int, default=15,
                        help="the number of adaption epochs on evaluation few-shot episodes")
    parser.add_argument("--inner_steps", type=int, default=15, help="the number of adaption steps")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lm_learning_rate", type=float, default=1e-5)
    parser.add_argument("--prompt_learning_rate", type=float, default=5e-5)
    parser.add_argument("--no_train", type=int, default=0)
    parser.add_argument("--no_eval", type=int, default=0)
    parser.add_argument("--no_load", type=int, default=0)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    args.data_path = args.data_path + args.dataset + ".json"
    config_str = str(args.n_way) + 'way' + str(args.k_shot) + 'shot_' + str(args.inner_steps) + 'ada' + \
                 str(args.seed) + 'seed' + str(args.prompt_template) + 'template'

    tprint(config_str)

    args.output_dir = os.path.join(args.output_dir, args.dataset + '_' + config_str + '')
    set_seed(args.seed)

    stored_episodes_dir = os.path.join(args.stored_episodes_dir, args.dataset,
                                       str(args.n_way) + 'way' + str(args.k_shot) + 'shot')
    if not (os.path.exists(os.path.join(stored_episodes_dir, "train.json")) and
            os.path.exists(os.path.join(stored_episodes_dir, "val.json")) and
            os.path.exists(os.path.join(stored_episodes_dir, "test.json"))):
        train_data, val_data, test_data, train_classes, val_classes, test_classes = loader.load_dataset(args)

    else:
        train_data, val_data, test_data = [], [], []

    train_episodes = loader.build_episodes(train_data, n_way=args.n_way, k_shot=args.k_shot, l_query=args.l_query,
                                           n_episodes=6000, stored_episodes=os.path.join(stored_episodes_dir,
                                                                                          "train.json"))
    val_episodes = loader.build_episodes(val_data, n_way=args.n_way, k_shot=args.k_shot, l_query=args.l_query,
                                         n_episodes=1000, stored_episodes=os.path.join(stored_episodes_dir,
                                                                                       "val.json"))

    test_episodes = loader.build_episodes(test_data, n_way=args.n_way, k_shot=args.k_shot, l_query=args.l_query,
                                          n_episodes=3000, stored_episodes=os.path.join(stored_episodes_dir,
                                                                                        "test.json"))

    n_gpu = torch.cuda.device_count()
    per_gpu_train_batch_size = args.batch_size // n_gpu

    print("\n=======================================================")
    print("==================Meta training stage==================")
    print("=======================================================\n")

    wrapper = MetaTransformerModelWrapper(args)

    if args.no_train == 0:
        wrapper.train(train_data=train_episodes,
                      eval_data=val_episodes,
                      task_output_dir=args.output_dir,
                      per_gpu_train_batch_size=per_gpu_train_batch_size,
                      n_gpu=n_gpu,
                      eval_every_step=args.eval_every_step,
                      n_adapt_epochs=args.n_adapt_epochs,
                      n_train_epochs=args.train_epochs,
                      weight_decay=0.1,
                      n_inner_steps=3,
                      lm_learning_rate=args.lm_learning_rate,
                      prompt_learning_rate=args.prompt_learning_rate,
                      max_grad_norm=args.clip_grad
                      )

    if args.no_eval == 0:
        print("\n=======================================================")
        print("==================Meta testing stage===================")
        print("=======================================================\n")

        print("Loading best model")
        pretrained_model_path = os.path.join(args.output_dir, 'best') if args.no_load == 0 else None
        average_scores = wrapper.eval(eval_data=test_episodes,
                                      # classes=test_classes,
                                      pretrained_model_path=pretrained_model_path,
                                      n_eval_episodes=args.test_episodes,
                                      per_gpu_eval_batch_size=per_gpu_train_batch_size,
                                      n_gpu=n_gpu,
                                      n_adapt_epochs=args.n_adapt_epochs,
                                      lm_learning_rate=args.lm_learning_rate,
                                      prompt_learning_rate=args.prompt_learning_rate,
                                      )
        print(average_scores)
        print("\n=======================================================")


if __name__ == "__main__":
    main()
