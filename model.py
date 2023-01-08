import json
import jsonpickle
import os
from typing import List, Dict, Optional
import copy

import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, Dataset
from tqdm import trange, tqdm
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup
from transformers.data.metrics import simple_accuracy

from utils import tprint
from dataloader import TASK_CLASSES, MODEL_CLASSES

from meta.algrithm import MAML


class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        tensors.values()

        assert all(next(iter(tensors.values())).size(0) == tensor.size(0) for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)


class ContinuousPrompt(torch.nn.Module):
    def __init__(self, args, tokenizer):
        super(ContinuousPrompt, self).__init__()
        self.config = args
        self.tokenizer = tokenizer
        self.embed_size = args.embed_size
        self.hidden_size = self.embed_size
        if hasattr(self.config, 'prompt_template'):
            self.prompt_length = TASK_CLASSES[args.dataset]['prompt_length'][self.config.prompt_template]
        else:
            self.prompt_length = TASK_CLASSES[args.dataset]['prompt_length'][0]

        config_class = MODEL_CLASSES[self.config.pretrained_model]['config']
        model_config = config_class.from_pretrained(
            args.model_type)

        model_class = MODEL_CLASSES[self.config.pretrained_model]['mlm']
        self.model = model_class.from_pretrained(
            args.model_type,
            config=model_config)

        self.prompt_embeddings = torch.nn.Embedding(self.prompt_length, self.embed_size)

        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size,
                                       num_layers=2,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))


    def forward(self, inputs_embeds=None, attention_mask=None, token_type_ids=None, labels=None,
                output_hidden_states=False):
        return self.model(inputs_embeds=inputs_embeds,
                          attention_mask=attention_mask,
                          labels=labels,
                          token_type_ids=token_type_ids,
                          output_hidden_states=output_hidden_states)


class TransformerModelWrapper:
    """A wrapper around a Transformer-based language model."""

    def __init__(self, args):
        self.config = args
        tokenizer_class = MODEL_CLASSES[self.config.pretrained_model]['tokenizer']
        self.tokenizer = tokenizer_class.from_pretrained(
            args.model_type,
            cache_dir=args.pretrained_cache_dir if args.pretrained_cache_dir else None)
        self.model = ContinuousPrompt(args, self.tokenizer)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.cuda()
        self.prompt_template = args.prompt_template


    @classmethod
    def generate_verbalizer(cls, label_map):
        verbalizer = {}
        except_words = ['the', '&', 'and', ]
        convert_words = {}
        for label in label_map.keys():
            answers = label.split()
            for word in except_words:
                if word in answers:
                    answers.remove(word)
            for word in convert_words.keys():
                if word in answers:
                    answers.remove(word)
                    answers.append(convert_words[word])
            answers = [answer.lower() for answer in answers]
            verbalizer[label] = answers
        return verbalizer


    def refresh_label_map(self, label_map):
        self.label_map = label_map
        self.verbalizer = self.generate_verbalizer(label_map)
        self.mlm_logits_to_answer_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()


    def save(self, path: str) -> None:
        tprint("Saving models.")
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        model_to_save.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)

        state = {
            "prompt_embeddings": model_to_save.prompt_embeddings.state_dict(),
            "lstm_head": model_to_save.lstm_head.state_dict(),
            "mlp_head": model_to_save.mlp_head.state_dict()
        }

        save_path_file = os.path.join(path, "embeddings.pth")
        torch.save(state, save_path_file)


    @classmethod
    def from_pretrained(cls, path: str) -> 'TransformerModelWrapper':
        """Load a pretrained wrapper from a given path."""

        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)

        tokenizer_class = MODEL_CLASSES[wrapper.config.pretrained_model]['tokenizer']
        wrapper.tokenizer = tokenizer_class.from_pretrained(path)

        wrapper.model = ContinuousPrompt(wrapper.config, wrapper.tokenizer)
        model_class = MODEL_CLASSES[wrapper.config.pretrained_model]['mlm']
        wrapper.model.model = model_class.from_pretrained(path)

        save_path_file = os.path.join(path, "embeddings.pth")
        data = torch.load(save_path_file)
        wrapper.model.prompt_embeddings.load_state_dict(data["prompt_embeddings"])
        if "lstm_head" in data:
            assert ("mlp_head" in data)
            wrapper.model.lstm_head.load_state_dict(data["lstm_head"])
            wrapper.model.mlp_head.load_state_dict(data["mlp_head"])
        if "mlp" in data:
            wrapper.model.mlp_head.load_state_dict(data["mlp"])

        if torch.cuda.device_count() > 1:
            wrapper.model = torch.nn.DataParallel(wrapper.model)
        wrapper.model.cuda()
        return wrapper


    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, 'wrapper_config.json'), 'w') as f:
            f.write(jsonpickle.encode(self.config))


    @staticmethod
    def _load_config(path: str):
        with open(os.path.join(path, 'wrapper_config.json'), 'r') as f:
            return jsonpickle.decode(f.read())


    @staticmethod
    def eval_single_episode(episode,
                            pretrained_model,
                            per_gpu_eval_batch_size: int = 8,
                            n_gpu: int = 1,
                            n_adapt_epochs: int = 15,
                            gradient_accumulation_steps: int = 1,
                            weight_decay: float = 0.0,
                            lm_learning_rate: float = 1e-5,
                            prompt_learning_rate: float = 5e-5,
                            adam_epsilon: float = 1e-8,
                            max_grad_norm: float = 1,
                            ):
        task_wrapper = pretrained_model
        label_map = {label: i for i, label in enumerate(episode['labels'])}
        # print(label_map)
        task_wrapper.refresh_label_map(label_map)

        train_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        train_dataset = task_wrapper._generate_dataset(episode['support_set'])
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        t_total = n_adapt_epochs * len(train_dataloader)

        warmup_steps = int(t_total / 30)
        cur_model = task_wrapper.model.module if hasattr(task_wrapper.model, 'module') else task_wrapper.model
        optimizer, scheduler, embedding_optimizer, embedding_scheduler = task_wrapper.prepare_optimizer_scheduler(
            cur_model, t_total, weight_decay, lm_learning_rate, prompt_learning_rate, adam_epsilon, warmup_steps)

        task_wrapper.model.zero_grad()
        for i in range(n_adapt_epochs):
            for step, batch in enumerate(train_dataloader):
                task_wrapper.model.train()
                batch = {k: t.cuda() for k, t in batch.items()}
                loss = task_wrapper.mlm_train_step(batch)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    # TODO
                    torch.nn.utils.clip_grad_norm_(task_wrapper.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    embedding_optimizer.step()
                    embedding_scheduler.step()
                    task_wrapper.model.zero_grad()

        task_wrapper.model.eval()
        eval_dataset = task_wrapper._generate_dataset(episode['query_set'])
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu) * 6
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)
        preds = None
        out_label_ids, all_indices, question_ids = None, None, None
        for step, batch in enumerate(eval_dataloader):
            batch = {k: t.cuda() for k, t in batch.items()}
            labels = batch['labels']
            with torch.no_grad():
                logits = task_wrapper.mlm_eval_step(batch)
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        predictions = np.argmax(preds, axis=1)
        scores = {}
        metrics = ['acc']
        for metric in metrics:
            if metric == 'acc':
                scores[metric] = simple_accuracy(predictions, out_label_ids)

        return scores


    def eval(self,
             eval_data: List[InputExample],
             # classes: List[str],
             pretrained_model_path: str = None,
             n_eval_episodes: int = 100,
             per_gpu_eval_batch_size: int = 8,
             n_gpu: int = 1,
             n_adapt_epochs: int = 15,
             gradient_accumulation_steps: int = 1,
             weight_decay: float = 0.1,  #
             lm_learning_rate: float = 1e-5,
             prompt_learning_rate: float = 5e-5,
             adam_epsilon: float = 1e-8,
             max_grad_norm: float = 1,
             ) -> Dict:

        scores = []
        episode_iter = tqdm(eval_data[: n_eval_episodes], desc="Eval_episode")
        if pretrained_model_path is None:
            pretrained_wrapper = self.__class__(self.config)
        else:
            pretrained_wrapper = self.__class__.from_pretrained(pretrained_model_path)

        for step, episode in enumerate(episode_iter):
            task_wrapper = copy.deepcopy(pretrained_wrapper)
            episode_score = self.eval_single_episode(episode=episode,
                                                     pretrained_model=task_wrapper,
                                                     per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                                                     n_gpu=n_gpu,
                                                     n_adapt_epochs=n_adapt_epochs,
                                                     lm_learning_rate=lm_learning_rate,
                                                     prompt_learning_rate=prompt_learning_rate,
                                                     gradient_accumulation_steps=gradient_accumulation_steps,
                                                     weight_decay=weight_decay,
                                                     adam_epsilon=adam_epsilon,
                                                     max_grad_norm=max_grad_norm,
                                                     )
            scores.append(episode_score)

        def mean(nums):
            return sum(nums) / len(nums)

        average_scores = {}
        for key in scores[0].keys():
            average_scores[key] = mean([episode_score[key] for episode_score in scores])
        return average_scores


    def _generate_dataset(self, data: list, labelled: bool = True):
        features = self._convert_examples_to_features(data, labelled=labelled)
        feature_dict = {
            'input_ids': torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f["token_type_ids"] for f in features], dtype=torch.long),
            'labels': torch.tensor([f["label"] for f in features], dtype=torch.long),
            'mlm_labels': torch.tensor([f["mlm_labels"] for f in features], dtype=torch.long),
            'block_flag': torch.tensor([f["block_flag"] for f in features], dtype=torch.long)
        }
        return DictDataset(**feature_dict)


    def _convert_examples_to_features(self, examples: list, labelled: bool = True):
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index > 0 and ex_index % 10000 == 0:
                tprint("Writing example {}".format(ex_index))
            input_features = self.get_input_features(example, labelled=labelled)
            features.append(input_features)
        return features


    def generate_item(self, raw: list, task: dict):
        if hasattr(self.config, 'prompt_template'):
            prompt_template = self.config.prompt_template
        else:
            prompt_template = 0
        prompted_str = task['prompt'][prompt_template](raw)
        rough_block_flag = task['block_flag'][prompt_template]

        parts = [self.tokenizer.encode(string, add_special_tokens=False) for string in prompted_str]

        assert len(parts) == len(rough_block_flag)

        block_flag = [flag for text, flag in zip(parts, rough_block_flag) for _ in range(len(text))]
        token_ids = [id for ids in parts for id in ids]

        input_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(token_ids)
        block_flag = self.tokenizer.build_inputs_with_special_tokens(block_flag)

        block_flag = [item if item in [0, 1] else 0 for item in block_flag]
        assert len(input_ids) == len(block_flag)

        return input_ids, block_flag, token_type_ids


    def get_input_features(self, example: dict, labelled: bool):
        input_ids, block_flag, token_type_ids = self.generate_item(example['raw'], TASK_CLASSES[self.config.dataset])

        attention_mask = [1] * len(input_ids)
        padding_length = 512 - len(input_ids)

        if padding_length < -11:
            raise ValueError(f"Maximum sequence length is too small, got {len(input_ids)} input ids")

        input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        block_flag = block_flag + ([0] * padding_length)

        assert len(input_ids) == 512
        assert len(attention_mask) == 512
        assert len(token_type_ids) == 512
        assert len(block_flag) == 512

        label = self.label_map[example['label']] if example['label'] is not None else -100
        logits = [-1]

        if labelled:
            mlm_labels = self.get_mask_positions(input_ids)
        else:
            mlm_labels = [-1] * 512

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "label": label,
                "mlm_labels": mlm_labels,
                "logits": logits,
                "block_flag": block_flag}


    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        labels = [-1] * len(input_ids)
        for label_idx, input_id in enumerate(input_ids):
            if input_id == self.tokenizer.mask_token_id:
                labels[label_idx] = 1
        return labels


    def generate_default_inputs(self, batch: Dict[str, torch.Tensor], M=None) -> Dict[str, torch.Tensor]:

        input_ids = batch['input_ids']
        bz = batch['input_ids'].shape[0]
        block_flag = batch["block_flag"]
        if M is None:
            model = self.model.module if hasattr(self.model, 'module') else self.model
        else:
            model = M.module.module if hasattr(M.module, 'module') else M.module

        if self.config.pretrained_model == "albert":
            raw_embeds = model.model.albert.embeddings.word_embeddings(input_ids)
        elif self.config.pretrained_model == "bert":
            raw_embeds = model.model.bert.embeddings.word_embeddings(input_ids)
        elif self.config.pretrained_model == "roberta":
            raw_embeds = model.model.roberta.embeddings.word_embeddings(input_ids)

        replace_embeds = model.prompt_embeddings(
            torch.LongTensor(list(range(model.prompt_length))).cuda())
        replace_embeds = replace_embeds.unsqueeze(0)  # [batch_size, prompt_length, embed_size]

        replace_embeds = model.lstm_head(replace_embeds)[0]  # [batch_size, seq_len, 2 * hidden_dim]
        if model.prompt_length == 1:
            replace_embeds = model.mlp_head(replace_embeds)
        else:
            replace_embeds = model.mlp_head(replace_embeds).squeeze()

        blocked_indices = torch.nonzero(block_flag == 1).reshape((bz, model.prompt_length, 2))[:, :, 1]

        for bidx in range(bz):
            for i in range(blocked_indices.shape[1]):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]

        inputs = {'inputs_embeds': raw_embeds, 'attention_mask': batch['attention_mask']}

        if self.config.pretrained_model in ['bert']:
            inputs['token_type_ids'] = batch['token_type_ids']

        return inputs


    def mlm_train_step(self, labeled_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a MLM training step."""
        inputs = self.generate_default_inputs(labeled_batch)
        mlm_labels, labels = labeled_batch['mlm_labels'], labeled_batch['labels']
        outputs = self.model(**inputs, output_hidden_states=True)
        prediction_scores = self.convert_mlm_logits_to_cls_logits(mlm_labels, outputs[0])
        loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.verbalizer.keys())), labels.view(-1))
        return loss


    def mlm_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a MLM evaluation step."""
        inputs = self.generate_default_inputs(batch)
        outputs = self.model(**inputs)
        return self.convert_mlm_logits_to_cls_logits(batch['mlm_labels'], outputs[0])


    def convert_mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        masked_logits = logits[mlm_labels >= 0]
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
        return cls_logits


    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        m2c = self.mlm_logits_to_answer_logits_tensor.to(logits.device)
        filler_len = torch.tensor([len(self.verbalizer[label]) for label in self.verbalizer.keys()],
                                  dtype=torch.float)
        filler_len = filler_len.to(logits.device)
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        cls_logits = cls_logits * (m2c > 0).float()
        cls_logits = cls_logits.sum(axis=1) / filler_len
        return cls_logits


    def _build_mlm_logits_to_cls_logits_tensor(self):
        label_list = self.verbalizer.keys()
        max_num_answers = max([len(self.verbalizer[label]) for label in label_list])
        m2c_tensor = torch.ones([len(label_list), max_num_answers],
                                dtype=torch.long,
                                requires_grad=False) * -1

        # episode_label_mask_tensor = torch.ones([len(label_list)], dtype=torch.long)
        for label_idx, label in enumerate(label_list):
            answers = self.verbalizer[label]
            for answer_id, answer in enumerate(answers):
                verbalizer_id = self.tokenizer.encode(answer, add_special_tokens=False)[0]
                assert verbalizer_id != self.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, answer_id] = verbalizer_id
        return m2c_tensor


    @staticmethod
    def prepare_optimizer_scheduler(
            cur_model: torch.nn.Module,
            t_total: int,
            weight_decay: float = 0.0,
            lm_learning_rate: float = 1e-5,
            prompt_learning_rate: float = 5e-5,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
    ):
        """
            Prepare optimizer and schedule (linear warmup and decay)
        """
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in cur_model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in cur_model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        embedding_parameters = [
            {'params': [p for p in cur_model.lstm_head.parameters()]},
            {'params': [p for p in cur_model.mlp_head.parameters()]},
            {'params': [p for p in cur_model.prompt_embeddings.parameters()]},
            {'params': [cur_model.layer_lr] if hasattr(cur_model, 'layer_lr') else []}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=lm_learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        embedding_optimizer = AdamW(embedding_parameters, lr=prompt_learning_rate, eps=adam_epsilon)
        embedding_scheduler = get_linear_schedule_with_warmup(embedding_optimizer, num_warmup_steps=warmup_steps,
                                                              num_training_steps=t_total)
        return optimizer, scheduler, embedding_optimizer, embedding_scheduler


class MetaTransformerModelWrapper(TransformerModelWrapper):
    """ Model wrapper that intended for episode-style meta-learning """

    def __init__(self, args):
        super(MetaTransformerModelWrapper, self).__init__(args)

    def train(self,
              train_data: list,
              eval_data: list,
              task_output_dir: str,
              per_gpu_train_batch_size: int = 4,
              n_gpu: int = 1,
              n_train_epochs: int = 300,
              gradient_accumulation_steps: int = 1,
              eval_every_step: int = 100,
              n_adapt_epochs: int = 3,
              weight_decay: float = 0.1,
              learning_rate: float = 5e-5,
              lm_learning_rate: float = 1e-5,
              prompt_learning_rate: float = 5e-5,
              adam_epsilon: float = 1e-8,
              max_grad_norm: float = 1,
              logging_steps: int = 10,
              n_inner_steps: int = 3,
              max_steps=-1):
        """
        Meta training.
        :param train_data: the training episodes to use
        :param eval_data: the validation episodes to use
        :param task_output_dir: directory to save model checkpoints
        :param mode: meta learning mode
        :param per_gpu_train_batch_size: per_gpu_train_batch_size
        :param n_gpu: n_gpu
        :param n_train_epochs: the number of epochs to train
        :param gradient_accumulation_steps: the number of gradient accumulation steps before performing an update
        :param eval_every_step: the number of optimization steps between evaluation
        :param n_adapt_epochs: the number of adaption epochs on evaluation few-shot episodes
        :param weight_decay: the weight decay to use
        :param learning_rate: the learning rate to use for MAML/MAML++ adaption
        :param lm_learning_rate: the learning rate to use for LM param optimization
        :param prompt_learning_rate: the learning rate to use for prompt param optimization
        :param adam_epsilon: the epsilon used in Adam optimizaer
        :param max_grad_norm: the maximum norm for the gradient
        :param logging_steps: the number of steps after which logging information is printed
        :param n_inner_steps: the number of adaption steps
        :param max_steps: max training steps
        :return: best_global_step, best_loss
        """

        t_total = int(100 * n_train_epochs / gradient_accumulation_steps)
        print(f"Epoch {n_train_epochs}, "
              f"episode num: {len(train_data)}, "
              f"n episodes per epoch: {100}"
              f"total outer step {t_total}")

        cur_model = self.model.module if hasattr(self.model, 'module') else self.model
        warmup_steps = int(t_total / 30)
        optimizer, scheduler, embedding_optimizer, embedding_scheduler = self.prepare_optimizer_scheduler(
            cur_model, t_total, weight_decay, lm_learning_rate, prompt_learning_rate, adam_epsilon, warmup_steps,
        )

        self.loss_weights = self.get_per_step_loss_importance_vector(n_steps=n_inner_steps,
                                                                     n_epochs=n_train_epochs)
        self.maml_model = MAML(self.model,
                               lr=learning_rate,
                               first_order=False,
                               allow_nograd=True,
                               allow_unused=True)
        self.maml_model.train()
        writer = SummaryWriter(log_dir=os.path.join(self.config.output_dir, "writer_logs"))

        best_dev_acc = 0.0
        best_global_step = 0
        best_loss = 0.0
        early_stop_epoch = 0

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.maml_model.zero_grad()

        train_iterator = trange(int(n_train_epochs), desc="Epoch")
        for epoch_id in train_iterator:
            tprint(f'=== Start epoch {epoch_id} ===')
            # random.shuffle(train_data)
            batch_iterator = tqdm(train_data[epoch_id * 100: epoch_id * 100 + 100], desc="batch")

            for episode_id, episode in enumerate(batch_iterator):
                label_map = {label: i for i, label in enumerate(episode['labels'])}
                self.refresh_label_map(label_map)
                loss = self.inner_steps(episode, n_inner_steps, per_gpu_train_batch_size, n_gpu,
                                        epoch_id, episode_id, gradient_accumulation_steps)

                tr_loss += loss.item()
                if (episode_id + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    embedding_optimizer.step()
                    embedding_scheduler.step()
                    self.maml_model.zero_grad()
                    self.model.zero_grad()

                    global_step += 1
                    if logging_steps > 0 and global_step % logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        writer.add_scalar(" loss", loss_scalar, global_step=global_step)
                        writer.add_scalar(" lr", learning_rate_scalar, global_step=global_step)
                        logs['learning_rate'] = learning_rate_scalar
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss
                        print(json.dumps({**logs, **{'step': global_step}}))

                    if global_step % eval_every_step == 0 and global_step > 500:
                        self.save(task_output_dir)
                        eval_scores = self.eval(eval_data=eval_data,
                                                pretrained_model_path=task_output_dir,
                                                n_eval_episodes=self.config.val_episodes,
                                                per_gpu_eval_batch_size=per_gpu_train_batch_size,
                                                n_gpu=n_gpu,
                                                n_adapt_epochs=n_adapt_epochs,
                                                lm_learning_rate=lm_learning_rate,
                                                prompt_learning_rate=prompt_learning_rate)
                        if 'eval_shot_rate' in eval_scores.keys():
                            writer.add_scalar("val_shot_rate", eval_scores["eval_shot_rate"],
                                              global_step=global_step)
                        writer.add_scalar("val_acc", eval_scores["acc"], global_step=global_step)
                        if eval_scores["acc"] >= best_dev_acc:
                            if eval_scores["acc"] > best_dev_acc:
                                early_stop_epoch = 0
                            else:
                                early_stop_epoch += 1

                            best_dev_acc = eval_scores["acc"]
                            best_global_step = global_step
                            best_loss = tr_loss

                            tprint("best_dev_acc: %.4f | best_global_step: %d" % \
                                   (best_dev_acc, best_global_step))

                            self.save(os.path.join(task_output_dir, 'best'))
                            tprint("eval_data performance:")
                            tprint(eval_scores)
                        else:
                            early_stop_epoch += 1
                            tprint(eval_scores)
                            tprint(early_stop_epoch)

                if 0 < max_steps < global_step or early_stop_epoch >= self.config.patience:
                    batch_iterator.close()
                    break

            if 0 < max_steps < global_step or early_stop_epoch >= self.config.patience:
                train_iterator.close()
                break

        return best_global_step, best_loss


    def inner_steps(self, episode, n_steps, per_gpu_train_batch_size, n_gpu, cur_epoch=0, episode_id=0,
                    gradient_accumulation_steps=1):
        """
        Optimize model parameter according to MAML++ algorithm.
        :param episode: A few-shot episode
        :param n_steps: The number of MAML++ adaption steps
        :param per_gpu_train_batch_size: per_gpu_train_batch_size
        :param n_gpu: n_gpu
        :param cur_epoch: current training epoch
        :param episode_id: current episode id
        :param gradient_accumulation_steps: the number of gradient accumulation steps before performing an update
        :return: loss
        """
        if len(episode['support_set']) > 5:
            train_batch_size = len(episode['support_set'])//3
        else:
            train_batch_size = len(episode['support_set'])
        train_dataset = self._generate_dataset(episode['support_set'])
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        eval_dataset = self._generate_dataset(episode['query_set'])
        eval_batch_size = int(per_gpu_train_batch_size * max(1, n_gpu))
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        def compute_loss(batch, task_model):
            inputs = self.generate_default_inputs(batch, task_model)
            mlm_labels, labels = batch['mlm_labels'], batch['labels']
            outputs = task_model(**inputs)
            prediction_scores = self.convert_mlm_logits_to_cls_logits(mlm_labels, outputs[0])
            loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.verbalizer.keys())), labels.view(-1))
            return loss

        with torch.backends.cudnn.flags(enabled=False):
            all_loss = []
            for batch in eval_dataloader:
                task_model = self.maml_model.clone()  # torch.clone() for nn.Modules
                task_model.train()
                batch = {k: t.cuda() for k, t in batch.items()}
                step = 0
                while step < n_steps:
                    for i, train_batch in enumerate(train_dataloader):
                        if step == n_steps:
                            break
                        train_batch = {k: t.cuda() for k, t in train_batch.items()}
                        adapt_loss = compute_loss(train_batch, task_model)
                        if n_gpu > 1:
                            adapt_loss = adapt_loss.mean()  # mean() to average on multi-gpu parallel training
                        task_model.adapt(adapt_loss, step)
                        meta_loss = compute_loss(batch, task_model)
                        if n_gpu > 1:
                            meta_loss = meta_loss.mean()  # mean() to average on multi-gpu parallel training
                        # Update loss weights every 10 episodes 
                        meta_loss = meta_loss * self.loss_weights[int(cur_epoch * 10 + episode_id / 10)][step]
                        meta_loss = meta_loss / gradient_accumulation_steps
                        if step == n_steps - 1:
                            meta_loss.backward()
                        else:
                            meta_loss.backward(retain_graph=True)
                        step += 1
                all_loss.append(meta_loss)

        return sum(all_loss)


    @staticmethod
    def get_per_step_loss_importance_vector(n_steps, n_epochs):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :param n_steps: The number of MAML adaption steps
        :param n_epochs: Total training epochs
        :return: A tensor indicating (high-order) loss weights across MAML adaption steps
        """
        loss_weights = np.ones(shape=(n_epochs * 10, n_steps)) * (
                1.0 / n_steps)
        decay_rate = 1.0 / n_steps / n_epochs
        min_value_for_non_final_losses = 0.03 / n_steps

        for epoch in range(n_epochs * 10):
            for i in range(len(loss_weights[0]) - 1):
                curr_value = np.maximum(loss_weights[epoch][i] - (epoch * decay_rate), min_value_for_non_final_losses)
                loss_weights[epoch][i] = curr_value

            curr_value = np.minimum(
                loss_weights[epoch][-1] + (epoch * (n_steps - 1) * decay_rate),
                1.0 - ((n_steps - 1) * min_value_for_non_final_losses))
            loss_weights[epoch][-1] = curr_value

        loss_weights = torch.Tensor(loss_weights).cuda()
        return loss_weights

