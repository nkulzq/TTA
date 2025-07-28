from datasets import load_dataset, DatasetDict
from transformers import WhisperProcessor
from datasets import Audio
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperForConditionalGeneration
from functools import partial
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import csv
import torch.nn as nn
import numpy as np
import random
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F

from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

class VICReg(nn.Module):
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, eps=1e-4):
        super(VICReg, self).__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.eps = eps

    def forward(self, z1, z2):
        sim_loss = nn.functional.mse_loss(z1, z2)

        def variance_loss(z):
            std = torch.sqrt(z.var(dim=0) + self.eps)
            return torch.mean(torch.relu(1.0 - std))

        std_loss = variance_loss(z1) + variance_loss(z2)

        def off_diagonal(x):
            n, d = x.size()
            x = x - x.mean(dim=0)
            cov = (x.T @ x) / (n - 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            return (off_diag ** 2).sum() / d

        cov_loss = off_diagonal(z1) + off_diagonal(z2)

        loss = self.sim_coeff * sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return loss

def random_augment(audio, sample_rate):
    with torch.no_grad():
        x = audio.to(torch.float32)

        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            x *= gain

        if random.random() < 0.5:
            new_sr = int(sample_rate * random.uniform(0.9, 1.1))
            resample = T.Resample(orig_freq=sample_rate, new_freq=new_sr)
            x = resample(x)

        if random.random() < 0.5:
            lowpass = T.Vol(gain=random.uniform(0.5, 1.0))
            x = lowpass(x)

    return x

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        criterion = VICReg()
        labels = inputs.pop("labels")
        e_i = model.model.encoder(input_features=inputs["input_features_1"])[0][:,-1,:]
        e_j = model.model.encoder(input_features=inputs["input_features_2"])[0][:,-1,:]
        loss = criterion(e_i, e_j)
        return loss


common_voice = DatasetDict()

common_voice["train"] = load_dataset(
    "united-we-care/United-Syn-Med", split="train"
)
common_voice["test"] = load_dataset(
    "united-we-care/United-Syn-Med", split="test"
)

path = "/home/icml02/.cache/huggingface/hub/datasets--united-we-care--United-Syn-Med/snapshots/54b992a26c1b2b00eeace87aea61c3596e2e0c88/data/audio"
common_voice["train"] = common_voice["train"].map(
    lambda x: {"file_name": path+"/train/"+x['file_name']}
)
common_voice["test"] = common_voice["test"].map(
    lambda x: {"file_name": path+"/test/"+x['file_name']}
)

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="english", task="transcribe"
)

sampling_rate = processor.feature_extractor.sampling_rate
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=sampling_rate))

def prepare_dataset_train(example):
    audio = example["audio"]
    text = example["transcription"]
    waveform = torch.tensor(audio["array"])
    sample_rate = audio["sampling_rate"]

    # Create two differently augmented views
    aug1 = random_augment(waveform, sample_rate).squeeze(0).numpy()
    aug2 = random_augment(waveform, sample_rate).squeeze(0).numpy()

    # Process with processor
    processed_1 = processor(audio=aug1, sampling_rate=sample_rate, text=text)
    processed_2 = processor(audio=aug2, sampling_rate=sample_rate, text=text)
    example = {
        "input_features": np.concatenate([processed_1["input_features"],processed_2["input_features"]], axis=1),
        "labels": processed_2["labels"]
    }
    return example

def prepare_dataset_test(example):
    audio = example["file_name"]
    text = example["transcription"]
    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=text,
    )
    return example

common_voice["train"] = common_voice["train"].map(prepare_dataset_train, remove_columns=common_voice["train"].column_names)
common_voice["test"]  = common_voice["test"].map(prepare_dataset_test,  remove_columns=common_voice["test"].column_names)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if len(features[0]["input_features"]) != 2:
            input_features = [
                {"input_features": feature["input_features"][0]} for feature in features
            ]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        else:
            input_features_1 = [
                {"input_features": feature["input_features"][0][0]} for feature in features
            ]
            input_features_2 = [
                {"input_features": feature["input_features"][0][0]} for feature in features
            ]
            batch_1 = self.processor.feature_extractor.pad(input_features_1, return_tensors="pt")
            batch_2 = self.processor.feature_extractor.pad(input_features_2, return_tensors="pt")
            batch = {
                "input_features_1":batch_1["input_features"],
                "input_features_2":batch_2["input_features"],
            }

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")

normalizer = BasicTextNormalizer()

class Scorer():
    def __init__(self, ref, gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE"),
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f" % (m, sc))
                total_scores["Bleu"] = score
            else:
                print("%s: %0.3f" % (method, score))
                total_scores[method] = score

        print('*****DONE*****')
        for key, value in total_scores.items():
            print('{}:{}'.format(key, value))
        return total_scores

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# model.freeze_encoder()

model.config.use_cache = False

model.generate = partial(
    model.generate, language='english', task="transcribe", use_cache=True
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./wsm_simclr",
    per_device_train_batch_size=128,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=1000,
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=250,
    eval_steps=10,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

trainer = CustomSeq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.train()