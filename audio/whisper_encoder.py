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

from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # outputs = model(**inputs)
        # loss = outputs.loss
        # return (loss, outputs) if return_outputs else loss
        cross_attn_head_mask = torch.zeros((12,12,), dtype=torch.float32, device="cuda:0")
        outputs = model(**inputs, cross_attn_head_mask=cross_attn_head_mask, output_hidden_states=True)
        text_embeds = outputs["decoder_hidden_states"][-1][:,-1,:].detach().cpu()
        labels = inputs.pop("labels")
        audio_embeds = model.model.encoder(**inputs)[0][:,-1,:]
        from sklearn.cluster import KMeans
        clustering = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(text_embeds)
        labels = clustering.labels_
        min_samples = 3
        from collections import Counter
        label_counts = Counter(labels)
        satisfies_min_samples = all(count >= min_samples for count in label_counts.values())
        from torchclustermetrics import silhouette
        loss = 1 - silhouette.score(audio_embeds, labels, True)
        if not satisfies_min_samples:
            loss.detach()
            print("skip")
        return (loss, outputs) if return_outputs else loss


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
common_voice = common_voice.cast_column("file_name", Audio(sampling_rate=sampling_rate))

def prepare_dataset(example):
    audio = example["file_name"]
    text = example["transcription"]
    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=text,
    )
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example

common_voice = common_voice.map(
    prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1
)

max_input_length = 30.0

def is_audio_in_length_range(length):
    return length < max_input_length

common_voice["train"] = common_voice["train"].filter(
    is_audio_in_length_range,
    input_columns=["input_length"],
)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

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
    output_dir="./wsm_tta",
    per_device_train_batch_size=128,
    gradient_accumulation_steps=1,
    learning_rate=2e-6,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=20,
    max_steps=400,
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=5,
    logging_steps=1,
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
    processor=processor,
)

trainer.train()