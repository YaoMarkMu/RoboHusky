#!/usr/bin/env python
# coding=utf-8
# Copyright Qing-Long Zhang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""

import logging
import os
import sys
import warnings

from dataclasses import dataclass, field
from typing import Optional

import datasets
from PIL import PngImagePlugin, Image, ImageFile

import torch
from datasets import load_dataset, load_from_disk
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from husky.dist_utils import init_dist
from husky.conversation import get_conv_template
from husky.model.modeling_husky_multi import HuskyForConditionalGeneration

from husky.video_transformers import (
    GroupNormalize,
    GroupScale,
    GroupCenterCrop,
    Stack,
    ToTorchFormatTensor,
    get_index, GroupMultiScaleCrop,
)

from decord import VideoReader, cpu

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    Trainer,
    set_seed,
    default_data_collator,
    DataCollatorForSeq2Seq,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from transformers.utils.logging import (
    set_verbosity_info,
    set_verbosity,
    enable_default_handler,
    enable_explicit_format,
)

IGNORE_INDEX = -100
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<ImageContent>"
DEFAULT_IMG_START_TOKEN = "<img>"
DEFAULT_IMG_END_TOKEN = "</img>"

DEFAULT_VIDEO_START_TOKEN = "<vid>"
DEFAULT_VIDEO_END_TOKEN = "</vid>"

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.29.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    freeze_vision_model: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained vision model whose head dimensions are different."},
    )
    freeze_text_model: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained text model whose head dimensions are different."},
    )
    freeze_qformer: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained qformer model whose head dimensions are different."},
    )
    un_freeze_vision_embedding: bool = field(
        default=False,
        metadata={"help": "Will enable to tuning image patch_embedding when vision_model are frozen"},
    )
    un_freeze_video_embedding: bool = field(
        default=False,
        metadata={"help": "Will enable to tuning video patch_embedding when vision_model are frozen"},
    )
    use_lora: bool = field(
        default=False, metadata={"help": "add the LoRA adapters to the base model"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The data directory containing input files."})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacrebleu) on a jsonlines file."
        },
    )
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the metrics (sacrebleu) on a jsonlines file."},
    )
    image_path: Optional[str] = field(
        default=None,
        metadata={"help": "An optional image path"},
    )
    video_path: Optional[str] = field(
        default=None,
        metadata={"help": "An optional video path"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    val_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    conv_style: Optional[str] = field(
        default=None, metadata={"help": "prompt style for a conversation."}
    )
    save_data_path: Optional[str] = field(
        default=None, metadata={"help": "prompt style for a conversation."}
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the :obj:`decoder_start_token_id`.Useful for"
                " multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token needs to"
                " be the target language token.(Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        # accepting both json and jsonl file extensions, as
        # many jsonlines files actually have a .json extension
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension == "json", "`test_file` should be a json file."

def build_transform(is_train, input_size):
    if is_train:
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(input_size, scale=(0.5, 1.0), ratio=(3. / 4., 4. / 3.),
                                interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        crop_pct = 224 / 256
        size = int(input_size / crop_pct)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(size, interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    return transform

def load_video(video_path, num_segments=8):
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    # transform
    crop_size = 224
    scale_size = 224
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        GroupMultiScaleCrop(crop_size, [1, .875, .75, .66]),
        Stack(roll=False),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std)
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())
        images_group.append(img)
    video = transform(images_group)
    TC, H, W = video.shape
    video = video.reshape(TC // 3, 3, H, W).transpose(0, 1)  # [C, T, H, W]
    return video

def is_image(image_file):
    if image_file.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
        return True
    else:
        return False

def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    init_dist(launcher='slurm', backend='nccl', port=29502)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("tune_husky_chat", model_args, data_args)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 4. Get the datasets
    # you can either provide your own JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    """
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        ds = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            data_dir=data_args.data_dir,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]

        ds = load_dataset(
            "json" if extension == "jsonl" else extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in ds.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = ds["train"].train_test_split(data_args.train_val_split)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]
    """

    # 5. Load pretrained model, tokenizer, and image processor
    #
    # Distributed training: The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # add special token
    tokenizer.pad_token_id = 0
    if tokenizer.unk_token is None:
        tokenizer.add_special_tokens({"unk_token": DEFAULT_UNK_TOKEN})

    if data_args.conv_style is not None:
        tokens_list = [DEFAULT_IMG_START_TOKEN, DEFAULT_IMG_END_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_START_TOKEN,
                       DEFAULT_VIDEO_END_TOKEN]
    tokenizer.add_tokens(tokens_list, special_tokens=True)

    model = HuskyForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.config.use_cache = False
    num_queries = model.config.num_query_tokens

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.language_model.get_input_embeddings().weight.shape[0]

    if len(tokenizer) > embedding_size:
        model.language_model.resize_token_embeddings(len(tokenizer))
    model.config.text_config.vocab_size = len(tokenizer)

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    # """
    if model_args.freeze_vision_model:
        model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)
        # tuning image | video patch_embedding
        if model_args.un_freeze_vision_embedding:
            model.vision_model.embeddings.patch_embedding.weight.requires_grad = True
            model.vision_model.embeddings.class_embedding.requires_grad = True
            model.vision_model.embeddings.position_embedding.requires_grad = True
        if model_args.un_freeze_video_embedding:
            model.vision_model.video_embeddings.patch_embedding.weight.requires_grad = True
            model.vision_model.video_embeddings.class_embedding.requires_grad = True
            model.vision_model.video_embeddings.position_embedding.requires_grad = True

    if model_args.freeze_qformer:
        model.qformer = model.qformer.eval()
        _freeze_params(model.qformer)
        model.query_tokens.requires_grad = False

    if model_args.freeze_text_model:
        _freeze_params(model.language_model)

    if model_args.use_lora:
        training_args.ddp_find_unused_parameters = False
        _freeze_params(model)
        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.language_model = get_peft_model(model.language_model, lora_config)
        model.language_model.print_trainable_parameters()
    # """
    # if model_args.un_freeze_video_embedding:
    #     _freeze_params(model)
    #     model.vision_model.video_embeddings.patch_embedding.weight.requires_grad = True
    #     model.vision_model.video_embeddings.class_embedding.requires_grad = True
    #     model.vision_model.video_embeddings.position_embedding.requires_grad = True

    else:
        _freeze_params(model)
        if model_args.un_freeze_video_embedding:
            model.vision_model.video_embeddings.patch_embedding.weight.requires_grad = True
            model.vision_model.video_embeddings.class_embedding.requires_grad = True
            model.vision_model.video_embeddings.position_embedding.requires_grad = True
        # Only language_projection are tuned
        # model.language_projection.weight.requires_grad = True

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # 7. Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    """
    if training_args.do_train:
        column_names = ds["train"].column_names
    elif training_args.do_eval:
        column_names = ds["validation"].column_names
    elif training_args.do_predict:
        column_names = ds["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # set padding.
    padding = "max_length" if data_args.pad_to_max_length else False

    def format_inputs(sources):
        # Apply prompt templates
        conv = get_conv_template(data_args.conv_style).copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                # vision is only supported for the human input
                if role == conv.roles[0]:
                    value = sentence["value"]
                    if "<image>" in value:
                        if value.endswith("\n<image>"):
                            value = "<ImageContent>\n" + value.replace("\n<image>", "")
                        else:
                            value = value.replace("<image>", "<ImageContent>")
                        image_query = DEFAULT_IMG_START_TOKEN + DEFAULT_IMAGE_TOKEN * num_queries + DEFAULT_IMG_END_TOKEN
                        sentence["value"] = value.replace(DEFAULT_IMAGE_TOKEN, image_query)
                    elif "<video>" in value:
                        if value.endswith("\n<video>"):
                            value = "<ImageContent>\n" + value.replace("\n<video>", "")
                        else:
                            value = value.replace("<video>", "<ImageContent>")
                        video_query = DEFAULT_VIDEO_START_TOKEN + DEFAULT_IMAGE_TOKEN * num_queries + DEFAULT_VIDEO_END_TOKEN
                        sentence["value"] = value.replace(DEFAULT_IMAGE_TOKEN, video_query)

                elif role == conv.roles[1]:
                    sentence["value"] = sentence["value"].replace("image", "video")

                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        return conversations, conv

    def preprocess_function(examples, is_train=False, input_size=224, vision_type="image"):
        conversations, conv = format_inputs(examples['conversations'])
        model_inputs = tokenizer(
            conversations,
            max_length=data_args.max_seq_length,
            padding=padding,
            truncation=True,
            return_tensors="pt",
        )
        model_inputs.pop("token_type_ids", None)

        image_transform = build_transform(is_train=is_train, input_size=input_size)
        if vision_type == "image":
            if "coco" in data_args.image_path:
                images = [
                    Image.open(os.path.join(data_args.image_path, "COCO_train2014_" + image_file)) for image_file in
                    list(examples["image"])
                ]
            else:
                images = [
                    Image.open(os.path.join(data_args.image_path, image_file)) for image_file in
                    list(examples["image"])
                ]
            pixel_values = [image_transform(image) for image in images]

        elif vision_type == "video":
            pixel_values = [
                load_video(os.path.join(data_args.video_path, video + ".mp4")) for video in
                list(examples["image"])
            ]
        else:
            assert NotImplementedError

        model_inputs["pixel_values"] = torch.stack(pixel_values, dim=0)

        model_inputs["pixel_values"] = torch.stack(pixel_values, dim=0)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        targets = model_inputs["input_ids"].clone()

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
                target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX
            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX

        model_inputs["labels"] = targets
        return model_inputs

    def train_preprocess(examples):
        return preprocess_function(examples=examples, is_train=True)

    if training_args.do_train:
        if "train" not in ds:
            train_dataset = ds
        else:
            train_dataset = ds["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                train_preprocess,
                batched=True,
                batch_size=32,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        train_dataset.save_to_disk(data_args.save_data_path)
    """
    # Here we load the processed datasets directly
    # train_dataset = load_from_disk(data_args.save_data_path)
    # English image dataset
    cc3m_595k = load_from_disk("datasets/tokenized/chat_cc3m_595k")
    # lain_491k = load_from_disk("datasets/tokenized/lain_491k")

    # Chinese image dataset
    # cc3m_595k_zh = load_from_disk("datasets/tokenized/chat_cc3m_595k_zh")
    # mmc4_130k_zh = load_from_disk("datasets/tokenized/mmc4_130k_zh")
    # coco_5k_zh = load_from_disk("datasets/tokenized/coco_5k_zh")
    # chinese_food = load_from_disk("datasets/tokenized/chinese_food")

    # text_vqa = load_from_disk("datasets/tokenized/TextVQA")
    # text_caps = load_from_disk("datasets/tokenized/TextCaps")
    # infographic_vqa = load_from_disk("datasets/tokenized/infographicVQA")

    # concat english and chinese dataset
    # image_dataset = datasets.concatenate_datasets(
    #     [cc3m_595k, lain_491k, text_caps, mmc4_130k_zh, coco_5k_zh, text_vqa, infographic_vqa, chinese_food])

    # here we assume total training batch_size is 128
    # max_image_sample = int(len(image_dataset) // 128) * 128
    # image_dataset = image_dataset.select(range(0, max_image_sample))

    # load video dataset
    # vid_dataset = load_from_disk("datasets/tokenized/webvid2m")
    train_dataset = cc3m_595k
    train_dataset = train_dataset.select(range(64))
    # ego_4d = load_from_disk("datasets/tokenized/EGO4D")
    # # concat image and video dataset
    # train_dataset = datasets.concatenate_datasets([ego_4d, vid_dataset])
    # train_dataset = image_dataset

    # Data collator
    label_pad_token_id = IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # 8. Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 9. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        if model_args.use_lora:
            model.language_model.save_pretrained(training_args.output_dir)
        else:
            trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

if __name__ == "__main__":
    main()
