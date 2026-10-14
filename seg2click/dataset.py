import copy
import bisect
import json
import math
import os
import random
import re
import ast
import struct
import sys
from array import array
from pathlib import Path
from typing import Dict

import torch
import transformers
import yaml
from qwen_vl_utils import smart_resize, process_vision_info
from torch.utils.data import Dataset

from seg2click.constants import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    ACTION_PATTENS_XY,
    ADDITIONAL_SPECIAL_TOKENS,
    assistant_template,
    chat_template,
    grounding_system_message,
)
import torch.distributed as dist


def rank0_print(*args):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args)


def reformat_coordinates(text):
    """
    (1) Find all the coordinates in the text.
    (2) Replace the coordinates with the special tokens.
    (3) Return the new text and the coordinates as a list of (x, y), where x in [0, 1] and y in [0, 1].
    """
    epsilon = 0.001

    def adjust_coord(c):
        """
        Adjust coordinate if it is too close to 0 or 1.
        """
        if abs(c) < epsilon:
            return epsilon
        elif abs(c - 1) < epsilon:
            return 1 - epsilon
        return c

    all_matches = []
    for pattern in ACTION_PATTENS_XY:
        matches = list(re.finditer(pattern, text))
        for match in matches:
            all_matches.append((match.start(), match.groups()))
        if pattern == ACTION_PATTENS_XY[0]:
            target_text = f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}"
        else:
            target_text = f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}, {DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}"
        text = re.sub(pattern, target_text, text)

    coordinates = []
    all_matches.sort(key=lambda x: x[0])
    # Extract coordinates in order
    for _, groups in all_matches:
        # When two coordinate values are found, parse them as one (x, y) pair.
        if len(groups) == 2:
            x_str, y_str = groups
            x = adjust_coord(ast.literal_eval(x_str))
            y = adjust_coord(ast.literal_eval(y_str))
            coordinates.append((x, y))
        # When four coordinate values are found, parse them as two pairs.
        elif len(groups) == 4:
            x1_str, y1_str, x2_str, y2_str = groups
            x1 = adjust_coord(ast.literal_eval(x1_str))
            y1 = adjust_coord(ast.literal_eval(y1_str))
            x2 = adjust_coord(ast.literal_eval(x2_str))
            y2 = adjust_coord(ast.literal_eval(y2_str))
            coordinates.append((x1, y1))
            coordinates.append((x2, y2))

    return text, coordinates


def split_system_message_and_limit_turns(source, roles, default_system_message, max_conv_turns):
    """Remove an optional leading system message, then limit non-system turns."""
    source = list(source)
    system_message = default_system_message
    if source:
        first = source[0]
        raw_role = first.get("role", first.get("from"))
        if roles.get(raw_role, raw_role) == "system":
            system_message = first.get("content", first.get("value"))
            source = source[1:]

    if max_conv_turns is not None:
        if max_conv_turns < 0:
            raise ValueError("max_conv_turns must be non-negative or None")
        source = source[:max_conv_turns]
    return system_message, source


def get_token_index(image_processor, image, point_x, point_y):
    """
    Get the index of the visual token that contains the point (x, y).
    Args:
        image_processor: the image processor
        image: the image in PIL format
        point_x: the x coordinate of the point, in [0, 1].
        point_y: the y coordinate of the point, in [0, 1].
    """
    if len(image) != 1:
        raise ValueError(f"Expected 1 image, got {len(image)}")

    # get the original image size and the resized image size
    image = image[0]
    w, h = image.size
    px, py = w * point_x, h * point_y
    # rank0_print(f"px: {px}, py: {py}")
    # get the token index
    merge_patch_size = image_processor.patch_size * image_processor.merge_size
    x_index = math.floor(px / merge_patch_size)
    y_index = math.floor(py / merge_patch_size)

    visual_token_index = y_index * (w // merge_patch_size) + x_index

    # merge all above print into one line
    return visual_token_index


def get_multi_patch_labels(image_processor, image, bbox_gt):
    """
    Get the multi-patch labels for the bounding box.
    Args:
        image_processor: the image processor
        image: the image in PIL format
        bbox_gt: the bounding box in the format of (x_min, y_min, x_max, y_max) [0,1]
    """
    if len(image) != 1:
        raise ValueError(f"Expected 1 image, got {len(image)}")

    # Get the original image size and the resized image size
    image = image[0]
    w, h = image.size

    bbox_gt = [bbox_gt[0] * w, bbox_gt[1] * h, bbox_gt[2] * w, bbox_gt[3] * h]
    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = bbox_gt
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    merge_patch_size = image_processor.patch_size * image_processor.merge_size
    assert w % merge_patch_size == 0 and h % merge_patch_size == 0, (
        f"Image size {w}x{h} is not divisible by merge_patch_size {merge_patch_size}"
    )
    grid_h, grid_w = h // merge_patch_size, w // merge_patch_size

    binary_mask = torch.zeros(grid_h * grid_w)
    # Iterate through all patches, check if they overlap with the bounding box
    for y_idx in range(grid_h):
        for x_idx in range(grid_w):
            # Calculate patch boundaries
            patch_x_min = x_idx * merge_patch_size
            patch_y_min = y_idx * merge_patch_size
            patch_x_max = patch_x_min + merge_patch_size
            patch_y_max = patch_y_min + merge_patch_size

            # Check if patch overlaps with the bounding box
            if not (
                patch_x_max <= x_min
                or patch_x_min >= x_max
                or patch_y_max <= y_min
                or patch_y_min >= y_max
            ):
                # Calculate patch index in the flattened grid
                patch_idx = y_idx * grid_w + x_idx
                binary_mask[patch_idx] = 1

    return binary_mask


def token_index_to_coordinates(image_processor, visual_token_index, image_width, image_height):
    merge_patch_size = image_processor.patch_size * image_processor.merge_size
    x_index = visual_token_index % (image_width // merge_patch_size)
    y_index = visual_token_index // (image_width // merge_patch_size)
    px = x_index * merge_patch_size + merge_patch_size / 2
    py = y_index * merge_patch_size + merge_patch_size / 2
    return px, py


INDEX_MAGIC = b"GUIAIDX1"
INDEX_HEADER = struct.Struct("<8sQQ")


class InMemoryJsonSource:
    def __init__(self, name, records, images_folder):
        self.name = name
        self.records = records
        self.images_folder = str(images_folder)

    def __len__(self):
        return len(self.records)

    def get(self, index):
        return self.records[index]

    def record_id(self, index):
        return f"{self.name}:record:{index}"


class IndexedJsonlSource:
    """Random-access JSONL source backed by a GUIAIDX1 offset index."""

    def __init__(self, name, jsonl_path, index_path, images_folder):
        self.name = name
        self.jsonl_path = Path(jsonl_path).resolve()
        self.index_path = Path(index_path).resolve()
        images_path = Path(images_folder).resolve()
        self.images_folder = str(images_path)
        self._file_handle = None
        self._file_pid = None

        if not self.jsonl_path.is_file():
            raise FileNotFoundError(f"Missing JSONL file: {self.jsonl_path}")
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Missing GUIAIDX1 file: {self.index_path}")
        if not images_path.is_dir():
            raise FileNotFoundError(f"Missing images directory: {images_path}")
        with self.index_path.open("rb") as handle:
            header = handle.read(INDEX_HEADER.size)
            offset_bytes = handle.read()
        if len(header) != INDEX_HEADER.size:
            raise ValueError(f"Truncated GUIAIDX1 header: {self.index_path}")
        magic, record_count, jsonl_bytes = INDEX_HEADER.unpack(header)
        if magic != INDEX_MAGIC:
            raise ValueError(f"Unexpected index magic in {self.index_path}: {magic!r}")
        if self.jsonl_path.stat().st_size != jsonl_bytes:
            raise ValueError(
                f"JSONL size mismatch for {self.jsonl_path}: "
                f"{self.jsonl_path.stat().st_size} != {jsonl_bytes}"
            )
        if len(offset_bytes) != record_count * 8:
            raise ValueError(
                f"Offset table size mismatch for {self.index_path}: "
                f"{len(offset_bytes)} != {record_count * 8}"
            )
        offsets = array("Q")
        if offsets.itemsize != 8:
            raise RuntimeError("GUIAIDX1 requires an 8-byte unsigned integer array")
        offsets.frombytes(offset_bytes)
        if sys.byteorder != "little":
            offsets.byteswap()
        if offsets and offsets[-1] >= jsonl_bytes:
            raise ValueError(f"Offset outside JSONL in {self.index_path}")
        if any(left >= right for left, right in zip(offsets, offsets[1:])):
            raise ValueError(f"Offsets must be strictly increasing: {self.index_path}")
        self.offsets = offsets
        self.jsonl_bytes = jsonl_bytes

    def __len__(self):
        return len(self.offsets)

    def _open_for_current_process(self):
        process_id = os.getpid()
        if self._file_handle is None or self._file_pid != process_id:
            self.close()
            self._file_handle = self.jsonl_path.open("rb")
            self._file_pid = process_id
        return self._file_handle

    def get(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        handle = self._open_for_current_process()
        handle.seek(self.offsets[index])
        line = handle.readline()
        if not line:
            raise ValueError(
                f"Empty JSONL record at offset {self.offsets[index]} in {self.jsonl_path}"
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL record at offset {self.offsets[index]} in {self.jsonl_path}"
            ) from exc

    def __getitem__(self, index):
        return self.get(index)

    def record_id(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return f"{self.name}:offset:{self.offsets[index]}"

    def close(self):
        if self._file_handle is not None:
            self._file_handle.close()
        self._file_handle = None
        self._file_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_handle"] = None
        state["_file_pid"] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class LazySupervisedDataset(Dataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        processor: transformers.ProcessorMixin,
        data_path: str,
        data_args,
        include_pointer_targets: bool = True,
        include_bbox_targets: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self._sources = []
        self._cumulative_sizes = []
        expected_total_records = None
        self.pointer_pad_token_id = tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0]
        self.pointer_start_token_id = tokenizer.encode(DEFAULT_POINTER_START_TOKEN)[0]
        self.pointer_end_token_id = tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
        self.include_pointer_targets = include_pointer_targets
        self.include_bbox_targets = include_bbox_targets

        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                full_path = str(Path(f"{base_path}{file_name}.json").resolve())
                data_args.dataset_paths.append(full_path)
                rank0_print(f"Loading {full_path}")
                with open(full_path) as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                self._append_source(
                    InMemoryJsonSource(
                        file_name,
                        cur_data_dict,
                        Path(data_args.image_folder or "").resolve(),
                    )
                )
        elif data_path.endswith((".yaml", ".yml")):
            config_path = Path(data_path).resolve()
            with config_path.open() as file:
                yaml_data = yaml.safe_load(file)
                expected_total_records = yaml_data.get("expected_total_records")
                datasets = yaml_data.get("datasets")
                if not isinstance(datasets, list) or not datasets:
                    raise ValueError(f"No datasets found in {config_path}")
                config_dir = config_path.parent
                selected_split = getattr(data_args, "data_split", None)
                if selected_split is None:
                    selected_split = yaml_data.get("default_split")
                data_args.data_split = selected_split
                data_args.dataset_paths = []
                for dataset in datasets:
                    json_path = self._resolve_yaml_path(config_dir, dataset.get("json_path"))
                    images_folder = self._resolve_yaml_path(
                        config_dir, dataset.get("images_folder")
                    )
                    name = dataset.get("name", Path(json_path).stem)
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    split_indexes = dataset.get("split_indexes")
                    index_path = dataset.get("index_path")
                    if split_indexes is not None:
                        if selected_split is None:
                            raise ValueError(f"data_split or default_split is required for {name}")
                        if selected_split not in split_indexes:
                            raise ValueError(
                                f"Split {selected_split!r} is not configured for {name}"
                            )
                        index_path = split_indexes[selected_split]
                    data_args.dataset_paths.append(json_path)

                    if index_path is not None:
                        if sampling_strategy != "all":
                            raise ValueError(
                                f"Indexed source {name} does not support "
                                f"sampling_strategy={sampling_strategy!r}"
                            )
                        if not json_path.endswith(".jsonl"):
                            raise ValueError(f"Indexed source must use JSONL: {json_path}")
                        resolved_index_path = self._resolve_yaml_path(config_dir, index_path)
                        source = IndexedJsonlSource(
                            name=name,
                            jsonl_path=json_path,
                            index_path=resolved_index_path,
                            images_folder=images_folder,
                        )
                        expected_records = dataset.get("expected_records")
                        if isinstance(expected_records, dict):
                            expected_records = expected_records.get(selected_split)
                        if expected_records is not None and len(source) != expected_records:
                            raise ValueError(
                                f"Indexed record count mismatch for {name}/{selected_split}: "
                                f"{len(source)} != {expected_records}"
                            )
                        rank0_print(
                            f"Indexed {len(source)} samples from {json_path} "
                            f"using {resolved_index_path}"
                        )
                        self._append_source(source)
                        continue

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")
                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path) as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path) as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")
                    cur_data_dict = self._sample_records(cur_data_dict, sampling_strategy)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self._append_source(InMemoryJsonSource(name, cur_data_dict, images_folder))
        else:
            resolved_data_path = str(Path(data_path).resolve())
            data_args.dataset_paths = [resolved_data_path]
            rank0_print(f"Loading {resolved_data_path}")
            with open(resolved_data_path) as file:
                cur_data_dict = json.load(file)
            rank0_print(f"Loaded {len(cur_data_dict)} samples from {resolved_data_path}")
            self._append_source(
                InMemoryJsonSource(
                    Path(resolved_data_path).stem,
                    cur_data_dict,
                    Path(data_args.image_folder or "").resolve(),
                )
            )

        if expected_total_records is not None and len(self) != expected_total_records:
            raise ValueError(
                f"Dataset total record count mismatch for {data_path}: "
                f"{len(self)} != {expected_total_records}"
            )
        rank0_print(f"Loaded {len(self)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

    @staticmethod
    def _resolve_yaml_path(config_dir, value):
        if value is None:
            return str(config_dir)
        path = Path(value)
        if not path.is_absolute():
            path = config_dir / path
        return str(path.resolve())

    @staticmethod
    def _sample_records(records, sampling_strategy):
        sampling_number = None
        if ":" in sampling_strategy:
            sampling_strategy, sampling_number = sampling_strategy.split(":", 1)
            if "%" in sampling_number:
                percentage = int(sampling_number.split("%", 1)[0])
                sampling_number = math.ceil(percentage * len(records) / 100)
            else:
                sampling_number = int(sampling_number)
        if sampling_strategy == "all":
            return records
        if sampling_strategy == "first" and sampling_number is not None:
            return records[:sampling_number]
        if sampling_strategy == "end" and sampling_number is not None:
            return records[-sampling_number:]
        if sampling_strategy == "random" and sampling_number is not None:
            random.shuffle(records)
            return records[:sampling_number]
        raise ValueError(f"Invalid sampling strategy: {sampling_strategy!r}")

    def _append_source(self, source):
        if len(source) == 0:
            return
        self._sources.append(source)
        previous_size = self._cumulative_sizes[-1] if self._cumulative_sizes else 0
        self._cumulative_sizes.append(previous_size + len(source))

    def _locate_record(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_index = bisect.bisect_right(self._cumulative_sizes, index)
        source_start = self._cumulative_sizes[source_index - 1] if source_index > 0 else 0
        local_index = index - source_start
        source = self._sources[source_index]
        return source, local_index

    def _raw_record(self, index):
        source, local_index = self._locate_record(index)
        return (
            source.get(local_index),
            source.images_folder,
            source.record_id(local_index),
        )

    def __len__(self):
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def _require_in_memory_for_length_grouping(self):
        if any(isinstance(source, IndexedJsonlSource) for source in self._sources):
            raise RuntimeError(
                "Length grouping is disabled for indexed JSONL sources. "
                "Use --group_by_length False and --group_by_modality_length False."
            )

    @property
    def lengths(self):
        self._require_in_memory_for_length_grouping()
        length_list = []
        for source in self._sources:
            for sample in source.records:
                img_tokens = (
                    1200 * len(sample["image"])
                    if isinstance(sample["image"], list)
                    else 1200
                    if "image" in sample
                    else 0
                )
                length_list.append(
                    sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens
                )
        return length_list

    @property
    def modality_lengths(self):
        self._require_in_memory_for_length_grouping()
        length_list = []
        for source in self._sources:
            for sample in source.records:
                cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
                assert cur_len > 0, f"Conversation length is 0 for {sample}"
                img_tokens = (
                    1200 * len(sample["image"])
                    if isinstance(sample["image"], list)
                    else 1200
                    if "image" in sample
                    else 0
                )
                if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                    length_list.append(cur_len + img_tokens)
                else:
                    length_list.append(-cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self._get_item(i)

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        record, image_path, default_item_id = self._raw_record(i)
        sources = record

        if "image" in sources:
            image_file = record["image"]
            if isinstance(image_file, list):
                image_list = [
                    os.path.join(image_path, current_image) for current_image in image_file
                ]
            else:
                image_list = [os.path.join(image_path, image_file)]

            sources = copy.deepcopy(sources["conversations"])
        elif "video" in sources:
            raise NotImplementedError("Video is not supported for Qwen2VL")
        else:
            sources = copy.deepcopy(sources["conversations"])

        item_id = record.get("id", default_item_id)

        data_dict = self.preprocess_qwen2vl(
            sources, self.tokenizer, self.processor, image_list, id=item_id
        )
        if isinstance(i, int):
            processed = data_dict
            data_dict = {
                "input_ids": processed["input_ids"][0],
                "labels": processed["labels"][0],
                "pixel_values": processed["pixel_values"],
                "image_grid_thw": processed["image_grid_thw"],
            }
            data_dict["bbox_image_indices"] = processed.get("bbox_image_indices", [None])[0]
            data_dict["image_counts"] = processed.get("image_counts", [1])[0]
            if self.include_bbox_targets:
                data_dict["bbox_targets"] = processed["bbox_targets"][0]
                data_dict["bbox_image_sizes"] = processed.get("bbox_image_sizes", [None])[0]
            if self.include_pointer_targets:
                data_dict.update(
                    {
                        "coordinates": processed["coordinates"][0],
                        "visual_token_indices_of_coordinates": processed[
                            "visual_token_indices_of_coordinates"
                        ][0],
                        "multi_patch_labels": processed["multi_patch_labels"][0],
                    }
                )

        data_dict["id"] = item_id

        # Qwen's processor has already expanded every image placeholder into
        # its visual-token sequence, so input_ids already includes image tokens.
        input_length = len(data_dict["input_ids"])
        if input_length > self.tokenizer.model_max_length:
            raise ValueError(
                f"Sample {item_id} has {input_length} tokens, exceeding "
                f"model_max_length={self.tokenizer.model_max_length}"
            )

        return data_dict

    def preprocess_qwen2vl(
        self,
        source,  # conversations
        tokenizer: transformers.PreTrainedTokenizer,
        processor: transformers.ProcessorMixin,
        image: list,
        system_message: str = grounding_system_message,
        agent_mode: bool = True,
        chat_template: str = chat_template,
        assistant_template: str = assistant_template,
        id: int = None,
    ) -> Dict:
        roles = {"human": "user", "gpt": "assistant", "system": "system"}
        assistant_template = assistant_template if agent_mode else chat_template
        processor.tokenizer = tokenizer
        assert tokenizer.additional_special_tokens == ADDITIONAL_SPECIAL_TOKENS

        # Apply prompt templates
        pixel_values, image_grid_thw = None, None

        input_id, target = [], []
        coordinates = []
        visual_token_indices_of_coordinates = []
        multi_patch_labels = []
        bbox_targets = []
        bbox_image_indices = []
        bbox_image_sizes = []

        image_list = []
        # Keep the most recently decoded image batch available to the
        # following assistant turn.  Training records normally contain one
        # still image followed by several user/assistant action turns; the
        # baseline relies on that convention for multi-query supervision.
        image_inputs = []
        image_token_counts = []
        image_index = 0

        ## prepare the system message and limit non-system conversation turns
        system_message, source = split_system_message_and_limit_turns(
            source,
            roles,
            system_message,
            self.data_args.max_conv_turns,
        )
        system_input_id = tokenizer.apply_chat_template(
            conversation=[
                {"role": "system", "content": [{"type": "text", "text": system_message}]}
            ],
            chat_template=chat_template,
        )
        input_id += system_input_id
        target += [IGNORE_INDEX] * len(system_input_id)

        ## prepare user-assistant conversation
        for conv in source:
            # regularize the conversation format
            try:
                role = conv["role"]
                content = conv["content"]
            except Exception:
                role = conv["from"]
                content = conv["value"]
            role = roles.get(role, role)

            # Count the number of <image> tokens in the content
            image_count = content.count(DEFAULT_IMAGE_TOKEN)
            if image_count > 0:
                assert role == "user", "Images are only supported for user messages"
                # include image information regarding to current conversation turn
                image_placeholders = []
                for _ in range(image_count):
                    image_placeholders.append(
                        {
                            "type": "image",
                            "image": image[image_index],
                            "min_pixels": self.processor.image_processor.min_pixels,
                            "max_pixels": self.processor.image_processor.max_pixels,
                        }
                    )
                    image_index += 1

                content = content.replace(DEFAULT_IMAGE_TOKEN, "")
                conv = {
                    "role": role,
                    "content": image_placeholders + [{"type": "text", "text": content}],
                }

                image_inputs, _ = process_vision_info([conv])  # list of PIL.Image.Image
                image_list.extend(image_inputs)

                templated_conv = tokenizer.apply_chat_template(
                    conversation=[conv], chat_template=chat_template, tokenize=False
                )
                inputs = processor(text=[templated_conv], images=image_inputs, return_tensors="pt")
                image_token_counts.extend(
                    int(grid[0].item() * grid[1].item() * grid[2].item())
                    // int(processor.image_processor.merge_size**2)
                    for grid in inputs["image_grid_thw"]
                )

                if pixel_values is None and image_grid_thw is None:
                    pixel_values = inputs["pixel_values"]
                    image_grid_thw = inputs["image_grid_thw"]
                else:
                    pixel_values = torch.concat([pixel_values, inputs["pixel_values"]], dim=0)
                    image_grid_thw = torch.concat([image_grid_thw, inputs["image_grid_thw"]], dim=0)
            else:
                if role in ["user", "system"]:
                    conv = {"role": role, "content": [{"type": "text", "text": content}]}
                else:  # assistant
                    conv = {
                        "role": role,
                        "content": [{"type": "text", "text": content}],
                        "recipient": conv.get("recipient", "os"),
                        "end_turn": conv.get("end_turn", True),
                        "bbox_gt": conv.get("bbox_gt", None),
                    }
                    if conv["recipient"] == "os":
                        if len(image_list) == 0:
                            raise ValueError("No image found for visual grounding")
                        # replace the coordinates with the special tokens
                        text, coord = reformat_coordinates(conv["content"][0]["text"])
                        conv["content"][0]["text"] = text
                        # rank0_print(f"coord: {coord}")

                        current_image = image_list[-1]
                        current_image_index = len(image_list) - 1
                        if self.include_pointer_targets:
                            coordinates.extend(coord)
                        for point_x, point_y in coord:
                            if self.include_pointer_targets:
                                visual_token_index = get_token_index(
                                    processor.image_processor, [current_image], point_x, point_y
                                )
                                visual_token_index += sum(image_token_counts[:-1])
                                visual_token_indices_of_coordinates.append(visual_token_index)

                            if conv["bbox_gt"] is not None and (
                                self.include_pointer_targets or self.include_bbox_targets
                            ):
                                bbox = torch.as_tensor(conv["bbox_gt"], dtype=torch.float32)
                                if bbox.shape != (4,):
                                    raise ValueError(
                                        f"bbox_gt must contain four values, got {conv['bbox_gt']!r}"
                                    )
                                if not torch.isfinite(bbox).all():
                                    raise ValueError("bbox_gt contains non-finite values")
                                if torch.any(bbox < 0) or torch.any(bbox > 1):
                                    raise ValueError(
                                        f"bbox_gt must be normalized to [0, 1], got {bbox.tolist()}"
                                    )
                                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                                    raise ValueError(
                                        "bbox_gt must satisfy x_min < x_max and y_min < y_max, "
                                        f"got {bbox.tolist()}"
                                    )
                                bbox_targets.append(bbox)
                                bbox_image_sizes.append(current_image.size)
                                if self.include_pointer_targets:
                                    patch_mask = get_multi_patch_labels(
                                        processor.image_processor, [current_image], conv["bbox_gt"]
                                    )
                                    multi_patch_labels.append(patch_mask)
                                bbox_image_indices.append(current_image_index)

                templated_conv = tokenizer.apply_chat_template(
                    conversation=[conv],
                    chat_template=assistant_template,
                    tokenize=False,
                )
                inputs = processor(text=[templated_conv], return_tensors="pt")

            encode_id = inputs.input_ids[0].tolist()

            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        # make the labels of all pointer_end_token_id to be IGNORE_INDEX
        target = [IGNORE_INDEX if token == self.pointer_end_token_id else token for token in target]

        input_ids = torch.tensor([input_id], dtype=torch.long)
        targets = torch.tensor([target], dtype=torch.long)
        visual_token_indices_of_coordinates = (
            torch.tensor([visual_token_indices_of_coordinates], dtype=torch.long)
            if len(visual_token_indices_of_coordinates) > 0
            else [None]
        )
        coordinates = [coordinates] if len(coordinates) > 0 else [None]

        # process multi_patch_labels
        if len(multi_patch_labels) > 0:
            total_visual_tokens = sum(image_token_counts)
            global_masks = []
            for local_mask, image_idx in zip(multi_patch_labels, bbox_image_indices):
                before = sum(image_token_counts[:image_idx])
                after = total_visual_tokens - before - image_token_counts[image_idx]
                global_masks.append(
                    torch.cat(
                        [
                            torch.zeros(before, dtype=local_mask.dtype),
                            local_mask,
                            torch.zeros(after, dtype=local_mask.dtype),
                        ]
                    )
                )
            multi_patch_labels = [torch.stack(global_masks)]
        else:
            multi_patch_labels = [None]
        if len(bbox_targets) > 0:
            bbox_targets = [torch.stack(bbox_targets)]
        else:
            bbox_targets = [None]

        data_dict = {
            "input_ids": input_ids,  # tensor(bs x seq_len)
            "labels": targets,  # tensor(bs x seq_len)
        }

        if pixel_values is not None:
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_grid_thw

        # if len(coordinates[0]) != len(visual_token_indices_of_coordinates[0]):
        #     raise ValueError(f"The number of coordinates ({len(coordinates[0])}) does not match the number of image token indices ({len(visual_token_indices_of_coordinates[0])})")
        if self.include_pointer_targets:
            data_dict["coordinates"] = coordinates
            data_dict["visual_token_indices_of_coordinates"] = visual_token_indices_of_coordinates
            data_dict["multi_patch_labels"] = multi_patch_labels
        data_dict["bbox_image_indices"] = (
            [torch.as_tensor(bbox_image_indices, dtype=torch.long)]
            if bbox_image_indices
            else [None]
        )
        data_dict["image_counts"] = [len(image_token_counts)]
        data_dict["bbox_targets"] = bbox_targets
        data_dict["bbox_image_sizes"] = (
            [torch.as_tensor(bbox_image_sizes, dtype=torch.float32)] if bbox_image_sizes else [None]
        )

        return data_dict
