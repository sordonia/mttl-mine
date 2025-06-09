import os
from dataclasses import dataclass

from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.models.library.dataset_library import DatasetLibrary


@dataclass
class OrcaDataConfig(DatasetConfig):
    pass


@DataModule.register("orca", config_cls=OrcaDataConfig)
class OrcaDataModule(DataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset = DatasetLibrary.pull_dataset("Open-Orca/OpenOrca")

        def map_example(example):
            # Build the source from system_prompt and question
            if example["system_prompt"] and example["system_prompt"].strip():
                source = f"{example['system_prompt']}\n\n{example['question']}"
            else:
                source = example["question"]
            
            target = example["response"]
            
            return {
                "source": source,
                "target": target,
                "task_name": "orca",
                "task_source": "orca",
                "id": example["id"],
            }

        dataset = dataset.map(
            map_example,
            num_proc=n_proc,
        )

        self._task_to_id = {"orca": 0}
        self._task_names = ["orca"]

        # Create train/validation split since OpenOrca only has a train split
        self.train_dataset, self.dev_dataset = self.create_train_valid_split(dataset["train"])
        self.test_dataset = self.dev_dataset
