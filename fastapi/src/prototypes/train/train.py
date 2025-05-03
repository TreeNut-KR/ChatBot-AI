import os
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_from_disk
from accelerate import Accelerator
from dotenv import load_dotenv
from torch.utils.data import DataLoader


class MultiGPUTrainer:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다.")
        
        # 환경 변수 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        dotenv_path = os.path.join(parent_dir, '.env')
        load_dotenv(dotenv_path)
        
        self.num_gpus = torch.cuda.device_count()
        print(f"사용 가능한 GPU 개수: {self.num_gpus}")
        
        # GPU 메모리 설정
        self.max_memory = self.get_gpu_memory()
        print(f"GPU 메모리 설정: {self.max_memory}")
        
        # 모델 및 토크나이저 설정
        self.cache_dir = "./fastapi/ai_model"
        self.model_id = "meta-llama/Llama-3.1-8B-Instruct"
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")
        
        # BitsAndBytes 설정
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type = "nf4"
        )
        
        self.lora_config = LoraConfig(
            r = 8,
            lora_alpha = 32,
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout = 0.05,
            bias = "none",
            task_type = TaskType.CAUSAL_LM
        )
        
        # 오프로드 폴더 설정
        self.offload_folder = "./fastapi/ai_model/offload"
        os.makedirs(self.offload_folder, exist_ok = True)
        
        # Accelerator 초기화
        self.accelerator = Accelerator(cpu = False)

        # DeepSpeed 설정
        self.ds_config = {
            "train_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-5
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                }
            }
        }

        # 토크나이저 및 모델 로드
        print("토크나이저 로드 중...")
        self.tokenizer = self.load_tokenizer()
        print("모델 로드 중...")
        self.model = self.load_model()
        print("모델 로드 완료")

    def get_gpu_memory(self):
        """각 GPU의 사용 가능한 VRAM을 측정하여 반환"""
        max_memory = {}
        for i in range(self.num_gpus):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            reserved_memory = torch.cuda.memory_reserved(i)
            allocated_memory = torch.cuda.memory_allocated(i)
            free_memory = total_memory - (reserved_memory + allocated_memory)
            max_memory[i] = f"{free_memory // (1024 ** 3)}GiB"
        max_memory["cpu"] = "16GiB"
        return max_memory

    def load_tokenizer(self):
        """토크나이저 로드"""
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            token = self.hf_token,
            cache_dir = self.cache_dir
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_model(self):
        """모델 로드 및 PEFT 설정"""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token = self.hf_token,
            cache_dir = self.cache_dir,
            torch_dtype = torch.float16,
            device_map = None,
            offload_folder = self.offload_folder,
            max_memory = self.max_memory,
            low_cpu_mem_usage = True,
            quantization_config = self.quantization_config
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, self.lora_config)
        model.gradient_checkpointing_enable()

        # 모델을 Accelerator와 함께 병렬화
        model = self.accelerator.prepare(model)
        return model
        
    def preprocess_function(self, examples):
        """데이터 전처리"""
        inputs = [f"질문: {q} 답변:" for q in examples["instruction"]]
        model_inputs = self.tokenizer(inputs, max_length = 512, truncation = True, padding = "max_length")

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples["output"], max_length = 512, truncation = True, padding = "max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train_model(self):
        """모델 학습 실행"""
        print("데이터셋 로드 중...")
        dataset_path = "./fastapi/datasets/maywell/ko_wikidata_QA"
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"데이터셋 경로를 찾을 수 없습니다: {dataset_path}")
        
        dataset = load_from_disk(dataset_path)
        train_data = dataset["train"]
        sampled_data = train_data.select(range(len(train_data)))
        
        print("데이터 전처리 중...")
        tokenized_dataset = sampled_data.map(
            self.preprocess_function,
            batched = True,
            remove_columns = sampled_data.column_names
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer = self.tokenizer,
            mlm = False
        )

        training_args = TrainingArguments(
            output_dir = self.offload_folder,
            learning_rate = 2e-4,
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            max_steps = 1000,
            warmup_steps = 100,
            logging_steps = 10,
            save_steps = 200,
            fp16 = True,
            gradient_checkpointing = True,
            logging_dir = "./logs",
            dataloader_num_workers = 1,
            dataloader_pin_memory = False,
            optim = "paged_adamw_32bit",
            remove_unused_columns = False,
            deepspeed = self.ds_config  # DeepSpeed 활성화
        )

        trainer = Trainer(
            model = self.model,
            args = training_args,
            train_dataset = tokenized_dataset,
            data_collator = data_collator,
            tokenizer = self.tokenizer
        )

        print("학습 시작...")
        trainer.train()
        print("모델 저장 중...")
        trainer.save_model()
        print("학습 완료!")

def main():
    trainer = MultiGPUTrainer()
    trainer.train_model()

if __name__  ==  "__main__":
    main()
