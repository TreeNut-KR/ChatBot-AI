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
from torch.utils.data import DataLoader, DistributedSampler

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
        
        # AI_Llama_8B.py와 동일한 설정
        self.cache_dir = "./fastapi/ai_model"
        self.model_id = "meta-llama/Llama-3.1-8B-Instruct"
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")
        
        # BitsAndBytes 설정
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.lora_config = LoraConfig(
            r=8,  # 랭크
            lora_alpha=32,  # 알파 값
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # GPU 메모리 설정
        self.max_memory = {
            0: "10GiB",  # 12GB GPU에 10GiB 메모리 할당
            1: "6GiB",   # 8GB GPU에 6GiB 메모리 할당
            "cpu": "16GiB"
        }
        
        # 오프로드 폴더 설정
        self.offload_folder = "./fastapi/ai_model/offload"
        os.makedirs(self.offload_folder, exist_ok=True)
        
        # Accelerator 초기화
        self.accelerator = Accelerator(cpu=False)

        print("토크나이저 로드 중...")
        self.tokenizer = self.load_tokenizer()
        print("모델 로드 중...")
        self.model = self.load_model()
        
    def load_tokenizer(self):
        """토크나이저 로드"""
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            token=self.hf_token,
            cache_dir=self.cache_dir
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_model(self):
        """모델 로드 및 PEFT 설정"""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=self.hf_token,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            device_map=None,  # auto를 사용하지 않음
            offload_folder=self.offload_folder,
            max_memory=self.max_memory,
            low_cpu_mem_usage=True,
            quantization_config=self.quantization_config
        )
        # 모델 구조 출력 (디버깅용)
        print(model)

        # LoRA 및 기타 설정
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, self.lora_config)
        model.gradient_checkpointing_enable()

        # 모델을 Accelerator와 함께 병렬화
        model = self.accelerator.prepare(model)

        return model
        
    def preprocess_function(self, examples):
        """데이터 전처리"""
        try:
            inputs = [f"질문: {q} 답변:" for q in examples["instruction"]]
            model_inputs = self.tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(examples["output"], max_length=512, truncation=True, padding="max_length")

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        except Exception as e:
            print(f"데이터 전처리 중 오류 발생: {str(e)}")
            raise

    def train_model(self):
        """모델 학습 실행"""
        try:
            print("데이터셋 로드 중...")
            dataset_path = "./fastapi/datasets/maywell/ko_wikidata_QA"
            dataset = load_from_disk(dataset_path)
            train_data = dataset["train"]
            train_size = len(train_data)
            sample_size = int(train_size)
            sampled_data = train_data.select(range(sample_size))
            
            print("데이터 전처리 중...")
            tokenized_dataset = sampled_data.map(
                self.preprocess_function,
                batched=True,
                remove_columns=sampled_data.column_names
            )

            # 분산 샘플러 설정 (각 GPU에 적절한 데이터 분배)
            train_sampler = DistributedSampler(tokenized_dataset, num_replicas=self.num_gpus, rank=self.accelerator.process_index)

            # 데이터 로더 설정
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            train_dataloader = DataLoader(
                tokenized_dataset,
                sampler=train_sampler,
                batch_size=4,
                collate_fn=data_collator
            )

            # 학습 인자 설정
            training_args = TrainingArguments(
                output_dir=self.offload_folder,
                learning_rate=2e-4,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=16,
                max_steps=1000,
                warmup_steps=100,
                logging_steps=10,
                save_steps=200,
                fp16=True,  # 혼합 정밀도 설정
                gradient_checkpointing=True,
                logging_dir="./logs",
                dataloader_num_workers=1,
                dataloader_pin_memory=False,
                ddp_find_unused_parameters=False,
                optim="paged_adamw_32bit",
                local_rank=self.accelerator.process_index  # 분산 학습을 위한 설정
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )

            print("학습 시작...")
            trainer.train()
            
            print("모델 저장 중...")
            trainer.save_model()
            print("학습 완료!")

        except Exception as e:
            print(f"학습 중 오류 발생: {str(e)}")
            raise

if __name__ == "__main__":
    trainer = MultiGPUTrainer()
    trainer.train_model()