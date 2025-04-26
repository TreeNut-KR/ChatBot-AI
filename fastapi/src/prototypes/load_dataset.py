import os
from datasets import load_dataset

def download_and_save_dataset():
    dataset_name="maywell/ko_wikidata_QA"
    save_path=f"./fastapi/datasets/{dataset_name}"

    # 폴더가 없는 경우 생성
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"폴더 생성 완료: {save_path}")
    
    # 데이터셋 불러오기
    dataset=load_dataset(dataset_name)
    
    # 데이터셋 구조 확인
    print(dataset)
    # 첫 번째 데이터셋의 예시 확인
    print(dataset['train'][0])
    
    # 데이터셋 저장
    dataset.save_to_disk(save_path)
    print(f"데이터셋 저장 완료: {save_path}")

if __name__ == "__main__":
    download_and_save_dataset()
