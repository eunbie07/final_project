from google.cloud import aiplatform
from google.cloud.aiplatform.jobs import ModelTuningJob

# ... (이전과 동일한 설정) ...
PROJECT_ID = "virtual-muse-466706-v2"
LOCATION = "us-central1"
GCS_METADATA_PATH = "gs://my-room-finetune-bucket/metadata.jsonl"
TUNED_MODEL_NAME = "korean-room-style-model-v1"
JOB_DISPLAY_NAME = "korean-room-style-tuning-job-1"
SOURCE_MODEL_ID = "imagen-3.0-fast-generate-001"
# ----------------------------------------------------------------------

print(f"Vertex AI를 초기화합니다... (Project: {PROJECT_ID})")
aiplatform.init(project=PROJECT_ID, location=LOCATION)

print(f"'{TUNED_MODEL_NAME}' 모델의 파인튜닝 작업을 시작합니다...")
job = ModelTuningJob.run(
    display_name=JOB_DISPLAY_NAME,
    source_model=SOURCE_MODEL_ID,
    training_dataset=GCS_METADATA_PATH,
    tuned_model_display_name=TUNED_MODEL_NAME,
    hyperparameters={
        "train_steps": 300
    }
)

print("\n파인튜닝 작업이 성공적으로 제출되었습니다!")
print("Google Cloud 콘솔에서 아래 작업 ID를 통해 진행 상황을 확인하세요.")
print(f"작업 리소스 이름: {job.resource_name}")