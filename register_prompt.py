import mlflow
from datetime import datetime
import os

# Configure logging
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Double curly braces for prompt variable placeholders
grading_prompt_template = """<s>[INST] <<SYS>>
Your task is to choose the MOST suitable option among a set of options I provide, about a code which will also be provided. Give your output as a json with a single field "answer". Do not output anything else. Strictly follow this output format at any cost.
<</SYS>>

### Context : 
{{ context }}

### Code : 
{{ code }}

### Task : 
{{ task }}

### Options : 
{{ options }}

### Response : The required output in json format is : [/INST]"""


def register_prompt():
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("code-grader")
        with mlflow.start_run(run_name="prompt_registration"):
            # Log prompt as an artifact
            prompt_path = "grading_prompt_template.txt"
            with open(prompt_path, "w") as f:
                f.write(grading_prompt_template)
            mlflow.log_artifact(prompt_path, artifact_path="prompts")
            os.remove(prompt_path)

            # Log prompt metadata
            current_date = datetime.today().strftime("%Y-%m-%d")
            metadata = {
                "name": "DPO_Prompt",
                "commit_message": "Prompt For Fine-Tuning Using DPO",
                "author": "saurav@cse.iitb.ac.in",
                "date": current_date,
            }
            mlflow.log_dict(metadata, "prompt_metadata.json")

            # Log tags
            tags = {
                "task": "grading",
                "model_type": "DPO",
                "language": "cpp",
            }
            for key, value in tags.items():
                mlflow.set_tag(key, value)

            logger.info(f"Successfully registered prompt 'DPO_Prompt' with metadata")
    except Exception as e:
        logger.error(f"Failed to register prompt: {e}")
        raise


if __name__ == "__main__":
    register_prompt()
