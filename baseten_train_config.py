"""
Baseten SSH Training Config.

Push this to Baseten, SSH in, train directly on GPU.

Usage:
    uvx truss login
    uvx truss ssh setup
    uvx truss train push baseten_train_config.py
    ssh training-job-<job_id>-0.ssh.baseten.co

Inside the SSH session:
    cd $BT_WORKING_DIR
    python train.py --data ./data/dataset.jsonl --tokenizer ./tokenizer.json --epochs 5

To transfer dataset + code:
    scp -r ./data training-job-<job_id>-0.ssh.baseten.co:$BT_WORKING_DIR/
    scp ./tokenizer.json training-job-<job_id>-0.ssh.baseten.co:$BT_WORKING_DIR/
"""

from truss_train import TrainingProject, TrainingJob, Image, Compute, Runtime
from truss_train.definitions import (
    InteractiveSession,
    InteractiveSessionTrigger,
    InteractiveSessionProvider,
)
from truss.base.truss_config import AcceleratorSpec

training_job = TrainingJob(
    image=Image(
        base_image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"
    ),
    compute=Compute(
        # GPU options: T4 (free tier), L4, A100, H100, H200
        # T4 is cheapest, cukup untuk model ~80M params
        accelerator=AcceleratorSpec(accelerator="T4", count=1),
    ),
    runtime=Runtime(
        # Optional: auto-run training on start
        # start_commands=[
        #     "pip install -r requirements.txt",
        #     "python train.py --data ./data/dataset.jsonl --tokenizer ./tokenizer.json --epochs 5",
        # ],
        # Better: SSH in manually so you can monitor & iterate
        start_commands=[],
    ),
    interactive_session=InteractiveSession(
        trigger=InteractiveSessionTrigger.ON_STARTUP,
        session_provider=InteractiveSessionProvider.SSH,
    ),
)

training_project = TrainingProject(
    name="empathy-transformer-training",
    job=training_job,
)
