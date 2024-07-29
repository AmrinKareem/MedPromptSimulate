import os
import pathlib
from globals import record, experience
from . import MMLU
from .embed_problems import embed_file
from .mmlu_paths import mmlu_data_dir, mmlu_generations_dir

model_name = "gpt-4-1106-preview"


def generate(dataset_name: str):
    # dev_problem = f"mmlu_{dataset_name}_val"
    test_problem = f"mmlu_{dataset_name}_test"

    # if not os.path.exists(str(mmlu_data_dir / dev_problem) + ".json.gz"):
    #     embed_file(str(mmlu_data_dir / dev_problem) + ".json")

    ################CHANGED CODE STARTS HERE##################
    #@amrin: We only look at one patient at a time so the test problem has only one question at a time. We need to make this a loop to receive queries in real time and generate responses in real time.
    if not os.path.exists(str(mmlu_data_dir / test_problem) + ".json.gz"):
        embed_file(str(mmlu_data_dir / test_problem) + ".json")

    # check if there are enough records and experience to retrieve examples
    if record < 3 and experience < 3:
        MMLU.generate_solutions_without_rank(
        test_problem, run_name=f"{test_problem}/cot", model=model_name
        )
    elif record < 3 and experience >= 3:
        MMLU.run_cot_without_rank(
            test_problem,
            run_name=f"{test_problem}/cot_knn",
            examples=str(
                mmlu_generations_dir / f"expt" / f"{test_problem}" / "cot_knn" / "experience"
            ),
            mode="knn",
            num_examples=3,
            num_repeat=15,
            max_thread=50,
            model=model_name,
        )
    elif record >= 3 and experience < 3:
        MMLU.run_cot_without_rank(
            test_problem,
            run_name=f"{test_problem}/cot_knn",
            examples=str(
                mmlu_generations_dir / f"expt" / f"{test_problem}" / "cot_knn" / "result"
            ),
            mode="knn",
            num_examples=3,
            num_repeat=15,
            max_thread=50,
            model=model_name,
        )
    else:
        MMLU.run_cot_without_rank(
            test_problem,
            run_name=f"{test_problem}/cot_knn",
            examples=[str(
                mmlu_generations_dir / f"expt" / f"{test_problem}" / "cot_knn" / "result"),
             str(mmlu_generations_dir / f"expt" / f"{test_problem}" / "cot_knn" / "experience")],
            mode="knn",
            num_examples=3,
            num_repeat=15,
            max_thread=50,
            model=model_name,
        )
    # MMLU.run_cot_without_rank(
    #     test_problem,
    #     run_name=f"{test_problem}/cot_via_knn",
    #     mode="knn",
    #     num_examples=5,
    #     num_repeat=15,
    #     max_thread=50,
    #     model=model_name,
    # )

     ########EXAMPLES NEED TO COME FROM COMMON MEDICAL RECORD LIBRARY BASED ON K WHERE K IS THE NUMBER OF CASES IN THE LIBRARY########

    if False:
        # Logprobs not currently available in OpenAI API
        MMLU.run_logprobs(
            test_problem,
            run_name=f"{test_problem}/logprobs5",
            num_examples=5,
            num_repeat=10,
            max_thread=50,
            model=model_name,
        )
