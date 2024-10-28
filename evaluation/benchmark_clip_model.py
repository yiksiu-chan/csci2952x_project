from functools import partial
from io import open_code
from unibench import Evaluator
from unibench.models_zoo.wrappers.clip import ClipModel
import open_clip
import torch

import sys
import os
sys.path.append(os.path.abspath("../training"))
import CustomCLIP


device = "cuda" if torch.cuda.is_available() else "cpu"

custom_clip = CustomCLIP(
            "openai-community/gpt2", 
            "google/vit-huge-patch14-224-in21k", 
            embedding_dim=1024,
            use_peft=True
        ).to(device)

tokenizer = open_clip.get_tokenizer("ViTamin-L")

model = partial(
    ClipModel,
    model=custom_clip,
    model_name="clip-small-text-large-vision",
    tokenizer=tokenizer,
    input_resolution=custom_clip.visual.image_size[0],
    logit_scale=custom_clip.logit_scale,
)


eval = Evaluator(benchmarks_dir="../benchmark_data/insert_benchmark_folder_here")

eval.add_model(model=model)
eval.update_benchmark_list(["benchmark_name"])
eval.update_model_list(["clip-small-text-large-vision"])
eval.evaluate()
