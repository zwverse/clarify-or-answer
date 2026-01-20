
import os, sys
import wandb
if 'WANDB_PROJECT' not in os.environ:
    os.environ["WANDB_PROJECT"] = "grpo_tool_undefined"  # name your W&B project

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trl'))

from rewards import (
    ground_truth_similarity_reward,
    question_focused_relevance_reward, question_format_reward, ambiguity_resolution_reward, novelty_reward
)
from GRPO_GRTrainer import GRPOGRTrainer

from accelerate import Accelerator
from multimodalAmbiguityDataset import MultimodalAmbiguityDataset
accelerator = Accelerator()
import shutil

from transformers import (
    HfArgumentParser
)
from dataclasses import dataclass, field
from trl import (
    ModelConfig,
    GRPOConfig,
    ScriptArguments,
    get_peft_config,
)

wandb.init(
    project=os.environ.get("WANDB_PROJECT", "grpo_tool_undefined"),
    group="experiment_1", # Shared by both GPUs
    name=f"rank_{accelerator.process_index}",
    job_type="train"
)

@dataclass
class VLToolGRPOConfig(GRPOConfig):
    """
    Extended arguments for VLTool GRPO training script, including InternVL3-specific parameters.
    """
    eval_only: bool = field(default=False, metadata={"help": "Whether to only run evaluation."})
    setting: str = field(default="rr_use_external_grounding_tool", metadata={"help": "Experiment setting."})
    project_root_path: str = field(default="", metadata={"help": "The root path of the project."})
    python_path_for_dino: str = field(default="/usr/local/anaconda3/envs/grpo_env/bin/python", metadata={"help": "The conda environment path."})
    train_data_path: str = field(default="", metadata={"help": "Path to training data."})
    train_image_folder_path: str = field(default="", metadata={"help": "Path to training images."})
    eval_data_path: str = field(default="", metadata={"help": "Path to evaluation data."})
    eval_image_folder_path: str = field(default="", metadata={"help": "Path to evaluation images."})
    max_turns: int = field(default=2, metadata={"help": "Max conversational turns."})
    tool_port_starting_num: int = field(default=8020, metadata={"help": "Starting port number for tools."})

    # InternVL3-specific parameters
    downsample_ratio: float = field(default=0.5, metadata={"help": "Downsample ratio for vision inputs."})
    mlp_path: str = field(default=None, metadata={"help": "Path to pretrained MLP projector."})
    force_image_size: int = field(default=448, metadata={"help": "Forced resize image dimension."})
    dynamic_image_size: bool = field(default=True, metadata={"help": "Enable dynamic image resizing."})
    vision_select_layer: int = field(default=-1, metadata={"help": "Selected vision model layer for extraction."})
    min_dynamic_patch: int = field(default=8, metadata={"help": "Minimum dynamic patch size."})
    max_dynamic_patch: int = field(default=6, metadata={"help": "Maximum dynamic patch size."})
    use_thumbnail: bool = field(default=True, metadata={"help": "Use thumbnails instead of full images."})
    ps_version: str = field(default='v2', metadata={"help": "Position embedding scheme version."})
    drop_path_rate: float = field(default=0.1, metadata={"help": "Drop path rate for stochastic depth in ViT."})
    use_backbone_lora: int = field(default=0, metadata={"help": "LoRA rank for vision backbone."})
    use_llm_lora: int = field(default=0, metadata={"help": "LoRA rank for LLM."})
    freeze_backbone: bool = field(default=True, metadata={"help": "Freeze vision backbone parameters."})
    freeze_llm: bool = field(default=False, metadata={"help": "Freeze LLM parameters."})
    unfreeze_vit_layers: int = field(default=0, metadata={"help": "Number of ViT layers to unfreeze from the end."})
    freeze_mlp: bool = field(default=False, metadata={"help": "Freeze MLP layers."})
    unfreeze_lm_head: bool = field(default=False, metadata={"help": "Unfreeze the language model head."})
    conv_style: str = field(default="internvl2_5", metadata={"help": "Convolution style for the model."})
    down_sample_ratio: float = field(default=0.5, metadata={"help": "Downsample ratio for the model."})

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, VLToolGRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    if accelerator.is_main_process:

        shutil.rmtree(training_args.output_dir, ignore_errors=True)

    peft_config = get_peft_config(model_args)
    prompt_suffix = ''
    prompt = ''
    if script_args.dataset_name == 'rr':
        assert training_args.max_turns == 1

        # clarification question generation
        # grpo
        if 'grpo' in training_args.setting:
            prompt = '''You write exactly one clarifying question. Do not answer the original question.

Given an image and an original question, ask for the single missing fact that would determine the answer.

Input: (1) image, (2) original question.

Output: exactly one short clarification question without any prefix.

Rules:
1. Don't just repeat the original question. Ask for information not visible in the image but knowable by the user.
2. Don’t infer from visible cues; avoid “What does X mean?” or “What might be happening?”.
3. Be specific and minimal—target the variable that would flip the answer.
4. No lists, explanations, avoid definitions or speculation.

Now generate the clarification question.'''
            prompt_suffix = ''

        # direct ask
        if 'direct_ask' in training_args.setting:
            prompt = '''You write exactly one clarifying question. Do not answer the original question.

Given an image and an original question, ask for the single missing fact that would determine the answer.

Input: (1) image, (2) original question.

Output: exactly one short clarification question without any prefix.

Rules:
1. Don't just repeat the original question. Ask for information not visible in the image but knowable by the user.
2. Don’t infer from visible cues, avoid “What does X mean?” or “What might be happening?”.
3. Be specific and minimal—target the variable that would flip the answer.
4. No lists, explanations, avoid definitions or speculation.
            '''
            prompt_suffix = ''

        # icl
        if 'icl' in training_args.setting:
            prompt = '''You write exactly one clarifying question. Do not answer the original question.

Given an image and an original question, ask for the single missing fact that would determine the answer.

Input: (1) image, (2) original question.

Output: exactly one short clarification question without any prefix.

Rules:
1. Don't just repeat the original question. Ask for information not visible in the image but knowable by the user.
2. Don’t infer from visible cues, avoid “What does X mean?” or “What might be happening?”.
3. Be specific and minimal—target the variable that would flip the answer.
4. No lists, explanations, avoid definitions or speculation.

Examples:
Original Question: "The sound made by that man is very loud. Is his behavior polite?"
Clarification Question: "Which country is this taking place in?"

Original Question: "Do I see a front face or a side face?"
Clarification Question: "That depends on which figure you perceive. Are you focusing on the person formed by the black shape or the face formed in the white space?"

Now generate the clarification question.'''
            prompt_suffix = ''

        # pipeline level evaluation
        # baseline
        if 'baseline_r1' in training_args.setting:
            prompt = """You are a visual assistant. You answer the question directly whenever the image provides enough information. Only ask a clarification question if the answer would differ depending on missing information.
Input: (1) image, (2) original question.
Output (exactly one): final answer or one short clarification question
"""

        # CoA pipeline
        if 'pipeline_no_r1' in training_args.setting:
            prompt = """You are a visual assistant. Answer the user's question using the image.
Input: (1) image, (2) original question.
Output: final answer"""

        if 'pipeline_yes_r1' in training_args.setting:
            prompt = '''You are a visual assistant. Ask exactly one concise clarification question that, once answered, would change the final answer.

Input: (1) image, (2) original question.

Output: exactly one short clarification question without any prefix.
Rules:
- Ask only one question.
- Do not answer the original question.
- Do not ask about details clearly visible in the image.'''

        if 'r2' in training_args.setting:
            prompt = """You are a visual assistant. Answer the user's question using the image.
Input: (1) image, (2) original question.
Output: final answer"""

        tools = ['BoundingboxBrushTool:bbox-brush']

        if ',' in training_args.train_data_path:
            assert ',' in training_args.train_image_folder_path, "Please provide the same number of image folders as the number of train data paths."
            train_dataset = {}
            train_data_paths = training_args.train_data_path.split(',')
            train_image_folder_paths = training_args.train_image_folder_path.split(',')
            train_dataset = MultimodalAmbiguityDataset(train_data_paths, train_image_folder_paths, prompt = prompt, prompt_suffix = prompt_suffix)
        else:
            train_dataset = MultimodalAmbiguityDataset(training_args.train_data_path, training_args.train_image_folder_path, prompt = prompt, prompt_suffix = prompt_suffix)

        if ',' in training_args.eval_data_path:
            assert ',' in training_args.eval_image_folder_path, "Please provide the same number of image folders as the number of eval data paths."
            eval_dataset = {}
            eval_data_paths = training_args.eval_data_path.split(',')
            eval_image_folder_paths = training_args.eval_image_folder_path.split(',')
            for i, eval_data_path in enumerate(eval_data_paths):
                eval_dataset[eval_data_path] = MultimodalAmbiguityDataset(eval_data_path, eval_image_folder_paths[i], prompt = prompt, prompt_suffix = prompt_suffix)
        else:
            eval_dataset = MultimodalAmbiguityDataset(training_args.eval_data_path, training_args.eval_image_folder_path, prompt = prompt, prompt_suffix = prompt_suffix, limits=None) #30 if not training_args.eval_only else

        if 'baseline_r1' in training_args.setting:
            REWARD_FUNCS_REGISTRY = {
                "ambiguity_resolution_reward": ambiguity_resolution_reward,
            }
        else:
            REWARD_FUNCS_REGISTRY = {
                "question_format_reward": question_format_reward,
                "question_focused_relevance_reward": question_focused_relevance_reward,
                "novelty_reward": novelty_reward,
                "ground_truth_similarity_reward": ground_truth_similarity_reward,
                'ambiguity_resolution_reward': ambiguity_resolution_reward,
            }

        reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in REWARD_FUNCS_REGISTRY.keys()]

    else:
        raise ValueError(f"Dataset name in {training_args.setting} is not supported.")    
    
    end_flag = False
    if os.path.exists(training_args.output_dir):
        checkpoint_list = [d for d in os.listdir(training_args.output_dir) if d.endswith('end_of_training.txt')]
        if len(checkpoint_list) > 0:
            print(f"Training has been finished. Please remove {training_args.output_dir} to continue training.")
            end_flag = True
        

    if not end_flag:
        ################
        # Training
        ################
        trainer = accelerator.prepare(GRPOGRTrainer( 
            args=training_args,
            model=  model_args.model_name_or_path, 
            tool_names = tools,
            reward_funcs= reward_funcs, 
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,   
        ))

        if training_args.eval_only:
            trainer.evaluate()
        else:
            trainer.train()

        # save a mark for the end of training
        with open(os.path.join(training_args.output_dir, "end_of_training.txt"), "w") as f:
            f.write("Training finished.\n")
            
        
