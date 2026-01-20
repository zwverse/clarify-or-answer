"""Reward functions for GRPO-GR training."""

import re


import numpy as np
import json
import re
from collections import defaultdict

import nltk
from nltk.translate.bleu_score import sentence_bleu

from callGPT import call_gpt
from typing import Dict, Set
from itertools import combinations

from concurrent.futures import ThreadPoolExecutor
import re


KEYWORDS: Dict[str, Set[str]] = {
    "Cultural Background": {
        "country","city","state","region","location","place","area","nation","where",
        "origin","cultural","tradition","custom","law","legal","illegal","rules", "culture"
    },
    "Temporal Information": {
        "before","after","prior","following","earlier","later","recently","soon","now",
        "current","time","date","day","night","morning","afternoon","evening",
        "today","tomorrow","yesterday"
    },
    "Location and Orientation": {
        "left","right","up","down","forward","backward","toward","away","direction",
        "push","pull","open","close","above","below","enter","exit"
    },
    "Attributes": {
        "hot","cold","on","off","high","low","big","small","large","short","long",
        "heavy","light","temperature","speed","limit","safe","dangerous","bright",
        "dark","clean","dirty","full","empty"
    },
    "Relationships": {
        "who","which","person","man","woman","boy","girl","friend","family","colleague",
        "partner","teacher","student","owner","driver","passenger","speaker","listener",
        "helper"
    }
}

VAGUE_PATTERNS = [
    r"\b(tell me more|explain|elaborate|give (me )?details|describe|more info|clarify|can you clarify|what's going on)\b",
    r"\b(i don't understand|not sure|confused|what do you mean)\b",
    r"\b(any|anything|something|everything)\b",
    r"\b(why)\b$"  # bare "why?" without a target
]

SPECIFIC_GENERIC_PATTERNS = [
    # explicit choice / alternatives
    r"\b(either|neither)\b",
    r"\b\w+\s+or\s+\w+\b",                     # A or B
    # tight wh-question targeting a property
    r"^\s*(which|what)\s+\w+(?:\s+\w+){0,3}\?\s*$",
    r"\b(what|which)\s+(one|type|kind|side|part|version|value)\b",
    r"\b(who|what)\s+(is|was)\s+(it|this|that|they|he|she)\b",
    r"\b(what|which)\s+\w+\s+(does|is|was|are|were)\b",
    # comparative cues
    r"\b(before|after|earlier|later|sooner|bigger|smaller|higher|lower|more|less)\b",
    # motion/orientation (generic)
    r"\b(going|moving|facing|turning|pointing)\b",
    # state/status queries
    r"\b(is|was)\s+(it|this|that|he|she|they)\s+\w+\b",
    # purpose/intent
    r"\b(what|which)\s+(for|purpose|reason|goal)\b",
    # reference resolution
    r"\bwhich\s+(one|of\s+(them|these|those))\b"
]

_VAGUE = [re.compile(p, re.I) for p in VAGUE_PATTERNS]
_SPEC  = [re.compile(p, re.I) for p in SPECIFIC_GENERIC_PATTERNS]

clarification_response_prompt = f"""
You are a simulator of realistic user replies to clarification questions in a visual Q&A conversation.

Given:
- An image
- The original user question
- A clarification question

Your task:
- Generate exactly {{num_responses}} **different, plausible factual answers** to the clarification question, as if they came from different people in different contexts.
- Each answer must be **short** (one sentence or less) and **fact-based**.
- Avoid explanations, opinions, or reasoning. Just state the fact.
- At least two answers should lead to **different possible final answers** to the original question.
- Responses must be realistic given the image and the type of clarification question.

Format:
Return the answers as a numbered list.

Examples:

Clarification Q: "In which country is this gesture being used?"
Good answers:
1. United States.
2. Brazil.
3. Japan.
Bad answers:
- It seems friendly so probably not offensive.   (← explanation)
- I think it's Brazil because of the buildings.  (← reasoning)

Clarification Q: "Is this moment before they arrived or after they are leaving?"
Good answers:
1. Before they arrived.
2. Just after they left.
3. As they were arriving.

Original question:
{{original_question}}

Clarification question:
{{clarification_question}}

Return the responses as a list, one per line.
"""

final_answer_prompt = f"""
You are a visual Q&A assistant. Given an image, the original question, and a clarification response, 
answer the original question as accurately as possible. Base your answer only on visible image content and the given clarification response. 
Be concise: one short sentence. No explanations.

Original question:
{{original_question}}

Clarification question:
{{clarification_question}}

Clarification response:
{{clarification_response}}
"""

check_final_answers_prompt = f"""
    Original question: {{original_question}}
    Answer A: {{answer_a}}
    Answer B: {{answer_b}}

    Do these answers convey the same meaning in this context?
    Reply "same" or "different".
"""

def question_format_reward(prompts, completions, gt_answer, **kwargs):
    """Reward if generated response contains correct answer."""
    print("prompt", prompts)
    print("gt_answer", gt_answer)
    print("original_questions", kwargs["question"])
    print("clarification_requests", completions)
    rewards = []
    for response in completions:
        reward = 0.5 if is_valid_question(response) else -0.5
        rewards.append(reward)
    return rewards

def is_valid_question(response):
   q = response.lower().strip()
   if "\n" in q or "```" in q: return False
   if not q.endswith("?") : return False
   wc = len(re.findall(r"\b\w+(?:[-']\w+)*\b", q))
   return 4 <= wc <= 50


def question_focused_relevance_reward(prompts, completions, gt_answer, **kwargs):
    rewards = []
    for i in range(len(completions)):
        response = completions[i].lower()
        category = kwargs["category"][i]
        if category == "":
            reward = 0
        elif is_valid_question(response):
            is_targeted = False
            is_relevant = any(k in response for k in KEYWORDS.get(category))

            for pat in _SPEC:
                if pat.search(response):
                    is_targeted = True

            if is_relevant and is_targeted:
                reward = 0.3
            elif is_relevant or is_targeted:
                reward = 0.1
            else:
                reward = -0.1
        else:
            reward = 0

        rewards.append(reward)
    return rewards

STOP = {"the","a","an","and","or","of","to","in","on","for","is","are","was","were","be","this","that","it","with","from","by","at","as","do","does","did"}

def content_tokens(s):
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP}

def jaccard(a, b):
    inter = len(a & b); union = len(a | b) or 1
    return inter / union

def novelty_reward(prompts, completions, gt_answer, **kwargs):
    orig_q = kwargs["question"][0]
    rewards = []
    for response in completions:
        reward = 0
        if is_valid_question(response):
            clar_q = response.lower().strip()
            reward = get_novelty_reward(orig_q, clar_q)
        rewards.append(reward)
    return rewards

def get_novelty_reward(orig_q, clar_q):
    A = content_tokens(orig_q)
    B = content_tokens(clar_q)
    J = jaccard(A, B)  # 0..1

    # If the clarification introduces very few new tokens, penalize.
    new_ratio = len(B - A) / max(1, len(B))

    # Heuristic thresholds (tune):
    if J >= 0.8 and new_ratio < 0.2:
        reward = -0.3          # near-echo
    elif J >= 0.6 and new_ratio < 0.3:
        reward = -0.1           # mild echo
    elif new_ratio >= 0.3 or J < 0.6:
        reward = 0.3         # rewards novelty a bit
    return reward

def ground_truth_similarity_reward(prompts, completions, gt_answer, **kwargs):
    rewards = []
    original_question = kwargs["question"]
    for i in range(len(completions)):
        completion = completions[i].lower()
        if gt_answer[i] == "":
            reward = 0.0
        elif is_valid_question(completion):
            prefix = "clarification question:"
            response = completion.split(prefix, 1)[1].strip() if prefix in completion else completion
            prompt = f"""You are responsible for proofreading the clarification question, you need to give a score to the model’s clarification question by referring to the standard clarification question, based on the given original question. The full score is 1 point and the minimum score is 0 points. Please output the score in the json form "{{score: <score>}}". The evaluation criteria require that the closer the model’s clarification question is to the standard clarification question, the higher the score.
    Question: {original_question[i]}
    Standard clarification question: {gt_answer[i]}
    Model’s clarification question: {response}"""

            tempt = 0
            reward = 0.0
            while tempt <3:
                tempt+=1
                try:
                    output = re.sub(r'[^\w\s\.]', '', call_gpt(prompt).strip().lower())
                    output_num = output.split("score")[-1][:5]

                    # remove any text that is not a number
                    output = re.sub(r'[^\d.]', '', output_num)
                    reward = float(output)
                    break
                except Exception as e:
                    print("Error in gpt_score_reward: ", e)
        else:
            reward = 0
        rewards.append(reward)
    return rewards

def ambiguity_resolution_reward(prompts, completions, gt_answer, **kwargs):
    rewards = []
    try:
        for i in range(len(completions)):
            response = completions[i].lower()
            if response.endswith("?"):
                final_answers = []
                prompt_to_get_response = clarification_response_prompt.format(
                    num_responses=2,
                    original_question=kwargs["question"][i],
                    clarification_question=response,
                )
                clarification_responses = re.sub(r'[^\w\s\.]', '', call_gpt(prompt_to_get_response, kwargs["image"][i]).strip().lower())
                print('clarification_responses', clarification_responses)
                outputs = [line.strip() for line in clarification_responses.splitlines() if line.strip()]
                for output in outputs:
                    prompt_to_get_final_answer = final_answer_prompt.format(
                        original_question=kwargs["question"][i],
                        clarification_question=response,
                        clarification_response=output,
                    )
                    original = call_gpt(prompt_to_get_final_answer, kwargs["image"][i]).strip().lower()
                    final_answer = re.sub(r'[^\w\s\.]', '', original)
                    final_answers.append(final_answer)
                print('final_answers', final_answers)
                if check_ambiguity_resolved(final_answers, kwargs["question"][i], kwargs["image"][i]):
                    # resolve the ambiguity
                    reward = 0.5
                else:
                    reward = -0.3
                rewards.append(reward)
            else:
                rewards.append(0)
    except Exception as e:
        print(e)
        print("an except happened when calculating the ambiguity resolution reward")
        rewards.append(0)

    return rewards

def same_meaning(a, b, orig_q, image_path, caller_type):
    prompt = check_final_answers_prompt.format(
        original_question=orig_q,
        answer_a = a,
        answer_b = b,
    )
    resp = re.sub(r'[^\w\s\.]', '', call_gpt(prompt, image_path, caller_type=caller_type).strip().lower())
    print("same_meaning", resp)
    return resp.strip().lower() == "same"

def check_ambiguity_resolved(final_answers, orig_q, image_path, caller_type=None,):
    # final_answers: list of 2 strings
    different_found = False
    if len(final_answers) == 2:
        a1 = final_answers[0]
        a2 = final_answers[1]
        if not same_meaning(a1, a2, orig_q, image_path, caller_type):
            different_found = True
    return different_found

def gpt_score_reward(prompts, completions, gt_answer, **kwargs):
    """Reward based on gpt-4o evaluation between generated response and correct answer."""
    rewards = []
    # pattern = r'<answer>(.*?)<answer_end>'  # generated by chatGPT
    for i in range(len(gt_answer)):
        if 'dataset' in kwargs and 'ovd' in kwargs['dataset'][i]:
            rewards.append(0.0)
            continue
        response, answer = completions[i], gt_answer[i]
        question = prompts[i][0]['content'][-1]['text'].split('Please answer question: ')[-1]
        question = kwargs['question']

        reward = 0.0
        if '<answer>' in response:
            match_pattern = response.split('<answer>', 1)
            predicted_content = match_pattern[-1].split('</answer>')[0]
            prompt = f"""You are responsible for proofreading the answers, you need to give a score to the model’s answer by referring to the standard answer, based on the given question. The full score is 1 point and the minimum score is 0 points. Please output the score in the json form "{{score: <score>}}". The evaluation criteria require that the closer the model’s answer is to the standard answer, the higher the score.
Question: {question} 
Standard answer: {answer} 
Model’s answer: {predicted_content}"""

            tempt = 0
            while tempt <3:
                tempt+=1
                try:
                    output = re.sub(r'[^\w\s\.]', '', call_gpt(prompt).strip().lower())
                    output_num = output.split("score")[-1][:5]
                    
                    # remove any text that is not a number 
                    output = re.sub(r'[^\d.]', '', output_num)
                    output = float(output)
                    break
                except Exception as e:
                    print("Error in gpt_score_reward: ", e)
                    output =  0
            reward += output
        rewards.append(reward)
    
    return rewards