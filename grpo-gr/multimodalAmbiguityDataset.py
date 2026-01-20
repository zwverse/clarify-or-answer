from torch.utils.data import DataLoader, Dataset
import jsonlines
import os
class MultimodalAmbiguityDataset(DataLoader):
    def __init__(self, data_path, image_folder_path, prompt, limits=None, prompt_suffix = ''):
        # sample data: {"question": "What is across from the truck?", "answer": "tv", "image": "000000410123.jpg", "width": 640, "height": 354, "bboxs": [[343.87, 94.03, 396.81, 126.06]], "dataset": "vsr", "split": "train"}
        if type(data_path) == str:
            self.dataset = list(jsonlines.open(data_path))
            self.image_folder_path = image_folder_path
        else:
            self.dataset = []
            for i, dp in enumerate(data_path):
                
                _dataset =  list(jsonlines.open(dp))
                for d in _dataset:
                    d['image'] = os.path.join(image_folder_path[i], d['image'])
                self.dataset+= _dataset
            self.image_folder_path = ''
        if limits:
            self.dataset = self.dataset[:limits]
        
        self.prompt = prompt
        self.prompt_suffix = prompt_suffix

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        question = data["question"]
        # answer = str(data["answer"])
        if data["dataset"] == "clarification":
            answer = data["clarification_request"]
            category = data["category"]
        elif data["dataset"] == "clearVQA":
            answer = data["clarification_request"]
            category = ""
        else:
            answer = ""
            category = ""
        image =  os.path.join(self.image_folder_path, data["image"])
        # "messages":{
        # "role": "user"
        # "content": [
        #       {"type": "image" , "image": image_path[0]},
        #       {"type": "text", "text": task},
        # ],

        if "real_question" in data:
            needs_clarification = 'yes'
        else:
            needs_clarification = 'no'
        returns =  {"message": 
                    [
                        {
                            "role": "user", 
                            "content": [
                                    {"type": "image", "image": image}, 
                                    {"type": "text", "text": self.prompt+ "\nQuestion: " + question + self.prompt_suffix} #
                                ]
                        } 
                    ], 
                "gt_answer": answer,
                "image": image,
                "dataset": data["dataset"],
                "category": category,
                "question": question,
                "needs_clarification": needs_clarification
                }
        if 'key_words' in data:
            returns['key_words'] = data['key_words']
        if 'bboxs' in data:
            returns['bboxs'] = data['bboxs']
        if 'width' in data:
            returns['width'] = data['width']
        if 'height' in data:
            returns['height'] = data['height']
        
        # if self.prompt_suffix != '':
        #     returns['message'].append(
        #         {
        #             "role": "assistant",
        #             "content": [
        #                 {"type": "text", "text": "Answer: <think_start>"}
        #             ]
        #         }
        #     )
        return returns
        
        
        # return {"prompt": self.prompt+task, "gt_answer": answer, "image": image}