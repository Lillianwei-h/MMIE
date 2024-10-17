import os
import json

def get_dataset(input_dir, file_name, temp_output_path=None):
    data_path = os.path.join(input_dir,file_name)
    with open(data_path, 'r') as f:
        data = json.load(f)

    processed_data = []
    for d in data:
        processed_data_dict = {}
        id = d['id']
        model = d['model']
        question = d['question']
        answer = d['answer']
        gt_answer = d['gt_answer'][0]['text']
        question_text = ""
        answer_text = ""
        image_no = 0
        images = []

        for q in question:
            if q['text'] is not None:
                question_text+=q['text']+'\n'
            if q['image'] is not None:
                image_path = os.path.join(input_dir,q['image'])
                if os.path.exists(image_path):
                    question_text+=f"Image-{image_no}: <image>\n"
                    image_no+=1
                    images.append(image_path)
                else:
                    print(f"{image_path} not found!")
        
        if isinstance(answer,str):
            answer = [{"text": answer,"image": None}]

        for a in answer:
            if a['text'] is not None:
                answer_text+=a['text']+'\n'
            if image_no>2:
                break
            if a['image'] is not None:
                image_path = os.path.join(input_dir,a['image'])
                if os.path.exists(image_path):
                    answer_text+=f"Image-{image_no}: <image>\n"
                    image_no+=1
                    images.append(image_path)
                else:
                    print(f"{image_path} not found!")
            if image_no>2:
                break
            

        processed_data_dict['id'] = id
        processed_data_dict['model'] = model
        processed_data_dict['question'] = question_text
        processed_data_dict['answer'] = answer_text
        processed_data_dict['gt_answer'] = gt_answer
        processed_data_dict['images'] = images
        if len(images) <= 4:
            processed_data.append(processed_data_dict)

    if os.path.exists(temp_output_path):
        print("Find temp data!")
        with open(temp_output_path, 'r') as f:
            temp_data = json.load(f)
        i = 0
        for td in temp_data:
            if 'gpt_feedback' in td:
                assert(processed_data[i]['id'] == td['id'])
                processed_data[i]['gpt_feedback'] = td['gpt_feedback']
                i+=1
            else:
                break

    return processed_data
