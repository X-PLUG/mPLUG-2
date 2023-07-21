import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import re_train_dataset, re_eval_dataset, pretrain_dataset_4m, coco_dataset, nocaps_dataset, coco_caption, coco_dataset_scst

from dataset.randaugment import RandomAugment

# Video Stuff
from dataset.video_utils import video_transforms, volume_transforms
from dataset.video_pretrain_dataset import pretrain_dataset_video, pretrain_eval_dataset_video
from dataset.video_downstream_datasets import (
    video_retrieval_dataset_train, video_retrieval_dataset_eval,
    video_qa_dataset, 
    video_caption_dataset,
)

def create_dataset(dataset, config, epoch=None):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    pretrain_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])    
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
    
    video_pretrain_transform = transforms.Compose([
        video_transforms.Resize((int(config['image_res'] * 1.14), int(config['image_res'] * 1.14))),
        # video_transforms.ShortSideScale(int(config['image_res'] * 256 // 224)),
        video_transforms.TemporalConsistentRandomAugment(N = 2, M = 5, augs = ['Identity', 'Contrast', 'Brightness', 
            'Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'HorizontalFlip']),
        video_transforms.RandomCrop(config['image_res']),
        volume_transforms.ClipToTensor(channel_nb=3),
        video_transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    video_train_transform = transforms.Compose([
        video_transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0), interpolation="bicubic"),
        video_transforms.RandomHorizontalFlip(),
        video_transforms.TemporalConsistentRandomAugment(N = 2, M = 5, augs = ['Identity', 'Contrast', 'Brightness', 
            'Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        volume_transforms.ClipToTensor(channel_nb=3),
        video_transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    video_train_transform_no_randaug = transforms.Compose([
        video_transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0), interpolation="bicubic"),
        video_transforms.RandomHorizontalFlip(),
        volume_transforms.ClipToTensor(channel_nb=3),
        video_transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    video_test_transform = transforms.Compose([
        video_transforms.Resize((config['image_res'], config['image_res'])),
        volume_transforms.ClipToTensor(channel_nb=3),
        video_transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    if dataset=='pretrain':
        dataset = pretrain_dataset_4m(config['train_file'], pretrain_transform, read_local_data=config['read_local_data'], epoch=epoch)
        return dataset

    elif dataset=='re':
        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset
               
    elif dataset=='vqa': 
        train_dataset = vqa_dataset(config['train_file'], train_transform, config['vqa_root'], config['vg_root'], config['gqa_root'], split='train', read_local_data=config['read_local_data'], add_ocr=config['add_ocr'], add_object=config['add_object']) 
        vqa_test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'], config['gqa_root'], split='test', answer_list=config['answer_list'], read_local_data=config['read_local_data'], add_ocr=config['add_ocr'], add_object=config['add_object'])       
        vqa_val_dataset = vqa_dataset(config['val_file'], test_transform, config['vqa_root'], config['vg_root'], config['gqa_root'],split='test', answer_list=config['answer_list'], read_local_data=config['read_local_data'], add_ocr=config['add_ocr'], add_object=config['add_object'])       
        return train_dataset, vqa_val_dataset, vqa_test_dataset
    elif dataset== 'nocaps':
        val_dataset = nocaps_dataset(config['val_file'], test_transform, config['nocaps_root'], max_words=config['max_length'], read_local_data=config['read_local_data'], is_train=False, add_object=config['add_object'])
        test_dataset = nocaps_dataset(config['test_file'], test_transform, config['nocaps_root'], max_words=config['max_length'], read_local_data=config['read_local_data'], is_train=False, add_object=config['add_object'])
        return val_dataset, test_dataset
    elif dataset== 'coco':
        train_dataset = coco_dataset(config['train_file'], train_transform, config['coco_root'], max_words=config['max_length'], read_local_data=config['read_local_data'], is_train=True, add_object=config['add_object'])
        val_dataset = coco_dataset(config['val_file'], test_transform, config['coco_root'], max_words=config['max_length'], read_local_data=config['read_local_data'], is_train=False, add_object=config['add_object'])
        test_dataset = coco_dataset(config['test_file'], test_transform, config['coco_root'], max_words=config['max_length'], read_local_data=config['read_local_data'], is_train=False, add_object=config['add_object'])
        return train_dataset, val_dataset, test_dataset
    elif dataset== 'coco_scst':
        train_dataset = coco_dataset_scst(config['train_file'], train_transform, config['coco_root'], max_words=config['max_length'], read_local_data=config['read_local_data'], is_train=True, add_object=config['add_object'])
        val_dataset = coco_dataset(config['val_file'], test_transform, config['coco_root'], max_words=config['max_length'], read_local_data=config['read_local_data'], is_train=False, add_object=config['add_object'])
        test_dataset = coco_dataset(config['test_file'], test_transform, config['coco_root'], max_words=config['max_length'], read_local_data=config['read_local_data'], is_train=False, add_object=config['add_object'])
        return train_dataset, val_dataset, test_dataset
    elif dataset=='nlvr':   
        train_dataset = nlvr_dataset(config['train_file'], train_transform, config['image_root'])  
        val_dataset = nlvr_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = nlvr_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset        
               
    elif dataset=='ve':   
        train_dataset = ve_dataset(config['train_file'], train_transform, config['image_root'])  
        val_dataset = ve_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = ve_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset     

    elif 'vg_' in dataset:
        if 'uni' in dataset:
            train_dataset = build_uni_training_dataset(args=config)
            val_dataset = build_vg_dataset(split='val',args=config,dataset_name='unc')
            eval_dataset = 'unc'
        else:
            train_dataset = build_vg_dataset(split='train',args=config,dataset_name=dataset[3:])
            val_dataset = build_vg_dataset(split='val',args=config,dataset_name=dataset[3:])
            eval_dataset = dataset[3:]
        eval_split = {
            'unc':['testA','testB'],
            'unc+':['testA','testB'],
            'gref_umd':['test']
        }
        test_datasets = {split:build_vg_dataset(split=split,args=config,dataset_name=eval_dataset) for split in eval_split[eval_dataset]}
        return train_dataset, val_dataset,test_datasets


    elif dataset == "video_retrieval":
        train_dataset = video_retrieval_dataset_train(config["train_file"], video_train_transform, config["video_root"],
            num_frames=config["model_num_frames"], has_multi_vision_gt=config.get("has_multi_vision_gt", False), 
            is_paragraph_retrieval=config.get("is_paragraph_retrieval", False), read_local_data=config["read_local_data"])
        val_dataset = video_retrieval_dataset_eval(config["test_file"], video_test_transform, config["video_root"],
            num_frames=config.get("test_num_frames", config["model_num_frames"]), has_multi_vision_gt=config.get("has_multi_vision_gt", False), 
            is_paragraph_retrieval=config.get("is_paragraph_retrieval", False), read_local_data=config["read_local_data"])
        return train_dataset, val_dataset

    elif dataset == "video_qa":
        train_dataset = video_qa_dataset(config["train_file"], video_train_transform, config["video_root"],
            num_frames=config["model_num_frames"], eos=config["eos"], split="train", 
            read_local_data=config["read_local_data"])
        val_dataset = video_qa_dataset(config["test_file"], video_test_transform, config["video_root"],
            num_frames=config["model_num_frames"], eos=config["eos"], split="test", answer_list=config["answer_list"],
            read_local_data=config["read_local_data"])
        return train_dataset, val_dataset
    
    
    elif dataset == "video_qa_no_randaug":
        train_dataset = video_qa_dataset(config["train_file"], video_train_transform_no_randaug, config["video_root"],
            num_frames=config["model_num_frames"], eos=config["eos"], split="train", 
            read_local_data=config["read_local_data"])
        val_dataset = video_qa_dataset(config["test_file"], video_test_transform, config["video_root"],
            num_frames=config["model_num_frames"], eos=config["eos"], split="test", answer_list=config["answer_list"],
            read_local_data=config["read_local_data"])
        return train_dataset, val_dataset
    
    
    
    elif dataset == "video_caption":
        train_dataset = video_caption_dataset(config["train_file"], video_train_transform, config["video_root"],
            num_frames=config["model_num_frames"], split="train", read_local_data=True)
        val_dataset = video_caption_dataset(config["test_file"], video_test_transform, config["video_root"],
            num_frames=config["model_num_frames"], split="test", read_local_data=True)
        return train_dataset, val_dataset
    
    
    elif dataset == "video_caption_no_randaug":
        train_dataset = video_caption_dataset(config["train_file"], video_train_transform_no_randaug, config["video_root"],
            num_frames=config["model_num_frames"], split="train", read_local_data=True)
        val_dataset = video_caption_dataset(config["test_file"], video_test_transform, config["video_root"],
            num_frames=config["model_num_frames"], split="test", read_local_data=True)
        return train_dataset, val_dataset



def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n

def nocaps_collate_fn(batch):
    image_list, image_id_list = [], []
    for image, image_id in batch:
        image_list.append(image)
        image_id_list.append(image_id)
    return torch.stack(image_list,dim=0), image_id_list

def coco_collate_fn(batch):
    image_list, caption_list, object_labels, image_id_list, gold_caption_list = [], [], [], [], []
    for image, caption, object_label, image_id, gold_caption in batch:
        image_list.append(image)
        caption_list.append(caption)
        image_id_list.append(image_id)
        gold_caption_list.append(gold_caption)
        object_labels.append(object_label)
    return torch.stack(image_list,dim=0), caption_list, object_labels, image_id_list, gold_caption_list


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    


def create_pretrain_sampler(datasets, shuffle, num_tasks, global_rank):
    samplers = []
    for dataset in datasets:
        sampler = torch.utils.data.DistributedSampler(dataset[0], num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_pretrain_loader(datasets, samplers, video_batch_size, num_workers, is_train, collate_fn, suffix=True):
    loaders = []
    data_types = []
    nums = list(range(len(datasets)))
    for dataset, sampler, i in zip(datasets, samplers, nums):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        data_type = dataset[1]
        loader = DataLoader(
            dataset[0],
            batch_size=video_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
        if suffix:
            data_types.append(data_type + "_" + str(i))
        else:
            data_types.append(data_type)

    return loaders, data_types
