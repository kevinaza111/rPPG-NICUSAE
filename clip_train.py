import os
import random
import sys
import time
import fnmatch
import math

import imageio
import timm
import numpy as np
import torch
from torch import nn, Tensor
import argparse
import torch.utils
import torch.optim as optim
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from pre_model import Finetunemodel
import torchvision.transforms as transforms
from transformers import ViTModel, CLIPVisionModel, CLIPTextModel, CLIPTokenizer
from torchmetrics.classification import MultilabelAveragePrecision
from torchvision.transforms import Compose
from tqdm import tqdm

# {dark:0, light:1}

# ------------------------------evaluation standard------------------------------
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# ------------------------------dataloader------------------------------
class CLIPLoader_video(torch.utils.data.Dataset):
    def __init__(self, root, type, frames):
        self.root = os.path.expanduser(root)
        self.type = type
        self.frames = frames

        if self.type == 'train':
            dark_root = os.path.join(self.root, 'Dark')
            dark_file_list = [f for f in os.listdir(dark_root) if f.endswith('.mp4')]
            dark_file_list = sorted(dark_file_list)
            dark_file_list = dark_file_list[:int(len(dark_file_list) * 0.8)]
            dark_label_list = [[dark_root, 0] for _ in range(len(dark_file_list))]

            light_root = os.path.join(self.root, 'Light')
            light_file_list = [f for f in os.listdir(light_root) if f.endswith('.mp4')]
            light_file_list = sorted(light_file_list)
            light_file_list = light_file_list[:int(len(light_file_list) * 0.8)]
            light_label_list = [[light_root, 1] for _ in range(len(light_file_list))]

            self.file_list = dark_file_list + light_file_list
            self.label_list = dark_label_list + light_label_list
        elif self.type == 'test':
            dark_root = os.path.join(self.root, 'Dark')
            dark_file_list = [f for f in os.listdir(dark_root) if f.endswith('.mp4')]
            dark_file_list = sorted(dark_file_list)
            dark_file_list = dark_file_list[int(len(dark_file_list) * 0.8):]
            dark_label_list = [[dark_root, 0] for _ in range(len(dark_file_list))]

            light_root = os.path.join(self.root, 'Light')
            light_file_list = [f for f in os.listdir(light_root) if f.endswith('.mp4')]
            light_file_list = sorted(light_file_list)
            light_file_list = light_file_list[int(len(light_file_list) * 0.8):]
            light_label_list = [[light_root, 1] for _ in range(len(light_file_list))]

            self.file_list = dark_file_list + light_file_list
            self.label_list = dark_label_list + light_label_list

        else:
            raise ValueError('The value of type is wrong, it must be train or test')

        self.count = len(self.file_list)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        video_root = os.path.join(self.label_list[index][0], self.file_list[index])
        reader = imageio.get_reader(video_root)
        total_frames = reader.count_frames()
        image_index = sorted(random.sample(range(total_frames), self.frames))
        images = torch.stack([self.transform(reader.get_data(i)) for i in image_index])
        # images:torch.Size([self.frames, 3, H, W])

        return images, self.label[index][1]

    def __len__(self):
        return self.count

class CLIPLoader_images(torch.utils.data.Dataset):
    def __init__(self, root, type, frames):
        self.root = os.path.expanduser(root)
        self.type = type
        self.frames = frames

        if self.type == 'train':
            dark_root = os.path.join(self.root, 'Dark_enhancement', 'V2I')
            # dark_file_list = os.listdir(dark_root)
            dark_file_list = [f for f in os.listdir(dark_root) if not fnmatch.fnmatch(f, '*.json')]
            dark_file_list = sorted(dark_file_list)
            dark_file_list = dark_file_list[:int(len(dark_file_list) * 0.8)]
            dark_label_list = [[os.path.join(dark_root, filename), [0.0, 1.0]] for filename in dark_file_list]

            light_root = os.path.join(self.root, 'Light_enhancement')
            # light_file_list = os.listdir(light_root)
            light_file_list = [f for f in os.listdir(light_root) if not fnmatch.fnmatch(f, '*.json')]
            light_file_list = sorted(light_file_list)
            light_file_list = light_file_list[:int(len(light_file_list) * 0.8)]
            light_label_list = [[os.path.join(light_root, filename), [1.0, 0.0]] for filename in light_file_list]

            self.file_list = dark_file_list + light_file_list
            self.label_list = dark_label_list + light_label_list
        elif self.type == 'test':
            dark_root = os.path.join(self.root, 'Dark_enhancement', 'V2I')
            # dark_file_list = os.listdir(dark_root)
            dark_file_list = [f for f in os.listdir(dark_root) if not fnmatch.fnmatch(f, '*.json')]
            dark_file_list = sorted(dark_file_list)
            dark_file_list = dark_file_list[int(len(dark_file_list) * 0.8):]
            dark_label_list = [[os.path.join(dark_root, filename), [0.0, 1.0]] for filename in dark_file_list]

            light_root = os.path.join(self.root, 'Light_enhancement')
            # light_file_list = os.listdir(light_root)
            light_file_list = [f for f in os.listdir(light_root) if not fnmatch.fnmatch(f, '*.json')]
            light_file_list = sorted(light_file_list)
            light_file_list = light_file_list[int(len(light_file_list) * 0.8):]
            light_label_list = [[os.path.join(light_root, filename), [1.0, 0.0]] for filename in light_file_list]

            self.file_list = dark_file_list + light_file_list
            self.label_list = dark_label_list + light_label_list

        else:
            raise ValueError('The value of type is wrong, it must be train or test')

        self.count = len(self.file_list)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])

    def __getitem__(self, index):
        result_images = []
        images_root = self.label_list[index][0]
        images = os.listdir(images_root)
        # print('num_images : ', len(images), 'images_root : ', images_root)
        selected_images = sorted(random.choices(images, k=self.frames))
        for image in selected_images:
            img = Image.open(os.path.join(images_root, image)).convert('RGB')
            img = self.transform(img)
            result_images.append(img)
        result_images = torch.stack(result_images)
        # result_images:torch.Size([self.frames, 3, 224, 224])

        return result_images, torch.tensor(self.label_list[index][1])

    def __len__(self):
        return self.count

# ------------------------------Convit------------------------------
def get_prompt(class_list):
    temp_prompt = []
    for c in class_list:
        temp_prompt.append(c)
    return temp_prompt

def get_text_features(class_list):
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    act_prompt = get_prompt(class_list)
    texts = tokenizer(act_prompt, padding=True, return_tensors="pt")
    text_class = text_model(**texts).pooler_output.detach()
    return text_class

class PositionalEncoding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        out = self.dropout(x)
        return out

class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class Convit(torch.nn.Module):
    def __init__(self, class_embed):
        super(Convit, self).__init__()
        self.num_classes = 2
        self.embed_dim = 512

        self.backbone = timm.create_model('resnetv2_50', pretrained=True, num_classes=768)
        self.linear = nn.Linear(in_features=768, out_features=self.embed_dim, bias=False)
        self.pos_encod = PositionalEncoding(d_model=self.embed_dim)

        # self.image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        # self.linear2 = nn.Linear(in_features=self.backbone.config.hidden_size + self.embed_dim, out_features=self.embed_dim, bias=False)
        self.query_embed = nn.Parameter(class_embed)
        self.transformer = nn.Transformer(d_model=self.embed_dim, batch_first=True)
        self.group_linear = GroupWiseLinear(self.num_classes, self.embed_dim, bias=True)

    def forward(self, tensor):
        # 接收的tensor是归一化的(b,t,3,224,224)
        b, t, c, h, w = tensor.size()
        tensor = tensor.reshape(-1, c, h, w)
        x = self.backbone(tensor)
        x = x.reshape(b, t, -1)
        x = self.linear(x)
        x = self.pos_encod(x)

        query_embed = self.query_embed.unsqueeze(0).repeat(b, 1, 1)

        x = self.transformer(x, query_embed)
        x = self.group_linear(x)
        return x

# ------------------------------trainer------------------------------
def trainer(train_loader, test_loader, model, device, optimizer, scheduler, criterion, eval_metric, args):
    for index in tqdm(range(args.epochs)):
        model.train()
        loss_meter = AverageMeter()
        start_time = time.time()

        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss, data.shape[0])

        elapsed_time = time.time() - start_time
        scheduler.step()
        print("Epoch [" + str(index + 1) + "]"
              + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
              + " loss: " + "{:.4f}".format(loss_meter.avg))

        if (index + 1) % args.test_every == 0:
            model.eval()
            evaluation_meter = AverageMeter()
            for data, label in test_loader:
                data = data.to(device)
                label = label.long().to(device)
                with torch.no_grad():
                    pred = model(data)
                pred_eval = eval_metric(pred, label)
                evaluation_meter.update(pred_eval.item(), data.shape[0])
            print("[INFO] Evaluation Metric: {:.2f}".format(evaluation_meter.avg * 100), flush=True)

    torch.save(model.state_dict(), args.save_path)

# ------------------------------main------------------------------
parser = argparse.ArgumentParser("CLIP")
parser.add_argument('--data_root', type=str, default=r' ', help='location of the data corpus')
parser.add_argument('--save_path', type=str, default=r' ')
parser.add_argument('--frames', type=int, default=8, help='The number of frames in a video')
parser.add_argument('--epochs', type=int, default=20, help='epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--test_every', type=int, default=5, help='test the model every this number of epochs')
parser.add_argument('--device', type=str, default='cuda:0', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()

device = torch.device(args.device)
train_set = CLIPLoader_images(root=args.data_root, type='train', frames=args.frames)
train_sampler = torch.utils.data.RandomSampler(train_set, num_samples=500)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=args.batch_size,
    sampler=train_sampler,
    shuffle=False
)

test_set = CLIPLoader_images(root=args.data_root, type='test', frames=args.frames)
test_sampler = torch.utils.data.RandomSampler(test_set)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=args.batch_size,
    sampler=test_sampler,
    shuffle=False
)

class_list = ['baby facial skin is bright', 'baby facial skin is dimly lit']
class_embed = get_text_features(class_list)

my_convit = Convit(class_embed).to(device)
optimizer = optim.Adam(my_convit.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
criterion = torch.nn.BCEWithLogitsLoss()
eval_metric = MultilabelAveragePrecision(num_labels=2, average='micro')
memory = sum(p.numel() for p in my_convit.parameters() if p.requires_grad) / 268435456
print('Parameter Space: ABS: {:.2f}'.format(memory) + " GB")

trainer(train_loader,
        test_loader,
        my_convit,
        device,
        optimizer,
        scheduler,
        criterion,
        eval_metric,
        args
)