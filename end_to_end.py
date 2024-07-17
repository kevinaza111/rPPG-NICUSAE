import os
import argparse
import random
import torchvision.transforms as transforms
from PIL import Image
import time
import math
from torch import nn, Tensor
from transformers import ViTModel, CLIPVisionModel, CLIPTextModel, CLIPTokenizer
import numpy as np
import torch
from config import get_config
import timm
from pre_model import Finetunemodel
from dataset import data_loader
from neural_methods import trainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict
from torch.utils.data import DataLoader

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Create a general generator for use with the validation dataloader,
# the test dataloader, and the unsupervised dataloader
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
# Create a training generator to isolate the train dataloader from
# other dataloaders and better control non-deterministic behavior
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)

# ------------------------------CLIP------------------------------
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

# ------------------------------------------------------------

# ------------------------------SCI_dataloader------------------------------
batch_w = 600
batch_h = 400
class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, task):
        self.low_img_dir = img_dir
        self.task = task
        self.train_low_data_names = []

        for root, dirs, names in os.walk(self.low_img_dir):
            for name in names:
                self.train_low_data_names.append(os.path.join(root, name))

        self.train_low_data_names.sort()
        self.count = len(self.train_low_data_names)

        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        return img_norm

    def __getitem__(self, index):

        low = self.load_images_transform(self.train_low_data_names[index])

        h = low.shape[0]
        w = low.shape[1]
        #
        h_offset = random.randint(0, max(0, h - batch_h - 1))
        w_offset = random.randint(0, max(0, w - batch_w - 1))
        #
        # if self.task != 'test':
        #     low = low[h_offset:h_offset + batch_h, w_offset:w_offset + batch_w]

        low = np.asarray(low, dtype=np.float32)
        low = np.transpose(low[:, :, :], (2, 0, 1))

        img_name = self.train_low_data_names[index].split('\\')[-1]
        # if self.task == 'test':
        #     # img_name = self.train_low_data_names[index].split('\\')[-1]
        #     return torch.from_numpy(low), img_name

        return torch.from_numpy(low), img_name

    def __len__(self):
        return self.count
# ------------------------------------------------------------

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False, default="./configs/infer_configs/NBHR-rPPG_NICU_FORMER_BASIC.yaml", type=str, help="The name of the model.")
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--save_path', type=str, default='./temp')
    parser.add_argument('--model', type=str, default='./weights/medium.pt', help='location of the data corpus')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu device id')
    return parser

def train_and_test(config, data_loader_dict):
    """Trains the model."""
    print("config.MODEL.NAME:{}".format(config.MODEL.NAME))
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'Former':
        model_trainer = trainer.FormerTrainer.FormerTrainer(config, data_loader_dict)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.train(data_loader_dict)
    model_trainer.test(data_loader_dict)


def test(config, data_loader_dict):
    """Tests the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'Former':
        model_trainer = trainer.FormerTrainer.FormerTrainer(config, data_loader_dict)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.test(data_loader_dict)


def unsupervised_method_inference(config, data_loader):
    if not config.UNSUPERVISED.METHOD:
        raise ValueError("Please set unsupervised method in yaml!")
    for unsupervised_method in config.UNSUPERVISED.METHOD:
        if unsupervised_method == "POS":
            unsupervised_predict(config, data_loader, "POS")
        elif unsupervised_method == "CHROM":
            unsupervised_predict(config, data_loader, "CHROM")
        elif unsupervised_method == "ICA":
            unsupervised_predict(config, data_loader, "ICA")
        elif unsupervised_method == "GREEN":
            unsupervised_predict(config, data_loader, "GREEN")
        elif unsupervised_method == "LGI":
            unsupervised_predict(config, data_loader, "LGI")
        elif unsupervised_method == "PBV":
            unsupervised_predict(config, data_loader, "PBV")
        else:
            raise ValueError("Not supported unsupervised method!")

def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im = im.resize((640, 480))
    im.save(path, 'JPEG')
    # np.save(path, image_numpy)
    # return im

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # configurations.
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')

# ------------------------------SCI------------------------------
    args = parser.parse_args()
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')
    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=True, num_workers=0)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    device = torch.device(args.device)

    model = Finetunemodel(args.model).to(device)
    model.eval()

    class_list = ['baby facial skin is bright', 'baby facial skin is dimly lit']
    class_embed = get_text_features(class_list)
    pre_model = Convit(class_embed).to(device)
    pre_model.load_state_dict(torch.load(r'./weights/convit_03.pt', map_location=device))
    pre_model.eval()
    data_list = []
    with torch.no_grad():
        probs = None
        for i, (input, image_name) in enumerate(test_queue):
            input = input.to(device)
            if i==0:
                tensor = transform(input.squeeze(0)).unsqueeze(0).unsqueeze(0).to(device)
                probs = pre_model(tensor)
                probs = probs.softmax(dim=-1)

            image_name = image_name[0].split('/')[-1].split('.')[0]
            i, r = model(input)
            # if probs[0][0] < 0.1:
            #    probs[0][0] = probs[0][0] + 0.5
            r = input + (r - input) * (1 - probs[0][0])

            u_name = "%s.jpg" % (image_name)
            print('processing {}'.format(u_name), ' score:', probs[0][0])
            u_path = save_path + '/' + u_name
            data_list.append(save_images(r, u_path))
# ------------------------------------------------------------

    data_loader_dict = dict()   # dictionary of data loaders
    test_loader = data_loader.NICUrPPG_frameLoader.NICUrPPGLoader
    test_data = test_loader(
        name="test",
        data_path=config.TEST.DATA.DATA_PATH,
        config_data=config.TEST.DATA)
    data_loader_dict["test"] = DataLoader(
        dataset=test_data,
        num_workers=16,
        batch_size=config.INFERENCE.BATCH_SIZE,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=general_generator)

    test(config, data_loader_dict)
