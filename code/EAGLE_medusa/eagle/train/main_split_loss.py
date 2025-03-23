import argparse

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/group/ossdphi_algo_scratch_08/haidhuan/models/llama2/Llama-2-70b-chat-hf')
# /group/ossmodelzoo/haidhuan/Llama-2-7b-chat-hf    /group/ossdphi_algo_scratch_08/haidhuan/models/vicuna/vicuna-7b-v1.3
# /group/ossdphi_algo_scratch_08/haidhuan/models/llama2/Llama-2-13b-chat-hf /group/ossdphi_algo_scratch_08/haidhuan/models/vicuna/vicuna-13b-v1.3 
parser.add_argument('--configpath', type=str, default="/home/haidhuan/code/EAGLE_MOE/eagle/train/llama_2_chat_70B_config.json")
# vicuna_7B_config.json  vicuna_13B_config.json  llama_2_chat_7B_config.json  llama_2_chat_13B_config.json
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--bs', type=int, default=4) #total bs=16
parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
parser.add_argument('--tmpdir', type=str, default='/scratch1_nvme_1/workspace/haiduo/shareGPT/eagle_data/Llama-2-70b-chat-hf') #数据地址
# /scratch1_nvme_1/workspace/haiduo/shareGPT/eagle_data/vicuna-7b-v1.3
# /scratch1_nvme_1/workspace/haiduo/shareGPT/eagle_data/vicuna-13b-v1.3
# /group/ossdphi_algo_scratch_08/haidhuan/shareGPT/eagle_data/llama2_7b_chat
# /scratch1_nvme_1/workspace/haiduo/shareGPT/eagle_data/Llama-2-13b-chat-hf
parser.add_argument('--outdir', type=str, default='0')
parser.add_argument('--cpdir', type=str, default='/group/ossdphi_algo_scratch_08/haidhuan/output/eagle/checkpoints/llama2/70b/split_gpt_mlp') #改配置
parser.add_argument('--enable_wandb', type=bool, default=True)
args = parser.parse_args()

train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": 20,
    # Depending on your data and model size, the larger the model, the higher the sample efficiency. We recommend setting it between 20-40.
    "num_warmup_steps": 2000,
    "total_steps": 800000,
    "p_w": 0.1, #交叉熵损失权重
    "v_w": 1.0, #smoothL1损失权重
    "head_w": 0.1,
    "num_workers": 2,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 2048,
    # During training, truncating the training sequences means that the larger the setting, the more training data is used, and the better the effect, but it also consumes more VRAM.
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 5
}

import json
from safetensors import safe_open
# from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification
import os,sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch

torch.backends.cuda.matmul.allow_tf32 = True #TF32是NVIDIA提供的一种新的数值精度格式，专门设计用于Ampere架构 GPU。
from accelerate import Accelerator
from accelerate.utils import set_seed


set_seed(0)
# 针对MOE训练
from accelerate import DistributedDataParallelKwargs
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) #https://github.com/huggingface/accelerate/issues/24#issuecomment-814106927
accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=train_config["gradient_accumulation_steps"], kwargs_handlers=[ddp_kwargs])
# 环境不支持 bf16，考虑使用其他类型的混合精度，如 fp16
# accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=train_config["gradient_accumulation_steps"])
from pathlib import Path
# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))
from eagle.model.cnets import Model
from eagle.model.configs import EConfig
from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# import accelerate
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoConfig

if accelerator.is_main_process and args.enable_wandb:
    import wandb
    output_dir = "/group/ossdphi_algo_scratch_08/haidhuan/output/eagle/wandb"
    os.makedirs(output_dir, exist_ok=True)
    wandb.init(project="llama2_70b_split_gpt_mlp_gpu09", entity="haiduo", config=train_config, dir=output_dir)

baseconfig = AutoConfig.from_pretrained(args.basepath)

head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)

try:
    with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(args.basepath, head_path),
                   framework="pt",
                   device="cpu") as f:
        tensor_slice = f.get_slice("lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
except:
    with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    weights = torch.load(os.path.join(args.basepath, head_path))
    tensor = weights["lm_head.weight"].float()

head.weight.data = tensor
head.eval()

for param in head.parameters():
    param.requires_grad = False


def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # try:
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = data['hidden_state'][:train_config["max_len"]][None, :]
        input_ids = data['input_ids'][:train_config["max_len"]][None, :]
        loss_mask = data["loss_mask"][:train_config["max_len"]][None, :]

        # except:
        #     with open("error_path.txt", "w") as file:
        #         file.write(self.data[index])
        #     print('error path',self.data[index])

        length = hidden_state.shape[1]
        # length_q = data['query_ids'].shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target
        # sample = torch.cat((data['xs'],data['xb']))
        # sample=torch.cat((self.data[index]['x'],self.data[index]['logits']))
        # label = data['y']

        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


def top_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        #topk在输出张量的最后一个维度（即类别维度）上，获取前 maxk 个最大值及其索引。返回值是两个张量：一个是最大值，另一个是最大值的索引。
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


@torch.no_grad()
def getkacc_v1(model, data, head, max_length=5):  # eaglev1的实现
    hidden_states = data["hidden_states"]
    input_ids = data["input_ids"]
    # attention_mask=data["attention_mask"]
    loss_mask = data["loss_mask"]
    # sample_mask=data["sample_mask"]
    target = data["target"]  #"hidden_states"的左移一个token位置
    total = [0 for _ in range(max_length)]
    correct = [0 for _ in range(max_length)]
    bs, sl = hidden_states.shape[0], hidden_states.shape[1]
    target_headout = head(target)
    hidden_states_headout = head(hidden_states)

    for i in range(bs):
        for j in range(sl):

            single_hidden_states = hidden_states[i, :j]
            single_input_ids = input_ids[i, :j]

            single_hidden_states = single_hidden_states[None, :, :]
            single_input_ids = single_input_ids[None, :]
            for k in range(max_length):
                if loss_mask[i, single_hidden_states.shape[1] - 1] == 0:
                    break # 针对token位置的mask为0则不计算loss
                tmp_in_target_headout = hidden_states_headout[i, single_hidden_states.shape[1] - 1]
                tmp_out_target_headout = target_headout[i, single_hidden_states.shape[1] - 1]
                target_in_token = torch.argmax(tmp_in_target_headout)
                target_out_token = torch.argmax(tmp_out_target_headout)
                tmp_token = input_ids[i, single_hidden_states.shape[1] - 1]
                # tmp_sample_mask=sample_mask[i,single_hidden_states.shape[1]-1]
                if not (target_in_token == tmp_token): # 保证倒数第二层hidden state经过head后的token与对应的inputs的token一样
                    break  # 否则，表示target LLM生成的倒数第二层hidden state与inputs的token有误差，则对于后续Eagle的验证就没必要了
                out_hidden = model(single_hidden_states, input_ids=single_input_ids) #这里已经将后面的tokens砍掉了，所以没必要用attention_mask
                out_hidden = torch.mean(out_hidden, dim=0) # 修改
                last_hidden = out_hidden[:, -1]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout)
                total[k] += 1
                if token == target_out_token:
                    correct[k] += 1
                else:
                    for kk in range(k + 1, max_length): #一旦当前token不正确，后面的token都默认不正确，直接计数
                        total[kk] += 1
                    break
                # 继续下一个自回归草稿生成 最大序列为max_length
                single_hidden_states = torch.cat((single_hidden_states, out_hidden[:, -1:]), dim=1)
                single_input_ids = torch.cat((single_input_ids, torch.tensor([[token]]).to(single_input_ids.device)), dim=1)

    acc = [correct[i] / total[i] for i in range(len(correct))]
    return acc


@torch.no_grad()
def getkacc(model, data, head, max_length=5): #eagle v2的实现
    def generate(hidden_states, input_ids, head, max_length=4, use_cache=True):
        if use_cache:
            past_key_values = None
            for i in range(max_length):
                if past_key_values != None:
                    out_hidden, past_key_values = model(last_hidden, input_ids=token, past_key_values=past_key_values, use_cache=True)
                else:
                    out_hidden, past_key_values = model(hidden_states, input_ids=input_ids, use_cache=True)
                out_hidden = torch.mean(out_hidden, dim=0) # 修改
                last_hidden = out_hidden[:, -1:]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout, dim=-1)
                input_ids = torch.cat((input_ids, token), dim=1)
        else:
            raise NotImplementedError

        return input_ids

    hidden_states = data["hidden_states"]
    input_ids = data["input_ids"]
    loss_mask = data["loss_mask"]
    target = data["target"]
    total = [0 for _ in range(max_length)]
    correct = [0 for _ in range(max_length)]
    bs, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    target_headout = head(target)
    target_ids = target_headout.argmax(dim=2)

    for pre_len in range(1, seq_len):
        if loss_mask[:, pre_len].sum() == 0:
            continue
        pre_hidden_states = hidden_states[:, :pre_len]
        pre_input_ids = input_ids[:, :pre_len]
        outs = generate(pre_hidden_states, pre_input_ids, head, max_length=max_length)
        generate_ids = outs[:, pre_len:]
        for bid in range(bs):
            for k in range(max_length):
                if loss_mask[bid, pre_len + k] == 0:
                    break
                if pre_len + k >= seq_len:
                    break
                total[k] += 1
                if generate_ids[bid, k] == target_ids[bid, pre_len + k - 1]:
                    correct[k] += 1
                else:
                    for kk in range(k + 1, max_length):
                        total[kk] += 1
                    break
    acc = [correct[i] / total[i] for i in range(len(correct))]
    return acc

if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])
    else:
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
else:
    aug = None

datapath = list_files(train_config["datapath"])

traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]
# print('td',train_config["datapath"])
# print(datapath)
# exit()
traindataset = CustomDataset(traindatapath, transform=aug)
testdataset = CustomDataset(testdatapath)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True, collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False, collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)
# for batch_data in train_loader:
#     print(batch_data)

if accelerator.is_main_process:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

config = EConfig.from_pretrained(train_config["config_path"])
model = Model(config, load_emb=True, path=args.basepath) #初始化草稿模型

criterion = nn.SmoothL1Loss(reduction="none") #特征回归loss
optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

if is_warmup:
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(model, head, optimizer, train_loader, test_loader, scheduler)
    # model, head, optimizer, train_loader, test_loader = accelerator.prepare(model, head, optimizer, train_loader, test_loader)
else:
    model, head, optimizer, train_loader, test_loader = accelerator.prepare(model, head, optimizer, train_loader, test_loader)
# accelerator.load_state("checkpoints/state_5")
for epoch in range(num_epochs + 1):
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            predict = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
            with torch.no_grad():
                target_head = head(data["target"])
                target_p = nn.Softmax(dim=-1)(target_head)
                target_p = target_p.detach()
            out_head = head(predict)
            out_logp = nn.LogSoftmax(dim=-1)(out_head)

            loss_mask = data["loss_mask"][:, :, None]
            plogp = target_p * out_logp

            # top-1的loss:
            ploss1 = -torch.sum(torch.sum(loss_mask * plogp[0], 2)) / (loss_mask.sum()+1e-5)
            vloss1 = criterion(predict[0], data["target"])
            vloss1 = torch.sum(torch.mean(loss_mask * vloss1, 2)) / (loss_mask.sum()+1e-5)
            # top-2的loss:
            ploss2 = -torch.sum(torch.sum(loss_mask * plogp[1], 2)) / (loss_mask.sum()+1e-5)
            vloss2 = criterion(predict[1], data["target"])
            vloss2 = torch.sum(torch.mean(loss_mask * vloss2, 2)) / (loss_mask.sum()+1e-5)

            ploss = ploss1 + ploss2
            vloss = vloss1 + vloss2
            loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss

            # loss.backward()
            accelerator.backward(loss)
            accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            if is_warmup:
                scheduler.step()

        with torch.no_grad():
            _, predicted = torch.max(out_head, -1)
            _, target = torch.max(target_head, -1)

            ct = loss_mask.sum().item()  ## 计算有效的样本数 ct
            # top-1的acc:
            cc1 = ((predicted[0] == target) * loss_mask.squeeze()).sum().item()  ## 贪婪 计算预测正确的样本数 cc
            out_head1 = out_head[0].view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]

            # top-2的acc:
            cc2 = ((predicted[1] == target) * loss_mask.squeeze()).sum().item()  ## 贪婪 计算预测正确的样本数 cc
            out_head2 = out_head[1].view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]

            topkacc1 = top_accuracy(out_head1, target, (1, 2, 3))
            topkacc2 = top_accuracy(out_head2, target, (1, 2, 3))
            
            cc = cc1 + cc2
            topkacc = [topkacc1[i] + topkacc2[i] for i in range(len(topkacc1))]
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            ct = ct *2
            total += ct
            correct += cc
        if accelerator.is_main_process and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                       "train/ploss": ploss.item(), "train/loss": loss.item(), "train/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct
            
            if args.enable_wandb:
                wandb.log(logdict)
                # for id,i in enumerate(top_3acc):
                #     wandb.log({f'train/top_{id+1}_acc':topkacc[id].item()/ct})

        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1
        # # debug
        # if num_batches == 10:
        #     break

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    if accelerator.is_local_main_process:
        for id, i in enumerate(top_3acc):
            if args.enable_wandb:
                wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
    if accelerator.is_local_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        print('Train Accuracy: {:.2f}%'.format(100 * correct / total))
        if args.enable_wandb:
            wandb.log({"train/epochacc": correct / total, "train/epochloss": epoch_loss})

    if (epoch + 1) % train_config["save_freq"]:
        top_3acc = [0 for _ in range(3)]
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.eval()

        k_acc = [[] for i in range(5)]
        for batch_idx, data in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                if batch_idx < 10:
                    acces = getkacc(model, data, head, max_length=5) # 这里的max_length为tree depth或者说草稿的最大tokens数
                    for i in range(len(acces)):
                        k_acc[i].append(acces[i])
                predict = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
                target_head = head(data["target"])
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
                out_head = head(predict)
                out_logp = nn.LogSoftmax(dim=-1)(out_head)
                loss_mask = data["loss_mask"][:, :, None]
                plogp = target_p * out_logp

                # top-1的loss:
                ploss1 = -torch.sum(torch.sum(loss_mask * plogp[0], 2)) / (loss_mask.sum()+1e-5)
                vloss1 = criterion(predict[0], data["target"])
                vloss1 = torch.sum(torch.mean(loss_mask * vloss1, 2)) / (loss_mask.sum()+1e-5)
                # top-2的loss:
                ploss2 = -torch.sum(torch.sum(loss_mask * plogp[1], 2)) / (loss_mask.sum()+1e-5)
                vloss2 = criterion(predict[1], data["target"])
                vloss2 = torch.sum(torch.mean(loss_mask * vloss2, 2)) / (loss_mask.sum()+1e-5)

                ploss = ploss1 + ploss2
                vloss = vloss1 + vloss2

                loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
                _, predicted = torch.max(out_head, -1)
                _, target = torch.max(target_head, -1)
                ct = loss_mask.sum().item()
                # top-1的acc:
                cc1 = ((predicted[0] == target) * loss_mask.squeeze()).sum().item()  ## 贪婪 计算预测正确的样本数 cc
                out_head1 = out_head[0].view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]

                # top-2的acc:
                cc2 = ((predicted[1] == target) * loss_mask.squeeze()).sum().item()  ## 贪婪 计算预测正确的样本数 cc
                out_head2 = out_head[1].view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
                target = target.view(-1)[loss_mask.view(-1) == 1]
                
                topkacc1 = top_accuracy(out_head1, target, (1, 2, 3))
                topkacc2 = top_accuracy(out_head2, target, (1, 2, 3))
                
                cc = cc1 + cc2
                topkacc = [topkacc1[i] + topkacc2[i] for i in range(len(topkacc1))]
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]
                ct = ct *2
                total += ct
                correct += cc
            epoch_loss += loss.item()
            num_batches += 1

        mean_acces = []
        for id, i in enumerate(k_acc):
            mean_acc = np.array(i).mean()
            mean_acc = torch.tensor(mean_acc).cuda()
            mean_acces.append(mean_acc)

        mean_acces = accelerator.gather_for_metrics(mean_acces)
        if accelerator.is_local_main_process:
            for id, i in enumerate(mean_acces):
                mean_acc = i.mean().item()
                if args.enable_wandb:
                    wandb.log({f"test/{id}_acc": mean_acc})

        correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
        correct, total = accelerator.gather_for_metrics((correct, total))
        correct, total = correct.sum().item(), total.sum().item()
        top_3acc = accelerator.gather_for_metrics(top_3acc)
        if accelerator.is_local_main_process:
            for id, i in enumerate(top_3acc):
                if args.enable_wandb:
                    wandb.log({f'test/top_{id + 1}_acc': i.sum().item() / total})
        epoch_loss /= num_batches
        if accelerator.is_local_main_process:
            print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
            print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
            if args.enable_wandb:
                wandb.log({"test/epochacc": correct / total, "test/epochloss": epoch_loss})
            # accelerator.save_model(model, f"checkpoints/model_{epoch}")
            # accelerator.save_state(output_dir=f"{args.outdir}/state_{epoch}")
            # os.system(f"cp -r {args.outdir} {args.cpdir}")
            if epoch>18:
                accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}", safe_serialization=True)
