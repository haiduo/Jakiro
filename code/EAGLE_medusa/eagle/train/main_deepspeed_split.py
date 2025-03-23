import argparse
import deepspeed

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/group/ossdphi_algo_scratch_08/haidhuan/models/llama2/Llama-2-70b-chat-hf')
parser.add_argument('--configpath', type=str, default="/home/haidhuan/code/EAGLE_MOE/eagle/train/llama_2_chat_70B_config.json")
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--bs', type=int, default=4) #total bs=16
parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
parser.add_argument('--tmpdir', type=str, default='/scratch1_nvme_1/workspace/haiduo/shareGPT/eagle_data/Llama-2-70b-chat-hf')
parser.add_argument('--outdir', type=str, default='0')
parser.add_argument('--cpdir', type=str, default='/group/ossdphi_algo_scratch_08/haidhuan/output/eagle/checkpoints/llama2/70b/split_gpt_mlp')
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser.add_argument('--enable_wandb', type=bool, default=True)
parser.add_argument('--deepspeed_config', default="/home/haidhuan/code/EAGLE_MOE/eagle/train/ds_config.json")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()
import json

train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": 20,
    "num_warmup_steps": 2000,
    "total_steps": 800000,
    "p_w": 0.1,
    "v_w": 1.0,
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
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 5
}

from safetensors import safe_open
# from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification
import os,sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed, DummyOptim, DummyScheduler

set_seed(0)
# # 针对MOE训练
# from accelerate import DistributedDataParallelKwargs
# ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) #https://github.com/huggingface/accelerate/issues/24#issuecomment-814106927
# accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=train_config["gradient_accumulation_steps"], kwargs_handlers=[ddp_kwargs])
# 环境不支持 bf16，考虑使用其他类型的混合精度，如 fp16
accelerator = Accelerator(mixed_precision="fp16")
from pathlib import Path
# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))
from eagle.model.cnets import Model
from eagle.model.configs import EConfig
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# import accelerate
import numpy as np
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup

deepspeed.init_distributed()
rank = torch.distributed.get_rank()
if rank == 0 and args.enable_wandb:
    import wandb
    output_dir = "/group/ossdphi_algo_scratch_08/haidhuan/output/eagle/wandb"
    os.makedirs(output_dir, exist_ok=True)
    wandb.init(project="llama2_70b_split_gpt_mlp_gpu09", entity="haiduo", config=train_config, dir=output_dir)

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

head = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False)
head.weight.data = tensor

def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path, followlinks=True):
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
        batch_loss_mask = torch.tensor([item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor([item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
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

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


@torch.no_grad()
def getkacc(model, data, max_length=5):
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

    hidden_states = data["hidden_states"].half().to(rank)
    input_ids = data["input_ids"].to(rank)
    loss_mask = data["loss_mask"].to(rank)
    target = data["target"].half().to(rank)
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
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False, collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)
# for batch_data in train_loader:
#     print(batch_data)

if rank == 0:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

config = EConfig.from_pretrained(train_config["config_path"])
model = Model(config, path=args.basepath, load_emb=True)

criterion = nn.SmoothL1Loss(reduction="none")

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

model_engine, optimizer, train_loader, _ = deepspeed.initialize(args=args,
                                                                model=model,
                                                                model_parameters=model.parameters(),
                                                                training_data=traindataset,
                                                                collate_fn=DataCollatorWithPadding()
                                                                )

head_engine, _, test_loader, _ = deepspeed.initialize(args=args,
                                                      model=head,
                                                      model_parameters=head.parameters(),
                                                      training_data=testdataset,
                                                      collate_fn=DataCollatorWithPadding()
                                                      )

for param in head.parameters():
    param.requires_grad = False

for epoch in range(num_epochs):
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    for batch_idx, data in enumerate(train_loader):
        model.zero_grad()
        predict = model_engine(data["hidden_states"].to(rank), input_ids=data["input_ids"].to(rank), 
                               attention_mask=data["attention_mask"].to(rank))
        with torch.no_grad():
            target_head = head_engine(data["target"].to(rank))
            target_p = nn.Softmax(dim=2)(target_head)
            target_p = target_p.detach()
        out_head = head_engine(predict)
        out_logp = nn.LogSoftmax(dim=2)(out_head)

        loss_mask = data["loss_mask"][:, :, None].to(rank)
        plogp = target_p * out_logp

        # top-1的loss:
        ploss1 = -torch.sum(torch.sum(loss_mask * plogp[0], 2)) / (loss_mask.sum()+1e-5)
        vloss1 = criterion(predict[0], data["target"].to(rank))
        vloss1 = torch.sum(torch.mean(loss_mask * vloss1, 2)) / (loss_mask.sum()+1e-5)
        # top-2的loss:
        ploss2 = -torch.sum(torch.sum(loss_mask * plogp[1], 2)) / (loss_mask.sum()+1e-5)
        vloss2 = criterion(predict[1], data["target"].to(rank))
        vloss2 = torch.sum(torch.mean(loss_mask * vloss2, 2)) / (loss_mask.sum()+1e-5)

        ploss = ploss1 + ploss2
        vloss = vloss1 + vloss2
        loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss

        # loss.backward()
        model_engine.backward(loss)
        # accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])

        model_engine.step()

        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)

            ct = loss_mask.sum().item()

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
        if rank == 0 and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                       "train/ploss": ploss.item(), "train/loss": loss.item(), "train/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct
            wandb.log(logdict)
            # for id,i in enumerate(top_3acc):
            #     wandb.log({f'train/top_{id+1}_acc':topkacc[id].item()/ct})

        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    if accelerator.is_local_main_process:
        for id, i in enumerate(top_3acc):
            wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
    if accelerator.is_local_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        print('Train Accuracy: {:.2f}%'.format(100 * correct / (total + 1e-5)))
        wandb.log({"train/epochacc": correct / (total + 1e-5), "train/epochloss": epoch_loss})
    
    if (epoch + 1) % train_config["save_freq"]:
        top_3acc = [0 for _ in range(3)]
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0

        k_acc = [[] for i in range(5)]
        for batch_idx, data in enumerate(test_loader):
            with torch.no_grad():
                if batch_idx < 10:
                    acces = getkacc(model, data, max_length=5)
                    for i in range(len(acces)):
                        k_acc[i].append(acces[i])
                predict = model(data["hidden_states"].half().to(rank), input_ids=data["input_ids"].to(rank),
                                attention_mask=data["attention_mask"].half().to(rank))
                target_head = head_engine(data["target"].half().to(rank))
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
                out_head = head_engine(predict)
                out_logp = nn.LogSoftmax(dim=2)(out_head)
                loss_mask = data["loss_mask"][:, :, None].half().to(rank)
                plogp = target_p * out_logp

                # top-1的loss:
                ploss1 = -torch.sum(torch.sum(loss_mask * plogp[0], 2)) / (loss_mask.sum()+1e-5)
                vloss1 = criterion(predict[0], data["target"].to(rank))
                vloss1 = torch.sum(torch.mean(loss_mask * vloss1, 2)) / (loss_mask.sum()+1e-5)
                # top-2的loss:
                ploss2 = -torch.sum(torch.sum(loss_mask * plogp[1], 2)) / (loss_mask.sum()+1e-5)
                vloss2 = criterion(predict[1], data["target"].to(rank))
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
            del ploss, vloss
            break

        mean_acces = []
        for id, i in enumerate(k_acc):
            mean_acc = np.array(i).mean()
            mean_acc = torch.tensor(mean_acc).cuda()
            mean_acces.append(mean_acc)

        mean_acces = accelerator.gather_for_metrics(mean_acces)
        if accelerator.is_local_main_process:
            for id, i in enumerate(mean_acces):
                mean_acc = i.mean().item()
                wandb.log({f"test/{id}_acc": mean_acc})

        correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
        correct, total = accelerator.gather_for_metrics((correct, total))
        correct, total = correct.sum().item(), total.sum().item()
        top_3acc = accelerator.gather_for_metrics(top_3acc)
        if accelerator.is_local_main_process:
            for id, i in enumerate(top_3acc):
                wandb.log({f'test/top_{id + 1}_acc': i.sum().item() / (total + 1e-5)})
        epoch_loss /= num_batches
        if accelerator.is_local_main_process:
            print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
            print('Test Accuracy: {:.2f}%'.format(100 * correct / (total + 1e-5)))
            if args.enable_wandb:
                wandb.log({"test/epochacc": correct / (total + 1e-5), "test/epochloss": epoch_loss})
        if epoch % 10 == 0 and epoch>18:
            model_engine.save_16bit_model(f"{args.cpdir}/state_{epoch}")
            deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{args.cpdir}/state_{epoch}")
