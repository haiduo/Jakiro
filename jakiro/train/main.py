import argparse
import torch.nn.functional as F
import json
from safetensors import safe_open
# from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification
import os,sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/home/haiduo/model/vicuna/vicuna-7b-v1.3') # target model path
parser.add_argument('--configpath', type=str, default="/home/haiduo/code/Jakiro/jakiro/train/vicuna_7B_config.json")
parser.add_argument('--lr', type=float, default=9e-5)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--wd', type=float, default=1e-3)
parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
parser.add_argument('--tmpdir', type=str, default='/home/haiduo/data/eagle_data/vicuna-7b-v1.3') # data path
parser.add_argument('--cpdir', type=str, default='/home/haiduo/code/Jakiro/jakiro/outputs/ckpts') # checkpoint path
parser.add_argument('--enable_wandb', type=bool, default=False)
parser.add_argument('--debug', type=bool, default=False)
args = parser.parse_args()

train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "wd": args.wd,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": 20,
    # Depending on your data and model size, the larger the model, the higher the sample efficiency. We recommend setting it between 20-40.
    "num_warmup_steps": 2000,
    "total_steps": 800000,
    "p_w": 0.1,
    "v_w": 1.0,
    "const": 0.5,
    "head_w": 0.1,
    "num_workers": 0,
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


torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

from accelerate.utils import DistributedType
class CustomAccelerator(Accelerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_state_dict(self, model, unwrap=True):
        if self.distributed_type == DistributedType.DEEPSPEED:
            if self.deepspeed_config["zero_optimization"]["stage"] == 3:
                if model.zero_gather_16bit_weights_on_model_save():
                    state_dict = model._zero3_consolidated_16bit_state_dict()
                else:
                    raise ValueError(
                        "Cannot get 16bit model weights because `stage3_gather_16bit_weights_on_model_save` in DeepSpeed config is False. "
                        "To save the model weights in 16bit, set `stage3_gather_16bit_weights_on_model_save` to True in DeepSpeed config file or "
                        "set `zero3_save_16bit_model` to True when using `accelerate config`. "
                        "To save the full checkpoint, run `model.save_checkpoint(save_dir)` and use `zero_to_fp32.py` to recover weights."
                    )
            else:
                from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
                state_dict = clone_tensors_for_torch_save(self.unwrap_model(model).state_dict())
        elif self.distributed_type == DistributedType.FSDP:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            if unwrap:
                model = self.unwrap_model(model)
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = model.state_dict()
        else:
            if unwrap:
                model = self.unwrap_model(model)
            state_dict = model.state_dict()

        # No need to save head and embedding layers
        if len(state_dict) <=1: 
            return None
        else:
            eagle_state_dict = {k: v for k, v in state_dict.items() if "embed" not in k and "Head" not in k and "head" not in k}
            return eagle_state_dict
    
set_seed(0)
#### original #####
accelerator = CustomAccelerator(mixed_precision='bf16', gradient_accumulation_steps=train_config["gradient_accumulation_steps"])

# ### updated ### 
# ### for MOE training: "self.gradient_checkpointing = False"  # turn it off if use "find_unused_parameters=True"
# from accelerate import DistributedDataParallelKwargs
# ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) #https://github.com/huggingface/accelerate/issues/24#issuecomment-814106927
# accelerator = CustomAccelerator(mixed_precision='bf16', gradient_accumulation_steps=train_config["gradient_accumulation_steps"], kwargs_handlers=[ddp_kwargs])
# # The environment does not support bf16, consider using other types of mixed precision, such as fp16
# # accelerator = CustomAccelerator(mixed_precision='bf16', gradient_accumulation_steps=train_config["gradient_accumulation_steps"])

from pathlib import Path
# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))
from jakiro.model.cnets import Model
from jakiro.model.configs import EConfig
from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# import accelerate
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoConfig

if accelerator.is_main_process and args.enable_wandb:
    import wandb
    output_dir = "/home/haiduo/code/Jakiro/jakiro/outputs/wandb"
    os.makedirs(output_dir, exist_ok=True)
    wandb.init(project="jakiro_vicuan-7b_medusa", entity="haiduo", config=train_config, dir=output_dir)

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

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def compute_loss(target, target_p, predict, loss_mask):
    out_head = head(predict)
    out_logp = nn.LogSoftmax(dim=-1)(out_head)
    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum() + 1e-5)
    vloss = criterion(predict, target)
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum() + 1e-5)
    return vloss, ploss, out_head


@torch.no_grad()
def getkacc(model, data, head, max_length=5):
    def generate(hidden_states, input_ids, head, max_length=4, use_cache=True):
        if use_cache:
            past_key_values = None
            for i in range(max_length):
                if past_key_values != None:
                    out_hidden, past_key_values = model(last_hidden, input_ids=token, past_key_values=past_key_values, use_cache=True)
                else:
                    out_hidden, past_key_values = model(hidden_states, input_ids=input_ids, use_cache=True)

                out_hidden = out_hidden[0] # Only validated for the top-1 expert
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

traindataset = CustomDataset(traindatapath, transform=aug)
testdataset = CustomDataset(testdatapath)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True,
                          collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                          pin_memory=True)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                         collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)

if accelerator.is_main_process:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

config = EConfig.from_pretrained(train_config["config_path"])
model = Model(config, load_emb=True, path=args.basepath)

criterion = nn.SmoothL1Loss(reduction="none")
optimizer = optim.AdamW(model.parameters(), weight_decay=train_config["wd"], lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

# #### (1) The total number of samples and batch size provided by the data loader #####
total_samples = len(train_loader.dataset)  # Total number of samples in the dataset
batch_size = train_loader.batch_size       # Number of samples per batch
steps_per_epoch = np.math.ceil(total_samples / batch_size)  # Number of steps per epoch
total_steps = num_epochs * steps_per_epoch #Dynamically calculate total_steps based on num_epochs
# num_warmup_steps = int(0.1 * total_steps)  # Calculate the warmup phase ratio based on the total number of steps
print("############## traindataset ################:", len(traindataset))
print("############## train_loader ################:", len(train_loader)) 
print("############## steps_per_epoch ################:", steps_per_epoch)
print("############## total_steps ################:", total_steps)


# #### (2) Dynamically adjust num_epochs based on total_steps #####
# steps_per_epoch = len(train_loader)  # Number of steps per epoch
# print("############## traindataset ################:", len(traindataset))
# print("############## train_loader ################:", len(train_loader)) 
# num_epochs = np.math.ceil(total_steps / steps_per_epoch)  # Calculate the number of epochs required
# print("############## all num_epochs ################:", num_epochs)

#### alter copsin ####
from transformers import get_cosine_schedule_with_warmup
if is_warmup:
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(model, head, optimizer, train_loader, test_loader, scheduler)
    # model, head, optimizer, train_loader, test_loader = accelerator.prepare(model, head, optimizer, train_loader, test_loader)
else:
    model, head, optimizer, train_loader, test_loader = accelerator.prepare(model, head, optimizer, train_loader, test_loader)

# #### original ####
# if is_warmup:
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps) #liner
#     model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(model, head, optimizer, train_loader, test_loader, scheduler)
#     # model, head, optimizer, train_loader, test_loader = accelerator.prepare(model, head, optimizer, train_loader, test_loader)
# else:
#     model, head, optimizer, train_loader, test_loader = accelerator.prepare(model, head, optimizer, train_loader, test_loader)


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
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()

            ### Updated ###
            out_head = head(predict)
            out_logp = nn.LogSoftmax(dim=-1)(out_head)

            loss_mask = data["loss_mask"][:, :, None]
            plogp0 = target_p * out_logp[0]

            # moe loss:
            ploss1 = -torch.sum(torch.sum(loss_mask * plogp0, 2)) / (loss_mask.sum()+1e-5) # classification loss
            vloss1 = criterion(predict[0], data["target"])  # regression loss
            vloss1 = torch.sum(torch.mean(loss_mask * vloss1, 2)) / (loss_mask.sum()+1e-5)
            # const loss:
            loss_mask1 = loss_mask[:, 1:, :]
            plogp1 = target_p[:, 1:, :] * out_logp[1][:,:-1,:]
            ploss2 = -torch.sum(torch.sum(loss_mask1 * plogp1, 2)) / (loss_mask1.sum()+1e-5) # classification loss
            vloss2 = criterion(predict[1][:,:-1,:], data["target"][:, 1:, :]) # regression loss
            vloss2 = torch.sum(torch.mean(loss_mask1 * vloss2, 2)) / (loss_mask1.sum()+1e-5)

            ploss = ploss1 + ploss2 * train_config["const"]
            vloss = vloss1 + vloss2 * train_config["const"]
            loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
            
            # loss.backward()
            accelerator.backward(loss)
            accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            if is_warmup:
                scheduler.step()
            
            ### original ###
            # loss_mask = data["loss_mask"][:, :, None]
            # vloss, ploss, out_head = compute_loss(data["target"], target_p, predict, loss_mask)
            # loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
            # # loss.backward()
            # accelerator.backward(loss)
            # accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
            # optimizer.step()
            # if is_warmup:
            #     scheduler.step()

        with torch.no_grad():
            ### Updated ####
            _, predicted = torch.max(out_head, -1)
            _, target = torch.max(target_head, -1)
        
            ct = loss_mask.sum().item()  ## Calculate the effective number of samples ct of moe
            # moe: acc:
            cc1 = ((predicted[0] == target) * loss_mask.squeeze()).sum().item()
            out_head1 = out_head[0].view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            # const: acc:
            loss_mask1 = loss_mask[:, 1:, :]
            ct1 = loss_mask1.sum().item()  ## Calculate the effective number of samples ct of const
            cc2 = ((predicted[1][:,:-1] == target[:, 1:]) * loss_mask1.squeeze()).sum().item()
            out_head2 = out_head[1][:,:-1,:].reshape(-1, (target_head.shape[-1]))[loss_mask1.reshape(-1) == 1]

            target = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc1 = top_accuracy(out_head1, target, (1, 2, 3))
            topkacc2 = top_accuracy(out_head2[:-1,:], target[1:], (1, 2, 3))
            
            cc = cc1 + cc2
            topkacc = [topkacc1[i] + topkacc2[i] for i in range(len(topkacc1))]
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            ct = ct + ct1
            total += ct
            correct += cc

            # ### original ####
            # _, predicted = torch.max(out_head, 2)
            # _, target = torch.max(target_head, 2)
            # ct = loss_mask.sum().item()
            # cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            # out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            # target = target.view(-1)[loss_mask.view(-1) == 1]
            # topkacc = top_accuracy(out_head, target, (1, 2, 3))
            # for top_i in range(len(topkacc)):
            #     top_3acc[top_i] += topkacc[top_i]
            # total += ct
            # correct += cc

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
        # debug
        if args.debug and num_batches == 10:
            break

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
                    acces = getkacc(model, data, head, max_length=5)
                    for i in range(len(acces)):
                        k_acc[i].append(acces[i])
                predict = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
                predict = predict[0] # Only validated for the moe expert
                target_head = head(data["target"])
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
                loss_mask = data["loss_mask"][:, :, None]
                vloss, ploss, out_head = compute_loss(data["target"], target_p, predict, loss_mask)
                loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
                _, predicted = torch.max(out_head, 2)
                _, target = torch.max(target_head, 2)
                ct = loss_mask.sum().item()
                cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
                out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
                target = target.view(-1)[loss_mask.view(-1) == 1]
                topkacc = top_accuracy(out_head, target, (1, 2, 3))
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]
                total += ct
                correct += cc
            epoch_loss += loss.item()
            num_batches += 1

            # ## debug
            # if args.debug and num_batches == 1:
            #     break
            if batch_idx > 0: # Speed ​​up training and test only one batch
                break

        mean_acces = []
        for id, i in enumerate(k_acc):
            # print("i:", i)
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
        if not isinstance(top_3acc, torch.Tensor):
            top_3acc = torch.tensor(top_3acc, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(top_3acc, dtype=torch.float32)
        top_3acc = accelerator.gather_for_metrics(top_3acc)
        if accelerator.is_local_main_process:
            for id, i in enumerate(top_3acc):
                if args.enable_wandb:
                    wandb.log({f'test/top_{id + 1}_acc': i.sum().item() / total})
        epoch_loss /= (num_batches+1e-6)

        if accelerator.is_local_main_process:
            print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
            print('Test Accuracy: {:.2f}%'.format(100 * correct / (total+1e-6)))
            if args.enable_wandb:
                wandb.log({"test/epochacc": correct / total, "test/epochloss": epoch_loss})
            if epoch>18 or epoch <1:
                accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}", safe_serialization=False)

            print(f"finish {epoch}-epoch save state!")
