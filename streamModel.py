import tarfile
import math
import lzma
import multiprocessing
import re
import subprocess
import csv
import os
import itertools
import torch
import pickle
total = []
reserve = []
vLen = 0
results = []
with open("vocab.txt", "r") as csvfile:
	reader = csv.reader(csvfile)
	readList = list(reader) # change contents to floats
	if len(readList) > 0:
		vLen = len(readList[0])
	csvfile.close()
device = "cuda"
c = 1025
b2 = 16
h = 0
max_batch_size = 500000
min_batch_size = 32000
rate = int(round((max_batch_size ** 2 - min_batch_size ** 2) / (2 * 4000000000)))
starting_steps = math.floor((max_batch_size - min_batch_size) / rate + 1)
b = min_batch_size
agg_size = 800
max_rate = 6e-4
min_rate = 6e-5
warmup = 2000
max_iters = 600000
class MaskedAttention(torch.nn.Module):
	def __init__(self, emb_dim, heads, dims):
		super().__init__()
		self.heads = heads
		self.qMat = torch.nn.Linear(emb_dim, heads * dims, bias=False)
		self.kMat = torch.nn.Linear(emb_dim, heads * dims, bias=False)
		self.vMat = torch.nn.Linear(emb_dim, heads * dims, bias=False)
		self.oMat = torch.nn.Linear(heads * dims, emb_dim, bias=False)
	def forward(self, x):
		# x: (batch_size, seq_len, emb_dim)
		# return: (batch_size, seq_len, emb_dim)
	   
		q = self.qMat(x).view(x.size(0), x.size(1), self.heads, -1).transpose(-3, -2)
		k = self.kMat(x).view(x.size(0), x.size(1), self.heads, -1).transpose(-3, -2)
		v = self.vMat(x).view(x.size(0), x.size(1), self.heads, -1).transpose(-3, -2)
		# attention: (batch_size, heads, seq_len, emb_dim)
		attention = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.1).transpose(-3, -2).contiguous()
		attention = self.oMat(attention.view(attention.size(0), attention.size(1), -1))
		return attention
class Layer(torch.nn.Module):
	def __init__(self, emb_dim, heads, dims):
		super().__init__()
		self.dropout = torch.nn.Dropout(0.1)
		self.heads = heads
		self.attention = MaskedAttention(emb_dim, heads, dims)
		self.normalize = torch.nn.LayerNorm([emb_dim])
		self.linear = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
		self.normalize2 = torch.nn.LayerNorm([emb_dim])
	def forward(self, x):
		# x: (batch_size, seq_len, emb_dim)
		attention = self.attention(x)
		norm = self.normalize(x + self.dropout(attention))
		lin = self.linear(norm)
		norm2 = self.normalize2(norm + self.dropout(lin))
		return norm2
class Network(torch.nn.Module):
	def  __init__(self):
		super().__init__()
		self.dropout = torch.nn.Dropout(0.1)
		self.emb = torch.nn.Embedding(50560, 768, padding_idx=50559)
		self.register_buffer("pos", (torch.fmod(torch.arange(self.emb.embedding_dim), 2).unsqueeze(0) * torch.sin(torch.arange(c - 1).flip(0).unsqueeze(1) / torch.pow(torch.tensor(10000), torch.floor(torch.arange(self.emb.embedding_dim) / 2).unsqueeze(0) * 2 / self.emb.embedding_dim)) + (1 - torch.fmod(torch.arange(self.emb.embedding_dim), 2).unsqueeze(0)) * torch.cos(torch.arange(c - 1).flip(0).unsqueeze(1) / torch.pow(torch.tensor(10000), torch.floor(torch.arange(self.emb.embedding_dim) / 2).unsqueeze(0) * 2 / self.emb.embedding_dim))))
		self.temp = [Layer(self.emb.embedding_dim, 12, 64) for _ in range(12)]
		self.layers = torch.nn.Sequential(*self.temp)
		self.linear3 = torch.nn.Linear(self.emb.embedding_dim, 50560, bias=False)
		self.emb.weight = self.linear3.weight
	def forward(self, x):
		embed = self.dropout(self.emb(x) + self.pos[:x.size(-1), :].unsqueeze(0))
		layer = self.layers(embed)
		lin3 = self.linear3(layer)
		return torch.log_softmax(lin3, dim=-1)
class EvalMaskedAttention(torch.nn.Module):
	def __init__(self, emb_dim, heads, dims):
		super().__init__()
		self.heads = heads
		self.qMat = torch.nn.Linear(emb_dim, heads * dims, bias=False)
		self.kMat = torch.nn.Linear(emb_dim, heads * dims, bias=False)
		self.vMat = torch.nn.Linear(emb_dim, heads * dims, bias=False)
		self.oMat = torch.nn.Linear(heads * dims, emb_dim, bias=False)
	def forward(self, x):
		# x: (batch_size, seq_len, emb_dim)
		# return: (batch_size, seq_len, emb_dim)
		q = self.qMat(x).view(x.size(0), x.size(1), self.heads, -1).transpose(-3, -2)
		k = self.kMat(x).view(x.size(0), x.size(1), self.heads, -1).transpose(-3, -2)
		v = self.vMat(x).view(x.size(0), x.size(1), self.heads, -1).transpose(-3, -2)
		# attention: (batch_size, heads, seq_len, emb_dim)
		attention = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(-3, -2).contiguous()
		attention = self.oMat(attention.view(attention.size(0), attention.size(1), -1))
		return attention
class EvalLayer(torch.nn.Module):
	def __init__(self, emb_dim, heads, dims):
		super().__init__()
		self.heads = heads
		self.attention = EvalMaskedAttention(emb_dim, heads, dims)
		self.normalize = torch.nn.LayerNorm([emb_dim])
		self.linear = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
		self.normalize2 = torch.nn.LayerNorm([emb_dim])
	def forward(self, x):
		# x: (batch_size, seq_len, emb_dim)
		attention = self.attention(x)
		norm = self.normalize(x + attention)
		lin = self.linear(norm)
		norm2 = self.normalize2(norm + lin)
		return norm2
class EvalNetwork(torch.nn.Module):
	def  __init__(self):
		super().__init__()
		self.emb = torch.nn.Embedding(50560, 768, padding_idx=50559)
		self.register_buffer("pos", (torch.fmod(torch.arange(self.emb.embedding_dim), 2).unsqueeze(0) * torch.sin(torch.arange(c - 1).flip(0).unsqueeze(1) / torch.pow(torch.tensor(10000), torch.floor(torch.arange(self.emb.embedding_dim) / 2).unsqueeze(0) * 2 / self.emb.embedding_dim)) + (1 - torch.fmod(torch.arange(self.emb.embedding_dim), 2).unsqueeze(0)) * torch.cos(torch.arange(c - 1).flip(0).unsqueeze(1) / torch.pow(torch.tensor(10000), torch.floor(torch.arange(self.emb.embedding_dim) / 2).unsqueeze(0) * 2 / self.emb.embedding_dim))))
		self.temp = [EvalLayer(self.emb.embedding_dim, 12, 64) for _ in range(12)]
		self.layers = torch.nn.Sequential(*self.temp)
		self.linear3 = torch.nn.Linear(self.emb.embedding_dim, 50560, bias=False)
		self.emb.weight = self.linear3.weight
	def forward(self, x):
		embed = self.emb(x) + self.pos[:x.size(-1), :].unsqueeze(0)
		layer = self.layers(embed)
		lin3 = self.linear3(layer)
		return torch.log_softmax(lin3, dim=-1)
r = 0
torch.set_float32_matmul_precision('high')
model = torch.compile(Network().to(device))
lFunc = torch.nn.NLLLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
s = 0
processed = 0
r = 0
content = []
temp5 = 0
processed = 0
def preprocess1(x):
	return lzma.decompress(x).decode()
def preprocess2(x):
	return re.split(r'(?<!\x00)(?:(?:\x00*[^\x00]*\.txt\x00{56}[0-9]{7}\x000{7}\x000{7}\x00[0-9]{11}\x000{11}\x00[0-9]{6}\x00 0\x00{100}ustar  \x00{65}0{7}\x000{7}\x00{168})|(?:\x00*$))', x)
def preprocess3(x):
	return x.replace("\x00", "")
def preprocess4(x):
	return [i for i in x if len(i) != 0]
for p in range(21):
	with tarfile.open("urlsf_subset{:02d}.tar".format(p), "r") as tar:
		content = list(map(lambda x:tar.extractfile(x).read(), list(tar.getmembers())))
		with multiprocessing.Pool() as pool:
			content = list(pool.map(preprocess1, content))
		with multiprocessing.Pool() as pool:
			content = list(pool.map(preprocess2, content))
		content = list(itertools.chain.from_iterable(content))
		with multiprocessing.Pool() as pool:
			content = list(pool.map(preprocess3, content))
		content = list(map(lambda x:content[x:min(len(content), x + 1000)], list(range(0, len(content), 1000))))
		with multiprocessing.Pool() as pool:
			content = list(pool.map(preprocess4, content))
		content = list(itertools.chain.from_iterable(content))
		content = list(map(lambda x:content[x:min(len(content), x + 10000)], list(range(0, len(content), 10000))))
		for i, batch in enumerate(content):
			with open(f"{i}_raw.txt", "w") as file:
				file.write(",".join(list(map(lambda x:"\"{a}\"".format(a=x.replace("\"", "\"\"")) if "," in x or "\n" in x or "\"" in x else x, batch))))
				file.close()
		chunks = len(content)
		for z in range(chunks):
			total = []
			with open(f"{z}_raw.txt", "r") as rawFile:
				rawReader = csv.reader(rawFile) # change contents to floats
				total = list(rawReader)[0]
				rawFile.close()
			os.remove(f"{z}_raw.txt")
			results = list(map(lambda x:total[x:min(x + 100, len(total))], list(range(0, len(total), 100))))
			for k, batch in enumerate(results):
				with open(f"{k}.txt", "w") as file:
					file.write(",".join(list(map(lambda x:"\"{a}\"".format(a=x.replace("\"", "\"\"")) if "," in x or "\n" in x or "\"" in x else x, batch))))
					file.close()
			processes = []
			for i in range(len(results)):
				processes = processes + [subprocess.Popen([r"./test", f"{i}.txt", "vocab.txt"])]
			error_results = [q.wait() for q in processes]
			if not all([q == 0 for q in error_results]):
				raise RuntimeError()
			additional = set()
			allVocab = []
			temp = []
			for i in range(len(results)):
				allVocab = allVocab + [[]]
				with open(f"{i}_encoded.txt", "r") as file:
					temp = temp + [file.read()[:-1].split("\n")]
					file.close()
				with open(f"{i}_vocab.txt", "r") as file:
					reader = csv.reader(file)
					readList = list(reader)
					if len(readList) > 0:
						allVocab[i] = readList[0]
						additional.update(allVocab[i])
					file.close()
			additional = list(additional)
			temp = list(map(lambda y:list(map(lambda x:x.split(","), y)), temp))
			temp = list(map(lambda z:list(map(lambda x:list(map(lambda y:int(y), x)), z)), temp))
			if len(additional) > 0:
				indices = list(range(len(allVocab)))
				temp = list(map(lambda x:list(map(lambda y:list(map(lambda z:z if z - vLen < 0 else (additional.index(allVocab[x][z - vLen]) + vLen), y)), temp[x])), indices))
				with open("vocab.txt", "a") as file:
					file.write("," + ",".join(list(map(lambda x:"\"{a}\"".format(a=x.replace("\"", "\"\"")) if "," in x or "\n" in x or "\"" in x else x, additional))))
					file.close()
				vLen += len(additional)
			temp = list(itertools.chain.from_iterable(temp))
			for i in range(len(allVocab)):
				os.remove(f"{i}_encoded.txt")
				os.remove(f"{i}_vocab.txt")
				os.remove(f"{i}.txt")
			reserve = reserve + temp
			remaining = sum(list(map(lambda x:len(x) - 1, reserve)))
			mega = 0
			if h * agg_size < starting_steps:
				mega = int(min(starting_steps - h * agg_size, agg_size) * (2 * min_batch_size + rate * (h * agg_size + min(starting_steps, (h + 1) * agg_size) - 1)) / 2 + max(0, (h + 1) * agg_size - starting_steps) * max_batch_size)
			else:
				mega = int(agg_size * max_batch_size)
			while remaining >= mega:
				count = list(map(lambda x:len(x) - 1, reserve))
				count = torch.cumsum(torch.tensor(count), 0)
				loaction = torch.min(torch.arange(count.size(0)).double().masked_fill(count <= mega, float("Inf"))).item()
				if loaction != float("Inf"):
					loaction = int(loaction)
					send = reserve[:loaction]
					if loaction != 0:
						if mega - count[loaction - 1] != 0:
							send = send + [reserve[loaction][:mega + 1 - count[loaction - 1]]]
							reserve[loaction] = reserve[loaction][mega - count[loaction - 1]:]
						reserve = reserve[loaction:]
					else:
						send = send + [reserve[0][:mega + 1]]
						reserve[0] = reserve[0][mega:]
				else:
					send = reserve
					reserve = []
				isample = list(map(lambda x:x[:-1], send))
				osample = list(map(lambda x:x[1:], send))
				isample = list(itertools.chain.from_iterable(list(map(lambda x:list(map(lambda y:x[y:min(len(x), y + c - 1)], list(range(0, len(x), c - 1)))), isample))))
				osample = list(itertools.chain.from_iterable(list(map(lambda x:list(map(lambda y:x[y:min(len(x), y + c - 1)], list(range(0, len(x), c - 1)))), osample))))
				shuffleOrder = list(torch.randperm(len(osample)))
				isample = list(map(lambda x:isample[x], shuffleOrder))
				osample = list(map(lambda x:osample[x], shuffleOrder))
				isample2 = []
				osample2 = []
				scount = list(map(lambda x:len(x), osample))
				scount = torch.cumsum(torch.tensor(scount), 0)
				sloaction = torch.min(torch.arange(scount.size(0)).double().masked_fill(scount <= b, float("Inf"))).item()
				while len(osample) > 0:
					if sloaction != float("Inf"):
						sloaction = int(sloaction)
						isample2 = isample2 + [isample[:sloaction]]
						osample2 = osample2 + [osample[:sloaction]]
						if sloaction != 0:
							if b - scount[sloaction - 1] != 0:
								isample2[-1] = isample2[-1] + [isample[sloaction][:b - scount[sloaction - 1]]]
								osample2[-1] = osample2[-1] + [osample[sloaction][:b - scount[sloaction - 1]]]
								isample[sloaction] = isample[sloaction][b - scount[sloaction - 1]:]
								osample[sloaction] = osample[sloaction][b - scount[sloaction - 1]:]
							isample = isample[sloaction:]
							osample = osample[sloaction:]
						else:
							isample2[-1] = isample2[-1] + [isample[0][:b]]
							osample2[-1] = osample2[-1] + [osample[0][:b]]
							isample[0] = isample[0][b:]
							osample[0] = osample[0][b:]
					else:
						isample2 = isample2 + [isample]
						osample2 = osample2 + [osample]
						isample = []
						osample = []
					b = min(b + rate, max_batch_size)
					if len(osample) > 0:
						scount = list(map(lambda x:len(x), osample))
						scount = torch.cumsum(torch.tensor(scount), 0)
						sloaction = torch.min(torch.arange(scount.size(0)).double().masked_fill(scount <= b, float("Inf"))).item()
				isample = isample2
				osample = osample2
				isample = list(map(lambda x:torch.tensor(list(map(lambda y:y + ([50559] * ((-len(y)) % (c - 1))), x))).unsqueeze(0), isample))
				osample = list(map(lambda x:torch.tensor(list(map(lambda y:y + ([50559] * ((-len(y)) % (c - 1))), x))).unsqueeze(0), osample))
				maxLength = max(list(map(lambda x:x.size(1), osample)))
				t0 = torch.cat(list(map(lambda x:torch.cat((x, torch.full((1, maxLength - x.size(1), c - 1), 50559)), 1), isample)), 0).to(torch.int32)
				t1 = torch.cat(list(map(lambda x:torch.cat((x, torch.full((1, maxLength - x.size(1), c - 1), 50559)), 1), osample)), 0).to(torch.int32)
				torch.save(t0, f"ishard_{h}.pt")
				torch.save(t1, f"oshard_{h}.pt")
				h = h + 1
				for x in range(t0.size(0)):
					model.zero_grad()
					lloss = 0
					total_batch_size = (t0[x, :, :] != 50559).sum().item()
					for y in range(0, int(min(t1.size(1), torch.min(torch.arange(t1.size(1)).double().masked_fill(t1[x, :, 0] != 50559, float("Inf"))).item())), b2):
						output = t1[x, y:min(y + b2, int(min(t1.size(1), torch.min(torch.arange(t1.size(1)).double().masked_fill(t1[x, :, 0] != 50559, float("Inf"))).item()))), :].long().reshape(-1)
						inputt = t0[x, y:min(y + b2, int(min(t1.size(1), torch.min(torch.arange(t1.size(1)).double().masked_fill(t1[x, :, 0] != 50559, float("Inf"))).item()))), :].to(device)
						omask = output != 50559
						#print(model(input).size())
						#print(output.size())
						#print(model(input).view(input.size(0), -1).view(input.size(0), input.size(-1)).shape)
						temp5 = omask.sum().item() / total_batch_size
						with torch.autocast(device_type=device, dtype=torch.bfloat16):
							loss = lFunc(model(inputt).reshape(-1, 50560)[omask.to(device), :], output[omask].to(device)) * temp5
							lloss += loss.item()
						#print(input.size())
						#loss = 
						loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
					if r < warmup:
						learning = max_rate * (r + 1) / (warmup + 1)
						for param in optimizer.param_groups:
							param["lr"] = learning
					elif r > max_iters:
						for param in optimizer.param_groups:
							param["lr"] = min_rate
					else:
						learning = min_rate + 0.5 * (1.0 + math.cos(math.pi * (r - warmup) / (max_iters - warmup))) * (max_rate - min_rate)
						for param in optimizer.param_groups:
							param["lr"] = learning
					optimizer.step()
					r = r + 1
					if r % 100 == 0:
						print(lloss)
				remaining = sum(list(map(lambda x:len(x) - 1, reserve)))
				mega = 0
				if h * agg_size < starting_steps:
					mega = int(min(starting_steps - h * agg_size, agg_size) * (2 * min_batch_size + rate * (h * agg_size + min(starting_steps, (h + 1) * agg_size) - 1)) / 2 + max(0, (h + 1) * agg_size - starting_steps) * max_batch_size)
				else:
					mega = int(agg_size * max_batch_size)
			if p == 20 and z == chunks - 1 and remaining > 0:
				send = reserve
				reserve = []
				isample = list(map(lambda x:x[:-1], send))
				osample = list(map(lambda x:x[1:], send))
				isample = list(itertools.chain.from_iterable(list(map(lambda x:list(map(lambda y:x[y:min(len(x), y + c - 1)], list(range(0, len(x), c - 1)))), isample))))
				osample = list(itertools.chain.from_iterable(list(map(lambda x:list(map(lambda y:x[y:min(len(x), y + c - 1)], list(range(0, len(x), c - 1)))), osample))))
				shuffleOrder = list(torch.randperm(len(osample)))
				isample = list(map(lambda x:isample[x], shuffleOrder))
				osample = list(map(lambda x:osample[x], shuffleOrder))
				isample2 = []
				osample2 = []
				scount = list(map(lambda x:len(x), osample))
				scount = torch.cumsum(torch.tensor(scount), 0)
				sloaction = torch.min(torch.arange(scount.size(0)).double().masked_fill(scount <= b, float("Inf"))).item()
				while len(osample) > 0:
					if sloaction != float("Inf"):
						sloaction = int(sloaction)
						isample2 = isample2 + [isample[:sloaction]]
						osample2 = osample2 + [osample[:sloaction]]
						if sloaction != 0:
							if b - scount[sloaction - 1] != 0:
								isample2[-1] = isample2[-1] + [isample[sloaction][:b - scount[sloaction - 1]]]
								osample2[-1] = osample2[-1] + [osample[sloaction][:b - scount[sloaction - 1]]]
								isample[sloaction] = isample[sloaction][b - scount[sloaction - 1]:]
								osample[sloaction] = osample[sloaction][b - scount[sloaction - 1]:]
							isample = isample[sloaction:]
							osample = osample[sloaction:]
						else:
							isample2[-1] = isample2[-1] + [isample[0][:b]]
							osample2[-1] = osample2[-1] + [osample[0][:b]]
							isample[0] = isample[0][b:]
							osample[0] = osample[0][b:]
					else:
						isample2 = isample2 + [isample]
						osample2 = osample2 + [osample]
						isample = []
						osample = []
					b = min(b + rate, max_batch_size)
					if len(osample) > 0:
						scount = list(map(lambda x:len(x), osample))
						scount = torch.cumsum(torch.tensor(scount), 0)
						sloaction = torch.min(torch.arange(scount.size(0)).double().masked_fill(scount <= b, float("Inf"))).item()
				isample = isample2
				osample = osample2
				isample = list(map(lambda x:torch.tensor(list(map(lambda y:y + ([50559] * ((-len(y)) % (c - 1))), x))).unsqueeze(0), isample))
				osample = list(map(lambda x:torch.tensor(list(map(lambda y:y + ([50559] * ((-len(y)) % (c - 1))), x))).unsqueeze(0), osample))
				maxLength = max(list(map(lambda x:x.size(1), osample)))
				t0 = torch.cat(list(map(lambda x:torch.cat((x, torch.full((1, maxLength - x.size(1), c - 1), 50559)), 1), isample)), 0).to(torch.int32)
				t1 = torch.cat(list(map(lambda x:torch.cat((x, torch.full((1, maxLength - x.size(1), c - 1), 50559)), 1), osample)), 0).to(torch.int32)
				torch.save(t0, f"ishard_{h}.pt")
				torch.save(t1, f"oshard_{h}.pt")
				h = h + 1
				for x in range(t0.size(0)):
					model.zero_grad()
					lloss = 0
					total_batch_size = (t0[x, :, :] != 50559).sum().item()
					for y in range(0, int(min(t1.size(1), torch.min(torch.arange(t1.size(1)).double().masked_fill(t1[x, :, 0] != 50559, float("Inf"))).item())), b2):
						output = t1[x, y:min(y + b2, int(min(t1.size(1), torch.min(torch.arange(t1.size(1)).double().masked_fill(t1[x, :, 0] != 50559, float("Inf"))).item()))), :].long().reshape(-1)
						inputt = t0[x, y:min(y + b2, int(min(t1.size(1), torch.min(torch.arange(t1.size(1)).double().masked_fill(t1[x, :, 0] != 50559, float("Inf"))).item()))), :].to(device)
						omask = output != 50559
						#print(model(input).size())
						#print(output.size())
						#print(model(input).view(input.size(0), -1).view(input.size(0), input.size(-1)).shape)
						temp5 = omask.sum().item() / total_batch_size
						with torch.autocast(device_type=device, dtype=torch.bfloat16):
							loss = lFunc(model(inputt).reshape(-1, 50560)[omask.to(device), :], output[omask].to(device)) * temp5
							lloss += loss.item()
						#print(input.size())
						#loss = 
						loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
					if r < warmup:
						learning = max_rate * (r + 1) / (warmup + 1)
						for param in optimizer.param_groups:
							param["lr"] = learning
					elif r > max_iters:
						for param in optimizer.param_groups:
							param["lr"] = min_rate
					else:
						learning = min_rate + 0.5 * (1.0 + math.cos(math.pi * (r - warmup) / (max_iters - warmup))) * (max_rate - min_rate)
						for param in optimizer.param_groups:
							param["lr"] = learning
					optimizer.step()
					r = r + 1
					if r % 100 == 0:
						print(lloss)
		tar.close()
eval_model = torch.compile(EvalNetwork())
eval_model.load_state_dict(model.to("cpu").state_dict())
torch.save(eval_model, "gpt_model.pth")
def generate():
	torch.set_float32_matmul_precision('high')
	device = "cuda"
	eval_model = torch.compile(torch.load("gpt_model.pth", weights_only=False).to(device))
	c = 1025
	v = []
	with open("vocab.txt", "r") as csvDataFile:
		csvReader = csv.reader(csvDataFile)
		for row in csvReader:
			for i in row:
				v.append(i)
		csvDataFile.close()
	t = torch.multinomial(torch.exp(eval_model(torch.tensor([[50559]]).to(device))[:, -1, :]), 1)
	total = b""
	output = "�"
	if t[0, 0].item() < len(v):
		output = v[t[0, 0].item()]
	output = re.split(r'(\\x[0-9]{1,3})', output)
	for i in range(len(output)):
		if output[i][:2] == "\\x" and len(output[i]) > 2 and ord(output[i][2]) > 47 and ord(output[i][2]) < 58:
			temp = 0
			end = 0
			for j in output[i][2:]:
				temp = temp * 10 + (ord(j) - 48)
				if temp > 255:
					temp = int((temp - (ord(j) - 48)) / 10)
					break
				else:
					end = end + 1
			temp2 = output[i][2 + end:]
			output[i] = bytes([temp])
			output[i] = output[i] + bytes(temp2, encoding="utf-8")
		else:
			output[i] = bytes(output[i], encoding="utf-8")
	total = total + b"".join(output)
	if len(total) > 0 and total[-1] >> 7 == 0x00:
		output = total.decode("utf-8", errors="replace")
		total = b""
	elif len(total) > 0 and (total[-1] >> 5 == 0x06 or total[-1] >> 4 == 0x0e or total[-1] >> 3 == 0x1e):
		output = total[:-1].decode("utf-8", errors="replace")
		total = total[-1:]
	elif len(total) > 1 and (total[-2] >> 4 == 0x0e or total[-2] >> 3 == 0x1e):
		output = total[:-2].decode("utf-8", errors="replace")
		total = total[-2:]
	elif len(total) > 2 and total[-3] >> 3 == 0x1e:
		output = total[:-3].decode("utf-8", errors="replace")
		total = total[-3:]
	else:
		output = total.decode("utf-8", errors="replace")
		total = b""
	print(output, end="")
	while True:
		output = torch.multinomial(torch.exp(eval_model(t)[:, -1, :]), 1)
		if t.size(-1) == c - 1:
			t = t.roll(-1, -1)
		else:
			t = torch.cat((t, torch.tensor([[0]]).to(device)), -1)
		t[0, -1] = output.sum()
		if output.item() < len(v):
			output = v[output.item()]
		else:
			output = "�"
		output = re.split(r'(\\x[0-9]{1,3})', output)
		for i in range(len(output)):
			if output[i][:2] == "\\x" and len(output[i]) > 2 and ord(output[i][2]) > 47 and ord(output[i][2]) < 58:
				temp = 0
				end = 0
				for j in output[i][2:]:
					temp = temp * 10 + (ord(j) - 48)
					if temp > 255:
						temp = int((temp - (ord(j) - 48)) / 10)
						break
					else:
						end = end + 1
				temp2 = output[i][2 + end:]
				output[i] = bytes([temp])
				output[i] = output[i] + bytes(temp2, encoding="utf-8")
			else:
				output[i] = bytes(output[i], encoding="utf-8")
		total = total + b"".join(output)
		if len(total) > 0 and total[-1] >> 7 == 0x00:
			output = total.decode("utf-8", errors="replace")
			total = b""
		elif len(total) > 0 and (total[-1] >> 5 == 0x06 or total[-1] >> 4 == 0x0e or total[-1] >> 3 == 0x1e):
			output = total[:-1].decode("utf-8", errors="replace")
			total = total[-1:]
		elif len(total) > 1 and (total[-2] >> 4 == 0x0e or total[-2] >> 3 == 0x1e):
			output = total[:-2].decode("utf-8", errors="replace")
			total = total[-2:]
		elif len(total) > 2 and total[-3] >> 3 == 0x1e:
			output = total[:-3].decode("utf-8", errors="replace")
			total = total[-3:]
		else:
			output = total.decode("utf-8", errors="replace")
			total = b""
		print(output, end="")
with open("gpt.pkl", "wb") as f:
	pickle.dump(generate, f)
	f.close()
generate()