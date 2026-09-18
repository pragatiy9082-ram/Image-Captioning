import os, re, json, random
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

SEED = 42
random.seed(SEED); torch.manual_seed(SEED)

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, 'captions_500.csv')
IMG_DIR = os.path.join(BASE, 'images')
MODEL_OUT = os.path.join(BASE, 'caption_model.pth')
META_OUT = os.path.join(BASE, 'metadata.json')

# Captions in this educational dataset follow: "A/An COLOR OBJECT is shown in a LOCATION."
def parse_caption(c):
    c = c.lower().strip().rstrip('.')
    m = re.match(r'^(?:a|an) (\w+) (\w+) is shown in (?:a|an) (\w+)$', c)
    if not m:
        raise ValueError(f'Unexpected caption format: {c}')
    return m.group(1), m.group(2), m.group(3)


df = pd.read_csv(CSV)
rows = []
for _, r in df.iterrows():
    color, obj, loc = parse_caption(r['caption'])
    rows.append((r['image'], color, obj, loc))

df[['color','object','location']] = pd.DataFrame([x[1:] for x in rows], index=df.index)
classes = {k: sorted(df[k].unique().tolist()) for k in ['color','object','location']}
index = {k:{v:i for i,v in enumerate(classes[k])} for k in classes}

with open(META_OUT, 'w') as f:
    json.dump({'classes': classes, 'image_size': 64}, f, indent=2)

class CaptionDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.frame)
    def __getitem__(self, i):
        r = self.frame.iloc[i]
        img = Image.open(os.path.join(IMG_DIR, r['image'])).convert('RGB')
        img = self.transform(img)
        y = torch.tensor([index['color'][r['color']], index['object'][r['object']], index['location'][r['location']]], dtype=torch.long)
        return img, y

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(p=0.35),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
val_transform = transforms.Compose([
    transforms.Resize((64,64)), transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# Fixed split so results are reproducible.
g = torch.Generator().manual_seed(SEED)
perm = torch.randperm(len(df), generator=g).tolist()
cut = int(0.8*len(df)); train_idx, val_idx = perm[:cut], perm[cut:]
train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
train_ds = CaptionDataset(train_df, transform)
val_ds = CaptionDataset(val_df, val_transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

class CaptionNet(nn.Module):
    def __init__(self, nc, no, nl):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,192,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(192,128), nn.ReLU(), nn.Dropout(0.15))
        self.color = nn.Linear(128,nc)
        self.object = nn.Linear(128,no)
        self.location = nn.Linear(128,nl)
    def forward(self,x):
        z=self.fc(self.features(x))
        return self.color(z), self.object(z), self.location(z)

model = CaptionNet(len(classes['color']),len(classes['object']),len(classes['location']))
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def accuracy(out, y): return (out.argmax(1)==y).float().mean().item()

best = -1
for epoch in range(1,11):
    model.train(); total=0; tr_acc=[0,0,0]
    for x,y in train_loader:
        opt.zero_grad(); outs=model(x)
        loss=sum(loss_fn(o,y[:,i]) for i,o in enumerate(outs))
        loss.backward(); opt.step()
        total += loss.item()
        for i,o in enumerate(outs): tr_acc[i]+=accuracy(o,y[:,i])
    model.eval(); va=[0,0,0]; n=0
    with torch.no_grad():
        for x,y in val_loader:
            outs=model(x); bs=x.size(0); n+=1
            for i,o in enumerate(outs): va[i]+=accuracy(o,y[:,i])
    tr=[a/len(train_loader) for a in tr_acc]; va=[a/n for a in va]
    score=sum(va)/3
    print(f'Epoch {epoch:02d}/10 | loss {total/len(train_loader):.4f} | train {tr[0]:.3f}/{tr[1]:.3f}/{tr[2]:.3f} | val {va[0]:.3f}/{va[1]:.3f}/{va[2]:.3f}')
    if score > best:
        best=score
        torch.save(model.state_dict(), MODEL_OUT)

print(f'Best validation average accuracy: {best:.4f}')
print(f'Saved: {MODEL_OUT}')
print(f'Saved: {META_OUT}')
