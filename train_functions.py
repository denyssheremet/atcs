from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import pickle

def train_epoch(model, loader, optimizer, loss_module):
    model.train()
    losses, accs = [], []
    
    for premises, hypotheses, targets in tqdm(loader):

        premises = premises.cuda()
        hypotheses = hypotheses.cuda()
        targets = targets.cuda()
            
        predictions = model((premises, hypotheses, targets))
        
        optimizer.zero_grad()
        loss = loss_module(predictions, targets)
        acc = (predictions.argmax(axis=-1) == targets).float().mean()
        
        losses.append(loss)
        accs.append(acc)
        
        loss.backward()
        
        optimizer.step()
    
    train_loss = torch.tensor(losses).mean()
    train_acc = torch.tensor(accs).mean()
    return train_loss, train_acc


# measure accuracy
def evaluate(model, loader):
    model.eval()
    total_correct = 0.
    total = 0.
    bs = loader.batch_size
    
    
    for premises, hypotheses, targets in loader:
        premises = premises.cuda()
        hypotheses = hypotheses.cuda()
        targets = targets.cuda()
        
        with torch.no_grad():
            predictions = model((premises, hypotheses, targets)).argmax(axis=-1)
        total_correct += (predictions==targets).float().sum()
        total += bs
        
    acc = total_correct / total
    return acc

def update_lr(optimizer, new_lr):
    # update the learning rate for the optimizer
    for g in optimizer.param_groups:
        g['lr'] = new_lr

def train_loop(model, optimizer, loss_module, train_loader, val_loader, checkpoint_path, writer):
    lr = 0.1
    last_acc, best_acc = -1, -1
    epoch = -1
    try:
        os.remove(checkpoint_path) # need to remove it first, torch.save doesn't overwrite
    except:
        pass
    
    while lr > 1e-5:
        epoch += 1
        print(f'lr: {lr}')
        #train and evaluate
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_module)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)

        acc = evaluate(model, val_loader)
        writer.add_scalar("Acc/eval", acc, epoch)
        print(f'acc: {acc}')
        
        
        # learning rate decay
        lr = lr * 0.99
        update_lr(optimizer, lr)
        
        # save best checkpoint if necessary
        if acc > best_acc:
            best_acc = acc
            try:
                os.remove(checkpoint_path) # need to remove it first, torch.save doesn't overwrite
            except:
                pass
            model.save_model(checkpoint_path)
        
        # if val acc goes down, divide lr by 5
        if acc < last_acc:
            lr = lr / 5.
            update_lr(optimizer, lr)
            
        last_acc = acc
    writer.flush()
    
    # load the best model
    model.load_model(checkpoint_path)