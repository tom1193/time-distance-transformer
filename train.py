import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import math

def train(config, config_id, model, epoch_range, last_global_step, train_loader, val_loader, device, criterion,
          best_metric, optimizer, scheduler, stopper, writer, model_dir, checkpoint_dir):
    model.train()
    global_step = last_global_step
    best_metric_epoch = 0
    start_epoch, epochs = epoch_range
    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch} =============")
        # epoch_acc = 0
        # epoch_loss = 0
        step = 0

        for batch in tqdm(train_loader):
            data, times, label = (
                batch["img_seq"].to(device),
                batch["times"].to(device),
                batch["label"].to(device),
            )
            output = model(data, times)
            label = F.one_hot(label, num_classes=2).to(torch.float32)
            loss = criterion(output, label)

            # check if loss becomes nan
            if math.isnan(loss):
                checkpoint_path = os.path.join(checkpoint_dir, f"isnan_step{global_step}.tar")
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_metric': best_metric,
                }, checkpoint_path)
                print(f'Saved NaN state at step {global_step}')
                return

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            acc = (output.argmax(dim=1) == label.argmax(dim=1)).float().mean()
            # epoch_acc += acc / len(train_loader)
            # epoch_loss += loss / len(train_loader)

            writer.add_scalar("Acc/train", acc, global_step)
            writer.add_scalar("Loss/train", loss, global_step)

            # Validation
            if global_step % config["val_interval"] == 0:
                val_loss, val_acc = validate(model, val_loader, device, criterion, writer, global_step)
                
                # update stopping criteria every val period
                stopper.step(val_acc.cpu().numpy())
                
                if val_acc > best_metric:
                    best_metric = val_acc
                    best_metric_epoch = epoch
                    model_path = os.path.join(model_dir, f"best_model.pth")
                    torch.save(model.state_dict(), model_path)
                    print("Saved new best model")
                print(f"{config_id}:"
                    f"\n Current: epoch {epoch}, mean acc {val_acc:.4f}"
                  f"\n Best: epoch {best_metric_epoch}, mean acc {best_metric:.4f}")
            
            # Early stopping
            if stopper.acc_check_stop():
                return
            
            step += 1
            global_step += 1
            
        # Save checkpoints
        if epoch % config["checkpoint_interval"] == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch{epoch}.tar")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
            }, checkpoint_path)

def validate(model, val_loader, device, criterion, writer, global_step):
    model.eval()
    val_acc = 0
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
        # batch = next(val_iterator)
            data, times, label = (
                batch["img_seq"].to(device),
                batch["times"].to(device),
                batch["label"].to(device),
            )
            output = model(data, times)
            label = F.one_hot(label, num_classes=2).to(torch.float32)
            loss = criterion(output, label)

            acc = (output.argmax(dim=1) == label.argmax(dim=1)).float().mean()
            val_acc += acc
            val_loss += loss
            
    val_acc /= len(val_loader)
    val_loss /= len(val_loader)
    writer.add_scalar('Acc/val', val_acc, global_step)
    writer.add_scalar('Loss/val', val_loss, global_step)
    return val_loss, val_acc

def pretrain_MAE(config, config_id, vit, mae, epoch_range, last_global_step, train_loader, val_loader, device,
          best_metric, optimizer, scheduler, stopper, writer, model_dir, checkpoint_dir):
    mae.train()
    global_step = last_global_step
    best_metric_epoch = 0
    start_epoch, epochs = epoch_range
    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch} =============")
        # epoch_acc = 0
        # epoch_loss = 0
        step = 0

        for batch in tqdm(train_loader):
            data, times = (
                batch["img_seq"].to(device),
                batch["times"].to(device),
            )
            loss = mae(data, times)
            
            # check if loss becomes nan
            if math.isnan(loss):
                checkpoint_path = os.path.join(checkpoint_dir, f"isnan_step{global_step}.tar")
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': mae.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_metric': best_metric,
                }, checkpoint_path)
                print(f'Saved NaN state at step {global_step}')
                return

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # writer.add_scalar("Acc/train", acc, global_step)
            writer.add_scalar("Loss/train", loss, global_step)

            # Validation
            if global_step % config["val_interval"] == 0:
                val_loss = validate_MAE(mae, val_loader, device, writer, global_step)
                
                # update stopping criteria every val period
                stopper.step(val_loss.cpu().numpy())
                
                if val_loss < best_metric:
                    best_metric = val_loss
                    best_metric_epoch = epoch
                    mae_path = os.path.join(model_dir, f"best_pretrain_mae.pth")
                    vit_path = os.path.join(model_dir, f"best_pretrain_model.pth")
                    torch.save(mae.state_dict(), mae_path)
                    torch.save(vit.state_dict(), vit_path)
                    print("Saved new best model")
                print(f"{config_id}:"
                    f"\n Current: epoch {epoch}, mean acc {val_loss:.4f}"
                  f"\n Best: epoch {best_metric_epoch}, mean acc {best_metric:.4f}")
            
            # Early stopping
            if stopper.loss_check_stop():
                return
            
            step += 1
            global_step += 1
            
        # Save checkpoints
        if epoch % config["checkpoint_interval"] == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch{epoch}.tar")
            torch.save({
                'epoch': epoch,
                'model_state_dict': mae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
            }, checkpoint_path)

def validate_MAE(model, val_loader, device, writer, global_step):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
        # batch = next(val_iterator)
            data, times = (
                batch["img_seq"].to(device),
                batch["times"].to(device),
            )
            loss = model(data, times)
            val_loss += loss
            
    val_loss /= len(val_loader)
    # writer.add_scalar('Acc/val', val_acc, global_step)
    writer.add_scalar('Loss/val', val_loss, global_step)
    return val_loss
