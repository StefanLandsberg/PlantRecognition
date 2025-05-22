def train_one_epoch_mlp(model, dataloader, criterion, optimizer, scaler, scheduler, global_epoch, fold_num, device, writer, num_classes, ema_model):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Track per-epoch metrics
    epoch_metrics = {
        'loss': [],
        'acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    pbar = tqdm(dataloader, desc=f'Fold {fold_num} - Epoch {global_epoch}')
    for batch_idx, (features, labels, _) in enumerate(pbar):
        features, labels = features.to(device), labels.to(device)
        features = features.float()  # Ensure features are float32
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features, labels=labels if MLP_USE_ARCFACE else None)
        loss = criterion(outputs, labels)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    # Calculate epoch metrics
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    # Log metrics
    if writer is not None:
        writer.add_scalar(f'fold_{fold_num}/train_loss', epoch_loss, global_epoch)
        writer.add_scalar(f'fold_{fold_num}/train_acc', epoch_acc, global_epoch)
    
    # Store metrics
    epoch_metrics['loss'].append(epoch_loss)
    epoch_metrics['acc'].append(epoch_acc)
    
    return epoch_metrics

def validate_one_epoch_mlp(model, dataloader, criterion, device, global_epoch, writer, num_classes, scheduler=None, swa_model=None, ema_model=None, return_preds=False, fold_num=0):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels, _ in tqdm(dataloader, desc=f'Validation - Fold {fold_num}'):
            features, labels = features.to(device), labels.to(device)
            features = features.float()  # Ensure features are float32
            
            # Forward pass
            outputs = model(features, labels=labels if MLP_USE_ARCFACE else None)
            loss = criterion(outputs, labels)
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # Calculate validation metrics
    val_loss = total_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    # Log metrics
    if writer is not None:
        writer.add_scalar(f'fold_{fold_num}/val_loss', val_loss, global_epoch)
        writer.add_scalar(f'fold_{fold_num}/val_acc', val_acc, global_epoch)
    
    # Calculate and log the gap
    if writer is not None:
        train_metrics = writer.get_scalar(f'fold_{fold_num}/train_acc', global_epoch)
        if train_metrics is not None:
            gap = train_metrics - val_acc
            writer.add_scalar(f'fold_{fold_num}/train_val_gap', gap, global_epoch)
            print(f"\n{TermColors.CYAN}Train-Val Gap: {gap:.2f}%{TermColors.ENDC}")
            
            # Print gap analysis
            if gap < 5:
                print(f"{TermColors.YELLOW}Warning: Small gap might indicate underfitting{TermColors.ENDC}")
            elif gap > 15:
                print(f"{TermColors.YELLOW}Warning: Large gap might indicate overfitting{TermColors.ENDC}")
            else:
                print(f"{TermColors.GREEN}Gap is in healthy range{TermColors.ENDC}")
    
    return val_loss, val_acc 