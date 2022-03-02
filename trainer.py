import torch
from tqdm import tqdm 

class Trainer:
    def __init__(self):
        pass

    def compile(self, train_augmentation, validation_augmentation, lr, decay_fn, loss_fn, metric_dict, device): 
        self.train_augmentation = train_augmentation
        self.validation_augmentation = validation_augmentation
        self.lr = lr
        self.decay_fn = decay_fn 
        self.device = device
        self.loss_fn = loss_fn
        self.metric_dict = metric_dict

    def _get_optimizer(self, model, lr): 
        return torch.optim.Adam(model.parameters(), lr=lr)

    def _get_scheduler(self, decay_fn):
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda= decay_fn)

    def fit(self, model, train_loader, validation_loader, num_epoch, save_config, track=False):
        self.optimizer = self._get_optimizer(model, self.lr)
        self.scheduler = self._get_scheduler(self.decay_fn)
        self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(1, num_epoch+1):
            self._training_step(model, train_loader, self.loss_fn)
            val_loss = self._validation_step(model, validation_loader, self.loss_fn, self.metric_dict)

            if epoch % save_config["freq"] == 0:
                torch.save(model, save_config["path"])
            if track: 
                self.scheduler.step(val_loss.item())
            else: 
                self.scheduler.step()

    def _training_step(self, model, train_loader, loss_fn): 
        model.train()
        loop = tqdm(train_loader)
        total_loss = 0 

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=self.device)
            targets = targets.to(device=self.device)

            data, targets = self.train_augmentation(data, targets)
            targets = targets.type(torch.LongTensor).to(device=self.device)

            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(targets, predictions)   # .to(device=self.device)

            # backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()            

            total_loss += loss.item() * data.size(0)
            loop.set_postfix(loss=loss.item())

        print("train_loss:", total_loss/len(train_loader.dataset))

    @torch.no_grad()
    def _validation_step(self, model, validation_loader, loss_fn, metric_dict): 
        model.eval()
        test_loss = 0
        metric_eval_dict = {k:0 for k in metric_dict}

        for data, targets in validation_loader:
            data = data.to(device=self.device)
            targets = targets.to(device=self.device)

            data, targets = self.validation_augmentation(data, targets)
            targets = targets.type(torch.LongTensor).to(device=self.device)

            predictions = model(data)
            loss = loss_fn(targets, predictions)
            test_loss += loss.item() * targets.size(0)

            for metric_name in metric_dict: 
                metric_eval_dict[metric_name] += metric_dict[metric_name](targets, predictions)*targets.size(0)
    
        size = len(validation_loader.dataset)
        test_loss /= size

        print({k:(metric_eval_dict[k]/size).item() for k in metric_eval_dict})
        return test_loss
