from tqdm import tqdm
from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, EarlyStoppingV2, adjust_learning_rate, adjust_learning_rate_v2, visual, Buffer
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')
import torch.nn.functional as F

class PostProcessingNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, delta=0.1, mode='add'):
        super(PostProcessingNet, self).__init__()
        self.mode = mode
        self.delta = delta
        
        print(f'PostProcessingNet initialized with mode: {mode}, delta: {delta} (Norm Enabled)')

        if mode in ['add', 'mul']:
            # MLP structure for 'add' and 'mul' (Non-linear)
            self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
            self.bn1 = torch.nn.InstanceNorm1d(hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.bn2 = torch.nn.InstanceNorm1d(hidden_dim)
            self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        elif mode == 'affine':
            # Linear structure for 'affine' (Rotation/Linear Mixing)
            # y = (I + dA)x + db
            # Added LayerNorm for input normalization
            self.norm = nn.LayerNorm(state_dim)
            self.linear = torch.nn.Linear(state_dim, action_dim)
            # Initialize to ZERO for stability (start as identity transformation)
            # This prevents random noise from degrading performance initially
            nn.init.zeros_(self.linear.weight)
            nn.init.constant_(self.linear.bias, 0)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x):
        if self.mode in ['add', 'mul']:
            x = F.relu(self.fc1(x))
            x = self.bn1(x)
            x = F.relu(self.fc2(x))
            x = self.bn2(x)
            x = self.fc3(x)
            
            # Tanh activation specific logic
            out = torch.tanh(x) * self.delta
            
            if self.mode == 'mul':
                # Mul mode: returns scaling factor (1 + delta * output)
                return out + 1
            else:
                # Add mode: returns offset (delta * output)
                return out
                
        elif self.mode == 'affine':
            # Affine mode: just linear projection
            # No tanh to allow full linear transformation range, but scaled by delta for control
            # Apply LayerNorm before linear layer
            x = self.norm(x)
            return self.linear(x) * self.delta

class Exp_Combined_Norm(Exp_Basic):
    def __init__(self, args, adapter_mode='add'):
        # Temporarily store adapter mode in args for _build_model consumption if needed, 
        # but cleaner to pass it explicitly where used.
        self.adapter_mode = adapter_mode
        super(Exp_Combined_Norm, self).__init__(args)
        self.opt = self._select_optimizer()
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=0.00001)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _get_preds(self, batch_x, batch_y, batch_x_mark=None, batch_y_mark=None, i=None):

        if 'TST' in self.args.model:
            outputs = self.model(batch_x.permute(0,2,1), batch_x_mark, batch_y_mark)
            outputs = outputs.permute(0,2,1)
        elif "Period" in self.args.model or "Fre" in self.args.model:
            outputs = self.model(batch_x)
        elif 'iTrans' in self.args.model or 'DeepBooTS' in self.args.model:
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        elif 'former' in self.args.model or 'Linear' in self.args.model:
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            outputs = self.model(batch_x, dec_inp)
        elif 'rnn' in self.args.model:
            if len(batch_x.shape)<3:
                batch_x = batch_x.unsqueeze(0)
                batch_y = batch_y.unsqueeze(0)
            batch_x = batch_x.permute(0,2,1)
            batch_y = batch_y.permute(0,2,1)
            outputs = self.model(batch_x, batch_x_mark, batch_y_mark, batch_y)
        elif "Lade" in self.args.model or "Four" in self.args.model:
            outputs = self.model(batch_x)
        else:
            if len(batch_x.shape)<3:
                batch_x = batch_x.unsqueeze(0)
                batch_y = batch_y.unsqueeze(0)
            batch_x = batch_x.permute(0,2,1)
            batch_y = batch_y.permute(0,2,1)
            batch_x = batch_x.unsqueeze(-1)
            batch_y = batch_y.unsqueeze(-1)
            if self.args.model in ('STID','Mvstgn'):
                outputs = self.model(batch_x, batch_y_mark, batch_y_mark)
            else:
                outputs = self.model(batch_x, batch_y_mark, batch_y_mark, ycl=None, batch_seen=i)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        return outputs, batch_y    

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._get_preds(batch_x, batch_y, batch_x_mark, batch_y_mark, i)
                
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                self.opt.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)


                outputs, batch_y = self._get_preds(batch_x, batch_y, batch_x_mark, batch_y_mark, i)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.4f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.opt.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.4f} Vali Loss: {3:.4f} Test Loss: {4:.4f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.opt, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
        

    def test(self, setting, test=0, online=False, visual_ts=False):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        criterion = self._select_criterion()
        buffer = Buffer(self.args.seq_len, self.args.pred_len)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                batch_x_mark = None
                batch_y_mark = None
            else:
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

            
            outputs, batch_y = self._get_preds(batch_x, batch_y, batch_x_mark, batch_y_mark, i)
            
            if online:
                past_x, past_y =  buffer(batch_y)
                if past_x is not None:
                    self.model.train()
                    pred_past_y, _ = self._get_preds(past_x, past_y, None, None, i)
                    loss = criterion(pred_past_y, past_y)
                    loss.backward()
                    self.opt.step()       
                    self.opt.zero_grad()
                    self.model.eval()
            
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            if test_data.scale and self.args.inverse:
                outputs = test_data.inverse_transform(outputs)
                batch_y = test_data.inverse_transform(batch_y)

            pred = outputs
            true = batch_y

            preds.append(pred)
            trues.append(true)

            if visual_ts and i % 20 == 0:
                input = batch_x.detach().cpu().numpy()
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{0:4f}, mae:{1:4f}, rmse:{2:4f}, mape:{3:4f}, mspe:{4:4f},'.format(mse, mae, rmse, mape, mspe))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + " \n")
        f.write('mse:{0:4f}, mae:{1:4f}, rmse:{2:4f}, mape:{3:4f}, mspe:{4:4f},'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

        return


    def vali_post(self, post_net_x, post_net_y, vali_loader, criterion):
        total_loss = []
        mae_loss = []
        self.model.eval()
        post_net_x.eval()
        post_net_y.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                x_state = batch_x.view(batch_x.shape[0],-1)
                
                # Apply Adapter X
                infos_x = post_net_x(x_state)
                infos_x = infos_x.view(*batch_x.shape)
                if self.adapter_mode in ['mul']:
                    batch_x = batch_x * infos_x
                else: # add, affine
                    batch_x = batch_x + infos_x
                
                outputs, batch_y = self._get_preds(batch_x, batch_y, batch_x_mark, batch_y_mark, i)
                
                # Apply Adapter Y
                y_state = outputs.reshape(outputs.shape[0],-1)
                infos_y = post_net_y(y_state)
                infos_y = infos_y.view(*outputs.shape)
                
                if self.adapter_mode in ['mul']:
                    outputs = outputs * infos_y
                else: # add, affine
                    outputs = outputs + infos_y
                
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                mae= F.l1_loss(pred, true)

                total_loss.append(loss)
                mae_loss.append(mae)
        total_loss = np.average(total_loss)
        mae_loss = np.average(mae_loss)
        post_net_x.train()
        post_net_y.train()
        return total_loss, mae_loss
   
    def post_train(self, setting, vali_set = False):
        
        print(f"load model ... (Mode: {self.adapter_mode}) - Norm Version")
        
        # Load the base model checkpoint
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))
        self.model.eval()
        
        # Create PostProcessingNets with the correct mode
        post_net_x = PostProcessingNet(self.args.seq_len * self.args.enc_in, 
                                         self.args.d_model, 
                                         self.args.seq_len * self.args.enc_in,
                                         self.args.delta,
                                         mode=self.adapter_mode).to(self.device)
        self.post_optim_x = torch.optim.Adam(post_net_x.parameters(), lr=self.args.post_train_lr)
        
        post_net_y = PostProcessingNet(self.args.seq_len * self.args.enc_in, 
                                         self.args.d_model, 
                                         self.args.pred_len * self.args.enc_in,
                                         self.args.delta,
                                         mode=self.adapter_mode).to(self.device)
        self.post_optim_y = torch.optim.Adam(post_net_y.parameters(), lr=self.args.post_train_lr)
        
        if vali_set:
            _, data_loader = data_provider(self.args, 'val')
        else:
            _, val_loader = data_provider(self.args, 'val')
            _, data_loader = data_provider(self.args, 'train')

        _, test_loader = data_provider(self.args, 'test')
        criterion = torch.nn.MSELoss()
        early_stopping = EarlyStoppingV2(patience=self.args.patience, verbose=True)
        
        for episode in range(self.args.num_episodes):
            loss_list = []
            with tqdm(total=len(data_loader), desc=f'Iteration {episode} ({self.adapter_mode})') as pbar:
                for step, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                    
                    self.post_optim_x.zero_grad()
                    self.post_optim_y.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                    # Adapter X Forward
                    x_state = batch_x.view(batch_x.shape[0],-1)
                    infos_x = post_net_x(x_state)
                    infos_x = infos_x.view(*batch_x.shape)
                    
                    if self.adapter_mode in ['mul']:
                        batch_x = batch_x * infos_x
                    else: # add, affine
                        batch_x = batch_x + infos_x
                    
                    outputs, batch_y = self._get_preds(batch_x, batch_y, batch_x_mark, batch_y_mark, episode)
                    
                    # Adapter Y Forward
                    y_state = outputs.reshape(outputs.shape[0],-1)
                    infos_y = post_net_y(y_state) 
                    infos_y = infos_y.view(*outputs.shape)
                    
                    if self.adapter_mode in ['mul']:
                        outputs = outputs * infos_y
                    else: # add, affine
                        outputs = outputs + infos_y
                    
                    loss = criterion(outputs, batch_y)
                    loss_list.append(loss.item())
                    loss.backward()
                    self.post_optim_x.step()
                    self.post_optim_y.step()

                    if (step+1) % 10 == 0:
                        pbar.set_postfix({'step': '%d' % (step+1), 'loss': '%.4f' % np.mean(loss_list)})
                    pbar.update(1)
            
            if not vali_set:
                val_loss, mae_loss = self.vali_post(post_net_x, post_net_y, val_loader, criterion)
                print('val_loss=%.4f' % val_loss, 'mae_loss=%.4f' % mae_loss)
                
                path = os.path.join(self.args.checkpoints, setting) 
                # Save with mode suffix
                suffix = f"_{self.adapter_mode}_norm"
                early_stopping(val_loss, (post_net_x, post_net_y), path, name=f'post_ckpt_{vali_set}{suffix}.pth')
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                    
            test_loss, mae_loss = self.vali_post(post_net_x, post_net_y, test_loader, criterion)
            print('test_loss=%.4f' % test_loss, 'mae_loss=%.4f' % mae_loss)
            
            adjust_learning_rate_v2((self.post_optim_x, self.post_optim_y), episode + 1, self.args)
        
        return test_loss

    def test2(self, setting, continue_train=False, post_process=False, vali_set=False, post_noise = False, online=False, visual_ts=False):
        ''' Test with Post Processing enabled '''
        test_data, test_loader = self._get_data(flag='test')
        
        print('loading checkpoint.pth ...')
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))
        self.model.eval()

        if post_process:
            suffix = f"_{self.adapter_mode}_norm"
            print(f'loading post_ckpt{suffix}.pth ...')
            
            post_net_x = PostProcessingNet(self.args.seq_len * self.args.enc_in, 
                                         self.args.d_model, 
                                         self.args.seq_len * self.args.enc_in,
                                            self.args.delta,
                                            mode=self.adapter_mode
                                         ).to(self.device)
            post_net_x.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, f'0_post_ckpt_{vali_set}{suffix}.pth')))
            
            post_net_y = PostProcessingNet(self.args.seq_len * self.args.enc_in, 
                                         self.args.d_model, 
                                         self.args.pred_len * self.args.enc_in,
                                            self.args.delta,
                                            mode=self.adapter_mode
                                         ).to(self.device)
            post_net_y.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, f'1_post_ckpt_{vali_set}{suffix}.pth')))
        
        criterion = self._select_criterion() 
        test_loss, mae_loss = self.vali_post(post_net_x, post_net_y, test_loader, criterion)
        print(f'[{self.adapter_mode}-Norm] final test_loss={test_loss:.4f}, mae_loss={mae_loss:.4f}') 

        return test_loss, mae_loss
