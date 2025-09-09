import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from scipy import sparse as sp
from thop import profile
import wandb
import os
import random
def set_env(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# Graph WaveNet은 lr, batch_size 제외하고 튜닝없이 그대로 쓰는 걸 추천, 
parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda',help='')
parser.add_argument('--data',type=str,default='EMPACTLABS/data/ours',help='data path')
parser.add_argument('--adjdata',type=str,default='data/ours/adj.npz',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=48,help='')
parser.add_argument('--nhid',type=int,default=16,help='')
parser.add_argument('--in_dim',type=int,default=7,help='inputs dimension , 피쳐 개수로 설정')
parser.add_argument('--num_nodes',type=int,default=396,help='number of nscodes')
parser.add_argument('--batch_size',type=int,default=16,help='batch size')
parser.add_argument('--learning_rate',type=float,default=1e-3,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./checkpoint',help='save path')
parser.add_argument('--paitence', type=int, default=10, help='early stopping patience')
parser.add_argument('--gat', action='store_true', help='use GAT')
parser.add_argument('--num_heads', type=int, default=1, help='number of heads') # [2,4]로 튜닝 추천
parser.add_argument('--gat_dropout', type=int, default=0.5, help='number of heads') # [0.1, 0.3, 0.5]로 튜닝 추천
parser.add_argument('--target_feature_index', type=int, default=0, help='target feature index') # 0: occupancy, 1:demand, 2:has_fast_charger, 3:has_slow_charger
args = parser.parse_args() 
def main():
    #set seed
    set_env(7)
    wandb.init(
        project="Graph-WaveNet",  # Change to your project name
        name="exp_ours_base", # Change to your experiment name
        config=vars(args)
    )
    # garage 폴더 없으면 생성
    os.makedirs(args.save, exist_ok=True)
    #load data 
    device = torch.device(args.device)
    adj_mx = sp.load_npz(args.adjdata)
    print("adj_mx shape: ", adj_mx.shape)
    dataloader = util.load_and_preprocess_dataset(args.data, args.batch_size, args.target_feature_index)
    a, b, adj_mx = util.load_adj(args.adjdata, args.adjtype, adj_mx)      
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    if args.gat:
        from engine1 import trainer
        engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit, num_heads=args.num_heads, gat_dropout=args.gat_dropout, layernorm=True)
    else : 
        from engine import trainer
        engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit)

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    best_valid_loss = float('inf')  # 최고 성능 추적
    
    for i in range(1,args.epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,args.target_feature_index,:,:]) 
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []


        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,args.target_feature_index,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)
        # Log to wandb
        wandb.log({
            "epoch": i,
            "train/loss": mtrain_loss,
            "train/mape": mtrain_mape,
            "train/rmse": mtrain_rmse,
            "val/loss": mvalid_loss,
            "val/mape": mvalid_mape,
            "val/rmse": mvalid_rmse,
            "train/time": t2-t1,
            "val/time": s2-s1
        })
        # Best model 저장 (valid loss가 개선되었을 때만)
        if mvalid_loss < best_valid_loss:
            best_valid_loss = mvalid_loss
            best_model_path = os.path.join(args.save, "GWN_best.pt")
            torch.save(engine.model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {i} with valid loss: {mvalid_loss:.4f}")
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # testing
    bestid = np.argmin(his_loss)
    best_model_path = os.path.join(args.save, "GWN_best.pt")
    engine.model.load_state_dict(torch.load(best_model_path), strict=False)


    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,args.target_feature_index,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        
        with torch.no_grad():
            preds = engine.model(testx).transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    print("Training finished")
    # print("The valid loss on best model is", str(round(his_loss[bestid],4)))


    amae = []
    amape = []
    armse = []
    import pickle
    pred_full = scaler.inverse_transform(yhat[:,args.target_feature_index,:,:]) 
    print(f"pred_full shape: {pred_full.shape}")
    # pickle.dump(pred_full.cpu().numpy(), open('./checkpoint/pred_gwn.pkl', 'wb'))
    tmp = scaler.inverse_transform(yhat[:,args.target_feature_index,:,:])
    print(f"tmp shape: {tmp.shape}")
    with open(f'.checkpoint/pred.pkl', 'wb') as f:
        pickle.dump(tmp.cpu().numpy(), f) 
    for i in range(args.seq_length): # [0, 47]
        pred = scaler.inverse_transform(yhat[:,args.target_feature_index,:,i])
        real = realy[:,:,i]
        metrics = util.metric(pred,real)


        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    wandb.log({
        "test/mae": np.mean(amae),
        "test/mape": np.mean(amape),
        "test/rmse": np.mean(armse)
    })
    wandb.finish()


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
