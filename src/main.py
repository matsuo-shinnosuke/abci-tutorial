from torchvision.models import resnet18
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import copy

from arguments import parse_option
from loader import set_loader
from utils import *

def set_model(args):
    model = resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    return model

def set_optimizer(model, args):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    return optimizer

def set_criterion(args):
    criterion = nn.CrossEntropyLoss()
    return criterion

def main():
    args = parse_option()
    set_reproducibility(seed=args.seed)
    args.device = set_device(args)
    set_loger(path=f'{args.output_dir}/log.txt')
    logging.info(args)   
    
    train_loader, val_loader, test_loader = set_loader(args)
    model = set_model(args).to(args.device)
    criterion = set_criterion(args)
    optimizer = set_optimizer(model, args)
    
    # ---- train ----
    if args.is_train:
        history = {
            'best_epoch': 0, 'best_acc': 0,
            'train_loss': [], 'val_loss': [], 'test_loss': [],
            'train_acc': [], 'val_acc': [], 'test_acc': [], 
            'train_cm': [], 'val_cm': [], 'test_cm': [], 
        }
        
        state = {
            'model_latest': None, 'optimizer_latest': None,
            'model_best': None, 'optimizer_best': None,
            'history': None, 'args': args,
        }

        for epoch in range(args.num_epochs):
            # ----
            model.train()
            loss_meter = AverageMeter()
            gt, pred = [], []
            for batch in tqdm(train_loader, leave=False):
                X, y = batch['X'].to(args.device), batch['y'].to(args.device)

                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss_meter.update(loss.item(), X.size(0))
                gt.extend(y.cpu().detach().numpy())
                pred.extend(logits.argmax(-1).cpu().detach().numpy())

            history['train_loss'].append(loss_meter.avg)
            gt, pred = np.array(gt), np.array(pred)
            history['train_acc'].append((gt==pred).mean()*100)
            history['train_cm'].append(confusion_matrix(y_true=gt, y_pred=pred))

            # ----
            model.eval()
            loss_meter = AverageMeter()
            gt, pred = [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, leave=False):
                    X, y = batch['X'].to(args.device), batch['y'].to(args.device)

                    logits = model(X)
                    loss = criterion(logits, y)

                    loss_meter.update(loss.item(), X.size(0))
                    gt.extend(y.cpu().detach().numpy())
                    pred.extend(logits.argmax(-1).cpu().detach().numpy())

            history['val_loss'].append(loss_meter.avg)
            gt, pred = np.array(gt), np.array(pred)
            history['val_acc'].append((gt==pred).mean()*100)
            history['val_cm'].append(confusion_matrix(y_true=gt, y_pred=pred))


            # ----
            if history['best_acc'] < history['val_acc'][-1]:
                history['best_epoch'] = epoch
                history['best_acc'] = history['val_acc'][-1]

                state['model_best'] = copy.deepcopy(model.state_dict())
                state['optimizer_best'] = copy.deepcopy(optimizer.state_dict())
                plot_confusion_matrix(
                    cm=history['train_cm'][-1],path=f'{args.output_dir}/cm_train.png',
                    title='train confusion matrix: acc=%.2f' % history['train_acc'][-1]
                )
                plot_confusion_matrix(
                    cm=history['val_cm'][-1],path=f'{args.output_dir}/cm_val.png',
                    title='val confusion matrix: acc=%.2f' % history['val_acc'][-1]
                )

            plot_loss_curve(history, args.output_dir)
            plot_acc_curve(history, args.output_dir)

            state['model_latest'] = copy.deepcopy(model.state_dict())
            state['optimizer_latest'] = copy.deepcopy(optimizer.state_dict())
            state['history'] = history
            torch.save(state, f'{args.output_dir}/state.pkl')

            logging.info('[%d/%d]: train loss: %.3f, acc: %.2f, val loss: %.3f, acc: %.2f'
                % (epoch+1, args.num_epochs, 
                    history['train_loss'][-1], history['train_acc'][-1],
                    history['val_loss'][-1], history['val_acc'][-1]))
        
    # ---- test ----
    if args.is_test:
        state = torch.load(f'{args.output_dir}/state.pkl')
        history = state['history']
        model.load_state_dict(state['model_best'])
        feature_extractor = nn.Sequential(*list(model.children())[:-1])

        model.eval()
        loss_meter = AverageMeter()
        gt, pred = [], []
        features = []
        with torch.no_grad():
            for batch in tqdm(test_loader, leave=False):
                X, y = batch['X'].to(args.device), batch['y'].to(args.device)

                f = feature_extractor(X).squeeze()
                logits = model(X)
                loss = criterion(logits, y)

                loss_meter.update(loss.item(), X.size(0))
                features.extend(f.cpu().detach().numpy())
                gt.extend(y.cpu().detach().numpy())
                pred.extend(logits.argmax(-1).cpu().detach().numpy())

        history['test_loss'].append(loss_meter.avg)
        gt, pred = np.array(gt), np.array(pred)
        history['test_acc'].append((gt==pred).mean()*100)
        history['test_cm'].append(confusion_matrix(y_true=gt, y_pred=pred))

        plot_confusion_matrix(
            cm=history['test_cm'][-1],path=f'{args.output_dir}/cm_test.png',
            title='test confusion matrix: acc=%.2f' % history['test_acc'][-1]
        )

        features = np.array(features)
        plot_feature_space(features, gt, args.output_dir)

        logging.info('test loss: %.3f, acc: %.2f'
            % (history['test_loss'][-1], history['test_acc'][-1]))


if __name__ == '__main__':
    main()