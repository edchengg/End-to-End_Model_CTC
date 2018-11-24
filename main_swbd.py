import os
import speech
import dataloader_swbd as loader

import pygsheets
import functions.ctc as ctc
import time
from logger import Logger
import argparse
import torch
import torch.optim as optim
import pickle

def int_to_char(character_set):
    with open(character_set, 'rb') as f:
        return pickle.load(f)

def load_data(batch_size, subset=0, log=True):

    train_ldr = loader.make_loader('swbd/swbd_train_nodup_data.json',batch_size, subset=subset, log=log)
    dev_ldr = loader.make_loader('swbd/swbd_dev_data.json', batch_size, log=log)

    return train_ldr, dev_ldr


def load_pretraned_dict_different_name(target_model, pretrained_model_dir):
    checkpoints = torch.load(pretrained_model_dir)
    pretrained_dict = checkpoints['state_dict']
    target_model_dict = target_model.state_dict()

    new_dict = {}

    # Filtered pretrained dict
    pretrained_dict_encoder1 = {k: v for k, v in pretrained_dict.items()
                                if 'encoder1' in k}
    pretrained_dict_encoder2 = {k: v for k, v in pretrained_dict.items()
                                if 'encoder2' in k}
    # create new dict for name changing purpose
    for key, val in pretrained_dict_encoder1.items():
        new_key = 'rnn1' + key[len('encoder1'):]
        new_dict[new_key] = val

    for key, val in pretrained_dict_encoder2.items():
        new_key = 'rnn2' + key[len('encoder2'):]
        new_dict[new_key] = val

    target_model_dict.update(new_dict)
    target_model.load_state_dict(target_model_dict)

    return target_model

def save_checkpoint(state, filename):
    torch.save(state, filename)

def write_google_sheet_loss_epoch(CONFIG, epoch, cer, loss):
    # authorization
    gc = pygsheets.authorize(service_file='asr-exp-4dd17862a7f6.json')
    # open the google spreadsheet
    sh = gc.open('WSJ_ASR')
    # select the sheet 3 save CER
    wks1 = sh[2]
    wks1.update_cell((CONFIG['id'] + 1, epoch + 2), cer * 100, parse=None)
    # select the sheet 4 save loss
    wks2 = sh[3]
    wks2.update_cell((CONFIG['id'] + 1, epoch + 2), loss, parse=None)

def write_google_sheet(CONFIG, eval_loss, eval_cer, test_loss, test_cer, stop_epoch):
    # authorization
    gc = pygsheets.authorize(service_file='asr-exp-4dd17862a7f6.json')

    values_list = [CONFIG['id'], eval_loss, eval_cer, stop_epoch, test_loss, test_cer]


    for key, val in CONFIG['optimizer'].items():
        values_list.append(str(val))

    for i in ['epoch', 'batch_size', 'cuda', 'pre_train', 'pretrained_id']:
        values_list.append(str(CONFIG[i]))

    # open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)
    sh = gc.open('WSJ_ASR')

    # select the sheet 5 for asr loss record
    wks = sh[4]

    # update the first sheet with df, starting at cell B2.
    wks.update_row(index=CONFIG['id']+1, values=values_list)

def run(CONFIG):

    torch.manual_seed(9669)

    # Initialize checkpoints dir
    save_chkpt = CONFIG['save_path'] + '/chkpt_' + str(CONFIG['id']) + '.pth.tar'
    best_save_chkpt = CONFIG['save_path'] + '/best_chkpt_' + str(CONFIG['id']) + '.pth.tar'

    # Initialize logger dir
    if CONFIG['end_log']:
        if not os.path.exists(CONFIG['save_path'] + '/log'):
            os.mkdir(CONFIG['save_path'] + '/log')
        log_dir = CONFIG['save_path'] + '/log/' + 'logger_' + str(CONFIG['id']) + '.pkl'

        logger = Logger(log_dir)

    # Prepare dataloader and preprocessor
    train_ldr, dev_ldr = load_data(CONFIG['batch_size'], CONFIG['subset'])
    int2char = int_to_char('swbd/int2char.swbd.pkl')
    print(int2char)
    print(len(int2char))

    model = Model(input_dim=123, num_class=len(int2char), CONFIG=CONFIG)

    # Load pretrain model
    if CONFIG['pre_train']:
        pretrained_model_dir = CONFIG['pretrained_save_path'] + '/asr_best_chkpt_' + str(CONFIG['pretrained_id']) + '.pth.tar'
        checkpoints = torch.load(pretrained_model_dir)
        pretrained_dict = checkpoints['state_dict']
        model.load_state_dict(pretrained_dict)

    # Turn on CUDA
    if CONFIG['cuda']:
        model = model.cuda()

    # Initialize optimizer SGD or ADAM
    optimizer = None
    if CONFIG['optimizer']['opt'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=CONFIG['optimizer']['lr'],
                                   momentum=CONFIG['optimizer']['mom'],
                                   weight_decay=CONFIG['optimizer']['l2'],
                                   nesterov=CONFIG['optimizer']['nes'])
    elif CONFIG['optimizer']['opt'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['optimizer']['lr'],
                                    betas=(CONFIG['optimizer']['beta1'], CONFIG['optimizer']['beta2']),
                                    eps=CONFIG['optimizer']['eps'],
                                    weight_decay=CONFIG['optimizer']['l2'])

    # Set best_cer to infinite
    best_cer = float('inf')

    # Set starting epochs for resume setting
    START_EPOCH = 0

    # Resume training from a previous checkpoints
    if CONFIG['resume']:
        checkpoint = torch.load(save_chkpt)
        START_EPOCH = checkpoint['epoch']
        best_cer = checkpoint['best_cer']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Initialize a scheduler for learning rate decay
    if CONFIG['optimizer']['lr_decay']:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG['optimizer']['step'],
                                                    gamma=CONFIG['optimizer']['factor'])
    try:
        print('Starts training')
        for epoch in range(START_EPOCH, CONFIG['epoch']):
            start_time = time.time()
            # Learning rate decay
            if CONFIG['optimizer']['lr_decay']:
                if epoch >= 10:
                    scheduler.step()
            # Training
            train_loss, eval_loss, eval_cer = train(train_ldr, dev_ldr, model, optimizer,  int2char)
            # Evaluating
            end_epoch_eval_loss, end_epoch_eval_cer = eval(dev_ldr, model, int2char)
            # Print stats
            echo(epoch, train_loss, end_epoch_eval_loss, end_epoch_eval_cer, start_time)

            # Save model in each epoch
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'best_cer': best_cer,
                             'optimizer': optimizer.state_dict()},
                              save_chkpt)

            # Save model in each of 10 epoch
            count_epoch = epoch+1
            if count_epoch <= 20:
                if count_epoch % 2 == 0:
                    save_checkpoint({'epoch': epoch + 1,
                                 'state_dict': model.state_dict(),
                                 'best_cer': best_cer,
                                 'optimizer': optimizer.state_dict()},
                                CONFIG['save_path'] + '/chkpt_' + str(CONFIG['id']) + '_' +
                                str(count_epoch)+'.pth.tar')
            else:
                if count_epoch % 10 == 0:
                    save_checkpoint({'epoch': epoch + 1,
                                     'state_dict': model.state_dict(),
                                     'best_cer': best_cer,
                                     'optimizer': optimizer.state_dict()},
                                    CONFIG['save_path'] + '/chkpt_' + str(CONFIG['id']) + '_' +
                                    str(count_epoch) + '.pth.tar')


            # Save best model
            if end_epoch_eval_cer < best_cer:
                save_checkpoint({'state_dict': model.state_dict(),
                                 'epoch': epoch + 1,
                                 'eval_cer': end_epoch_eval_cer,
                                 'eval_loss': end_epoch_eval_loss},
                                best_save_chkpt)
                best_cer = end_epoch_eval_cer
            # Save loss into logger
            if CONFIG['end_log']:
                eval_loss.append(end_epoch_eval_loss)
                eval_cer.append(end_epoch_eval_cer)
                logger.add_loss(epoch=epoch, loss_list=eval_loss)
                logger.add_cer(epoch=epoch, cer_list=eval_cer)
                logger.save()
    except KeyboardInterrupt:
        print('=' * 89)
        print('Exiting from training early')

    # Load best model and calculate test CER
    checkpoint = torch.load(best_save_chkpt)
    model_state_dict = checkpoint['state_dict']
    model.load_state_dict(model_state_dict)
    eval_loss, eval_cer = eval(dev_ldr, model, int2char, beam_size=10)
    stop_epoch = checkpoint['epoch']
    # Evaluate on test set
    #test_loss, test_cer = eval(test_ldr, model, int2char, beam_size=10)
    test_loss, test_cer = 0, 0
    print('=' * 89)
    print('======= Test Model =======')
    print('Early stopping epoch: %s' % stop_epoch)
    print('Eval Loss: %s' % eval_loss)
    print('CER: %s' % eval_cer)
    print('Test Loss: %s' % test_loss)
    print('CER: %s' % test_cer)
    # Write loss to the google sheet
    write_google_sheet(CONFIG, eval_loss, eval_cer, test_loss, test_cer, stop_epoch)

def echo(epoch, train_loss, eval_loss, cer, start_time):
    print('=' * 89)
    print('Epoch: %s' % (epoch + 1))
    print('Train Loss: %s' % train_loss)
    print('Eval Loss: %s' % eval_loss)
    print('CER: %s' % cer)
    print('Time: %s' % (time.time() - start_time))
    print('=' * 89)


def train(train_data_loader, eval_data_loader, model, optimizer, int_to_char, EVAL=False):
    print('=' * 89)
    log_interval = len(train_data_loader) // 10
    eval_interval = len(train_data_loader) // 4
    total_loss = []
    start_time = time.time()
    total_eval_loss = []
    total_eval_cer = []

    for idx, batch in enumerate(train_data_loader):
        inputs, labels = batch
        loss = model.train_model(inputs, labels, optimizer)
        elapsed = time.time() - start_time
        if idx % log_interval == 0 and idx > 0:
            print('| {:5d}/{:5d} batches | avg s/batch {:5.2f} | loss {:5.2f}'.format(
              idx, len(train_data_loader),
                elapsed/log_interval, loss
            ))
            start_time = time.time()
        total_loss.append(loss)
        if idx % eval_interval == 0 and idx > 0:
            if EVAL:
                eval_loss, eval_cer = eval(eval_data_loader, model,  int_to_char)
                total_eval_loss.append(eval_loss)
                total_eval_cer.append(eval_cer)

    avg_loss = sum(total_loss) / len(total_loss)
    return avg_loss, total_eval_loss, total_eval_cer


def decode(seq, int_to_char):
    text = [int_to_char[s] for s in seq]
    return text

def eval(data_loader, model, int_to_char, beam_size=1):
    total_loss = []
    all_preds = []
    all_labels = []

    for batch in data_loader:
        inputs, labels = batch
        loss, preds = model.eval_model(inputs, labels, beam_size)
        total_loss.append(loss)
        all_preds.extend(preds)
        all_labels.extend(labels)

    avg_loss = sum(total_loss)/len(total_loss)

    results = [(decode(l, int_to_char), decode(p, int_to_char))
               for l, p in zip(all_labels, all_preds)]
    cer = speech.compute_cer(results)

    return avg_loss, cer

def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser(description='End-to-End ASR CTC')

    # Model settings
    parser.add_argument('--layer', type=int, default=1)


    # Optimizer
    parser.add_argument('--opt', type=int, default=0,
                        help='optimizer: SGD=0, ADAM=1')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--mom', type=float, default=0)
    parser.add_argument('--nes', type=str2bool, default=False)
    parser.add_argument('--beta1', type=float, default=0.99)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--l2', type=float, default=0.00001)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=10)
    parser.add_argument('--beam', type=int, default=10)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--factor', type=float, default=0.95)
    parser.add_argument('--lr_decay', type=str2bool, default=True)

    # training
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--save', type=str, default='save/asr')
    parser.add_argument('--id', type=int, default=0,
                        help='Experiment ID')
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--pre_train', type=str2bool, default=False)
    parser.add_argument('--pre_id', type=int, default=0)
    parser.add_argument('--pre_save', type=str, default='save/asr')
    parser.add_argument('--end_logger', type=str2bool, default=True)
    parser.add_argument('--tensorboard', type=str2bool, default=False)
    parser.add_argument('--subset', type=int, default=0)
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = get_args()

    if not os.path.isdir(args.save):
        os.mkdir(args.save)


    config = {
        'optimizer': {
            'opt': 'SGD' if args.opt == 0 else 'Adam',
            'lr': args.lr,
            'mom': args.mom,
            'nes': args.nes,
            'beta1': args.beta1,
            'beta2': args.beta2,
            'l2': args.l2,
            'eps': args.eps,
            'clip': args.clip,
            'beam': args.beam,
            'step': args.step,
            'factor': args.factor,
            'lr_decay': args.lr_decay
        },
        'epoch': args.epoch,
        'batch_size': args.batch,
        'cuda': args.cuda,
        'save_path': args.save,
        'id': args.id,
        'resume': args.resume,
        'pre_train': args.pre_train,
        'pretrained_save_path': args.pre_save,
        'pretrained_id': args.pre_id,
        'subset': args.subset,
        'end_log': args.end_logger,
    }

    if args.layer == 3:
        from model_swbd_3layers import Model
    elif args.layer == 5:
        from model_swbd_5layers import Model

    run(CONFIG=config)
