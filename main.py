import os
import speech
import speech.loader as loader
from model import Model
import pygsheets
import functions.ctc as ctc
from tensorboardX import SummaryWriter
import time
import argparse
import torch
import torch.optim as optim

def load_data(train_set, dev_set, test_set, batch_size):

    preproc = loader.Preprocessor(train_set)
    train_ldr = loader.make_loader(train_set, preproc, batch_size)
    dev_ldr = loader.make_loader(dev_set, preproc, batch_size)
    test_ldr = loader.make_loader(test_set, preproc, batch_size)

    return train_ldr, dev_ldr, test_ldr, preproc

def load_pretraned_dict_different_name(target_model, pretrained_model_dir, name_in_pretrained, name_in_target):
    checkpoints = torch.load(pretrained_model_dir)
    pretrained_dict = checkpoints['state_dict']
    target_model_dict = target_model.state_dict()
    new_dict = {}

    # Filtered pretrained dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if name_in_pretrained in k}

    # create new dict for name changing purpose
    for key, val in pretrained_dict.items():
        new_key = name_in_target + key[len(name_in_pretrained):]
        new_dict[new_key] = val

    target_model_dict.update(new_dict)
    target_model.load_state_dict(target_model_dict)

    return target_model

def save_checkpoint(state, filename):
    torch.save(state, filename)

def write_google_sheet(CONFIG, test_loss, test_cer):
    # authorization
    gc = pygsheets.authorize(service_file='asr-exp-4dd17862a7f6.json')

    values_list = [CONFIG['id'], test_loss, test_cer]

    for key, val in CONFIG['model'].items():
        values_list.append(str(val))

    for key, val in CONFIG['optimizer'].items():
        values_list.append(str(val))

    for i in ['epoch', 'batch_size', 'cuda','multi_gpu', 'pre_train', 'pretrained_id']:
        values_list.append(str(CONFIG[i]))

    # open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)
    sh = gc.open('WSJ_ASR')

    # select the first sheet
    wks = sh[0]

    # update the first sheet with df, starting at cell B2.
    wks.insert_rows(row=CONFIG['id']+1, number=1, values=values_list)

def run(CONFIG):

    if CONFIG['tensorboard']:
        writer = SummaryWriter(os.path.join('runs', str(CONFIG['id'])))
    save_chkpt = CONFIG['save_path'] + '/chkpt_' + str(CONFIG['id']) + '.pth.tar'
    best_save_chkpt = CONFIG['save_path'] + '/best_chkpt_' + str(CONFIG['id']) + '.pth.tar'

    train_ldr, dev_ldr, test_ldr, preproc = load_data(CONFIG['data']['train_path'],
                                                      CONFIG['data']['dev_path'],
                                                      CONFIG['data']['test_path'],
                                                      CONFIG['batch_size'])

    PARALLEL = True if CONFIG['multi_gpu'] > 0 else False
    model = Model(input_dim=preproc.input_dim, num_class=preproc.vocab_size, CONFIG=CONFIG)

    if CONFIG['pre_train']:
        pretrained_model_dir = CONFIG['pretrained_save_path'] + '/best_chkpt_' + str(CONFIG['pretrained_id']) + '.pth.tar'
        model = load_pretraned_dict_different_name(model, pretrained_model_dir, 'encoder', 'rnn')

    if CONFIG['multi_gpu'] > 0:
        devices = [i for i in range(CONFIG['multi_gpu'])]
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=devices)
    elif CONFIG['cuda']:
        model = model.cuda()

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

    best_cer = float('inf')

    START_EPOCH = 0

    if CONFIG['resume']:
        checkpoint = torch.load(save_chkpt)
        START_EPOCH = checkpoint['epoch']
        best_cer = checkpoint['best_cer']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    try:
        print('Starts training')
        for epoch in range(START_EPOCH, CONFIG['epoch']):
            start_time = time.time()
            train_loss = train(train_ldr, model, optimizer, PARALLEL)
            eval_loss, cer = eval(dev_ldr, model, preproc, PARALLEL)
            # print stats
            echo(epoch, train_loss, eval_loss, cer, start_time)
            # logger
            if CONFIG['tensorboard']:
                writer.add_scalars('data/loss',{'TRAIN_LOSS': train_loss,
                                               'EVAL_LOSS': eval_loss},
                                   epoch)
                writer.add_scalars('data/cer', {'CER': cer},
                               epoch)
            # save best model
            save_checkpoint({'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'best_cer': best_cer,
                             'optimizer': optimizer.state_dict()},
                              save_chkpt)

            # save best model
            if cer < best_cer:
                save_checkpoint({'state_dict': model.state_dict()},
                                best_save_chkpt)
                best_cer = cer
    except KeyboardInterrupt:
        print('=' * 89)
        print('Exiting from training early')

    # Load best model and calculate test CER
    checkpoint = torch.load(best_save_chkpt)
    model_state_dict = checkpoint['state_dict']
    model.load_state_dict(model_state_dict)
    test_loss, test_cer = eval(test_ldr, model, preproc)
    print('=' * 89)
    print('======= Test Model =======')
    print('Test Loss: %s' % test_loss)
    print('CER: %s' % test_cer)
    write_google_sheet(CONFIG, test_loss, test_cer)

def echo(epoch, train_loss, eval_loss, cer, start_time):
    print('=' * 89)
    print('Epoch: %s' % (epoch + 1))
    print('Train Loss: %s' % train_loss)
    print('Eval Loss: %s' % eval_loss)
    print('CER: %s' % cer)
    print('Time: %s' % (time.time() - start_time))
    print('=' * 89)


def train(data_loader, model, optimizer, PARALLEL=False):
    print('=' * 89)
    log_interval = len(data_loader) // 10
    total_loss = []
    start_time = time.time()

    for idx, batch in enumerate(data_loader):
        inputs, labels = batch
        loss = model.module.train_model(inputs, labels, optimizer) if PARALLEL \
            else model.train_model(inputs, labels, optimizer)
        elapsed = time.time() - start_time
        if idx % log_interval == 0 and idx > 0:
            print('| {:5d}/{:5d} batches | avg s/batch {:5.2f} | loss {:5.2f}'.format(
              idx, len(data_loader),
                elapsed/log_interval, loss
            ))
            start_time = time.time()
        total_loss.append(loss)

    avg_loss = sum(total_loss) / len(total_loss)
    return avg_loss

def eval(data_loader, model, preproc, PARALLEL=False):
    total_loss = []
    all_preds = []
    all_labels = []

    for batch in data_loader:
        inputs, labels = batch
        loss, preds = model.module.eval_model(inputs, labels) if PARALLEL \
            else model.eval_model(inputs, labels)
        total_loss.append(loss)
        all_preds.extend(preds)
        all_labels.extend(labels)

    avg_loss = sum(total_loss)/len(total_loss)
    results = [(preproc.decode(l), preproc.decode(p))
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
    # data address
    parser.add_argument('--train_path', type=str, default='data/train_si284.json')
    parser.add_argument('--dev_path', type=str, default='data/dev_93.json')
    parser.add_argument('--test_path', type=str, default='data/eval_92.json')

    # Model settings
    parser.add_argument('--model', type=int, default=0,
                        help='model: GRU=0, LSTM=1')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--bi', type=str2bool, default=False,
                        help='bidirectional')

    # Optimizer
    parser.add_argument('--opt', type=int, default=0,
                        help='optimizer: SGD=0, ADAM=1')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mom', type=float, default=0)
    parser.add_argument('--nes', type=str2bool, default=False)
    parser.add_argument('--beta1', type=float, default=0.99)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--beam', type=int, default=2)

    # training
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--cuda', type=str2bool, default=False)
    parser.add_argument('--save', type=str, default='save/asr')
    parser.add_argument('--id', type=int, default=0,
                        help='Experiment ID')
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--multi_gpu', type=int, default=0)
    parser.add_argument('--tensorboard', type=str2bool, default=False)
    parser.add_argument('--pre_train', type=str2bool, default=False)
    parser.add_argument('--pre_id', type=int, default=0)
    parser.add_argument('--pre_save', type=str, default='save/vae')

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = get_args()

    if not os.path.isdir(args.save):
        os.mkdir(args.save)


    config = {
        'data': {
            'train_path': args.train_path,
            'dev_path': args.dev_path,
            'test_path': args.test_path
        },
        'model': {
            'model_type': 'GRU' if args.model == 0 else 'LSTM',
            'layer': args.layer,
            'dropout': args.dropout,
            'hid_dim': args.hid_dim,
            'bi': args.bi
        },
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
            'beam': args.beam
        },
        'epoch': args.epoch,
        'batch_size': args.batch,
        'cuda': args.cuda,
        'save_path': args.save,
        'id': args.id,
        'resume': args.resume,
        'multi_gpu': args.multi_gpu,
        'tensorboard': args.tensorboard,
        'pre_train': args.pre_train,
        'pretrained_save_path': args.pre_save,
        'pretrained_id': args.pre_id
    }

    run(CONFIG=config)
