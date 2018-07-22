import os
import speech
import speech.loader as loader
from model_vae import VAE
from tensorboardX import SummaryWriter
import time
import argparse
import torch
import torch.optim as optim
import pygsheets

def load_data(train_set, dev_set, test_set, batch_size):

    preproc = loader.Preprocessor(train_set)
    train_ldr = loader.make_loader(train_set, preproc, batch_size)
    dev_ldr = loader.make_loader(dev_set, preproc, batch_size)
    test_ldr = loader.make_loader(test_set, preproc, batch_size)

    return train_ldr, dev_ldr, test_ldr, preproc

def save_checkpoint(state, filename):
    torch.save(state, filename)

def write_google_sheet(CONFIG, test_loss, test_recon_loss, test_kl_divergence):
    # authorization
    gc = pygsheets.authorize(service_file='asr-exp-4dd17862a7f6.json')

    values_list = [CONFIG['id'], test_loss, test_recon_loss, test_kl_divergence]

    list_name = ['encoder', 'decoder', 'model', 'optimizer']
    for name in list_name:
        for key, val in CONFIG[name].items():
            values_list.append(str(val))

    for i in ['epoch', 'batch_size', 'cuda','multi_gpu']:
        values_list.append(str(CONFIG[i]))

    # open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)
    sh = gc.open('WSJ_ASR')

    # select the first sheet
    wks = sh[1]

    # update the first sheet with df, starting at cell B2.
    wks.insert_rows(row=CONFIG['id']+1, number=1, values=values_list)

def run(CONFIG):

    if CONFIG['tensorboard']:
        writer = SummaryWriter(os.path.join('runs', str(CONFIG['id'])))
    save_chkpt = CONFIG['save_path'] + '/vae_chkpt_' + str(CONFIG['id']) + '.pth.tar'
    best_save_chkpt = CONFIG['save_path'] + '/vae_best_chkpt_' + str(CONFIG['id']) + '.pth.tar'

    train_ldr, dev_ldr, test_ldr, preproc = load_data(CONFIG['data']['train_path'],
                                                      CONFIG['data']['dev_path'],
                                                      CONFIG['data']['test_path'],
                                                      CONFIG['batch_size'])

    
    model = VAE(input_dim=preproc.input_dim, CONFIG=CONFIG)
    
    PARALLEL = True if CONFIG['multi_gpu'] > 0 else False
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

    lowest_loss = float('inf')

    START_EPOCH = 0

    if CONFIG['resume']:
        checkpoint = torch.load(save_chkpt)
        START_EPOCH = checkpoint['epoch']
        lowest_loss = checkpoint['lowest_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    try:
        print('Starts training')
        for epoch in range(START_EPOCH, CONFIG['epoch']):
            start_time = time.time()
            train_loss, train_recon_loss, train_kl_divergence = train(train_ldr, model, optimizer, LAMBDA=CONFIG['optimizer']['lambda'], PARALLEL=PARALLEL)
            eval_loss, eval_recon_loss, eval_kl_divergence = eval(dev_ldr, model, LAMBDA=CONFIG['optimizer']['lambda'], PARALLEL=PARALLEL)
            # print stats
            echo(epoch, train_loss, train_recon_loss, train_kl_divergence,
                    eval_loss, eval_recon_loss, eval_kl_divergence, start_time)
            # logger
            if CONFIG['tensorboard']:
                writer.add_scalars('data/loss',{'TRAIN_LOSS': train_loss,
                                               'EVAL_LOSS': eval_loss},
                                   epoch)
                writer.add_scalars('data/recon_loss', {'TRAIN_RECON_LOSS': train_recon_loss,
                                                       'EVAL_RECON_LOSS': eval_recon_loss},
                                   epoch)
                writer.add_scalars('data/kl_divergence', {'TRAIN_KL_DIVERGENCE': train_kl_divergence,
                                                          'EVAL_KL_DIVERGENCE': eval_kl_divergence},
                                   epoch)
            # save best model
            save_checkpoint({'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'lowest_loss': lowest_loss,
                             'optimizer': optimizer.state_dict()},
                              save_chkpt)

            # save best model
            if eval_loss < lowest_loss:
                save_checkpoint({'state_dict': model.state_dict()},
                                best_save_chkpt)
                lowest_loss = eval_loss
    except KeyboardInterrupt:
        print('=' * 89)
        print('Exiting from training early')

    # Load best model and calculate test CER
    checkpoint = torch.load(best_save_chkpt)
    model_state_dict = checkpoint['state_dict']
    model.load_state_dict(model_state_dict)
    test_loss, test_recon_loss, test_kl_divergence = eval(test_ldr, model, LAMBDA=CONFIG['optimizer']['lambda'], PARALLEL=PARALLEL)
    print('=' * 89)
    print('======= Test Model =======')
    print('Test Loss: %s' % test_loss)
    print('Test Recon Loss: %s' % test_recon_loss)
    print('Test KL Divergence: %s' % test_kl_divergence)

    write_google_sheet(CONFIG, test_loss, test_recon_loss, test_kl_divergence)
    # Plot spectrogram
    audio_file = '/share/data/speech/yangc1/wsj/40po031e.wav'
    loader.plot_ori_recon_specgram(audio_file, model, os.path.join(CONFIG['save_path'], 'spectrogram_'+str(CONFIG['id'])))
def echo(epoch, train_loss, train_recon_loss, train_kl_divergence,
                    eval_loss, eval_recon_loss, eval_kl_divergence, start_time):
    print('=' * 89)
    print('Epoch: %s' % (epoch + 1))
    print('| Train | loss {:5.2f} | recon {:5.2f} | kl {:5.2f}|'.format(
        train_loss, train_recon_loss, train_kl_divergence
    ))
    print('| Eval | loss {:5.2f} | recon {:5.2f} | kl {:5.2f}|'.format(
        eval_loss, eval_recon_loss, eval_kl_divergence
    ))
    print('Time: %s' % (time.time() - start_time))
    print('=' * 89)


def train(data_loader, model, optimizer, LAMBDA=1, PARALLEL=False):
    print('=' * 89)
    log_interval = len(data_loader) // 10
    total_loss = []
    total_kl_divergence = []
    total_recon_loss = []
    start_time = time.time()

    for idx, batch in enumerate(data_loader):
        inputs, labels = batch
        loss, recon_loss, kl_divergence = model.module.train_model(inputs, optimizer, LAMBDA) if PARALLEL \
            else model.train_model(inputs, optimizer, LAMBDA)
        elapsed = time.time() - start_time
        if idx % log_interval == 0 and idx > 0:
            print('| {:5d}/{:5d} batches | avg s/batch {:5.2f} | loss {:5.2f} | recon {:5.2f} | kl {:5.2f}|'.format(
              idx, len(data_loader),
                elapsed/log_interval, loss, recon_loss, kl_divergence
            ))
            start_time = time.time()
        total_loss.append(loss)
        total_recon_loss.append(recon_loss)
        total_kl_divergence.append(kl_divergence)

    avg_loss = sum(total_loss) / len(total_loss)
    avg_recon_loss = sum(total_recon_loss) / len(total_recon_loss)
    avg_kl_divergence = sum(total_kl_divergence) / len(total_kl_divergence)
    return avg_loss, avg_recon_loss, avg_kl_divergence

def eval(data_loader, model, LAMBDA=1, PARALLEL=False):
    total_loss = []
    total_kl_divergence = []
    total_recon_loss = []

    for batch in data_loader:
        inputs, _ = batch
        loss, recon_loss, kl_divergence = model.module.eval_model(inputs, LAMBDA) if PARALLEL \
            else model.eval_model(inputs, LAMBDA)
        total_loss.append(loss)
        total_recon_loss.append(recon_loss)
        total_kl_divergence.append(kl_divergence)

    avg_loss = sum(total_loss) / len(total_loss)
    avg_recon_loss = sum(total_recon_loss) / len(total_recon_loss)
    avg_kl_divergence = sum(total_kl_divergence) / len(total_kl_divergence)
    return avg_loss, avg_recon_loss, avg_kl_divergence


def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser(description='VAE Seq2Seq Speech')
    # data address
    parser.add_argument('--train_path', type=str, default='data/train_si284.json')
    parser.add_argument('--dev_path', type=str, default='data/dev_93.json')
    parser.add_argument('--test_path', type=str, default='data/eval_92.json')

    # Model settings
    parser.add_argument('--lat', type=int, default=256)
    # Encoder
    parser.add_argument('--en_model', type=int, default=0,
                        help='model: GRU=0, LSTM=1')
    parser.add_argument('--en_layer', type=int, default=2)
    parser.add_argument('--en_dropout', type=float, default=0.2)
    parser.add_argument('--en_hid_dim', type=int, default=256)
    parser.add_argument('--en_bi', type=str2bool, default=False,
                        help='bidirectional')
    # Decoder
    parser.add_argument('--de_model', type=int, default=0,
                        help='model: GRU=0, LSTM=1')
    parser.add_argument('--de_layer', type=int, default=2)
    parser.add_argument('--de_dropout', type=float, default=0.2)
    parser.add_argument('--de_hid_dim', type=int, default=256)
    parser.add_argument('--de_bi', type=str2bool, default=False,
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
    parser.add_argument('--lam', type=float, default=1)

    # training
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--cuda', type=str2bool, default=False)
    parser.add_argument('--save', type=str, default='save/vae')
    parser.add_argument('--id', type=int, default=0,
                        help='Experiment ID')
    parser.add_argument('--resume', type=str, default=False)
    parser.add_argument('--multi_gpu', type=int, default=0)
    parser.add_argument('--tensorboard', type=str2bool, default=False)

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
        'encoder': {
            'model_type': 'GRU' if args.en_model == 0 else 'LSTM',
            'layer': args.en_layer,
            'dropout': args.en_dropout,
            'hid_dim': args.en_hid_dim,
            'bi': args.en_bi
        },
        'decoder': {
            'model_type': 'GRU' if args.de_model == 0 else 'LSTM',
            'layer': args.de_layer,
            'dropout': args.de_dropout,
            'hid_dim': args.de_hid_dim,
            'bi': args.de_bi
        },
        'model': {
            'latent_dim': args.lat
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
            'beam': args.beam,
            'lambda': args.lam
        },
        'epoch': args.epoch,
        'batch_size': args.batch,
        'cuda': args.cuda,
        'save_path': args.save,
        'id': args.id,
        'resume': args.resume,
        'multi_gpu': args.multi_gpu,
        'tensorboard':args.tensorboard
    }

    run(CONFIG=config)
