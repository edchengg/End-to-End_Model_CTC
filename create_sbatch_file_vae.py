import argparse
import os



def create_sequence_sh(file_name, gpu_partition, gpu_num,
                       series_name, shell_script_name, N_jobs):

    with open(file_name, 'w') as f:
        f.write('#!/bin/bash\n')
        for i in range(N_jobs):
            sh = ['sbatch', '-p', gpu_partition, '-c'+str(gpu_num),
                            '-J', series_name, '-d', 'singleton',
                            shell_script_name + '_' + str(i) + '.sh']
            shell_script = ' '.join(sh)
            f.write(shell_script + '\n')



def create_sbatch_sh(args, N_jobs, shell_script_name, python_file_name):
    for i in range(N_jobs):
        with open(shell_script_name + '_' + str(i) + '.sh', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('source /home-nfs/yangc1/speech/setup.sh\n')
            sh = ['python', python_file_name]
            for key, val in vars(args).items():
                sh.append('--' + key)
                if key == 'resume' and i != 0:
                    sh.append('True')
                elif key == 'resume' and i == 0:
                    sh.append('False')
                else:
                    sh.append(str(val))

            shell_script = ' '.join(sh)
            f.write(shell_script)


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
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--cuda', type=str2bool, default=False)
    parser.add_argument('--save', type=str, default='save/vae')
    parser.add_argument('--id', type=int, default=0,
                        help='Experiment ID')
    parser.add_argument('--resume', type=str, default=False)
    parser.add_argument('--multi_gpu', type=int, default=0)
    parser.add_argument('--tensorboard', type=str2bool, default=False)
    parser.add_argument('--N_jobs', type=int, default=4)
    parser.add_argument('--gpu_num', type=int, default=2)
    parser.add_argument('--series_name', type=str, default='vae_A')

    args = parser.parse_args()

    return args



if __name__ == "__main__":

    args = get_args()

    N_jobs = args.N_jobs
    gpu_partition = 'speech-gpu'
    gpu_num = args.gpu_num

    if args.multi_gpu == 0:
        assert gpu_num == 1
    else:
        assert gpu_num == args.multi_gpu
    assert args.cuda == True

    series_name = args.series_name

    python_file_name = '/home-nfs/yangc1/speech/main_vae.py'
    root = '/home-nfs/yangc1/speech/shell_script/vae/' + str(args.id)
    if not os.path.exists(root):
        os.makedirs(root)
    shell_script_name = root + '/job'
    file_name = root + '/vae_' + str(args.id) + '.sh'

    del args.gpu_num
    del args.N_jobs
    del args.series_name

    create_sbatch_sh(args, N_jobs, shell_script_name, python_file_name)
    create_sequence_sh(file_name, gpu_partition, gpu_num,
                       series_name, shell_script_name, N_jobs)





