import json
import speech
import speech.loader as loader
from model import Model
from tensorboardX import SummaryWriter

def run(EPOCH, BATCH_SIZE, CUDA):

    writer = SummaryWriter()
    start_and_end = True
    train_set = 'data/train_si284.json'
    dev_set = 'data/dev_93.json'
    batch_size = BATCH_SIZE

    preproc = loader.Preprocessor(train_set,
                                  start_and_end=start_and_end)

    train_ldr = loader.make_loader(train_set,
                                   preproc, batch_size)

    dev_ldr = loader.make_loader(dev_set,
                                 preproc, batch_size)


    model = Model(input_dim=161, num_class=preproc.vocab_size, CUDA=CUDA)

    if CUDA:
        model.cuda()

    for epoch in range(EPOCH):
        train_loss = train(train_ldr, model)
        eval_loss, cer = eval(dev_ldr, model, preproc)

        writer.add_scalars('data/loss',{'TRAIN_LOSS': train_loss,
                                        'EVAL_LOSS': eval_loss},
                            epoch)

def train(data_loader, model):
    total_loss = []
    for batch in data_loader:
        loss = model.train_model(batch)
        total_loss.append(loss)
    avg_loss = sum(total_loss) / len(total_loss)
    return avg_loss


def eval(data_loader, model, preproc):
    total_loss = []
    all_preds = []
    all_labels = []

    for batch in data_loader:
        loss = model.eval_model(batch)
        total_loss.append(loss)
        preds = model.infer(batch)
        all_preds.extend(preds)
        all_labels.extend(batch[1])

    avg_loss = sum(total_loss)/len(total_loss)
    results = [(preproc.decode(l), preproc.decode(p))
               for l, p in zip(all_labels, all_preds)]
    cer = speech.compute_cer(results)

    return avg_loss, cer






