import time
import os
import torch
import argparse
import json
import random
import torch.optim
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data

from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset
from data.Dubai_CC.DubaiCC import DubaiCCDataset
from data.WHU_CDC.WHUCDC import WHUCDCDataset
from model.model_encoder import Encoder, MiddleLayer
from model.model_decoder import DecoderTransformer
from utils import *


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(data_name, args):

    earlyStop = 0
    
    if os.path.exists(args.savepath)==False:
        os.makedirs(args.savepath)

    args.data_folder = './data/' + data_name + '/images/'
    args.token_folder = './data/' + data_name + '/tokens/'
    args.list_path = './data/' + data_name + '/'

    best_epoch  = 0
    best_bleu4  = 0 
    best_rouge  = 0
    best_cider  = 0
    best_meteor = 0
    best_avg    = 0

    start_epoch = 0
    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)

    if args.checkpoint is None:      
        encoder = Encoder(args.network, args)   
        encoder.fine_tune(args.fine_tune_encoder)     
        trainable_params = filter(lambda p: p.requires_grad, encoder.parameters())

        if(args.featureCNN == "True"):
            encoder_optimizer = torch.optim.Adam(params=trainable_params, lr=args.encoder_lr)
        
        # that variable name stays as a legacy
        encoder_trans = MiddleLayer(feature_size=[args.feat_size, args.feat_size, args.encoder_dim])
        encoder_trans_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, encoder_trans.parameters()),
                                            lr=args.encoder_lr, weight_decay=1e-4)
        decoder = DecoderTransformer(args=args, encoder_dim=args.encoder_dim, feature_dim=args.feature_dim, vocab_size=len(word_vocab), max_lengths=args.max_length, word_vocab=word_vocab, n_head=args.n_heads,
                                    n_layers= args.decoder_n_layers, dropout=args.dropout)
        decoder_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, decoder.parameters()), 
                                              lr=args.decoder_lr, weight_decay=1e-4)
    else:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder_trans = checkpoint['encoder_trans']
        encoder_trans_optimizer = checkpoint['encoder_trans_optimizer']
        encoder = checkpoint['encoder']
        if(args.featureCNN == "True"):
            encoder_optimizer = checkpoint['encoder_optimizer']
        if args.fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(args.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                            lr=args.encoder_lr)

    encoder = encoder.cuda()
    encoder_trans = encoder_trans.cuda()
    decoder = decoder.cuda()

    if (args.rankingLossCo != 0):
        ranking_loss_fn = DynamicTripletLoss(margin=0.2).cuda()

    lambda_r = args.rankingLossCo
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).cuda()

    g = torch.Generator()
    g.manual_seed(42)

    if data_name == 'LEVIR_CC':
        train_loader = data.DataLoader(
            LEVIRCCDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True,
            worker_init_fn=seed_worker, generator=g)
        val_loader = data.DataLoader(
            LEVIRCCDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif data_name == 'Dubai_CC':
        train_loader = data.DataLoader(
            DubaiCCDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True,
            worker_init_fn=seed_worker, generator=g)
        val_loader = data.DataLoader(
            DubaiCCDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif data_name == 'WHU_CDC':
        train_loader = data.DataLoader(
            WHUCDCDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True,
            worker_init_fn=seed_worker, generator=g)
        val_loader = data.DataLoader(
            WHUCDCDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

    
    if(args.featureCNN == "True"):
        encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=8, gamma=0.5) if args.fine_tune_encoder else None
    
    encoder_trans_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(encoder_trans_optimizer, T_max=args.num_epochs, eta_min=1e-6)
    decoder_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, T_max=args.num_epochs, eta_min=1e-6)

    index_i = 0
    hist = np.zeros((args.num_epochs * len(train_loader), 3))

    for epoch in range(start_epoch, args.num_epochs):        

        for _, (imgA, imgB, _, _, token, token_len, _) in enumerate(train_loader):
            start_time = time.time()
            decoder.train()
            encoder.train()
            encoder_trans.train()
            decoder_optimizer.zero_grad()
            encoder_trans_optimizer.zero_grad()

            if(args.featureCNN == "True"):
                if encoder_optimizer is not None:
                    encoder_optimizer.zero_grad()

            imgA = imgA.cuda()
            imgB = imgB.cuda()
            
            token = token.squeeze(1).cuda()
            token_len = token_len.cuda()

            feat1, feat2, mask = encoder(imgA, imgB)

            feat1, feat2 = encoder_trans(feat1, feat2, mask)
            scores, caps_sorted, decode_lengths, _, d12_vec, d21_vec = decoder(feat1, feat2, token, token_len)

            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss_ce = criterion(scores, targets.long())

            if (args.rankingLossCo != 0):
                loss_rank = ranking_loss_fn(d12_vec, d21_vec)
            else:
                loss_rank = 0

            loss = loss_ce + lambda_r * loss_rank
            loss.backward()
            
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(decoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_value_(encoder_trans.parameters(), args.grad_clip)
                encoder_grads = [p for p in encoder.parameters() if p.grad is not None]
                if encoder_grads:
                    torch.nn.utils.clip_grad_value_(encoder_grads, args.grad_clip)

            decoder_optimizer.step()
            encoder_trans_optimizer.step()
            if(args.featureCNN == "True"):
                if encoder_optimizer is not None:
                    encoder_optimizer.step()

            hist[index_i,0] = time.time() - start_time   
            hist[index_i,1] = loss.item()
            hist[index_i,2] = accuracy(scores, targets, 5)
            index_i += 1   

            if index_i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time: {3:.3f}\t'
                    'Loss: {4:.4f}\t'
                    'Top-5 Accuracy: {5:.3f}'.format(epoch, index_i %len(train_loader), len(train_loader),
                                            np.mean(hist[index_i-args.print_freq:index_i-1,0])*args.print_freq,
                                            np.mean(hist[index_i-args.print_freq:index_i-1,1]),
                                            np.mean(hist[index_i-args.print_freq:index_i-1,2])))

        decoder.eval()
        encoder_trans.eval()
        if encoder is not None:
            encoder.eval()

        nochange_list = ["the scene is the same as before ", "there is no difference ",
                         "the two scenes seem identical ", "no change has occurred ",
                         "almost nothing has changed "]

        val_start_time = time.time()
        references = list()
        hypotheses = list()
        
        with torch.no_grad():

            for ind, (imgA, imgB, token_all, _, _, _, _) in enumerate(val_loader):
                imgA = imgA.cuda()
                imgB = imgB.cuda()
                
                token_all = token_all.squeeze(0).cuda()

                feat1, feat2, mask = encoder(imgA, imgB)
                feat1, feat2 = encoder_trans(feat1, feat2, mask)
                seq = decoder.sample_beam(feat1, feat2, args.beam_size)

                img_token = token_all.tolist()
                img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}],
                        img_token))
                references.append(img_tokens)

                pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
                hypotheses.append(pred_seq)
                assert len(references) == len(hypotheses)

                pred_caption = ""
                ref_caption = ""
                for i in pred_seq:
                    pred_caption += (list(word_vocab.keys())[i]) + " "
                for j in img_tokens[0]:
                    ref_caption += (list(word_vocab.keys())[j]) + " "
                
                if ind == 0 or ind == 120:
                    print("Prediction:")
                    print(pred_caption)
                    print("Referance:")
                    print(ref_caption)

            val_time = time.time() - val_start_time
            score_dict = get_eval_score(references, hypotheses)
            Bleu_1 = score_dict['Bleu_1']
            Bleu_2 = score_dict['Bleu_2']
            Bleu_3 = score_dict['Bleu_3']
            Bleu_4 = score_dict['Bleu_4']
            Meteor = score_dict['METEOR']
            Rouge = score_dict['ROUGE_L']
            Cider = score_dict['CIDEr']
            Avg = (Bleu_4 + Meteor + Rouge + Cider) / 4
            print('Validation:\n' 'Time: {0:.3f}\t' 'AVG: {1:.4f}\t' 'BLEU-4: {2:.4f}\t' 'Cider: {3:.4f}\t' 'BLEU-1: {4:.4f}\t' 
                'BLEU-2: {5:.4f}\t' 'BLEU-3: {6:.4f}\t' 'Meteor: {7:.4f}\t' 'Rouge: {8:.4f}\t'
                .format(val_time, Avg, Bleu_4, Cider, Bleu_1, Bleu_2, Bleu_3, Meteor, Rouge))
        
        decoder_lr_scheduler.step()
        
        encoder_trans_lr_scheduler.step()
        if(args.featureCNN == "True"):
            if encoder_lr_scheduler is not None:
                encoder_lr_scheduler.step()

        if  Avg > best_avg:
            best_avg = Avg
            best_bleu4 = Bleu_4
            best_cider = Cider
            best_rouge = Rouge
            best_meteor = Meteor
            best_epoch = epoch
            print('New Best:\n' 'AVG: {0:.4f}\t' 'BLEU-4: {1:.4f}\t' 'Cider: {2:.4f}\t' 'Meteor: {3:.4f}\t' 'Rouge: {4:.4f}\t'
                .format(best_avg, best_bleu4, best_cider, best_meteor, best_rouge))
            earlyStop = 0

            print('Save Model')
            if(args.featureCNN == "True"):
                state = {
                    'epoch': epoch,
                    'avg_score': best_avg,
                    'bleu-4': Bleu_4,
                    'encoder_dict': encoder.state_dict(), 
                    'encoder_trans_dict': encoder_trans.state_dict(),   
                    'decoder_dict': decoder.state_dict(),
                    'encoder_optimizer': encoder_optimizer,
                    'encoder_trans_optimizer': encoder_trans_optimizer,
                    'decoder_optimizer': decoder_optimizer
                }
            else:
                state = {
                    'epoch': epoch,
                    'avg_score': best_avg,
                    'bleu-4': Bleu_4,
                    'encoder_dict': encoder.state_dict(), 
                    'encoder_trans_dict': encoder_trans.state_dict(),   
                    'decoder_dict': decoder.state_dict(),
                    'encoder_trans_optimizer': encoder_trans_optimizer,
                    'decoder_optimizer': decoder_optimizer
                }                
            model_name = str(args.model_name)+"_"+str(data_name)+'_batchsize_'+str(args.train_batchsize)+'_'+str(args.network)+'.pth'
            torch.save(state, os.path.join(args.savepath, model_name))
        else:
            earlyStop += 1
            print(f"No improvement since {earlyStop} epochs.\n")

            print('Old Best:\n' 'Epoch: {0:d}\t' 'AVG: {1:.4f}\t' 'BLEU-4: {2:.4f}\t' 'Cider: {3:.4f}\t' 'Meteor: {4:.4f}\t' 'Rouge: {5:.4f}\t'
                .format(best_epoch, best_avg, best_bleu4, best_cider, best_meteor, best_rouge))
    
        if(earlyStop == args.early_stop):
            break

def main(args):

    set_seed(42)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) 

    train(args.data_name, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Changes_to_Captions')

    # Data parameters
    parser.add_argument('--data_folder', default='./data/LEVIR_CC/images',help='folder with data files')
    parser.add_argument('--list_path', default='./data/LEVIR_CC/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./data/LEVIR_CC/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="LEVIR_CC",help='base name shared by data files.')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint, None if none.')
    parser.add_argument('--print_freq',type=int, default=50, help='print training/validation stats every __ batches')
   
    # Training parameters
    parser.add_argument('--fine_tune_encoder', type=bool, default=False, help='whether fine-tune encoder or not')    
    parser.add_argument('--train_batchsize', type=int, default=32, help='batch_size for training')
    parser.add_argument('--network', default='remoteClip', help='define the encoder to extract features')
    parser.add_argument('--encoder_dim',  type=int, default=1024, help='the dimension of extracted features using different network')
    parser.add_argument('--feat_size', default=16, help='define the output size of encoder to extract features')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--workers', type=int, default=8, help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--encoder_lr', type=float, default=8e-5, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=8e-5, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='clip gradients at an absolute value of.')
    parser.add_argument('--dropout', type=float, default=0.18, help='dropout')
    parser.add_argument('--early_stop', type=int, default=8, help='early stop')
    
    # Validation
    parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--beam_size', type=int, default=3, help='beam size for validation')
    parser.add_argument('--savepath', default="./models_checkpoint/")
    
    # Model parameters
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--decoder_n_layers', type=int, default=2)
    parser.add_argument('--feature_dim', type=int, default=768)

    #Ablation Parameters
    parser.add_argument('--mask', default = 'Self')
    parser.add_argument('--featureCNN', default = 'True')
    parser.add_argument('--featureCNNLayer', type=int, default = 3)
    parser.add_argument('--rankingLossCo', type=float, default = 0.2)
    parser.add_argument('--model_name', default = 'RemoteClip14_Dino_FeatureCNN')

    args = parser.parse_args()
    main(args)