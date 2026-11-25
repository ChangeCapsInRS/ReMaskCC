import sys
import os
import json
import argparse

sys.path.insert(0, os.path.abspath('.'))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default = 'LEVIR_CC', help= 'the name of the dataset')
parser.add_argument('--word_count_threshold', default=5, type=int)

SPECIAL_TOKENS = {
  '<NULL>': 0,
  '<UNK>': 1,
  '<START>': 2,
  '<END>': 3,
}

def main(args):

    args.data_folder = "./data/" + args.dataset
    
    if args.dataset == 'LEVIR_CC':
        input_captions_json = os.path.join(args.data_folder, 'LevirCCcaptions.json')
        save_dir = './data/LEVIR_CC/'
    elif args.dataset == 'Dubai_CC':
        input_captions_json = './data/Dubai_CC/DubaiCC/datasetDubaiCCPublic/description_jsontr_te_val'
        save_dir = './data/Dubai_CC/'
    elif args.dataset == 'WHU_CDC':
        input_captions_json = os.path.join(args.data_folder, 'whuCCcaptions.json')
        save_dir = './data/WHU_CDC/'
    
    output_vocab_json = 'vocab.json'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, 'tokens/')):
        os.makedirs(os.path.join(save_dir, 'tokens/'))
        
    print(f'Loading captions for {args.dataset}...')
    assert args.dataset in {'LEVIR_CC', 'Dubai_CC', 'WHU_CDC'}

    if args.dataset == 'LEVIR_CC' or args.dataset == 'WHU_CDC':
        with open(input_captions_json, 'r') as f:
            data = json.load(f)
            
        max_length = -1
        all_cap_tokens = []

        for img in data['images']:
            captions = []    
            for c in img['sentences']:
                assert len(c['raw']) > 0, 'error: some image has no caption'
                captions.append(c['raw'])
            
            tokens_list = []
            for cap in captions:
                cap_tokens = tokenize(cap,
                                    add_start_token=True,
                                    add_end_token=True,
                                    punct_to_keep=[';', ','],
                                    punct_to_remove=['?', '.'])
                tokens_list.append(cap_tokens)
                max_length = max(max_length, len(cap_tokens))
            
            split_name = img.get('split')
            
            if not split_name:
                filename_prefix = img['filename'].split('_')[0]
                if filename_prefix in ['train', 'val', 'test']:
                    split_name = filename_prefix
                else:
                    split_name = 'train' 

            all_cap_tokens.append((img['filename'], tokens_list, split_name))

        print('Saving captions and split lists...')
        
        f_train = open(os.path.join(save_dir, 'train.txt'), 'w')
        f_val = open(os.path.join(save_dir, 'val.txt'), 'w')
        f_test = open(os.path.join(save_dir, 'test.txt'), 'w')

        for img_name, tokens_list, split_name in all_cap_tokens:

            name_base = img_name.split('.')[0]
            tokens_json = json.dumps(tokens_list)
            
            with open(os.path.join(save_dir, 'tokens', name_base + '.txt'), 'w') as f:
                f.write(tokens_json)

            token_len = len(tokens_list)

            if split_name == 'train':
                for j in range(token_len):
                    f_train.write(img_name + '-' + str(j) + '\n')
            
            elif split_name == 'val':
                f_val.write(img_name + '\n')
            
            elif split_name == 'test':
                f_test.write(img_name + '\n')

        f_train.close()
        f_val.close()
        f_test.close()
    elif args.dataset == 'Dubai_CC': 
        filename = os.listdir(input_captions_json)
        max_length = -1
        all_cap_tokens = []

        for j in range(len(filename)):
            s_cap_tokens = []      
            caption_json = os.path.join(input_captions_json, filename[j])
            with open(caption_json, 'r') as f:
                data = json.load(f)
            for img in data['images']:
                captions = []
                for c in img['sentences']:
                    assert len(c['raw']) > 0, 'error: some image has no caption'
                    captions.append(c['raw'])
                tokens_list = []  
                for cap in captions:
                    cap_tokens = tokenize(cap,
                                        add_start_token=True,
                                        add_end_token=True,
                                        punct_to_keep=[';', ','],
                                        punct_to_remove=['?', '.'])
                    tokens_list.append(cap_tokens)
                    max_length = max(max_length, len(cap_tokens))
                s_cap_tokens.append((img['filename'], tokens_list))
                all_cap_tokens.append((img['filename'], tokens_list))
            
            print(f'Saving captions for part {filename[j]}')
            
            for img, tokens_list in s_cap_tokens:
                i = img.split('.')[0]
                token_len = len(tokens_list)
                tokens_list_json = json.dumps(tokens_list)
                
                with open(os.path.join(save_dir, 'tokens', i + '.txt'), 'w') as f:
                    f.write(tokens_list_json)

                # Dubai dosya ismi kontrolü (Train_xxxx.json gibi)
                split_prefix = filename[j].split('_')[0]
                
                if split_prefix == 'Train':
                    with open(os.path.join(save_dir, 'train.txt'), 'a') as f:
                        for s in range(token_len):
                            f.write(img + '-' + str(s) + '\n')

                elif split_prefix == 'Validation':
                    with open(os.path.join(save_dir, 'val.txt'), 'a') as f:
                        f.write(img + '\n')
                        
                elif split_prefix == 'Test':
                    with open(os.path.join(save_dir, 'test.txt'), 'a') as f:
                        f.write(img + '\n')

    print('max_length of the dataset:', max_length)
    
    if args.dataset == 'LEVIR_CC' or args.dataset == 'WHU_CDC':
        vocab_input = [(x[0], x[1]) for x in all_cap_tokens]
        print('Building vocab...')
        word_freq = build_vocab(vocab_input, args.word_count_threshold)
    else:
        print('Building vocab...')
        word_freq = build_vocab(all_cap_tokens, args.word_count_threshold)

    with open(os.path.join(save_dir, output_vocab_json), 'w') as f:
        json.dump(word_freq, f)
    
    print("Pre-processing completed successfully!")

def tokenize(s, delim=' ',add_start_token=True, 
    add_end_token=True,punct_to_keep=None, punct_to_remove=None):
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    tokens = [q for q in tokens if q != '']
    
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens

def build_vocab(sequences, min_token_count=1):
    token_to_count = {}
    for it in sequences:
        for seq in it[1]:
            for token in seq:
                if token not in token_to_count:
                    token_to_count[token] = 0
                token_to_count[token] += 1

    token_to_idx = {}
    for token, idx in SPECIAL_TOKENS.items():
        token_to_idx[token] = idx
    for token, count in sorted(token_to_count.items()):
        if token in token_to_idx.keys():
            continue
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx

def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Changes_to_Captions')

    parser.add_argument('--data_folder', default='./data/LEVIR_CC', help='folder with data files')
    parser.add_argument('--dataset', default='LEVIR_CC', help='dataset name: LEVIR_CC, WHU_CDC, Dubai_CC')
    parser.add_argument('--word_count_threshold', default=5, type=int, help='word threshold')

    args = parser.parse_args()
    main(args)