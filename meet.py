import argparse
import embeddings
import numpy as np
import sys
import torch
import torch.nn as nn
import torchvision

def combine_context(x, idx, k):
    #Hyper-parameters
    dimension = 300
    nums = len(idx)
    
    # comute result matrix
    result = np.zeros([nums, k, dimension])
    for i in range(nums):
        context = np.zeros([k, dimension])
        sim = x[idx[i], :].dot(x.T)
        # np.argsort : return index of sorted array(ascending order)
        order = np.argsort(-sim)
        for j in range(k):
            context[-j, :] = x[order[j], :]
        result[i, :, :] = context
    return result


def main():
    print(40*'=' + 'new round' + 40*'=')
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Map word embeddings in two languages into a shared space')
    # description - This argument gives a brief description of what the program does and how it works.
    parser.add_argument('src_input', help='the input source embeddings')
    # help - A brief description of what the argument does.
    parser.add_argument('trg_input', help='the input target embeddings')
    parser.add_argument('src_output', help='the output source embeddings')
    parser.add_argument('trg_output', help='the output target embeddings')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('-d', '--init_dictionary', default=sys.stdin.fileno(), metavar='DICTIONARY', help='the training dictionary file (defaults to stdin)')
    args = parser.parse_args()
    
    # Read input embeddings
    srcfile = open(args.src_input, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_input, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, dtype='float32')
    trg_words, z = embeddings.read(trgfile, dtype='float32')
    
    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}
    src_indices = []
    trg_indices = []
    dict_size = 5000
    
    # Read seed dictionary
    f = open(args.init_dictionary, encoding=args.encoding, errors='surrogateescape')
    for line in f:
        src, trg = line.split()
        try:
            src_ind = src_word2ind[src]
            trg_ind = trg_word2ind[trg]
            src_indices.append(src_ind)
            trg_indices.append(trg_ind)
        except KeyError:
            print('WARNING: OOV dictionary entry ({0} - {1})'.format(src, trg), file=sys.stderr)
        if len(src_indices) == dict_size:
            break
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print(device)
    
    # Hyper-parameters
    INPUT_SIZE = 300
    HIDDEN_SIZE = 512
    NUM_LAYERS = 1
    SEQ_LEN = 5
    OUTPUT_SIZE = 300
    CONTEXT_NUMS = 10
    BATCH_SIZE = 50
    LR = 0.001
    EPOCH = 500
    #LR 0.005 too quick LR 0.001 work
    print('Learning rate is {}'.format(LR))
    
    # Load data
    context_data = torch.from_numpy(combine_context(z, trg_indices, CONTEXT_NUMS)).float()
    trg_data = torch.from_numpy(np.array([np.expand_dims(row, axis=0) for row in z[trg_indices]])).float()
    src_data = torch.from_numpy(np.array([np.expand_dims(row, axis=0) for row in x[src_indices]])).float()
    #print(context_data.shape)
    #print(trg_data.shape)
    #print(src_data.shape)
    
    # Build model
    class RNN(nn.Module):
        def __init__(self, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE):
            super(RNN, self).__init__()
            
            self.rnn = nn.LSTM(
                         input_size = INPUT_SIZE,
                         hidden_size = HIDDEN_SIZE,
                         num_layers = NUM_LAYERS,
                         batch_first=True,
                       )
                                     
            self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        
        def forward(self, x):
            r_out, _ = self.rnn(x, None)
            
            # Decode the hidden state of the last time step
            out = self.fc(r_out[:, -1, :])
            return out
    
    model = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
    print(model)
    
    # Loss and optimizer
    loss_func = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Train the model
    TOTAL_STEP = context_data.shape[0]//BATCH_SIZE
    for epoch in range(EPOCH):
        for i in range(TOTAL_STEP):
            z = context_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :, :].to(device)
            z0 = trg_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :].to(device)
            x0 = src_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :].to(device)
            '''
            if i == 0:
                print(z.shape)
                print(z0.shape)
                print(x0.shape)
            '''
            # Forward pass
            outputs = model(z)
            loss = loss_func(outputs, x0)# + loss_func(outputs, z0)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #if (i+1) % 10 == 0:
            #    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, EPOCH, i+1, TOTAL_STEP, loss.item()))
        if (epoch+1) % 10 == 0:
            print('Epoch[{}/{}], Loss:{:.4f}'.format(epoch+1,EPOCH, loss.item()))
    #TODO loss jump
    #TODO model(z)

    # Write improved embeddings
    '''
    srcfile = open(args.src_output, mode='w', encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_output, mode='w', encoding=args.encoding, errors='surrogateescape')
    embeddings.write(src_words, x, srcfile)
    embeddings.write(trg_words, z, trgfile)
    srcfile.close()
    trgfile.close()
    '''

if __name__ == '__main__':
    main()
