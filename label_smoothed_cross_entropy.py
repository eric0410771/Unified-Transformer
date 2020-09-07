# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import numpy as np
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
import pickle
import os

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True, data_type = None):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    if data_type is not None:
        nll_loss = nll_loss * data_type
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss.sum()

def label_smoothed_nll_loss_v2(lprobs, target, epsilon, ignore_index = None):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim = -1, index = target)
    
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        
    return nll_loss
def draw_cross_single(attention_map, sample_id, source_tokens, target_tokens, source, target, predict, i2w, dir_name = "dual"):
    fig, ax = plt.subplots(figsize = (10,10))
    colors = [(1, 1, 1), (1, 0, 0)]
    n_bin = 100
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
    if(sample_id == 3232):
        img = attention_map.astype(float)[:,:-1]
    else:
        img = attention_map.astype(float)
    total_row = np.sum(img, axis = 1)
    img = (img.T/total_row).T
    def find_index(tokens):
        for i in range(len(tokens)):
            if tokens[i] == 2:
                break
        return i
        
    max_target_length = find_index(target_tokens)
    source_word_list = [i2w[i].replace("@@", "") for i in source_tokens]
    target_word_list = [i2w[i].replace("@@", "") for i in  target_tokens]
    if sample_id == 3232:
        source_word_list = source_word_list[:-1]
    
    img = img[:max_target_length]
    im = ax.imshow(img, cmap = cm)
    
    ax.set_xticks(np.arange(len(source_word_list)))
    ax.set_yticks(np.arange(max_target_length))
                
    ax.set_xticklabels(source_word_list, fontsize = 20)
    ax.set_yticklabels(target_word_list[:max_target_length], fontsize = 20)
    plt.title("Predicted result: " + predict , fontsize = 25)
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = 'right', rotation_mode = 'anchor')
                
    fig.tight_layout()
    plt.savefig(dir_name+"/"+dir_name+"_attention_map"+str(sample_id)+".png")

def draw_single(attention_map, data, predict_data, i2w, batch_index, counter, mask, prob, dir = 'mlm_tmp'):
    attention_map = attention_map.detach().cpu().numpy()
    img = attention_map[batch_index][:-1,:-1].astype(float)
    total_row = np.sum(img, axis = 1)
    font = 48
    colors = [(1, 1, 1), (1, 0, 0)]
    n_bin = 100
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
    img = (img.T/total_row).T
    if counter != 6 or counter != 171:
        fig, ax = plt.subplots(figsize = (len(data[batch_index]), len(data[batch_index])))
    else:
        fig, ax = plt.subplots(figsize = (8, 8))
    if counter == 6:
        im = ax.imshow(img[8:,8:], cmap =cm)
    elif counter == 171:
        im = ax.imshow(img[:8,:8], cmap = cm)
    else:
        im = ax.imshow(img)
    if counter == 171:
        mask = [5]
    correct_title = ""
    masked_title =  "Prediction     : "
    predict_title = "Ground-truth: "
    word_list = [i2w[i].replace("@@","").replace("&apos;","'") for i in data[batch_index]]
    word_list = np.asarray(word_list)
    word_list[mask] = '[MASK]'
    predict_word_list = []
    for index, i in enumerate(predict_data[batch_index]):
        if index in mask:
            predict_word_list.append(i2w[i].replace("@@", "").replace("&apos;","'"))
        else:
            predict_word_list.append(i2w[data[batch_index][index]].replace("@@", "").replace("&apos;","'"))
            
    
    if counter == 6:
        for index, i in enumerate(data[batch_index][8:]):
            if index != 5:
                correct_title += i2w[i].replace("@@", "").replace("&apos;","'") 
            else:
                correct_title += "[MASK]"        
            if index != len(data[batch_index][8:]):
                correct_title += " "
        for m in mask:
            if m >= 8:
                masked_title  += predict_word_list[m]
                predict_title += i2w[data[batch_index][m]].replace("@@", "").replace("&apos;","'")
    elif counter == 171:
        for index, i in enumerate(data[batch_index][:8]):
            if index not in mask:
                correct_title += i2w[i].replace("@@", "").replace("&apos;","'")
            else:
                correct_title += "[MASK]"
                        
            if index != len(data[batch_index][:8]):
                correct_title += " "
        masked_title += predict_word_list[5]
        predict_title += i2w[data[batch_index][5]].replace("@@","").replace("&apos;","'")
    else:
        for index, i  in enumerate(data[batch_index]):
            correct_title += i2w[i].replace("@@", "").replace("&apos;","'") + " "
            masked_title += word_list[index] + " "
            predict_title += predict_word_list[index] + " "
            
            if index != len(data[batch_index]):
                correct_title += " "
                masked_title += " "
                predict_title += " "
    
    plt.title(correct_title + "\n" + masked_title + "\n" + predict_title, fontsize = 44, loc='left')
    
    print("counter is ", counter)
    print(word_list)
    
    
    if counter == 1 or counter == 4 or counter == 17 or counter == 52 or True:
        for i in mask:
            if i < img.shape[0]:
                print("masked word is ", word_list[i])
                print(img[i])
            
                result = torch.exp(prob[batch_index, i])
                top = 3
                top_result = []
                top_probs = []
                
                for j in range(top):
                    index = torch.argmax(result).detach().cpu().item()
                    top_result.append(i2w[index].encode('utf-8'))
                    top_probs.append(result[index].detach().cpu().item())
                    result[index] = -1
                
                print(top_result)
                print(top_probs)
            
                
    
    print(word_list.tolist())
    print(predict_word_list)
    if counter == 6:
        ax.set_xticks(np.arange(len(word_list) - 1 - 8))
        ax.set_yticks(np.arange(len(word_list) - 1 - 8))
    
        ax.set_xticklabels(word_list[8:-1], fontsize = font)
        ax.set_yticklabels(word_list[8:-1], fontsize = font)
    elif counter == 171:
        ax.set_xticks(np.arange(8))
        ax.set_yticks(np.arange(8))
    
        ax.set_xticklabels(word_list[:8], fontsize = font)
        ax.set_yticklabels(word_list[:8], fontsize = font)
    else:
        ax.set_xticks(np.arange(len(word_list) - 1))
        ax.set_yticks(np.arange(len(word_list) - 1))
    
        ax.set_xticklabels(word_list[:-1], fontsize = 18)
        ax.set_yticklabels(word_list[:-1], fontsize = 18)       
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = 'right', rotation_mode = 'anchor')
                
    fig.tight_layout()
    plt.savefig(dir+"/attention_map"+str(counter)+".png")


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.gamma = 1
        self.gamma2 = 1
        self.counter = 0
        self.dict = None
        self.i2w = None
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on
    def set_dict(self, dict):
        self.dict = dict
        self.i2w = {}
        print("Enter")
        print(len(self.dict))
        for i, w in enumerate(self.dict):
            self.i2w[i] = w
        
    def forward(self, model, sample, dual = True, mlm = True, attention = False, cross_attention = True, consistency = False , reduce=True, test_lm = False, model2 = None):
        """Compute the loss for the given sample
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        '''
        print(sample.keys())
        print(sample['id'].shape)
        print(sample['net_input'])
        print(sample['target'].shape)
        '''
        con_loss = 0 
        future_loss = 0
        files = os.listdir('.')
        with open("ids.pickle", "rb") as f, open("result.pickle", "rb") as f2:
            ids = pickle.load(f)
            result = pickle.load(f2)
        trans_result = result["trans_result"]
        dual_result = result["dual_result"]
        def draw_attention_map(attention_maps, attention_maps2, sample_tokens, predict_sample_tokens, predict_sample_tokens2, mask, mlm_prob, prob, layer_index = 1):
            ## the shape of attention_maps is layer, batch size , length, length,
            length = 0
            data = sample_tokens.detach().cpu().numpy()
            predict_data = predict_sample_tokens.detach().cpu().numpy()
            predict_data2 = predict_sample_tokens2.detach().cpu().numpy()
            length = []

            for i in range(sample_tokens.size()[0]):
                tmp_length = 0
                for j in range(sample_tokens.size()[1]):
                    if sample_tokens[i,j] == self.dict['</s>']:
                        break
                tmp_length = j
                        
                miss = 0
                for j in range(tmp_length):
                    if data[i, j] != predict_sample_tokens[i, j]:
                        miss += 1
                
                if  miss < (data[i].shape[-1] * 0.3)/3 :
                    print(miss)
                    self.counter += 1
                    print("MLM")
                    draw_single(attention_map = attention_maps[layer_index - 1], data = data, predict_data = predict_data, i2w = self.i2w, batch_index = i, counter = self.counter, mask = mask, prob = mlm_prob, dir = 'mlm_tmp')
                    print("="*20)
                    print("LM")
                    draw_single(attention_map = attention_maps2[layer_index - 1], data = data, predict_data = predict_data2, i2w = self.i2w, batch_index = i, counter = self.counter, mask = mask, prob = prob, dir = 'tmp')
                    print("")
        
        
        
        def draw_cross_attention_map(attention_maps, attention_maps2, sample, layer_index = 1):
            sample_ids = sample['id'].detach().cpu().numpy()
            source_tokens = sample['net_input']['src_tokens'].detach().cpu().numpy()
            target_tokens = sample['target'].detach().cpu().numpy()
            source_length = source_tokens.shape[-1]
            target_length = target_tokens.shape[-1]
            attention_maps = attention_maps[layer_index-1].detach().cpu().numpy()
            attention_maps2 = attention_maps2[layer_index - 1].detach().cpu().numpy()
            def find_index(l, value):
                for i in range(len(l)):
                    if l[i] == value:
                        return i
                return -1   
            data = None
            
            """
            print("*" * 20)
            print("attention map ", attention_maps.shape)
            print("ids ", sample_ids.shape)
            print("source_length ",source_tokens.shape)
            print("target_length ",target_tokens.shape)
            """
            if self.counter < 14:
                for i, sample_id in enumerate(sample_ids):
                    if sample_id in ids and len(source_tokens[i]) < 10:
                        index = find_index(ids, sample_id)
                        draw_cross_single(attention_maps[i], sample_id, source_tokens[i], target_tokens[i], result['source'][index], result['target'][index], dual_result[index],self.i2w, dir_name = "dual")
                        draw_cross_single(attention_maps2[i], sample_id, source_tokens[i], target_tokens[i], result['source'][index], result['target'][index], trans_result[index],
                            self.i2w, dir_name = "trans")
                        
                        self.counter += 1                                                
        inverse_sample = None
        pad = torch.tensor([2] * sample['target'].size()[0], device = sample['target'].device).unsqueeze(-1)
        inverse_sample = {}
        inverse_sample['target'] = sample['net_input']['src_tokens']
        inverse_sample['id'] = sample['id']
        inverse_sample['nsentences'] = sample['nsentences']
        inverse_sample['ntokens'] = sample['net_input']['src_tokens'].size()[0] * sample['net_input']['src_tokens'].size()[1]
        #print("batch_size", sample["nsentences"])
        #print("length", sample['net_input']['src_tokens'].size()[1])
        
        net_input = {}
        net_input['prev_output_tokens'] = torch.cat((pad, sample['net_input']['src_tokens'][:,:-1]), dim = 1)
        net_input['src_tokens'] = sample['target']
        net_input['src_lengths'] = torch.tensor([sample['target'].size()[1]] * sample['nsentences'], device = sample['net_input']['src_lengths'].device)
        inverse_sample['net_input'] = net_input
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce = reduce)
        
        
            
        if dual:
            net_output2 = model(**inverse_sample['net_input'], inverse = True)
            loss2, nll_loss2 = self.compute_loss(model, net_output2, inverse_sample, reduce = reduce, data_type = sample['data_type'])
        if mlm:
            ratio = 0.3
            s_mask = np.random.choice(sample['net_input']['src_tokens'].size()[1], (int(sample['net_input']['src_tokens'].size()[1] * ratio), ),replace = False)
            t_mask = np.random.choice(inverse_sample['net_input']['src_tokens'].size()[1], (int(inverse_sample['net_input']['src_tokens'].size()[1] * ratio), ), replace = False)
            if model2 is not None and test_lm:
                with torch.no_grad():
                    t_net_output = model2.LM(sample['net_input']['prev_output_tokens'])
                    s_net_output = model2.LM(inverse_sample['net_input']['prev_output_tokens'], inverse = True)
                    s_loss, predict_sample_s2, prob_s2 = self.compute_mlm_loss(s_net_output[0], sample['net_input']['src_tokens'])
                    t_loss, predict_sample_t2, prob_t2 = self.compute_mlm_loss(t_net_output[0], inverse_sample['net_input']['src_tokens'])
            #for model that use mask language model
            s_net_output, s_embed = model.encodeMLM(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'], s_mask)
            t_net_output, t_embed = model.encodeMLM(inverse_sample['net_input']['src_tokens'], inverse_sample['net_input']['src_lengths'], t_mask, inverse = True)
            s_loss, predict_sample_s, prob_s = self.compute_mlm_loss(s_net_output, sample['net_input']['src_tokens'], data_type = sample['data_type'], reduce = True)
            t_loss, predict_sample_t, prob_t = self.compute_mlm_loss(t_net_output, inverse_sample['net_input']['src_tokens'], reduce = True)    
            ##for model that use language model
            #t_net_output = model.LM(sample['net_input']['prev_output_tokens'])
            #s_net_output = model.LM(inverse_sample['net_input']['prev_output_tokens'], inverse = True)
            #s_loss, predict_sample_s, prob_s = self.compute_mlm_loss(s_net_output[0], sample['net_input']['src_tokens'], data_type = sample['data_type'], reduce = True)
            #t_loss, predict_sample_t, prob_t = self.compute_mlm_loss(t_net_output[0], inverse_sample['net_input']['src_tokens'], reduce = True)    
                                      
            
            mlm_loss = t_loss + s_loss
        if attention:
            with torch.no_grad():
                s_attention_map = model.encode_attention(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'], s_mask, inverse = True)
                s_attention_map2 = model2.encode_attention(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'], s_mask, inverse = True)
                draw_attention_map(s_attention_map, s_attention_map2, sample['net_input']['src_tokens'], predict_sample_s, predict_sample_s2, s_mask, prob_s, prob_s2)
        if cross_attention:
            with torch.no_grad():
                s_t_cross_attention_map = model.encode_cross_attention(**sample['net_input'])
                s_t_cross_attention_map2 = model2.encode_cross_attention(**sample['net_input'])
                draw_cross_attention_map(s_t_cross_attention_map, s_t_cross_attention_map2, sample)
                '''
                if dual:
                    t_s_cross_attention_map = model.encode_cross_attention(**inverse_sample['net_input'])
                    draw_cross_attention_map(t_s_cross_attention_map, inverse_sample)
                ''' 
        if dual and mlm:  
            if sample['data_type'] is not None:
                #data_type = sample['data_type']
                #data_type = data_type.view(-1,1).repeat(1, predict_sample_s.size()[1])
                #future_loss = torch.sum(torch.sum((net_output2[2] - s_embed)**2, dim = -1) /512 * (1-data_type) )  + torch.sum((net_output[2] - t_embed)**2/512)
                pass
        if consistency:
            t_decoder_out, s_decoder_out = model.consistency(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'], sample['net_input']['prev_output_tokens'], 
                                                             inverse_sample['net_input']['src_tokens'], inverse_sample['net_input']['src_lengths'], inverse_sample['net_input']['prev_output_tokens'])
                                                             
            con_t_loss, _ = self.compute_loss(model, t_decoder_out, sample, reduce = reduce)
            con_s_loss, _ = self.compute_loss(model, s_decoder_out, inverse_sample, reduce = reduce)
            con_loss = con_t_loss + con_s_loss
            
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        if dual:
            trans_loss = loss + loss2
        else:
            trans_loss = loss
        
        if attention:
            return trans_loss * 0, sample_size, logging_output
        
        if dual:
            if mlm:
                return (mlm_loss), sample_size, logging_output
                #return 0 * (trans_loss), sample_size, logging_output
            else:
                return trans_loss, sample_size, logging_output
        else:
            if mlm:
                #return trans_loss , sample_size, logging_output
                return (trans_loss + mlm_loss + future_loss), sample_size, logging_output
            else:
                return trans_loss, sample_size, logging_output
        
    def compute_loss(self, model, net_output, sample, reduce=True, data_type = None):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        
        if data_type is not None:
            data_type = data_type.view(-1, 1).repeat(1,lprobs.size()[1])
            data_type = data_type.view(-1,1)
            ## de is mono so en to de, source need to subtract 1
            data_type = 1 - data_type
        
        lprobs = lprobs.view(-1, lprobs.size(-1))
        
        target = model.get_targets(sample, net_output).view(-1, 1)

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce, data_type = data_type,
        )
        return loss, nll_loss
        
    def self_compute_loss(self, model, net_output, sample, reduce = True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        
        
        target = model.get_targets(sample, net_output).view(-1, 1)
        predict_sentence = predict_sentence.view(taregt.size())
        
        
        nll_loss = label_smoothed_nll_loss_v2(
            lprobs, target, self.eps, ignore_index=self.padding_idx,
        )
        return nll_loss.view(sample['target'].size()), predict_sentences
    
    def compute_mlm_loss(self, enc_output, target, data_type = None, reduce = True):
        lprobs = utils.log_softmax(enc_output, dim = -1, onnx_trace = False)
        p_lprobs = lprobs.clone()
        
        if data_type is not None:
            data_type = data_type.view(-1, 1).repeat(1,lprobs.size()[1])
            data_type = data_type.view(-1,1)
            ## de is mono so en to de, source need to subtract 1
            data_type = 1 - data_type
            
        lprobs = lprobs.view(-1, lprobs.size(-1))
        predict_sentence = torch.argmax(lprobs, dim = -1)
        predict_sentence = predict_sentence.view(target.size())
        target = target.view(-1,1)
        
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index = self.padding_idx, data_type = data_type)
        
        return loss, predict_sentence, p_lprobs 
        
    
    
    def compute_semantic_loss(self, coarse, target):
        ## P(x->y | y)
        ## p(y | x->y)
        ## coarse size should be B x S x 512 
        ## target size should be B x T x 512

        if coarse.size()[0] != target.size()[0]:
            coarse = coarse.transpose(0,1)
        target = (target+1e-5) / (torch.sum(target ** 2, dim = -1, keepdim = True ) ** 0.5 + 1e-5) 
        coarse = (coarse+1e-5) / (torch.sum(coarse ** 2, dim = -1, keepdim = True) ** 0.5 + 1e-5)
        similarity_matrix = torch.bmm(target, coarse.transpose(1,2))  ## size = B x T x S 
        
        fn_similarity_matrix = self.softmax_normalization(similarity_matrix, along = 1)   ## size = B x T x S
        
        sn_similarity_matrix = self.softmax_normalization(fn_similarity_matrix, along = 2) ## size = B x T x S
        context_t = torch.bmm(sn_similarity_matrix, coarse) ## size = B x T x 512
        '''
        loss = 0
        total_p = 0
        for i in range(context_t.size()[0]):
            query = context_t[i:i+1]
            total_p += query
            qk = torch.cosine_similarity(query, target, dim = -1, eps = 1e-5) ## B x T
            qk = torch.log(torch.sum(torch.exp(qk), dim = -1)) ## B
            p = torch.exp(qk[i])/torch.sum(torch.exp(qk))
            loss += -torch.log(p)
        '''
        loss = torch.sum((context_t - target)**2)
        return loss
        
    def softmax_normalization(self, x, along = 0):
        x -= torch.max(x, dim = along, keepdim = True)[0]
        exp_x = torch.exp(x)
        sum_x = torch.sum(exp_x, dim = along, keepdim = True)
        
        return exp_x / sum_x
        
    
    def dot_product(self, query, key):
        return torch.sum(query * key, dim = -1)
    
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
