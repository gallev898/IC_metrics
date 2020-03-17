import os
import json
import numpy as np
import torch

from pycocoevalcap.COCOevaluator import COCOEvalCap, calculate_metrics


if __name__ == '__main__':
    rng = range(25000)

    f = open('results.txt', 'w+')

    spesific_mosel = ['train_show_and_tell']
    # spesific_mosel = ['unlikelihood_full_replace_16_16', 'run_batch_size_16', 'unlikelihood_noun_replace_16_16']
    specific_dic = ['beam_10', 'beam_1', 'top_k_5', 'top_p_0.8']
    # specific_dic = ['beam_1', 'beam_5', 'beam_10']

    for model in os.listdir('/home/gal/Desktop/Pycharm_projects/image_captioning/metrics/metrics_dics'):
        if model in spesific_mosel:
            for t in os.listdir('/home/gal/Desktop/Pycharm_projects/image_captioning/metrics/metrics_dics/{}/metrics'.format(model)):
                if any(t.__contains__(dic) for dic in specific_dic):
                    print('EVAL FOR: model: {}  dic: {}'.format(model, t))
                    dic = torch.load(
                        '/home/gal/Desktop/Pycharm_projects/image_captioning/metrics/metrics_dics/{}/metrics/{}'.format(model, t))

                    datasetGTS = dic['gt']
                    datasetRES = dic['hyp']
                    results = calculate_metrics(rng, datasetGTS, datasetRES)

                    f.write(model + '\n' + t + '\n')
                    json.dump(results, f)
                    f.write('\n')
                    f.write('----------------------------------------\n')

                    print results
                    print model
