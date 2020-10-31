"""
For data preparation. convert dataset to tsv file.

"""
import argparse
import csv
import pandas as pd
import ujson as json
import pickle
import codecs


def load_raw(loadpath, loadinfo):
    """
    load raw text file
    """
    print(loadinfo)
    with codecs.open(loadpath, 'r', encoding='utf8') as f:
        data = [line.strip('\n') for line in f]
    print('load raw text with {} lines.'.format(len(data)))
    return data

def dump_raw(data, savepath, saveinfo):
    print(saveinfo)
    with codecs.open(savepath, 'w', encoding='utf8') as f:
        for line in data:
            f.write(line + '\n')
    print('dump raw text with {} lines.'.format(len(data)))
    return

def loads_json(loadpath, loadinfo):
    with open(loadpath, 'r', encoding='utf-8') as fh:
        print(loadinfo)
        dataset = []
        for line in fh:
            example = json.loads(line)
            dataset.append(example)
        print('load json done\n')
    return dataset


def load_json(loadpath, loadinfo):
    with open(loadpath, 'r', encoding='utf-8') as fh:
        print(loadinfo)
        dataset = json.load(fh)
        print('load json done\n')
    return dataset


def dump_json(data, savepath, saveinfo):
    with open(savepath, 'w', encoding='utf-8') as fh:
        print(saveinfo)
        json.dump(data, fh)
        print('json save done')


def dumps_json(data, savepath, saveinfo):
    with open(savepath, 'w', encoding='utf-8') as fh:
        print(saveinfo)
        for example in data:
            fh.write(json.dumps(example) + '\n')
        print('json save done')


def dump_pickle(data, savepath, saveinfo):
    with open(savepath, 'wb') as fh:
        print(saveinfo)
        pickle.dump(data, fh)
        print('pickle save done')


def load_pickle(loadpath, loadinfo):
    with open(loadpath, 'rb') as fh:
        print(loadinfo)
        dataset = pickle.load(fh)
        print('load pickle done')
    return dataset


def add_blank_line():
    return

def write_txt2csv(txt_path, label, csv_path):
    with codecs.open(txt_path, 'r', encoding="utf-8") as fin:
        stripped = ([label, line.strip()] for line in fin)
        # stripped = ([label, line.encode('utf-8').strip()] for line in fin.readlines())
        with open(csv_path, 'w', encoding='utf8') as fout:
            mycsv = csv.writer(fout)
            mycsv.writerows(stripped)
    print('Write txt to csv done!')
    return


def tsv_combine_shuffle(data1, data2, data3, data4):
    # load in data
    data1 = pd.read_csv(data1, header=None)
    print('EM formal length {}'.format(len(data1)))
    data2 = pd.read_csv(data2, header=None)
    print('EM informal length {}'.format(len(data2)))
    data3 = pd.read_csv(data3, header=None)
    print('FR formal length {}'.format(len(data3)))
    data4 = pd.read_csv(data4, header=None)
    print('FR formal length {}'.format(len(data4)))
    # concat
    combined = pd.concat([data1, data2, data3, data4])
    # shuffle
    shuffled = combined.sample(frac=1)
    print('shuffled length {}'.format(len(shuffled)))
    # write into .tsv format
    data_bert = pd.DataFrame({
        'id':range(len(shuffled)),
        'label':shuffled[0],
        'alpha':['a']*shuffled.shape[0],
        'text':shuffled[1].replace(r'\n', ' ', regex=True)
    })
    return data_bert


def main(args):
    if args.parse:
        # for formal & informal files, read txt and transform to csv
        # need parser.add_argument('-genre', type=str, default='Entertainment_Music/', help='data genere. Entertainment_Music or Family_Relationship')
        files = ['informal', 'formal']
        modes = ['train/', 'test/']
        for mode in modes:
            for label, file in enumerate(files):
                txt_path = args.data + args.genre + mode + file
                csv_path = args.data + args.genre + mode + file + '.csv'
                # tsv_path = args.data + args.genre + mode + file + '.tsv'
                write_txt2csv(txt_path, label, csv_path)

    if args.combine:
        # for EM & FR, combine formal/informal.csv under train/ and test/. Then change into tsv file.
        modes = ['train/', 'test/']
        for mode in modes:
            em_f_csv_path = args.data + 'Entertainment_Music/' + mode + 'formal' + '.csv'
            em_inf_csv_path = args.data + 'Entertainment_Music/' + mode + 'informal' + '.csv'
            fr_f_csv_path = args.data + 'Family_Relationships/' + mode + 'formal' + '.csv'
            fr_inf_csv_path = args.data + 'Family_Relationships/' + mode + 'informal' + '.csv'
            out_tsv_path = '/research/king3/lijj/style_trans/bert_classifier/data/' + '{}.tsv'.format(mode[:-1])
            data_bert = tsv_combine_shuffle(em_f_csv_path, em_inf_csv_path, fr_f_csv_path, fr_inf_csv_path)
            data_bert.to_csv(out_tsv_path, sep='\t', index=False, header=False)

    if args.add_blank:
        files = ['train_all', 'train']
        for file in files:
            in_path = args.data + file + '.txt'
            out_path = args.data + file + '_pro.txt'
            with codecs.open(out_path, 'w', encoding='utf8') as fout:
                with codecs.open(in_path, 'r', encoding='utf8') as fin:
                    for line in fin:
                        fout.write(line)
                        fout.write('\n')
    return


def test():
    s1 = 'the sentence 1.\n'
    s2 = 'the sentence 2.\n'
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input preparation')
    parser.add_argument('-data', type=str, default='/research/king3/lijj/tesla/quora_data/', help='data root path')
    # parser.add_argument('-genre', type=str, default='Entertainment_Music/', help='data genere. Entertainment_Music or Family_Relationship')
    parser.add_argument('-parse', action='store_true', help='parse plain txt to csv file')
    parser.add_argument('-combine', action='store_true', help='combine corpus from two domains: EM and FR')
    parser.add_argument('-add_blank', action='store_true', help='add blank line between two sentence lines in txt file')
    args = parser.parse_args()

    main(args)
    # test()



