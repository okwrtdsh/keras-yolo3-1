from difflib import SequenceMatcher


def _main_(args):
    data = []
    with open(args.input, "r") as f:
        for line in f:
            data.append(line.rstrip().split(","))

    remap = {}
    for tm2, tm1, t, tp1 in zip(data, data[1:], data[2:], data[3:]):
        if tm1[1] == tp1[1] == tm2[1] and tm1[1] != '':
            if tm1[1] != t[1] and t[1] != '':
                s = SequenceMatcher(a=tm1[1], b=t[1])
                r = s.ratio()
                if r > 0.5:
                    print(t[0], '\t', '%.5f' % r, '\t',  repr(t[1]), '->', repr(tm1[1]))
                    remap[t[0]] = tm1[1]

    with open(args.output, "w") as f:
        for file, s in data:
            f.write("%s,%s\n" % (file, remap.get(file, s)))


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '-i',
        '--input',
        help='input path')
    argparser.add_argument(
        '-o',
        '--output',
        help='output path')
    args = argparser.parse_args()
    _main_(args)
