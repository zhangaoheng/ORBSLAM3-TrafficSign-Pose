import os, glob, sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                    '..', '..', 'data', 'test_pairs')
base = os.path.normpath(base)
split_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', 'data', 'extracted_data', 'split_by_map')
split_dir = os.path.normpath(split_dir)

fixed = 0
errors = []
for pair_dir in sorted(glob.glob(os.path.join(base, 'pair_*'))):
    pair_name = os.path.basename(pair_dir)
    parts = pair_name.replace('pair_', '').split('_')
    for seq_name, map_idx in [('seq1', 0), ('seq2', 1)]:
        map_id = parts[map_idx]
        for subdir in ['rgb', 'depth']:
            dst = os.path.join(pair_dir, seq_name, subdir)
            if not os.path.lexists(dst):
                continue
            src_sub = os.path.join(split_dir, f'map_{map_id}', subdir)
            real_src = os.path.realpath(src_sub)
            if not os.path.isdir(real_src):
                errors.append(f'{pair_name}/{seq_name}/{subdir}: src missing')
                continue
            try:
                if os.path.islink(dst):
                    os.unlink(dst)
                os.symlink(real_src, dst)
                fixed += 1
                print(f'  OK {pair_name}/{seq_name}/{subdir}')
            except Exception as e:
                errors.append(f'{pair_name}/{seq_name}/{subdir}: {e}')

print(f'\nFixed: {fixed}')
if errors:
    print(f'Errors ({len(errors)}):')
    for e in errors:
        print(f'  {e}')
