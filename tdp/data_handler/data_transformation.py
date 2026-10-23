def reindex(idx, old_root_idx, new_root_idx):
    if idx == old_root_idx:
        new_idx = new_root_idx
    else:
        new_idx = str(int(idx.split("_")[0]) + 1) + idx[idx.find("_") :]
    return new_idx


def add_root_and_reindex(text_string, nodelist, govlist, root_token):
    text_string = [root_token] + text_string
    old_root_idx = "-1_-1_-1"
    new_root_idx = "0_0_0"
    govlist = [reindex(idx, old_root_idx, new_root_idx) for idx in govlist]
    nodelist = [reindex(idx, old_root_idx, new_root_idx) for idx in nodelist]
    return (text_string, nodelist, govlist)
