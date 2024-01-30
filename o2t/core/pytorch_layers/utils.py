def get_value_by_key(node, key, default=None):
    if key in node.attrs:
        return node.attrs[key]
    else:
        return default
