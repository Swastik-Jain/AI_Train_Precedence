with open('or_tools/smart_optimizer.py', 'r') as f:
    code = f.read()

old_str = """            next_opts  = node_data.get('next', [])
            direction  = train.get('direction', 'DOWN')
            is_token   = node_data.get('token_block', False)

            if not next_opts:"""

new_str = """            direction  = train.get('direction', 'DOWN')
            next_opts = track_map.get(pos, {}).get('prev', []) if direction == 'UP' else track_map.get(pos, {}).get('next', [])
            is_token   = node_data.get('token_block', False)

            if not next_opts:"""

code = code.replace(old_str, new_str)
with open('or_tools/smart_optimizer.py', 'w') as f:
    f.write(code)
