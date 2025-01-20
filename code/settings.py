prompt_type = 'cse-sfp'  # ['eol', 'sth', 'sum', 'cse-sfp']
backbone = 'llama3-8b'  # ['opt-6.7b', 'llama2-7b', 'llama3-8b', 'mistral-7b']

if prompt_type == 'eol':
    manual_template = 'This sentence : "*sent_0*" means in one word:"'
elif prompt_type == 'sth':
    manual_template = 'This sentence : "*sent_0*" means something'
elif prompt_type == 'sum':
    manual_template = 'This sentence : "*sent_0*" can be summarized as'
elif prompt_type == 'cse-sfp':
    if backbone == 'mistral-7b':
        template_name = 'sth-sum'
        manual_template = 'This sentence : "*sent_0*" means something, it can be summarized as'
    elif backbone == 'opt-6.7b':
        template_name = 'sth-sum'
        manual_template = 'This sentence : "*sent_0*" means something, it can also be summarized as'
    else:
        template_name = 'eol-sum'
        manual_template = 'This sentence : "*sent_0*" means in one word, it can be summarized as'