# markdown中所包含的语法
# code: 长代码片段
# inline code:

# code
code = [
    '```',
    '    ',
]

# code (not markdown)
code_n = [
    '`',     # {'comment_id':327463297, 128768497}
]


# inline regex
inline = []
# inline code
inline.append(
    r'`[^`]*`',
)
# link
inline.append(r'!\[[^\[]*\]\([^\(]*\)')
inline.append(r'\[[^\[]*\]\([^\(]*\)')
inline.append(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


quote = [
    # quote
    '>',
    # 预设定
    '###',
    '##',
    '####'
]

