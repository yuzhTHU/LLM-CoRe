import re

def wrap_ansi(text, width=40):
    """
    每隔 width 个字符插入一个换行符，但忽略 ANSI 转义符的长度， 且保证不会在转义符中间换行。
    """
    # 匹配 ANSI 转义序列的正则表达式
    # \x1b 是 ESC 键，\[ 是 CSI 序列的开始
    # .*? 是非贪婪匹配中间的参数
    # [@-~] 是 CSI 序列的结束字符范围 (通常颜色以 'm' 结尾)
    ansi_escape_pattern = re.compile(r'(\x1b\[.*?[@-~])')
    
    # 使用正则切分字符串。
    # 因为正则中使用了括号 () 捕获组，split 会保留分隔符（即 ANSI 代码）
    parts = ansi_escape_pattern.split(text)
    
    result = []
    current_line_length = 0
    
    for part in parts:
        # 如果 part 是空的（split 有时会产生空串），跳过
        if not part:
            continue
        # 检查这部分是否是 ANSI 代码
        if ansi_escape_pattern.match(part):
            # 如果是颜色代码，直接加入结果，不增加长度计数，也不换行
            result.append(part)
        else:
            # 如果是普通文本，逐个字符处理
            for char in part:
                # 如果原文中已有换行符，重置计数器
                if char == '\n':
                    result.append(char)
                    current_line_length = 0
                    continue
                
                # 如果当前行长度达到限制，插入换行符并重置
                if current_line_length >= width:
                    result.append('\n')
                    current_line_length = 0
                
                result.append(char)
                
                # 在处理字符后，如果要计算宽字符（如中文），这里需要更复杂的逻辑
                # 这里简单地按字符数计算（1个汉字=1个长度）
                current_line_length += 1
                
    return "".join(result)
