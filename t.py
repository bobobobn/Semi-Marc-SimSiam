def min_operations_to_alternate(s: str) -> int:
    n = len(s)
    ops1 = ops2 = 0

    # Calculate operations needed for '010101...' pattern
    for i in range(n):
        expected_char = '0' if i % 2 == 0 else '1'
        if s[i] != expected_char:
            ops1 += 1

    # Calculate operations needed for '101010...' pattern
    for i in range(n):
        expected_char = '1' if i % 2 == 0 else '0'
        if s[i] != expected_char:
            ops2 += 1

    return min(ops1, ops2)


# 示例
s = "11100"
print(min_operations_to_alternate(s))  # 输出最少操作次数
