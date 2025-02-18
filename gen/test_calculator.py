from gen.calculator_grammar import create_calculator_grammar

def main():
    # 创建语法实例
    grammar = create_calculator_grammar()
    
    # 测试表达式
    test_expressions = [
        "1 + 2",
        "3 * (4 + 5)",
        "10 - 5 / 2",
        "(1 + 2) * 3"
    ]
    
    for expr in test_expressions:
        try:
            # 解析表达式
            parse_tree = grammar.parse(expr)
            print(f"\n表达式: {expr}")
            print("解析树:")
            print(parse_tree.pretty())
        except Exception as e:
            print(f"解析错误 '{expr}': {str(e)}")

if __name__ == "__main__":
    main()