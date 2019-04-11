import argparse

def main():

    # 1. argparse를 사용하는 첫 번째 단계는 ArgumentParser 클래스의 객체를 생성하는 것
    # ArgumentParser 클래스의 객체인 parser 생성
    parser = argparse.ArgumentParser()
    
    # 2.인자 추가하기
    # ArgumentParser에 프로그램 인자에 대한 정보를 채우려면 add_argument() 메서드를 호출
    parser.add_argument('--x', type=float, default=1.0,
            help='What is the first number?')
    
    parser.add_argument('--y', type=float, default=1.0,
            help='What is the second number?')
    
    parser.add_argument('--operation', type=str, default='add',
            help='What operation? (add, sub, mul, or div)')
    
    # 3. 인자 파싱하기
    # ArgumentParser는 parse_args() 메서드를 통해 인자 파싱 
    args = parser.parse_args()
    
    print(calc(args))

def calc(args):
    
    if args.operation == 'add':
        return args.x + args.y
    elif args.operation == 'sub':
        return args.x - args.y
    elif args.operation == 'mul':
        return args.x * args.y
    elif args.operation == 'div':
        return args.x / args.y

if __name__ == '__main__':
    main()

# 실행 시 
# python argparse_tutorial2.py --x=5 --y=2 --operation=mul
# python argparse_tutorial2.py -h
    
    