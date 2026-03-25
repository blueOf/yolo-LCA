n = int(input())
s = 0
for i in range(1, n):
    n -= i
    s += 1
    if s >= n:
        break
print(i)
