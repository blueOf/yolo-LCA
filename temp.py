n = int(input())
opt = []
for _ in range(n):
    opt.append(list(map(int, input().split())))

m = -1
M = 10**9

qujian = []
for i in range(n):
    qujian.append((opt[i][0] // (opt[i][1] + 1), opt[i][0] // opt[i][1]))
for i in range(n):
    m = max(m, qujian[i][0])
    M = min(M, qujian[i][1])

if m % 1 != 0:
    m = int(m) + 1
else:
    m = int(m)
M = int(M)
print(m, M)
