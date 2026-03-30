import math

n = int(input())
nums = list(map(int, input().split()))

a = [nums[i] - nums[n - 1 - i] for i in range(math.ceil(n / 2))]
ans = 0
for i in range(len(a) - 1):
    ans += abs(a[i])
    if a[i + 1] * a[i] > 0:
        if abs(a[i]) > abs(a[i + 1]):
            a[i + 1] = 0
        else:
            a[i + 1] -= a[i]
print(ans + abs(a[-1]))
