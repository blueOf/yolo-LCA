import bisect


def solve():
    n = int(input())
    a = list(map(int, input().split()))
    s = 0
    pre_sum = [0 for i in range(n + 1)]
    for i in range(1, n + 1):
        pre_sum[i] = pre_sum[i - 1] + a[i - 1]

    last = pre_sum[-1]
    first = 0
    if first > last:
        temp = last
        last = first
        first = temp
    ans = -1
    pre_sum.sort()
    print(first, last)
    last_index = bisect.bisect_left(pre_sum, last)
    first_index = bisect.bisect_left(pre_sum, first)
    print(first_index, last_index)
    l = pre_sum[first_index]
    r = pre_sum[first_index + 1]
    for i in range(first_index - 1, 0, -1):
        if (first_index - i) % 2 == 1:
            ans = max(ans, abs(r - pre_sum[i]))
            r = pre_sum[i]
        else:
            ans = max(ans, abs(l - pre_sum[i]))
            l = pre_sum[i]
    ans = max(ans, abs(l - pre_sum[0]))
    ans = max(ans, abs(r - pre_sum[0]))

    for i in range(first_index + 1, last_index + 1):
        ans = max(ans, abs(pre_sum[i] - pre_sum[i - 1]))

    l = pre_sum[last_index - 1]
    r = pre_sum[last_index]
    for i in range(last_index + 1, len(pre_sum) - 1):
        if (i - last_index) % 2 == 1:
            ans = max(ans, abs(l - pre_sum[i]))
            l = pre_sum[i]
        else:
            ans = max(ans, abs(r - pre_sum[i]))
            r = pre_sum[i]
    ans = max(ans, abs(l - pre_sum[len(pre_sum) - 1]))
    ans = max(ans, abs(r - pre_sum[len(pre_sum) - 1]))
    print(ans)


t = int(input())
for _ in range(t):
    solve()
