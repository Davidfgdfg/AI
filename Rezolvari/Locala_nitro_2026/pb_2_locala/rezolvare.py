from collections import deque

def solve(n):
    # stare: (c_stanga, a_stanga, c_insula, a_insula, c_dreapta, a_dreapta, barca)
    start = (n, n, 0, 0, 0, 0, 0)
    goal = (0, 0, 0, 0, n, n, 2)
    q = deque([(start, 0)])
    seen = {start}
    
    while q:
        s, d = q.popleft()
        if s[:6] == goal[:6]: return d
        
        b = s[6]
        curr_c, curr_a = s[b*2], s[b*2+1]
        
        for mc in range(3):
            for ma in range(3):
                if 1 <= mc + ma <= 2 and mc <= curr_c and ma <= curr_a:
                    for nxt in ([1] if b != 1 else [0, 2]):
                        ns = list(s)
                        ns[b*2] -= mc; ns[b*2+1] -= ma
                        ns[nxt*2] += mc; ns[nxt*2+1] += ma
                        ns[6] = nxt
                        t_ns = tuple(ns)
                        
                        if t_ns not in seen:
                            valid = True
                            for i in range(3):
                                c, a = ns[i*2], ns[i*2+1]
                                if a > 0 and c > a: valid = False
                                # In contextul n perechi, regula c <= a e suficienta
                            
                            if valid:
                                seen.add(t_ns)
                                q.append((t_ns, d + 1))
    return -1

print("subtaskID,datapointID,answer")
print(f"1,3,{solve(3)}")
for n in range(4, 9):
    print(f"2,{n},{solve(n)}")