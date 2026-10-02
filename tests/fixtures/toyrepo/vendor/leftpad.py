# Vendored code: the walker must skip it (denylisted directory).
def leftpad(s, n, ch=" "):
    return ch * max(n - len(s), 0) + s
