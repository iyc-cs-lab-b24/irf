def editDistance(str1, str2, m, n):
    if m == 0:
        return n
    if n == 0:
        return m
    if str1[m - 1] == str2[n - 1]:
        return editDistance(str1, str2, m - 1, n - 1)
    
    return 1 + min(editDistance(str1, str2, m, n - 1), 
                   editDistance(str1, str2, m - 1, n),  
                   editDistance(str1, str2, m - 1, n - 1))  

str1 = "write"
str2 = "right"
print("Edit Distance is:", editDistance(str1, str2, len(str1), len(str2)))