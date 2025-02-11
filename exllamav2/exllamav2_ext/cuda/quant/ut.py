# %%
s = "02468acegikmoqsu13579bdfhjlnprtv"
# print like zd = qqo | (qqq<<4) | (qqs<<8) | (qqu<<12) | (qqp<<16) | (qqr<<20) | (qqt<<24) | (qqv<<28);
output = "ze = qqv "
for i in range(1, len(s)):
    output += f"| (qq{s[len(s)-i-1]}<<{i})"
output += ";"
print(output)

# %%
