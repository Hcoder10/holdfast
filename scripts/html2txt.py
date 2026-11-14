import re, sys, html
src, dst = sys.argv[1], sys.argv[2]
t = open(src, encoding='utf8', errors='ignore').read()
t = re.sub(r'<script.*?</script>|<style.*?</style>', '', t, flags=re.S)
t = re.sub(r'<annotation[^>]*>.*?</annotation>', '', t, flags=re.S)
t = re.sub(r'</(p|div|tr|h\d|li|section|figcaption|table)>', '\n', t)
t = re.sub(r'</t[dh]>', ' | ', t)
t = re.sub(r'<[^>]+>', ' ', t)
t = html.unescape(t)
t = re.sub(r'[ \t]+', ' ', t); t = re.sub(r'\n\s*\n+', '\n', t)
open(dst, 'w', encoding='utf8').write(t)
print(len(t))
