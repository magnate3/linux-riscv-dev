from ipaddress import IPv6Address
# 创建IPv6Address对象
#ipv6 = IPv6Address('2001:0db8:85a3:0000:0000:8a2e:0370:7334')
ipv6 = IPv6Address('2009::a0a:c801')
 
# 直接使用str方法进行格式化
formatted_address = str(ipv6)
print(formatted_address)  # 输出: 2001:db8:85a3::8a2e:370:7334
 
# 使用IPv6Address对象的exploded属性进行格式化
formatted_address = ipv6.exploded
print(formatted_address)  # 输出: 2001:0db8:85a3:0000:0000:8a2e:0370:7334
 
# 如果你想要压缩的形式
formatted_address = ipv6.compressed
print(formatted_address)  # 输出: 2001:db8:85a3::8a2e:370:7334
num = 9999 
hex_str = format(num, 'x')
print(hex_str)  # 输出：270f
