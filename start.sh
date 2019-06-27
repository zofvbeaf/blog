#!/bin/bash
# if can't be accessed, do
# iptables -A OUTPUT -p tcp --sport 15722 -j ACCEPT
# iptables -A INPUT -p tcp --dport 15722 -j ACCEPT 
hexo server -i 192.168.1.52 -p 15722 --silent g -d

git add source/_posts/* && git push origin master && hexo g && hexo d
