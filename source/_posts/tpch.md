---
title: TPCH test scripts
tags: 
  - tpch
categories:
  - database
date: 2018-11-19 11:10:07
mathjax: true
---


Usage
========================================

```bash
cd tpch/dbgen
cp makefile.suite Makefile
# modify the Makefile
#  CC = gcc
#  DATABASE = ORACLE
#  MACHINE = LINUX
make
./dbgen -h
./dbgen -vf -s 1 
```
+ for postgresql
```
find -name "*.tbl" | xargs sed -i "s/|$//"

```


