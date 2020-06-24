---
layout: post
title:  "[E] The Important Skill for Everything."
date:   2020-06-05 14:41:00 +0800
categories: Tools Command
tags: SKILL E
author: Xlp
---
* content
{:toc}

## 01. Markdown
### 图片居中
> Q: 在GitHub Page和Jekyll中如何让图片或者文字居中显示？ 
 
```
# 图片居中显示
<div align="center"><img src=[local path or http path]></div>
# 文字居中显示
<div align="center">[something you want to show here]</div>
```




### 相对路径
> Q: 如何在GitHub Page上用相对路径引用图片？

在GitHub Page和Jekyll上引用图片时，使用相对地址相对来说很方便，例如，一般地，若图片与md文件在同级目录下时，可以使用 `./` 来表示；
若处于上一级目录时可使用 `../` 来表示。

在我的结构中，当在内容页显示图片时，由于其地址为 “xxx/xxx/xxx/2020/06/23/IMPALA/”，图片的相对路径与“2020”处于同一级目录，所以，
引用图片时的相对路径为 `../../../../`。


## 02. Vim
### 统计字符串出现的次数

- 统计从m行到n行中，字符串string出现的次数

```ruby
:m, ns/\<string\>//gn
```

- 在当前编辑的文件中统计字符串string出现的次数

```ruby
:%s/string//ng
```

- 统计字符串string在文件file中出现的行数

```ruby
cat file|grep -i string |wc -l
```

