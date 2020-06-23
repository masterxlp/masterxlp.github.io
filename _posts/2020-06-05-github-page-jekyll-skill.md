---
layout: post
title:  "[E] GitHub Page and Jekyll Skill"
date:   2020-06-05 14:41:00 +0800
categories: Tools
tags: GitHub E
---
* content
{:toc}

## Markdown
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