---
layout: post
title:  "[G] Data Structure"
date:   2020-07-13 10:23:00 +0800
categories: Data Structure
tags: 答题记录 代码实现 知识总结 G
author: Xlp
---
* content
{:toc}

> 题目来自：[LeetCode-cn](https://leetcode-cn.com/explore/learn)  
> 该部分主要记录对数据结构的总结及其结构的实现





## 队列

> 当你想要 **按顺序处理元素** 时，使用队列是一个很好的选择。

### 先入先出的数据结构
在 FIFO 数据结构中，将 **首先处理添加到队列中的第一个元素**。

队列是典型的 FIFO 数据结构。
插入（insert）操作也称为入队（enqueue），**新元素始终被添加在队列的末尾**。
删除（delete）操作也称为出队（dequeue），**每次仅移除队列的第一个元素**。

### 队列的缺点
队列在某些情况下效率很低：随着指针的移动，会浪费越来越多的空间。
当我们有空间限制时，这将是难以接受的。

示例：

<div align="center"><img src="../../../../image/队列/queue.jpeg" width="70%" height="70%"></div>
<div align="center">图1. 队列的入队和出队示意图</div>

### 循环队列
具体来说，我们可以使用 **固定大小的数组** 和 **两个指针** 来指示起始位置和结束位置，目的是 **重用** 之前提到的 被浪费的内存。

示例：

<iframe src="../../../../image/队列/循环队列演示.gif" width="70%" height="70%"></iframe>

### 循环队列的实现

```
# MyCode

class MyCircularQueue:

    def __init__(self, k: int):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        """
        self.size = k
        self.cur_size = 0
        
        self.head = -1
        self.tail = -1
        
        self.queue = [[] for _ in range(k)]

    def enQueue(self, value: int) -> bool:
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        """
        if not self.isFull():
            if self.isEmpty():
                self.queue[0] = value
                self.head, self.tail = 0, 0
            else:
                if self.tail < self.size - 1:
                    self.tail += 1
                else:
                    self.tail = 0
                self.queue[self.tail] = value
            self.cur_size += 1
            return True
        return False
        
        

    def deQueue(self) -> bool:
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        """
        if not self.isEmpty():
            self.queue[self.head] = []
            if self.head < self.size - 1:
                self.head += 1
            else:
                self.head = 0
            self.cur_size -= 1
            return True
        return False
                
        

    def Front(self) -> int:
        """
        Get the front item from the queue.
        """
        return self.queue[self.head] if not self.isEmpty() else -1
        

    def Rear(self) -> int:
        """
        Get the last item from the queue.
        """
        return self.queue[self.tail] if not self.isEmpty() else -1
        

    def isEmpty(self) -> bool:
        """
        Checks whether the circular queue is empty or not.
        """
        return self.cur_size == 0
        

    def isFull(self) -> bool:
        """
        Checks whether the circular queue is full or not.
        """
        return self.cur_size == self.size
        


# Your MyCircularQueue object will be instantiated and called as such:
# obj = MyCircularQueue(k)
# param_1 = obj.enQueue(value)
# param_2 = obj.deQueue()
# param_3 = obj.Front()
# param_4 = obj.Rear()
# param_5 = obj.isEmpty()
# param_6 = obj.isFull()
```

```
# 官方给出的 c++ 实现

class MyCircularQueue {
private:
    vector<int> data;
    int head;
    int tail;
    int size;
public:
    /** Initialize your data structure here. Set the size of the queue to be k. */
    MyCircularQueue(int k) {
        data.resize(k);
        head = -1;
        tail = -1;
        size = k;
    }
    
    /** Insert an element into the circular queue. Return true if the operation is successful. */
    bool enQueue(int value) {
        if (isFull()) {
            return false;
        }
        if (isEmpty()) {
            head = 0;
        }
        tail = (tail + 1) % size;
        data[tail] = value;
        return true;
    }
    
    /** Delete an element from the circular queue. Return true if the operation is successful. */
    bool deQueue() {
        if (isEmpty()) {
            return false;
        }
        if (head == tail) {
            head = -1;
            tail = -1;
            return true;
        }
        head = (head + 1) % size;
        return true;
    }
    
    /** Get the front item from the queue. */
    int Front() {
        if (isEmpty()) {
            return -1;
        }
        return data[head];
    }
    
    /** Get the last item from the queue. */
    int Rear() {
        if (isEmpty()) {
            return -1;
        }
        return data[tail];
    }
    
    /** Checks whether the circular queue is empty or not. */
    bool isEmpty() {
        return head == -1;
    }
    
    /** Checks whether the circular queue is full or not. */
    bool isFull() {
        return ((tail + 1) % size) == head;
    }
};

/**
 * Your MyCircularQueue object will be instantiated and called as such:
 * MyCircularQueue obj = new MyCircularQueue(k);
 * bool param_1 = obj.enQueue(value);
 * bool param_2 = obj.deQueue();
 * int param_3 = obj.Front();
 * int param_4 = obj.Rear();
 * bool param_5 = obj.isEmpty();
 * bool param_6 = obj.isFull();
 */
```

```
# 官方 c++ 代码的 Python 实现

class MyCircularQueue:

    def __init__(self, k: int):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        """
        self.size = k
        
        self.head = -1
        self.tail = -1
        
        self.queue = [[] for _ in range(k)]

    def enQueue(self, value: int) -> bool:
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        """
        if self.isFull():                         # 先处理特殊情况，再进行一般情况的处理
            return False
        
        if self.isEmpty():
            self.head = 0
        
        self.tail = (self.tail + 1) % self.size
        self.queue[self.tail] = value
        return True
        
        

    def deQueue(self) -> bool:
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        """
        if self.isEmpty():
            return False
        
        if self.head == self.tail:
            self.head = -1
            self.tail = -1
            return True
        self.head = (self.head + 1) % self.size     # 与我的逻辑处理的不同之处在于：利用取余来代替了 if 判断
        return True
                
        

    def Front(self) -> int:
        """
        Get the front item from the queue.
        """
        return self.queue[self.head] if not self.isEmpty() else -1
        

    def Rear(self) -> int:
        """
        Get the last item from the queue.
        """
        return self.queue[self.tail] if not self.isEmpty() else -1
        

    def isEmpty(self) -> bool:
        """
        Checks whether the circular queue is empty or not.
        """
        return self.head == -1
        

    def isFull(self) -> bool:
        """
        Checks whether the circular queue is full or not.
        """
        return (self.tail + 1) % self.size == self.head
        


# Your MyCircularQueue object will be instantiated and called as such:
# obj = MyCircularQueue(k)
# param_1 = obj.enQueue(value)
# param_2 = obj.deQueue()
# param_3 = obj.Front()
# param_4 = obj.Rear()
# param_5 = obj.isEmpty()
# param_6 = obj.isFull()
```

### 数据流中的移动平均值

> 给定一个整数数据流和一个窗口大小，根据该滑动窗口的大小，计算其所有整数的移动平均值。

> 示例：   
>> MovingAverage m = new MovingAverage(3);  
>> m.next(1) = 1  
>> m.next(10) = (1 + 10) / 2  
>> m.next(3) = (1 + 10 + 3) / 3  
>> m.next(5) = (10 + 3 + 5) / 3   

```
class MovingAverage:

    def __init__(self, size: int):
        """
        Initialize your data structure here.
        """
        self.size = size
        
        self.queue = []
        

    def next(self, val: int) -> float:
        if len(self.queue) < self.size:
            self.queue.append(val)
        else:
            self.queue = list(reversed(self.queue))
            self.queue.pop()
            self.queue = list(reversed(self.queue))
            self.queue.append(val)
        
        return sum(self.queue) / len(self.queue)
        


# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param_1 = obj.next(val)
```
