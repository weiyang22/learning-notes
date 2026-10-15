---
title: 02-UML 类图基础
tags:
  - UML
  - 类图
  - 设计模式可视化
description: 设计模式常用 UML 类图表示. 掌握 6 大关系和读图方法
last_updated: 2026-09-11
status: ✅ v1.0
chapter_id: 01-02
---

# 02. UML 类图基础

> 设计模式常用 UML 类图表示. 学会读图 = 学会快速理解模式的核心结构.

---

## 1. UML 类图基本元素

### 1.1 类的表示

```text
┌──────────────────────┐
│      ClassName        │  ← 类名 (粗体或首字母大写)
├──────────────────────┤
│ -field1: Type         │  ← 属性 (成员变量)
│ -field2: Type = value │
│ #protectedField       │  ← # 表示 protected
├──────────────────────┤
│ +method1(args): Ret   │  ← 方法 (成员函数)
│ +method2(): void      │
└──────────────────────┘

访问修饰符:
  +  public (公开)
  -  private (私有)
  #  protected (受保护)
  ~  package/default (默认, 包内可见)
```

### 1.2 接口 vs 抽象类

```text
接口 (Java/TypeScript):         抽象类 (Python/Java):
┌────────────────────┐         ┌────────────────────┐
│    <<interface>>   │         │  <<abstract>>       │
│       IDrawable     │         │      Shape          │
├────────────────────┤         ├────────────────────┤
│ +draw(): void       │         │ +area(): double      │
└────────────────────┘         │ +describe(): String   │
   (只有方法签名)               └────────────────────┘
                                (可以有部分实现)
```

---

## 2. 6 大关系

### 2.1 泛化 (Generalization) - 继承

**符号**: 实线 + 空心三角箭头 (箭头指向父类)

```text
┌──────┐                  ┌──────────┐
│ Dog  │ ──────▷─────────│  Animal  │
└──────┘   继承           └──────────┘
```

**代码**:

```python
class Animal: pass
class Dog(Animal): pass  # Dog 继承 Animal
```

### 2.2 实现 (Realization) - 实现接口

**符号**: 虚线 + 空心三角箭头 (箭头指向接口)

```text
┌──────┐                  ┌──────────┐
│ Dog  │ ─ - - - ▷ - - - -│ IDrawable│
└──────┘   实现           └──────────┘
```

**代码**:

```python
from abc import ABC, abstractmethod

class IDrawable(ABC):
    @abstractmethod
    def draw(self): pass

class Dog(IDrawable):
    def draw(self): print("画狗")
```

### 2.3 关联 (Association) - 知道

**符号**: 实线 (或带箭头)

```text
┌──────┐                  ┌──────┐
│Teacher│ ─────────────── │Student│
└──────┘   knows          └──────┘
```

**代码**:

```python
class Teacher:
    def __init__(self):
        self.students = []  # 关联: 老师知道学生
```

### 2.4 聚合 (Aggregation) - 弱拥有

**符号**: 空心菱形 (在整体一侧)

```text
┌────────┐                  ┌──────┐
│ Company│ ◇─────────────── │Employee│
└────────┘  has-a (弱)     └──────┘
```

**特点**: 整体和部分可独立存在. 公司没了, 员工还在.

```python
class Company:
    def __init__(self):
        self.employees = []  # 弱聚合, 员工不依赖公司

class Employee:
    pass
```

### 2.5 组合 (Composition) - 强拥有

**符号**: 实心菱形 (在整体一侧)

```text
┌──────┐                  ┌──────┐
│  Car │ ◆─────────────── │ Engine│
└──────┘  has-a (强)     └──────┘
```

**特点**: 整体销毁, 部分跟着销毁. 汽车报废, 引擎也没了.

```python
class Car:
    def __init__(self):
        self.engine = Engine()  # 强组合, 引擎是 Car 的一部分

class Engine:
    pass
```

### 2.6 依赖 (Dependency) - 使用

**符号**: 虚线 + 箭头 (箭头指向被依赖)

```text
┌──────┐                  ┌──────┐
│ ClassA│ - - - - -▷ - - - │ ClassB│
└──────┘  uses             └──────┘
```

**代码**:

```python
class ReportGenerator:
    def generate(self, data):
        printer = Printer()  # 临时依赖
        printer.print(data)
```

---

## 3. 关系对比表

| 关系 | 符号 | 强度 | 生命周期 | 代码体现 |
| :--- | :---: | :---: | :--- | :--- |
| 泛化 (继承) | ▷ | 强 | 同生命周期 | `class Dog(Animal)` |
| 实现 | ⇢ | 强 | 同生命周期 | `class Dog(IDrawable)` |
| 关联 | — | 弱 | 可独立 | 属性引用 |
| 聚合 | ◇ | 弱 | 可独立 | 集合属性 |
| 组合 | ◆ | 强 | 同生命周期 | 构造时创建 |
| 依赖 | ⇢ | 最弱 | 临时 | 方法参数 |

---

## 4. 实战: 画一个简单模式

### 单例模式 UML

```text
                  ┌──────────────────────┐
                  │     Singleton         │
                  ├──────────────────────┤
                  │ -instance: Singleton  │
                  │ -__new()              │
                  ├──────────────────────┤
                  │ +getInstance(): Singl. │
                  │ +businessMethod()     │
                  └──────────────────────┘
                            △
                            │ (实例化)
                  ┌──────────────────────┐
                  │       Client          │
                  │ -singleton: Singleton │
                  └──────────────────────┘
```

### 装饰器模式 UML

```text
                ┌───────────────┐
                │<<interface>>  │
                │  IComponent   │
                ├───────────────┤
                │ +operation()  │
                └──────△────────┘
                       │ (实现)
        ┌──────────────┴──────────────┐
        │                             │
┌───────────────┐              ┌───────────────┐
│  Component   │              │  Decorator    │
├───────────────┤              ├───────────────┤
│+operation()  │              │-component: IC│
└───────────────┘              │+operation()  │
                               └───────────────┘
                                       △
                                       │ (继承)
                              ┌───────────────┐
                              │ConcreteDec.  │
                              ├───────────────┤
                              │+operation()  │
                              │+addedBehavior│
                              └───────────────┘
```

---

## 5. 读图技巧

1. **从箭头出发**: 箭头方向 = 依赖方向 (箭头指向被依赖方)
2. **找核心类**: 通常是图中央, 被多个类引用
3. **看关系类型**: 6 种关系决定耦合度 (依赖 < 关联 < 聚合 < 组合 < 实现 < 泛化)
4. **看数字标识**:
   - `1` 表示 1 个
   - `0..*` 表示 0 到多个
   - `1..*` 表示 1 到多个

---

## 6. 总结

✅ **6 大关系**: 泛化 ▷ 实现 ⇢ 关联 — 聚合 ◇ 组合 ◆ 依赖 ⇢

✅ **箭头方向**: 指向被依赖方 (依赖方 → 被依赖方)

✅ **强度**: 依赖 < 关联 < 聚合 < 组合 < 实现 < 泛化

准备好读模式图了吗? 进入 [第02章 创建型模式](../第02章-创建型模式/)!
