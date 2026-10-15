---
title: LSP - 里氏替换原则 (Liskov Substitution Principle)
tags: [SOLID, 设计原则, LSP, 里氏替换, 多态, 继承]
description: 子类型必须能够替换它们的基类型,且不改变程序的正确性
last_updated: 2026-09-11
status: ✅ v1.0
chapter_id: 05-03
---

# LSP - 里氏替换原则 (Liskov Substitution Principle)

> **"Functions that use pointers or references to base classes must be able to use objects of derived classes without knowing it."**
> —— Barbara Liskov, *Data Abstraction and Hierarchy* (1987), 后续由 Jeannette Wing 形式化

里氏替换原则是 SOLID 五原则中 **最学术化、最难被"凭直觉"理解** 的一条。它由图灵奖得主 Barbara Liskov 在 1987 年的 OOPSLA 会议上提出(后来 Jeannette Wing 在 1993 年给出了更严格的数学定义),回答了一个看似简单实则深邃的问题:**"继承到底什么时候是安全的?"**

## 一句话定义

> **所有引用基类的地方必须能透明地使用其子类的对象,而程序的行为不会改变、正确性不会破坏。**

换句话说:子类应该 **完全实现** 父类的语义契约,而不是 **改变** 这个契约。如果父类方法约定 "调用后对象处于某状态",子类就不能违背这个约定;如果父类约定 "参数不能为 null",子类就不能 "进步地" 接受 null。

## 核心思想

### 1. LSP 不是关于继承语法,而是关于行为契约

很多程序员误以为 LSP 是 "子类能通过父类的类型检查就够了"。但 LSP 远比这严格——它关心的是 **行为** 而不是 **类型**。Java 的 `@Override` 注解只能保证 **语法兼容**(方法签名匹配),不能保证 **行为兼容**(返回值、状态、异常、副作用都符合父类契约)。

举几个 LSP 违反的典型例子:
- 父类 `add()` 约定 "添加元素后集合变大",子类的 `add()` 实现 "重复元素时静默忽略"——破坏了 "集合大小" 的不变量
- 父类 `read()` 约定 "不会修改对象状态",子类的 `read()` 实现 "同时往日志里写入数据"——破坏了 "查询操作无副作用" 的契约
- 父类 `setX()` 接受 "任意整数",子类的 `setX()` 抛出 "超过特定范围的异常"——收紧了前置条件

### 2. 子类型 vs 子类:细微但关键的区别

LSP 严格意义上说的是 **"子类型"(subtype)** 而非 **"子类"(subclass)**。在 Java/C++ 这类语言里,你用 `extends` 关键字建立的是 "继承关系",但只有当子类 **满足父类的所有行为契约** 时,它才同时也是 "子类型"。**"is-a" 的关系不是由继承语法定义的,而是由行为契约定义的**——这就是为什么 "Square extends Rectangle" 在 LSP 视角下是个错误。

### 3. 契约式设计(DBC)与 LSP

Bertrand Meyer 在他提出 OCP 的同一本书里,系统化了 "契约式设计"(Design by Contract)思想,与 LSP 高度契合:
- **前置条件(precondition)**:方法被调用前,调用方必须满足的条件。**子类可以放宽但不能收紧**(因为调用方按父类契约写的代码不会满足更严格的前置条件)。
- **后置条件(postcondition)**:方法返回后必须为真的条件。**子类可以收紧但不能放宽**(因为调用方依赖父类承诺的不变量)。
- **不变量(invariant)**:类在任何时候都必须保持的性质。**子类必须保持**(因为调用方随时可以观察这个不变量)。

如果子类违反了这些契约,就破坏了 LSP。例如:`Collections.sort(List)` 约定传入的 list 必须是可比较的(mutable),如果你重写 `sort` 让它接受 immutable list 才能工作——这就是收紧前置条件,LSP 违反。

### 4. LSP 与多态、OCP 的关系

LSP 是 OCP 的 **正确性保证**。OCP 要求 "通过子类扩展",但只有当子类满足 LSP 时,这种扩展才不会引入 bug。**如果 LSP 被破坏,OCP 就成了 "带 bug 的扩展能力"**——加一个子类可能让系统整体行为异常,而且这种 bug 极难调试,因为调用方写的是抽象类型,运行时才绑定到具体子类。

## Python/Java 正反对比

### ❌ 反例:经典 Square/Rectangle 问题

这是 LSP 文献中最常被引用的反例:

```java
public class Rectangle {
    protected int width;
    protected int height;

    public void setWidth(int width) {
        this.width = width;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public int getWidth() { return width; }
    public int getHeight() { return height; }

    public int getArea() {
        return width * height;
    }
}

// 反直觉的继承
public class Square extends Rectangle {
    @Override
    public void setWidth(int width) {
        this.width = width;
        this.height = width; // 必须同步修改 height 才能保持正方形
    }

    @Override
    public void setHeight(int height) {
        this.height = height;
        this.width = height; // 必须同步修改 width
    }
}

// 调用方:依赖 Rectangle 抽象
public void testRectangle(Rectangle r) {
    r.setWidth(5);
    r.setHeight(4);
    // 期望:面积为 20(5 * 4)
    assert r.getArea() == 20 : "面积应为 20,实际为 " + r.getArea();
}

// 调用
Rectangle r = new Rectangle(2, 3);
testRectangle(r); // ✅ 通过

Rectangle s = new Square(); // 多态调用,LSP 应该允许
testRectangle(s); // ❌ 面积是 16(因为 setWidth(5) 同时把 height 改为 5,setHeight(4) 又把 width 改为 4)
```

**问题分析:**
- `Square.setWidth` 不仅修改 width,还偷偷修改 height——**副作用违背父类契约**
- 调用方 `testRectangle` 完全按 `Rectangle` 的契约编写,没有用任何 `instanceof` 判断——按 LSP 应该是合法的
- 但实际上传 `Square` 会触发 bug,这就是 LSP 违反

### ✅ 正例:平行继承 / 共同抽象

```java
// 方案 1:让 Square 和 Rectangle 平行继承 Shape
public abstract class Shape {
    public abstract int getArea();
}

public class Rectangle extends Shape {
    private int width;
    private int height;
    public Rectangle(int width, int height) { this.width = width; this.height = height; }
    @Override public int getArea() { return width * height; }
    public void setWidth(int width) { this.width = width; }
    public void setHeight(int height) { this.height = height; }
}

public class Square extends Shape {
    private int side;
    public Square(int side) { this.side = side; }
    @Override public int getArea() { return side * side; }
    public void setSide(int side) { this.side = side; }
}

// 调用方
public void testShape(Shape s) {
    // 不假设具体子类的行为,只调用 Shape 公开的方法
    System.out.println("面积: " + s.getArea());
}

// 方案 2:用不可变对象消除 "setter 副作用" 的隐患
public final class ImmutableRectangle {
    private final int width;
    private final int height;

    public ImmutableRectangle(int width, int height) {
        this.width = width;
        this.height = height;
    }

    public int getArea() { return width * height; }
    // 没有 setter,根本不可能违反 LSP
}
```

### ❌ 反例:抛出父类未声明的异常

```java
public class FileReader {
    public String read(String path) throws IOException {
        return Files.readString(Paths.get(path));
    }
}

// LSP 违反:收紧后置条件,引入新的异常
public class StrictFileReader extends FileReader {
    @Override
    public String read(String path) throws IOException, SecurityException {
        if (!hasPermission(path)) {
            throw new SecurityException("无权访问: " + path);
        }
        return super.read(path);
    }
}

// 调用方
public void process(FileReader reader) {
    try {
        String content = reader.read("/data/file.txt");
        System.out.println(content);
    } catch (IOException e) { // 父类约定的异常
        System.err.println("IO 错误: " + e.getMessage());
    }
    // ❌ 如果传入 StrictFileReader,可能抛出 SecurityException 但未被捕获
}
```

### ✅ 正例:子类的异常应该是父类异常的子类

```java
public class FileReader {
    public String read(String path) throws IOException {
        return Files.readString(Paths.get(path));
    }
}

// 正确:SecurityException extends RuntimeException,不强制调用方处理
// 或者抛 IOException 的子类
public class AuditedFileReader extends FileReader {
    @Override
    public String read(String path) throws IOException { // 不引入新的 checked exception
        auditLog(path);
        return super.read(path);
    }
}
```

### Python 对比

```python
# ❌ 反例
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def set_width(self, w): self.width = w
    def set_height(self, h): self.height = h
    def area(self): return self.width * self.height

class Square(Rectangle):
    def set_width(self, w):
        self.width = w
        self.height = w  # 偷偷改 height
    def set_height(self, h):
        self.width = h
        self.height = h


# 测试
def test_rect(r: Rectangle):
    r.set_width(5)
    r.set_height(4)
    assert r.area() == 20, f"期望 20,实际 {r.area()}"

test_rect(Rectangle(2, 3))   # ✅
test_rect(Square(1))          # ❌ 16


# ✅ 正例
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def area(self): return self.width * self.height

class Square(Shape):
    def __init__(self, side):
        self.side = side
    def area(self): return self.side ** 2
```

## 实际应用

### 1. Java Collections Framework:LSP 的典范

Java 集合框架是 LSP 的最佳正面案例:

```java
// 抽象层:List 接口定义了清晰的契约
public interface List<E> extends Collection<E> {
    boolean add(E e);              // 契约:返回 true 表示集合因调用而改变
    void add(int index, E element); // 契约:index 必须在 [0, size()] 范围内
    E get(int index);              // 契约:index 必须在 [0, size()-1] 范围内
    E remove(int index);           // 契约:返回被移除的元素
    // ... 几十个方法
}

// ArrayList:基于数组的实现
public class ArrayList<E> implements List<E> {
    @Override public boolean add(E e) {
        // 数组扩容后追加
        // 始终遵守契约:add 后 size 增加 1
    }
}

// LinkedList:基于链表的实现
public class LinkedList<E> implements List<E> {
    @Override public boolean add(E e) {
        // 链表尾部追加
        // 始终遵守契约:add 后 size 增加 1
    }
}

// CopyOnWriteArrayList:线程安全的实现
public class CopyOnWriteArrayList<E> implements List<E> {
    @Override public boolean add(E e) {
        // 写时复制
        // 仍然遵守契约:add 后 size 增加 1
    }
}

// 调用方:可以透明替换任意实现
public void process(List<String> items) {
    items.add("hello");
    items.add(0, "world");
    String first = items.get(0);
    // 不管 items 是 ArrayList、LinkedList 还是 CopyOnWriteArrayList,行为都一致
}
```

这就是 LSP 的威力——JDK 提供了几十种 List 实现,但所有调用方写的是 `List<T>` 接口,可以随意切换实现而不影响代码正确性。

### 2. Spring 的 `JdbcTemplate` vs `NamedParameterJdbcTemplate`

`NamedParameterJdbcTemplate` 包装了 `JdbcTemplate`,添加了命名参数支持。它严格遵守 LSP——所有 `JdbcTemplate` 的方法在 `NamedParameterJdbcTemplate` 上行为一致,只是额外提供了新方法。这种 **"包装+扩展"** 模式是 LSP 的高级应用,本质上是装饰器模式。

### 3. 模板方法模式与 LSP

```java
public abstract class AbstractHandler {
    // 模板方法:定义算法骨架
    public final void handle(Request req) {
        validate(req);              // 钩子方法 1
        authenticate(req);          // 钩子方法 2
        Object result = doHandle(req); // 抽象方法:子类必须实现
        log(req, result);           // 钩子方法 3
    }

    protected void validate(Request req) { /* 默认空实现 */ }
    protected abstract Object doHandle(Request req);
    protected void log(Request req, Object result) { /* 默认日志 */ }
}

public class UserHandler extends AbstractHandler {
    @Override
    protected Object doHandle(Request req) {
        // 处理用户相关请求
        return userService.process(req);
    }
    // 子类不重写 validate 和 log,继承父类的默认行为——LSP 安全
}
```

模板方法模式严格遵守 LSP:子类只能通过重写钩子方法来扩展行为,不能修改父类 `handle()` 的算法骨架(`final` 方法)。

### 4. Java 8 函数式接口

Java 8 的 `Function<T, R>`、`Predicate<T>`、`Supplier<T>` 也遵循 LSP:

```java
Function<Integer, Integer> doubleIt = x -> x * 2;
Function<Integer, Integer> addOne = x -> x + 1;

// Function.andThen 组合:LSP 安全
Function<Integer, Integer> combined = doubleIt.andThen(addOne);
combined.apply(3); // 3 * 2 + 1 = 7

// 任何 Function<Integer, Integer> 实现都可以透明替换,行为一致
```

## 违反原则的危害

### 场景 1:HashMap/Hashtable 的历史包袱

Java 1.0 的 `Hashtable` 继承自 `Dictionary`,而 Java 1.2 引入的 `HashMap` 也实现 `Map` 接口。但 `Hashtable` 的很多方法(put、remove)是 **synchronized 的**,而 `HashMap` 不是。如果你在某个方法签名里用了 `Map` 类型,调用方可能传入 `Hashtable`,期望的并发行为就完全不对——这本质上是 LSP 违反的体现。后来的设计者意识到这个问题,干脆让 `Hashtable` 不再被推荐使用,新代码应该用 `ConcurrentHashMap`。

### 场景 2:日期 API 的混沌

Java 1.0 的 `java.util.Date` 设计混乱(可变、不直观、月份从 0 开始……)。Java 8 引入 `java.time` 包(LocalDate、Instant、ZonedDateTime)完全重做了日期 API。如果硬要让新 API 继承 `java.util.Date` 并保持向后兼容,必然导致 LSP 违反——所以 Java 团队选择让新旧 API **平行存在**,不建立继承关系,这是务实的做法。

### 场景 3:可变性带来的隐藏 bug

```java
public class Order {
    public BigDecimal calculateTotal() {
        // 不修改订单状态的纯计算
    }
}

public class AuditedOrder extends Order {
    @Override
    public BigDecimal calculateTotal() {
        auditLog.record("开始计算"); // ❌ LSP 违反:引入了副作用
        return super.calculateTotal();
    }
}
```

看起来 `AuditedOrder` 只是 "加了审计功能",但因为它在 `calculateTotal` 中偷偷写日志(改变对象状态),任何依赖 "calculateTotal 是无副作用的" 的代码都会出 bug。比如缓存层会把 `calculateTotal` 结果缓存起来,但实际上每次调用 `AuditedOrder.calculateTotal` 都会写入审计日志,导致日志爆炸。

## 判断标准

### 何时应该应用 LSP(检查继承关系)

✅ **三步检验法:**
1. 父类方法的前置条件,子类是否只放宽不收紧?
2. 父类方法的后置条件,子类是否只收紧不放宽?
3. 父类的不变量(invariant),子类是否始终保持?

✅ **更强的检验:** 在任何使用父类的地方,换成子类后单元测试是否依然通过?

✅ **更简单的直觉:** "子类是不是真的 'is-a' 父类?" —— Square "is-a" Rectangle 吗?事实上是反直觉的(正方形在数学上是矩形,但在程序行为上不是)。

### 何时不需要纠结 LSP

❌ **抽象类而非具体继承** —— 如果父类是 `abstract` 且没有完整实现,就不存在 LSP 问题(它本就是为继承而设计的)
❌ **测试代码** —— 测试中可能故意 mock 不同行为,这不在 LSP 考虑范围
❌ **历史包袱兼容** —— 偶尔为了兼容老代码,会写一些 "故意违反" 的子类,只要标注清楚就行

## 与其他原则的关系

### 与 OCP 的关系
OCP 要求 "通过子类扩展",LSP 保证 "子类扩展不会破坏正确性"。**LSP 是 OCP 在多态层面的正确性基石**。OCP 是 "能不能扩展",LSP 是 "扩展得对不对"。没有 LSP 的 OCP 是危险的——你可能加了 100 个子类,但其中任何一个都可能破坏系统行为。

### 与 ISP 的关系
LSP 关注 "子类型替换父类型",ISP 关注 "接口不要太大"。两者相辅相成:**接口越小越具体,LSP 越容易满足**。一个胖接口(如 `Worker { work, eat, sleep }`)很难让所有实现类都满足 LSP——`Robot` 实现 `Worker` 时 `eat()` 该怎么办?接口拆小后,每个接口的契约更精确,LSP 违反的风险更低。

### 与 SRP 的关系
LSP 要求子类不改变父类契约。**当父类职责过多(SRP 违反)时,它的 "契约" 也模糊不清**,子类 LSP 难以保证。所以 SRP 是 LSP 的前提——父类的职责清晰单一,LSP 才有意义。

### 协同案例
- **OCP + LSP**:策略模式(Strategy Pattern)——一族算法可互换,每种实现都严格遵守抽象接口契约
- **LSP + ISP**:接口隔离后,每个接口契约精确,子类型实现难度降低
- **SRP + LSP + OCP**:父类职责单一 → 契约清晰 → 子类容易正确实现 → OCP 扩展安全

## 实战思考

### Q1:LSP 是不是反对继承?

**答:** 不反对,但确实 **要求谨慎使用继承**。LSP 的真正含义是:**"如果你选择用继承,就必须严格遵守父类契约"**。如果发现两个类之间没有严格的 "is-a" 关系(行为层面),就应该:
- 用 **接口 + 实现**(composition) 替代继承
- 用 **抽象基类**(abstract class) 定义清晰的扩展点
- 用 **装饰器模式**(decorator) 在不修改原类的前提下扩展行为

Java 圈有句老话:**"Favor composition over inheritance"**(组合优于继承),这背后的理论基础就是 LSP——继承很容易不知不觉违反 LSP,而组合/接口实现则不容易。

### Q2:怎么快速判断一个继承关系是否违反 LSP?

**答:** 用 Robert Martin 的 "替换测试"(Substitutability Test):
> **"Replace every instance of the base class with an instance of the derived class. If the program still works correctly, the design obeys LSP."**

具体操作:
1. 写一段使用父类的代码(`Foo foo = ...; foo.doSomething();`)
2. 把 `Foo` 用子类实例替换(`Foo foo = new SubFoo();`)
3. 不修改调用方任何代码
4. 跑测试
5. 如果测试通过 → LSP 满足;如果有测试失败 → LSP 违反

这就是为什么 **单元测试是检验 LSP 的最佳手段**——给父类写的测试,应该能不加修改地跑过所有子类。

### Q3:LSP 和协变/逆变有什么关系?

**答:** 这是 LSP 在类型系统层面的精确化。Java 5 引入泛型时,加入了协变返回类型(covariant return types):

```java
class Animal {
    public Animal clone() { return new Animal(); }
}

class Dog extends Animal {
    @Override
    public Dog clone() { return new Dog(); } // ✅ 协变返回:Dog 是 Animal 的子类型
}
```

协变返回是 **LSP 友好的**——子类的 `clone()` 返回更具体的类型,调用方可以安全地用 `Dog` 接收而不必强制转型。

而 **逆变参数**(contravariant parameters)在 Java 中通过 `? super T` 实现:
```java
List<? super Integer> list = new ArrayList<Number>(); // 接受 Integer 的父类型
```

LSP 要求:**子类方法的参数类型可以是父类参数的超类(逆变),返回值类型可以是父类返回值的子类(协变)**。这正好对应 "前置条件放宽"、"后置条件收紧" 的契约设计思想。

## 小结

LSP 是 SOLID 中 **最考验设计直觉** 的一条原则。它不像 SRP 那么直观,也不像 OCP 那么有诱惑力,但它是大型系统长期健康度的 **隐性保障**。

LSP 的核心可以浓缩为一句话:**"子类必须严格遵守父类的行为契约,而不仅仅是类型兼容"**。当你设计一个继承关系时,问自己三个问题:
1. 子类有没有收紧父类的前置条件?
2. 子类有没有放宽父类的后置条件?
3. 子类有没有破坏父类的不变量?

如果三个答案都是 "没有",LSP 满足,OCP 才有意义。

LSP 的最大价值在 **演化中的系统** 才完全显现。一个 5 年的老系统,如果从一开始就严格遵守 LSP,加新功能只需要加新类;如果没有 LSP 保障,加新功能就要不停修改现有代码,而且每次修改都可能是高风险。LSP 就像代码的 "免疫系统"——平时感觉不到它的存在,一旦缺失,各种奇怪 bug 就会接踵而至。

下一章我们将学习 **ISP(接口隔离原则)**——它从接口维度告诉我们如何避免 "类被迫依赖自己不需要的方法",从而在更早的阶段预防 LSP 违反。
