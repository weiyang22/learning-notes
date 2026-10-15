---
title: 第 05 章 - 设计原则与 SOLID (索引)
tags: [SOLID, 设计原则, 设计哲学, 章节索引]
description: SOLID 五原则 + 其他经典设计原则的完整索引,系统化面向对象设计的思想体系
last_updated: 2026-09-11
status: ✅ v1.0
chapter_id: 05
---

# 第 05 章 设计原则与 SOLID

> **"Gather together the things that change for the same reasons. Separate those things that change for different reasons."**
> —— Robert C. Martin

设计原则是软件设计的 **"宪法"**——它告诉你在面对具体问题时 **"应该朝什么方向思考"**。本章系统讲解面向对象设计最重要的 **SOLID 五原则**,以及另外五条经典原则(KISS、YAGNI、DRY、组合优于继承、迪米特法则),共同构成完整的设计思想体系。

---

## 📑 本章目录

| 编号 | 文件 | 原则 | 一句话定义 |
|---|---|---|---|
| 05-01 | [`01-SRP-单一职责原则.md`](./01-SRP-单一职责原则.md) | **SRP** (Single Responsibility Principle) | 一个类只应该有一个引起它变化的原因 |
| 05-02 | [`02-OCP-开闭原则.md`](./02-OCP-开闭原则.md) | **OCP** (Open/Closed Principle) | 软件实体应该对扩展开放,对修改关闭 |
| 05-03 | [`03-LSP-里氏替换原则.md`](./03-LSP-里氏替换原则.md) | **LSP** (Liskov Substitution Principle) | 子类型必须能够替换它们的基类型 |
| 05-04 | [`04-ISP-接口隔离原则.md`](./04-ISP-接口隔离原则.md) | **ISP** (Interface Segregation Principle) | 不应该强迫客户依赖于它们不用的方法 |
| 05-05 | [`05-DIP-依赖倒置原则.md`](./05-DIP-依赖倒置原则.md) | **DIP** (Dependency Inversion Principle) | 高层模块不应依赖低层模块,二者都应依赖抽象 |
| 05-06 | [`06-其他设计原则.md`](./06-其他设计原则.md) | **KISS / YAGNI / DRY / 组合 / LoD** | 简单、节制、不重复、灵活、解耦 |

---

## 🎯 SOLID 五原则对比

| 原则 | 关注层 | 核心问题 | 关键反模式 | 主要工具 |
|---|---|---|---|---|
| **SRP** | 类 | "一个类该做什么?" | God Class / 万能类 | 类的拆分 |
| **OCP** | 扩展 | "如何不修改就扩展?" | switch-case 满天飞 | 抽象类 + 多态 |
| **LSP** | 继承 | "继承关系正确吗?" | Square 继承 Rectangle | 契约式设计、行为子类型 |
| **ISP** | 接口 | "接口要多大?" | 胖接口、UnsupportedOperationException | 角色接口 |
| **DIP** | 依赖 | "依赖谁?" | 高层直接 new 低层 | 抽象归属调用方、依赖注入 |

### 五大原则的内在逻辑

```
        ┌─────────────────────────────────────────────┐
        │              SRP (拆分类)                     │
        │         一个类只承担一个职责                    │
        └──────────────────┬──────────────────────────┘
                           │
                           ↓
        ┌─────────────────────────────────────────────┐
        │              OCP (可扩展)                     │
        │         通过抽象让扩展不动旧代码                │
        └──────────────────┬──────────────────────────┘
                           │
                           ↓
        ┌─────────────────────────────────────────────┐
        │              LSP (可替换)                     │
        │         子类正确实现父类契约                    │
        └──────────────────┬──────────────────────────┘
                           │
        ┌──────────────────┴──────────────────────────┐
        │                                             │
        ↓                                             ↓
┌────────────────────┐                    ┌────────────────────┐
│  ISP (小接口)        │                    │  DIP (依赖抽象)      │
│ 接口只暴露需要的方法 │                    │ 细节依赖抽象而非相反   │
└────────────────────┘                    └────────────────────┘
```

**SRP 是基础**——先按职责拆分类,才有后续原则的施展空间。
**OCP 是目标**——我们追求的就是系统可扩展而不修改老代码。
**LSP 是保证**——OCP 通过多态扩展,LSP 保证扩展的正确性。
**ISP 是策略**——接口设计要小而精,让 LSP 更容易满足。
**DIP 是机制**——通过依赖抽象(接口归属调用方),让前面四个原则真正落地。

---

## 📚 学习顺序建议

### 🥇 第一阶段:SOLO 学习(初学者)

按以下顺序学习,层层递进:

1. **SRP** ← 基础中的基础,理解"职责"的概念
2. **OCP** ← 理解"抽象+多态"的力量
3. **LSP** ← 深入理解"继承"的边界
4. **ISP** ← 接口设计的最佳实践
5. **DIP** ← SOLID 的集大成者

每学完一条,尝试在现有代码中找反例,体会违反原则的危害。

### 🥈 第二阶段:综合应用(进阶)

学完 SOLID 后,继续学习:

6. **其他设计原则**(KISS / YAGNI / DRY / 组合优于继承 / 迪米特法则)

这五条原则教你 **"克制"** 的设计哲学——什么时候 **不** 应用抽象。

### 🥉 第三阶段:实战落地(高级)

- 在 Spring 项目中,主动识别 SRP 违反的 God Class,做拆分
- 在设计新模块时,用 SOLID 五个原则审视 API 设计
- 阅读 JDK 源码(`InputStream`、`List`、`Executor` 等),体会 SOLID 在工业级框架中的应用

---

## 🔗 SOLID 与设计模式的关系

> **"SOLID 是原则,模式是具体实现"** —— Robert C. Martin

设计模式(GoF 23 种)是 SOLID 原则在 **特定场景下的具体实现**。理解了这个关系,你就不再是 "死记硬背模式",而是 **"根据原则推导模式"**。

### SOLID → 设计模式 对照表

| SOLID 原则 | 对应的设计模式 | 应用方式 |
|---|---|---|
| **SRP** | 外观模式(Facade)、中介者模式(Mediator) | 把多个职责聚合到统一入口 |
| **OCP** | 策略模式(Strategy)、模板方法(Template Method)、装饰器(Decorator) | 通过抽象和多态实现扩展 |
| **LSP** | 组合模式(Composite)、迭代器模式(Iterator) | 父类型与子类型行为完全一致 |
| **ISP** | 角色接口模式(Role Interface)、适配器模式(Adapter) | 接口按角色拆分 |
| **DIP** | 抽象工厂(Abstract Factory)、依赖注入(DI) | 抽象归属调用方 |

### 经典组合案例

**1. 策略模式(OCP + DIP):**
```java
// OCP:加新策略不动旧代码
// DIP:业务依赖 PricingStrategy 抽象,不依赖具体实现
public interface PricingStrategy { /* ... */ }
public class VipPricing implements PricingStrategy { /* ... */ }
public class DiscountPricing implements PricingStrategy { /* ... */ }
```

**2. 装饰器模式(OCP + LSP + DIP):**
```java
// OCP:叠加新功能不动旧代码
// LSP:每个装饰器都是 InputStream 的合法子类型
// DIP:调用方只依赖 InputStream 抽象
new BufferedReader(new InputStreamReader(new FileInputStream(path)))
```

**3. 模板方法模式(OCP + LSP):**
```java
// OCP:子类扩展算法步骤
// LSP:子类必须遵守父类定义的算法骨架
public abstract class AbstractHandler {
    public final void handle() {
        validate();           // 步骤 1(可重写)
        doHandle();           // 步骤 2(必须重写)
        log();                // 步骤 3(可重写)
    }
}
```

**4. 工厂模式 + 抽象工厂(DIP):**
```java
// DIP:业务依赖抽象工厂,不依赖具体产品
public interface PaymentFactory {
    PaymentProcessor createProcessor(String type);
}
```

### 学习路径建议

1. **先学 SOLID** ——理解"为什么"
2. **再学模式** ——理解"怎么做"
3. **互相印证** ——每学一个模式,问自己:"它体现了哪些 SOLID 原则?"

这样你就不会陷入 "为模式而模式" 的陷阱,而是 **根据问题选择模式,根据原则验证设计**。

---

## 🛠️ SOLID 在主流框架中的应用

### Spring Framework

| SOLID 原则 | Spring 实现 |
|---|---|
| SRP | `@Controller` / `@Service` / `@Repository` 各司其职 |
| OCP | 通过 `BeanPostProcessor`、`FactoryBean` 扩展点 |
| LSP | `BeanFactory` 接口及其多种实现 |
| ISP | Spring Data 的细粒度 `Repository` 子接口 |
| DIP | **IoC 容器 + 依赖注入** —— DIP 的工业级实现 |

### JDK

| SOLID 原则 | JDK 实现 |
|---|---|
| SRP | `java.nio.file.Path` 只表示路径,`Files` 只做操作 |
| OCP | `InputStream` 抽象 + 几十种实现(`FileInputStream` 等) |
| LSP | `List` 接口 + `ArrayList` / `LinkedList` / `CopyOnWriteArrayList` |
| ISP | `Iterable` / `Collection` / `List` / `Set` / `Queue` 分层接口 |
| DIP | JDBC(`DriverManager` 不依赖具体驱动)、SLF4J(日志门面) |

### Python 标准库

| SOLID 原则 | Python 实现 |
|---|---|
| SRP | `urllib.parse` 只解析 URL,`urllib.request` 只发请求 |
| OCP | `collections.abc` 抽象基类,新容器只需实现接口 |
| LSP | `MutableMapping` / `Mapping` 抽象基类 |
| ISP | `Iterator` / `Iterable` / `Generator` 区分 |
| DIP | `abc.ABC` + `Protocol`(结构性子类型) |

---

## ⚖️ SOLID 的权衡与边界

> **"任何原则都不是教条,都是在特定场景下的最优选择"** —— Andy Hunt

### 何时应该应用 SOLID

✅ **中大型项目**(代码量 > 1 万行)
✅ **多人协作团队**(3+ 人长期维护)
✅ **业务规则复杂**(支付、电商、金融等领域)
✅ **系统需要长期演化**(生命周期 > 3 年)

### 何时不应该过度应用 SOLID

❌ **小脚本/工具**(几百行一次性代码)
❌ **原型/概念验证**(快速验证想法)
❌ **性能极敏感的代码**(抽象层有方法调用开销)
❌ **业务稳定不变**(CRUD 表格系统,过度抽象没必要)

### SOLID 不是越多越好

**反模式:原则强迫症**

```java
// ❌ 过度设计:为简单业务堆砌原则
public interface StringWrapper {
    String getValue();
    void setValue(String value);
}
public class StringWrapperImpl implements StringWrapper {
    private String value;
    public String getValue() { return value; }
    public void setValue(String value) { this.value = value; }
}

// ✅ 简单直接
String name = "Alice";
```

**Martin Fowler 的话:**
> "Speculative generality is the enemy of good design."
> (投机性的通用化是优秀设计的敌人。)

**经验法则:**
- 第一次实现某功能 → 按最直观的方式写
- 第二次因为 "加新需求" 而修改时 → 抽象的时机到了
- 第三次重复 → 必须抽取

这叫 "Rule of Three"——既避免过度抽象,也避免重复代码。

---

## 📖 推荐阅读

### 经典著作

1. **《敏捷软件开发:原则、模式与实践》** —— Robert C. Martin
   - SOLID 原则的权威来源
   - 详细阐述每个原则的背景与案例

2. **《设计模式:可复用面向对象软件的基础》(GoF)** —— Erich Gamma 等
   - 23 种经典设计模式
   - SOLID 原则的具体实现

3. **《重构:改善既有代码的设计》** —— Martin Fowler
   - 教你如何把违反 SOLID 的代码重构为符合 SOLID
   - 大量实战技巧

4. **《Clean Code》** —— Robert C. Martin
   - 代码层面的 SOLID 实践
   - 命名、函数、注释、格式等细节

5. **《程序员修炼之道》** —— Andy Hunt & Dave Thomas
   - DRY、正交性、可逆性等原则
   - 实用主义的开发哲学

### 在线资源

- **Robert C. Martin 的博客**: https://blog.cleancoder.com/
- **Martin Fowler 的博客**: https://martinfowler.com/
- **B 站/YouTube SOLID 教程**(搜索 "SOLID Principles OOP")

---

## 🎓 学习检验清单

完成本章学习后,你可以回答以下问题吗?

### SRP
- [ ] 什么是 "变化的原因"? 怎么识别一个类的多个职责?
- [ ] SRP 和 "单一功能" 有什么区别?
- [ ] 什么时候拆分一个类?什么时候保持合并?

### OCP
- [ ] OCP 看似矛盾("对扩展开放、对修改关闭"),如何统一?
- [ ] 什么是 "变化点"?怎么识别需要抽象的业务规则?
- [ ] OCP 和 YAGNI 矛盾吗?如何平衡?

### LSP
- [ ] LSP 关注 "行为" 还是 "类型"?
- [ ] Square extends Rectangle 为什么违反 LSP?
- [ ] 子类能不能引入新的异常?能不能收紧前置条件?

### ISP
- [ ] 什么是 "胖接口"?它的三个危害是什么?
- [ ] 怎么判断一个接口是否需要拆分?
- [ ] ISP 和 "接口要多" 是一回事吗?

### DIP
- [ ] DIP 为什么叫 "倒置"?
- [ ] 抽象到底归属于调用方还是实现方?
- [ ] DIP 和 IoC、DI 是什么关系?

### 其他原则
- [ ] KISS、YAGNI、DRY 各自的关注点是什么?
- [ ] 组合和继承怎么选?有没有判断标准?
- [ ] 什么是 "迪米特法则"?链式调用为什么违反它?

---

## 🔮 下一章预告

设计原则是 **"道"**,设计模式是 **"术"**。

下一章我们将进入 **"第 06 章 - 经典设计模式"**,系统学习 GoF 23 种设计模式:
- 创建型模式(单例、工厂、建造者、原型、抽象工厂)
- 结构型模式(适配器、装饰器、代理、外观、桥接、组合、享元)
- 行为型模式(策略、模板方法、观察者、迭代器、责任链、命令、状态、访问者、中介者、备忘录、解释器)

每个模式都会用 **Python/Java 正反对比 + 实际应用** 的方式讲解,帮助你把 SOLID 原则落地到具体的设计方案中。

**让我们继续前行!** 🚀
