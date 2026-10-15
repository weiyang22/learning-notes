---
title: SRP - 单一职责原则 (Single Responsibility Principle)
tags: [SOLID, 设计原则, SRP, 单一职责, 面向对象]
description: 一个类只应该有一个引起它变化的原因——只能有一个职责
last_updated: 2026-09-11
status: ✅ v1.0
chapter_id: 05-01
---

# SRP - 单一职责原则 (Single Responsibility Principle)

> **"A class should have only one reason to change."**
> —— Robert C. Martin, *Agile Software Development: Principles, Patterns, and Practices* (2003)

单一职责原则是 SOLID 五原则之首,由 Robert C. Martin(绰号 "Uncle Bob")在 2003 年的著作中正式命名。它并非一个新思想——早在 70 年代就有类似表述——但 SRP 把它凝练成了一句可操作、可衡量、可争论的口号,并与后来四个原则(OCP、LSP、ISP、DIP)共同构成了面向对象设计的 "SOLID" 基础。

## 一句话定义

> **一个类(或模块、函数)应该只有一个引起它变化的原因,即只能承担一个职责。**

这里的 "职责" 不是 "方法" 的同义词,而是 **"变化的轴心"**(axis of change)。当你发现一个类会因为多种不同原因而被修改(业务规则变了要改、数据库换了要改、UI 改了要改、报表格式变了要改……),就说明它承担了多个职责,应该被拆分。

## 核心思想

### 1. 为什么是 "职责" 而不是 "功能"?

很多人误以为 SRP 是 "一个类只能有一个方法" 或 "只能做一件事"。这种理解太过狭隘——一个 UserService 自然有 `register`、`login`、`logout`、`updateProfile` 多个方法,但这些方法都围绕 "用户账户管理" 这 **一个职责** 展开,所以它们属于同一个类。

判断标准不在于方法数量,而在于 **"谁会要求我修改这个类"**。如果 A 业务方改了要求要改这个类,B 业务方改了要求也要改这个类,那 A 和 B 的需求就是两个独立的 "变化原因",就应该拆分。这就是 Martin 说的 "one reason to change" 的本质含义。

### 2. 职责的内聚性

SRP 在底层其实是 **内聚性原则**(cohesion)的极端体现。软件工程里讲的 "高内聚、低耦合",高内聚的极致就是 "类内所有元素都只为同一个职责服务"。一个类如果同时承担持久化、业务逻辑、UI 渲染、报表生成这四个职责,它的内聚度其实非常低——这些代码只是因为 "都跟用户有关" 这种弱关联被硬塞在一起,一旦任何一项需求变化,整个类都要重新测试。

### 3. 类的粒度与 SRP 的边界

SRP 不是鼓励无限拆分。把 `for` 循环里的两行代码拆成两个类就违背了 KISS 原则。SRP 的颗粒度应该匹配 **变化的频率和原因**——如果两段代码永远因为同样的原因一起变化,放在同一个类里反而更内聚。判断标准是:
- 这两段代码是否会被 **不同的 stakeholder**(产品经理、运维、安全审计……)要求修改?
- 修改其中一段时,是否需要确保另一段 **不会受影响**?
- 这两段代码能不能 **独立部署、独立测试**?

如果三个答案都是 "否",那它们就该在一起;如果答案是 "是",那 SRP 要求你拆分。

### 4. SRP 与微服务

有趣的是,SRP 的思想在架构层面被推到了极致——**康威定律与微服务**。一个微服务就是一个 "只有一个变化原因" 的部署单元。当我们说 "订单服务只管订单、库存服务只管库存" 时,其实就是把 SRP 应用到了服务粒度。所以你完全可以把 SRP 看作是从类到模块到服务到子系统都适用的一条统一原则,只是颗粒度不同。

## Python/Java 正反对比

### ❌ 反例:一个 God Class 承担 4 个职责

```java
// God Class - 典型的 "User" 类,承担了太多职责
public class User {
    private Long id;
    private String username;
    private String passwordHash;
    private String email;
    private LocalDateTime createdAt;

    // ===== 职责 1: 数据载体 (DTO/Entity) =====
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    // ... 几十个 getter/setter

    // ===== 职责 2: 认证逻辑 =====
    public boolean authenticate(String rawPassword) {
        return BCrypt.checkpw(rawPassword, this.passwordHash);
    }

    public String generatePasswordResetToken() {
        return UUID.randomUUID().toString();
    }

    public void changePassword(String oldPwd, String newPwd) {
        if (!authenticate(oldPwd)) throw new IllegalArgumentException("密码错误");
        this.passwordHash = BCrypt.hashpw(newPwd, BCrypt.gensalt());
    }

    // ===== 职责 3: 持久化 =====
    public void save() {
        try (Connection conn = DriverManager.getConnection(DB_URL, USER, PWD)) {
            PreparedStatement ps = conn.prepareStatement(
                "INSERT INTO users (id, username, password_hash, email) VALUES (?,?,?,?)");
            ps.setLong(1, id);
            ps.setString(2, username);
            ps.setString(3, passwordHash);
            ps.setString(4, email);
            ps.executeUpdate();
        } catch (SQLException e) {
            throw new RuntimeException("保存失败", e);
        }
    }

    public static User findById(Long id) {
        // JDBC 查询代码
    }

    // ===== 职责 4: 报表生成 =====
    public String generateProfileReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("用户: ").append(username).append("\n");
        sb.append("邮箱: ").append(email).append("\n");
        sb.append("注册时间: ").append(createdAt).append("\n");
        // ... 各种格式拼接
        return sb.toString();
    }

    public byte[] exportToPdf() {
        // iText/PDF 库依赖
    }
}
```

**问题分析:**
- 4 个完全不同的变化原因:密码算法升级要改、数据库迁移要改、报表格式调整要改、字段增减要改
- 这个类不能单元测试认证而不连数据库(持久化职责污染了认证逻辑)
- 任何修改都要重新部署整个类,影响所有 4 个职责的使用者
- 团队多人协作冲突严重——前端要改 getter/setter,后端要改 SQL,报表组要改 PDF,改的全是同一个文件

### ✅ 正例:按职责拆分

```java
// ===== 职责 1: 数据载体 (贫血模型) =====
public class User {
    private final Long id;
    private final String username;
    private final String email;
    private final LocalDateTime createdAt;

    public User(Long id, String username, String email, LocalDateTime createdAt) {
        this.id = id;
        this.username = username;
        this.email = email;
        this.createdAt = createdAt;
    }

    // 只有纯粹的 getters,无业务逻辑
    public Long getId() { return id; }
    public String getUsername() { return username; }
    public String getEmail() { return email; }
    public LocalDateTime getCreatedAt() { return createdAt; }
}

// ===== 职责 2: 认证服务 =====
@Service
public class UserAuthService {
    private final PasswordEncoder passwordEncoder;

    public UserAuthService(PasswordEncoder passwordEncoder) {
        this.passwordEncoder = passwordEncoder;
    }

    public boolean verifyPassword(User user, String rawPassword) {
        return passwordEncoder.matches(rawPassword, user.getPasswordHash());
    }

    public String generateResetToken(User user) {
        return UUID.randomUUID().toString();
    }

    public void changePassword(User user, String oldPwd, String newPwd) {
        if (!verifyPassword(user, oldPwd)) {
            throw new IllegalArgumentException("密码错误");
        }
        user.setPasswordHash(passwordEncoder.encode(newPwd));
    }
}

// ===== 职责 3: 仓储 (持久化) =====
@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    Optional<User> findByUsername(String username);
    Optional<User> findByEmail(String email);
}

// ===== 职责 4: 报表生成 =====
@Service
public class UserReportGenerator {
    public String generateProfileReport(User user) {
        return String.format(
            "用户: %s\n邮箱: %s\n注册时间: %s\n",
            user.getUsername(), user.getEmail(), user.getCreatedAt()
        );
    }

    public byte[] exportToPdf(User user) {
        // 调用 PDF 库,只关心这个职责
    }
}
```

**对比收益:**
- 每个类只有一个变化原因:密码算法升级只改 `UserAuthService`,数据库迁移只改 `UserRepository`,报表格式调整只改 `UserReportGenerator`
- 单元测试不再需要数据库:`UserAuthService` 可以用 mock `UserRepository` 单独测试
- 并行开发无冲突:4 个开发者改 4 个不同的文件
- 每个类的依赖收敛:`UserReportGenerator` 只依赖 PDF 库,不需要拖上 JDBC 驱动

### Python 对比

```python
# ❌ 反例
class User:
    def __init__(self, username, password_hash, email):
        self.username = username
        self.password_hash = password_hash
        self.email = email

    def verify_password(self, raw_password):
        return bcrypt.checkpw(raw_password.encode(), self.password_hash.encode())

    def save(self):
        conn = sqlite3.connect('app.db')
        conn.execute("INSERT INTO users VALUES (?, ?, ?)",
                     (self.username, self.password_hash, self.email))
        conn.commit()
        conn.close()

    def to_report(self):
        return f"User: {self.username}\nEmail: {self.email}"


# ✅ 正例
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class User:
    """纯数据载体"""
    id: int
    username: str
    email: str
    created_at: datetime


class UserAuthService:
    def __init__(self, password_encoder):
        self._encoder = password_encoder

    def verify_password(self, user: User, raw_password: str) -> bool:
        return self._encoder.matches(raw_password, user.password_hash)


class UserRepository(ABC):
    @abstractmethod
    def save(self, user: User) -> None: ...

    @abstractmethod
    def find_by_id(self, user_id: int) -> User | None: ...


class UserReportGenerator:
    def generate_profile_report(self, user: User) -> str:
        return f"User: {user.username}\nEmail: {user.email}"
```

## 实际应用

### 1. Spring 框架的注解分层

Spring 的注解 `@Controller`、`@Service`、`@Repository`、`@Component` 本身就在 **倡导 SRP**:
- `@Controller`:负责接收 HTTP 请求、参数校验、返回视图——只管 "Web 层"
- `@Service`:负责业务编排、事务管理——只管 "业务逻辑层"
- `@Repository`:负责数据访问、SQL 执行——只管 "持久化层"
- `@Component`:其他通用组件

一个常见反模式是把 SQL 写在 `@Service` 里(`userService.save()` 内嵌 JDBC 代码),这就让一个类同时承担了业务和持久化两个职责。

### 2. JDK 的 `java.nio.file.Files` vs `java.io.File`

`java.io.File` 类是个 **违反 SRP 的经典例子**:它既表示文件路径、又能执行 IO 操作、还能被序列化、还能比较路径……Java 7 引入 `java.nio.file.Files` 和 `java.nio.file.Path` 时,明确把 "路径表示" 和 "文件操作" 拆开了。`Path` 只表示路径,`Files` 只做静态 IO 操作,这就是 SRP 在 JDK 中的体现。

### 3. Spring Boot 的 `CommandLineRunner` 与 `ApplicationRunner`

如果你需要启动时执行一段逻辑,Spring 提供了两个接口;但每个 `Runner` 实现类 **只应该承担一个启动任务**。如果你把所有初始化逻辑塞进一个 `InitAllRunner`,那它就违背了 SRP——数据库初始化、缓存预热、消息订阅注册应该分开。

### 4. Java 8 的 `Function` / `Predicate` / `Supplier`

`java.util.function` 包下的这几个函数式接口,本身就是 SRP 的极致——`Function` 只负责 "输入到输出" 的转换,`Predicate` 只负责 "判断真假",`Supplier` 只负责 "提供值"。每个接口只承担一个语义角色。

## 违反原则的危害

### 场景 1:支付系统重构事故

某电商的 `OrderService` 类同时承担了:下单逻辑、库存扣减、支付调用、物流通知、发票生成、报表统计。某天业务方要求增加 "满 100 减 20" 的优惠规则,开发改了 `OrderService`,由于所有逻辑耦合在一起,这次改动意外触发了支付调用时机的 bug,导致部分订单 **重复扣款**。事后复盘发现,根源就是这个 God Class——它有 **6 个变化原因**,任何一次修改都是高风险。

### 场景 2:测试瘫痪

`UserService` 类内部直接 `new UserDao()`,而 `UserDao` 又直接 `new DataSource()`。结果测试 `UserService` 的 "用户名校验" 逻辑时,必须启动整个 Spring 容器、初始化连接池、准备测试数据库。一个简单单元测试需要 30 秒。**因为职责没分开,测试无法独立进行**——这是 SRP 违反对工程效率最直接的影响。

### 场景 3:并行开发冲突

团队 5 个人同时改 `PaymentService`,因为这个类里有支付宝、微信、银联三种支付方式(三种变化原因!)。Git 冲突频繁,合并成本极高。最后拆成 `AlipayService`、`WechatPayService`、`UnionPayService` 三个类后,并行开发恢复正常。

### 场景 4:微服务边界的混乱

把订单创建、库存扣减、支付回调、物流调度全塞进 `OrderService`,本质上是 **把多个微服务塞进了一个进程**。一旦库存子系统要独立扩容,发现它和订单逻辑强耦合,无法拆分。这就是 "在类层面违反 SRP" 在架构层面的放大版。

## 判断标准

### 何时应该应用 SRP(拆分)

✅ **一个类被多个 stakeholder 修改** —— 比如业务方、运维、安全审计会要求改同一个类
✅ **修改一个功能时,担心影响其他功能** —— 出现 "改一处坏一片" 的现象
✅ **类的导入列表非常长** —— 依赖了数据库、UI、PDF、HTTP 多种类库,说明职责混乱
✅ **类无法独立测试** —— 必须启动 Spring、连数据库、起 HTTP 才能测试
✅ **类难以命名** —— 名字中出现 "And"、"Or"、"Manager"、"Helper"、"Utils" 这类万能词
✅ **类的代码量过大** —— 一般超过 500 行就值得审视是否需要拆分

### 何时不应该过度拆分

❌ **为了拆分而拆分** —— 引入 5 个只有 3 行代码的类,反而增加阅读成本
❌ **稳定不变的代码** —— 几十年不会改的 enum、常量类,拆不拆无所谓
❌ **业务上必然耦合的概念** —— `Money` 类同时有金额和币种,这是不可拆分的内聚概念
❌ **性能敏感的紧耦合代码** —— 某些高频调用路径上,过度拆分类会增加方法调用开销

**经验法则:** 当你不确定要不要拆时,先用一段时间(几天/几周),等真正出现 "多个变化原因" 时再拆。过早拆分同样有害——它会让你陷入 "为了抽象而抽象" 的泥潭。

## 与其他原则的关系

### 与 OCP 的关系
OCP 要求 "对扩展开放、对修改关闭"。如果你有一个 God Class,那 OCP 几乎无法实现——任何扩展都要修改这个类。SRP 是 OCP 的 **前提**:只有职责清晰了,才能在不修改现有代码的前提下扩展新职责。

### 与 DIP 的关系
DIP 要求 "依赖抽象,不依赖具体"。当一个类承担多个职责时,它通常会直接依赖多个具体实现(`new MySQLConnection()`、`new SmtpEmailSender()`……)。拆分职责后,每个类只依赖与自己职责相关的抽象,DIP 更容易落地。

### 与 ISP 的关系
ISP 要求 "接口要小而精"。SRP 关注的是 **类** 的单一职责,ISP 关注的是 **接口** 的单一职责。两者是同一思想在不同抽象层次的应用——一个在类层面,一个在接口层面。

### 协同与权衡
SRP 不应该被推到极端。如果拆得过细,会产生 "类爆炸"(class explosion),反而损害可读性。**"适度 SRP"** 才是工程智慧——拆到 "每个类有一个清晰的、可以讲出故事的职责" 即可,不要拆到 "每个方法一个类" 的程度。

## 实战思考

### Q1:贫血模型(Anemic Domain Model)算违反 SRP 吗?

**答:** 严格说不算违反,但要警惕。贫血模型把 "数据" 和 "行为" 分离到不同的类——`User` 只有 getter/setter,所有行为在 `UserService` 里。DDD 派认为这违反 OOP,SRP 派认为这反而符合 SRP(数据类一个职责,服务类一个职责)。Martin 本人对此有过澄清:**"SRP 不要求把数据和行为放在一起,它要求的是变化原因唯一"**。贫血模型只要保证 `User` 和 `UserService` 各自只有一个变化原因,就不算违反 SRP。但如果 `UserService` 自己又同时管业务又管持久化,那就违反了。

### Q2:`Utils`、`Helper`、`Manager` 这种万能类要不要拆?

**答:** 一般建议拆。"Utils" 这种命名本身就意味着 "把杂七杂八的东西塞一起",几乎必然违反 SRP。**拆分的判断标准是:这个类里有没有两个方法会因为不同的原因被修改?** 如果 `StringUtils` 里的 `isEmpty()` 和 `formatJson()` 永远会一起被改,放一起没问题;但如果 `UserUtils` 里同时有用户名校验、密码加密、头像上传,那必须拆。**命名是信号**——当你发现自己想把一个类叫 `XxxManager` 时,先停下来想想是不是该拆成多个职责明确的类。

### Q3:重构一个存量 God Class 时,从哪里下手?

**答:** 经典做法是 **"Extract Class"(提炼类)** 重构手法:
1. 先列出 God Class 所有的 public 方法,按 "它们服务的对象" 分组(用户、订单、报表……)
2. 找到一组有内聚关系的方法(它们经常一起被调用、操作同一组字段),把它们连带相关字段一起抽到新类
3. 把调用方对老方法的引用改为对新类的引用
4. 跑测试,确保行为不变
5. 重复直到 God Class 瘦身到合理大小

**Martin Fowler 在《重构》第 3 版里描述这个手法时,特别强调 "先写测试,再做重构"**——没有测试覆盖的 God Class,重构风险极高。Spring 的 `@Service` + `@Autowired` 让这种重构相对安全,因为依赖注入会自动帮你切换引用。

## 小结

SRP 是 SOLID 中最基础、最容易理解、但也最容易被忽视的原则。它的核心不是 "一个类只做一个方法",而是 **"一个类只服务于一个变化原因"**。当你发现自己的类因为业务、数据库、UI、报表……多种原因被修改时,就该拆分了。

SRP 的红利不在于短期——拆分类确实会带来更多文件、更多依赖注入的样板代码。但长期看,它带来的是 **可测试性、可维护性、并行开发能力、故障隔离能力** 的全面提升。一个项目是否健康,看 God Class 的数量就大致知道。

记住 Uncle Bob 的话:**"Gather together the things that change for the same reasons. Separate those things that change for different reasons."**(把因相同原因而变化的事物聚合在一起,把因不同原因而变化的事物分离开。) 这是 SRP 的灵魂,也是整个 SOLID 思想的起点。

下一章我们将学习 **OCP(开闭原则)**——它告诉你怎样在 SRP 拆分好的基础上,做到 "扩展新功能时不动旧代码"。
