---
title: DIP - 依赖倒置原则 (Dependency Inversion Principle)
tags: [SOLID, 设计原则, DIP, 依赖倒置, IoC, 依赖注入]
description: 高层模块不应依赖低层模块,二者都应依赖抽象;抽象不应依赖细节,细节应依赖抽象
last_updated: 2026-09-11
status: ✅ v1.0
chapter_id: 05-05
---

# DIP - 依赖倒置原则 (Dependency Inversion Principle)

> **"A. High-level modules should not depend on low-level modules. Both should depend on abstractions.**
> **B. Abstractions should not depend on details. Details should depend on abstractions."**
> —— Robert C. Martin, *C++ Report* (1996), 后收录于《敏捷软件开发》

依赖倒置原则是 SOLID 的 **压轴原则**,也是连接前面四个原则的 **枢纽**。如果说 SRP 是基础、OCP 是目标、LSP 是保证、ISP 是策略,那 DIP 就是 **实现机制**——它告诉你 **"怎么用代码让前面四个原则真正落地"**。Spring 框架的 IoC 容器,本质就是 DIP 的工业级实现。

## 一句话定义

> **高层模块不应该依赖低层模块,二者都应该依赖于抽象;抽象不应该依赖于细节,细节应该依赖于抽象。**

DIP 包含两个子原则:
- **A**:高层 ↔ 低层,都依赖抽象(而不是高层依赖低层)
- **B**:抽象 ↔ 细节,细节依赖抽象(而不是抽象依赖细节)

注意 "倒置" 二字——传统依赖关系是 "上层依赖下层",DIP 把它 **倒过来**:"上层和下层都依赖中间的抽象层"。这就是 "依赖倒置" 的命名由来。

## 核心思想

### 1. 什么是 "高层" 和 "低层"?

- **高层模块**:业务逻辑层,代表系统的核心策略(如 `OrderService` 处理订单业务)
- **低层模块**:实现细节层,代表具体的实现机制(如 `MySQLOrderRepository` 把订单存进 MySQL)

传统的过程式设计中,高层直接调用低层:`OrderService` 直接 `new MySQLOrderRepository()`。这种设计的致命问题是:**业务逻辑被持久化细节绑架**——你想换成 PostgreSQL?改业务代码。你想做单元测试?必须连真实数据库。

DIP 主张:**在高层和低层之间引入抽象层(`OrderRepository` 接口),让高层依赖接口、低层实现接口**。这样:
- 高层(`OrderService` 业务逻辑)只关心接口,不关心 MySQL 还是 PostgreSQL
- 低层(`MySQLOrderRepository`、`PostgreSQLOrderRepository`)只关心实现接口,不关心业务
- 两者通过抽象层解耦,可以独立演化、独立测试、独立替换

### 2. 为什么叫 "倒置"?

```
传统依赖图:
    [OrderService]  ──depends on──>  [MySQLOrderRepository]

DIP 依赖图:
    [OrderService]  ──depends on──>  [OrderRepository]  <──implements──  [MySQLOrderRepository]
```

传统关系是 **单向的"上层依赖下层"**。DIP 引入抽象层后,**"上层和下层都指向抽象"**,从图上看就像把箭头方向 **倒过来** 了。这就是 "倒置" 的含义。

更精妙的是:**抽象层(`OrderRepository` 接口)的位置通常归属于高层模块所在的包**(在 Java 里,接口的所有权属于调用方,而不是实现方)。这是 OO 设计的 "好莱坞法则":**"Don't call us, we'll call you"**——抽象层定义 "我们想要什么",低层模块负责 "实现我们想要的"。

### 3. DIP 和 IoC(控制反转)的关系

**DIP 是原则,IoC 是模式**。IoC(Inversion of Control)是 DIP 的常见实现方式——通过 **控制反转容器**(Spring IoC、Guice、Dagger)把依赖的创建和管理从类内部移到外部容器,类只声明自己需要什么依赖,由容器负责注入。

IoC 又包含 **DI(Dependency Injection, 依赖注入)** 和 **Service Locator(服务定位器)** 两种主要实现:
- **DI**:构造函数注入、Setter 注入、字段注入
- **Service Locator**:`ServiceLocator.get(OrderRepository.class)` 主动查找

DI 是更现代、更可测的实践,Spring 从 2.5 开始全面拥抱 DI。

### 4. DIP 不只是 "换种方式 new"

很多人误以为 DIP 就是 "不用 new,改成注入"。这是浅层的理解。DIP 的深层含义是 **"让抽象归属于调用方"**:
- 错误的做法:`MySQLOrderRepository` 自己定义接口 `OrderRepository`,业务层依赖这个 "由实现方定义的" 接口
- 正确的做法:`OrderService` 自己定义接口 `OrderRepository`,`MySQLOrderRepository` 实现这个 "由调用方定义的" 接口

这种归属关系决定了接口的 **设计视角**——调用方定义的接口关注 "我要怎么用",实现方定义的接口关注 "我能提供什么"。DIP 要求前者,因为调用方才是真正理解业务需求的角色。

## Python/Java 正反对比

### ❌ 反例:高层直接依赖低层细节

```java
// 低层模块:具体的数据库实现
public class MySQLDatabase {
    public void connect(String url, String username, String password) {
        // 建立 MySQL 连接
    }

    public List<Map<String, Object>> query(String sql) {
        // 执行 SQL 查询,返回结果
        return new ArrayList<>();
    }

    public int update(String sql) {
        // 执行 SQL 更新
        return 0;
    }

    public void close() {
        // 关闭连接
    }
}

// 高层模块:业务逻辑
public class BusinessReportService {
    private MySQLDatabase database;

    public BusinessReportService() {
        this.database = new MySQLDatabase(); // ❌ 直接依赖具体实现
        this.database.connect("jdbc:mysql://localhost:3306/app", "root", "password");
    }

    public List<User> getActiveUsers() {
        // ❌ 业务代码里混入了 SQL 字符串!
        List<Map<String, Object>> rows = database.query(
            "SELECT id, username, email FROM users WHERE status = 'ACTIVE'"
        );

        // ❌ 还要手动把 Map 转成 User 对象
        List<User> users = new ArrayList<>();
        for (Map<String, Object> row : rows) {
            User u = new User();
            u.setId((Long) row.get("id"));
            u.setUsername((String) row.get("username"));
            u.setEmail((String) row.get("email"));
            users.add(u);
        }
        return users;
    }

    public byte[] exportToCsv() {
        // 想换成 PostgreSQL?改这里。
        // 想支持单元测试?必须能 mock MySQLDatabase——但它没有接口。
    }
}
```

**问题:**
- `BusinessReportService` 被死死绑定在 MySQL 上,换数据库要改业务代码
- 测试时必须启动真实 MySQL,无法单元测试业务逻辑
- SQL 字符串和业务转换代码混在业务层,职责混乱(SRP 也违反了)
- 想加缓存层?必须在 `BusinessReportService` 里手动判断,没有扩展点

### ✅ 正例:抽象归属调用方,细节依赖抽象

```java
// ===== 抽象层:由调用方(业务层)定义 =====
public interface UserRepository {
    Optional<User> findById(Long id);
    List<User> findActiveUsers();
    void save(User user);
    void delete(Long id);
}

// ===== 高层模块:业务逻辑,只依赖抽象 =====
@Service
public class UserService {
    private final UserRepository userRepository; // 依赖抽象
    private final NotificationService notifier;   // 依赖抽象

    // ✅ 构造函数注入(DI 的标准实践)
    public UserService(UserRepository userRepository, NotificationService notifier) {
        this.userRepository = userRepository;
        this.notifier = notifier;
    }

    public void registerUser(RegisterDto dto) {
        User user = new User(dto.getUsername(), dto.getEmail());
        userRepository.save(user); // 不关心 MySQL 还是 PostgreSQL
        notifier.sendWelcomeEmail(user);
    }

    public List<User> listActiveUsers() {
        return userRepository.findActiveUsers(); // 简洁的业务逻辑
    }
}

// ===== 低层模块:MySQL 实现,依赖抽象 =====
@Repository
public class MySQLUserRepository implements UserRepository {
    private final JdbcTemplate jdbc;

    public MySQLUserRepository(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    @Override
    public Optional<User> findById(Long id) {
        return jdbc.query(
            "SELECT id, username, email FROM users WHERE id = ?",
            new Object[]{id},
            (rs, rowNum) -> new User(rs.getLong("id"), rs.getString("username"), rs.getString("email"))
        ).stream().findFirst();
    }

    @Override
    public List<User> findActiveUsers() {
        return jdbc.query(
            "SELECT id, username, email FROM users WHERE status = 'ACTIVE'",
            (rs, rowNum) -> new User(rs.getLong("id"), rs.getString("username"), rs.getString("email"))
        );
    }

    // ...
}

// ===== 低层模块:PostgreSQL 实现,同样依赖抽象 =====
@Repository
public class PostgreSQLUserRepository implements UserRepository {
    // 类似实现,但用 PostgreSQL 特有的 SQL 语法
}

// ===== 另一个低层模块:内存实现(用于测试) =====
@Repository
@Profile("test")
public class InMemoryUserRepository implements UserRepository {
    private final Map<Long, User> store = new ConcurrentHashMap<>();

    @Override
    public Optional<User> findById(Long id) {
        return Optional.ofNullable(store.get(id));
    }
    // ...
}

// ===== 调用方:高层只管业务 =====
@RestController
public class UserController {
    private final UserService userService; // 间接依赖 UserRepository 抽象

    public UserController(UserService userService) {
        this.userService = userService;
    }

    @GetMapping("/active-users")
    public List<User> activeUsers() {
        return userService.listActiveUsers();
    }
}
```

**对比收益:**
- 高层 `UserService` 不依赖任何具体数据库——**换 PostgreSQL 不需要改业务代码一行**
- 单元测试时直接 mock `UserRepository` 接口,不需要启动数据库
- 新增 `MongoUserRepository`、`RedisUserRepository` 是 **扩展而非修改**——OCP 也满足了
- 抽象层 `UserRepository` 的归属清晰:**业务层定义接口,基础设施层实现接口**

### Python 对比

```python
# ❌ 反例
class MySQLDatabase:
    def __init__(self):
        self.connection = pymysql.connect(host='localhost', user='root', ...)

    def query(self, sql):
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
            return cursor.fetchall()


class UserService:
    def __init__(self):
        self.db = MySQLDatabase()  # ❌ 直接依赖具体实现

    def get_active_users(self):
        rows = self.db.query("SELECT * FROM users WHERE status='ACTIVE'")
        return [self._map_row(r) for r in rows]


# ✅ 正例
from abc import ABC, abstractmethod
from typing import Protocol


# 抽象层:由调用方定义(Protocol 是 Python 的结构性子类型)
class UserRepository(Protocol):
    def find_active_users(self) -> list[User]: ...
    def save(self, user: User) -> None: ...


class UserService:
    def __init__(self, user_repo: UserRepository):
        self._repo = user_repo  # 依赖抽象

    def get_active_users(self) -> list[User]:
        return self._repo.find_active_users()


# 低层实现
class MySQLUserRepository:
    def find_active_users(self) -> list[User]:
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE status='ACTIVE'")
            return [self._map_row(r) for r in cursor.fetchall()]
```

## 实际应用

### 1. Spring IoC:DIP 的工业级实现

Spring 框架的核心就是 DIP 的工业化:

```java
// 不需要 new,不需要手动管理依赖
@Service
public class OrderService {
    private final OrderRepository orderRepository;
    private final PaymentService paymentService;
    private final NotificationService notificationService;

    // ✅ 构造函数注入(推荐)
    public OrderService(OrderRepository orderRepository,
                        PaymentService paymentService,
                        NotificationService notificationService) {
        this.orderRepository = orderRepository;
        this.paymentService = paymentService;
        this.notificationService = notificationService;
    }

    @Transactional
    public Order createOrder(CreateOrderDto dto) {
        Order order = new Order(dto);
        orderRepository.save(order);
        paymentService.charge(order);
        notificationService.sendOrderConfirmation(order);
        return order;
    }
}
```

Spring 容器在启动时:
1. 扫描 `@Service`、`@Repository` 等注解
2. 创建所有 Bean 实例
3. 根据类型(和可选的 `@Qualifier`、`@Autowired`)自动注入依赖
4. 管理 Bean 的生命周期(`@PostConstruct`、`@PreDestroy`)

整个过程,`OrderService` 完全不知道自己的依赖是怎么创建的、在哪里——它只声明 "我需要这些东西",由容器负责满足。这就是 DIP 的完整实践。

### 2. Java JDBC:DriverManager 是 DIP 的早期实现

```java
// 高层:业务代码,只依赖 JDBC 抽象 API
public List<User> queryUsers(Connection conn) throws SQLException {
    PreparedStatement ps = conn.prepareStatement("SELECT * FROM users");
    ResultSet rs = ps.executeQuery();
    // ...
}

// DriverManager 不依赖任何具体数据库驱动
// 它通过 Class.forName 动态加载驱动
Connection conn = DriverManager.getConnection(
    "jdbc:mysql://localhost:3306/app",  // URL 决定用哪个驱动
    "user", "password"
);
```

JDBC 的 `DriverManager` 是 **"抽象属于调用方"** 的早期范例——Sun 公司定义了 JDBC 接口,各家数据库厂商(MySQL、PostgreSQL、Oracle)实现这些接口。业务代码只依赖 JDBC 接口,完全不依赖任何具体驱动。换数据库?改 URL 就行,业务代码不动。

### 3. SLF4J + Logback/Log4j:日志门面

```java
public class OrderService {
    private static final Logger log = LoggerFactory.getLogger(OrderService.class);
    // 依赖 SLF4J 抽象,不依赖 Logback 还是 Log4j

    public void createOrder(CreateOrderDto dto) {
        log.info("创建订单: {}", dto); // 业务代码不需要知道底层日志框架
    }
}
```

日志门面 SLF4J 定义抽象,Logback/Log4j/Log4j2 实现抽象。业务代码只 import `org.slf4j.Logger`,完全不关心底层实现。**想从 Logback 换到 Log4j2?** 改 pom.xml 依赖,业务代码一行不改。

### 4. Java EE / Jakarta EE 的依赖注入

CDI(Contexts and Dependency Injection)是 Java EE 标准的依赖注入规范:

```java
@ApplicationScoped
public class OrderService {
    @Inject
    private OrderRepository orderRepository;

    @Inject
    private PaymentService paymentService;
}
```

CDI 容器(Weld、OpenWebBeans)负责管理 Bean 生命周期和依赖注入,和 Spring IoC 类似,是 DIP 的另一种工业实现。

## 违反原则的危害

### 场景 1:换数据库变成换业务代码

某电商创业初期选了 MySQL,业务代码里到处都是 `new MySQLConnection()`、`MySQL jdbcTemplate.query(...)`。3 年后用户量增长,需要分库分表,MySQL 不够用了,要换 TiDB。但业务代码 30% 直接调用了 MySQL 特有的语法,改造工作量巨大。**根源就是早期没遵循 DIP**——业务层和基础设施层强耦合,数据库的更换变成了整个系统的重构。

### 场景 2:单元测试瘫痪

`PaymentService` 直接 `new AlipayClient()`(支付宝 SDK 的具体类),没有抽象层。测试 `PaymentService.charge()` 时,必须真的调用支付宝沙箱环境,或者手动 mock `AlipayClient` 的所有方法。但因为没有接口,只能 mock 具体类(用 PowerMock 之类的工具),测试代码冗长脆弱。**有抽象层的话,业务测试只需要 mock 接口,5 行代码搞定**。

### 场景 3:多团队并行开发受阻

`OrderService` 直接依赖 `UserService`、`InventoryService`、`PaymentService`、`LogisticsService` 的具体类。A 团队改 `UserService` 的内部实现,B 团队发现 `OrderService` 编译失败——因为 `UserService` 改了返回类型,A 团队的改动影响了 B 团队的工作。**有抽象接口的话,A 团队改实现不影响接口签名,B 团队完全不受影响**。

### 场景 4:技术债务的指数级累积

某项目初期为了 "快速上线",所有 Service 直接依赖具体 Repository,所有 Controller 直接依赖具体 Service。1 年后想引入缓存层?必须在 50 个地方手动加 `if (cache.has(key)) { ... } else { ... }`。**有了 DIP,在 Repository 之上加一层 `CachingUserRepository` 装饰器就能透明地加缓存,业务代码不动**。

## 判断标准

### 何时应该应用 DIP(引入抽象)

✅ **存在多种实现** —— 数据库、消息队列、缓存、支付渠道……这些天然有多种选择
✅ **需要单元测试** —— 任何想要单元测试的代码,都必须能 mock 掉外部依赖
✅ **业务规则可能变化** —— 业务逻辑中 "外部世界怎么连接" 的部分应该被抽象
✅ **多团队并行** —— 接口作为契约,让多个团队可以独立开发互不阻塞
✅ **模块化部署** —— 微服务之间通过 API(接口)而非实现类通信

### 何时不应该过度抽象

❌ **只有一种实现,且未来 5 年不会有第二种** —— 抽象一个永远只有一种实现的接口是浪费
❌ **追求性能极致的紧耦合代码** —— 某些 hot path 上,抽象层的方法调用开销不可接受
❌ **小型脚本/工具** —— 几百行的脚本,不需要 IoC 容器
❌ **过度依赖注入** —— 把 `String` 这种基础类型也抽象成接口是过度设计

**经验法则:** **"Rule of Three"** —— 当你第三次需要换某种实现时,就一定要抽象。第一次直接写,第二次发现重复,第三次就该抽接口了。

### 注入方式的选择

| 注入方式 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|
| **构造函数注入** | 强制依赖、不可变 | 不可变、易测试、明确依赖 | 参数过多时构造函数臃肿 |
| **Setter 注入** | 可选依赖 | 灵活 | 可能忘记注入,运行时 NPE |
| **字段注入** | 简单场景 | 简洁 | 不可测试、隐藏依赖 |
| **方法注入** | 罕见的特殊场景 | 灵活 | 不常见 |

**最佳实践:** 优先构造函数注入(显式、不可变、易测试);Setter 注入用于可选依赖;避免字段注入(除了 Spring 测试场景)。

## 与其他原则的关系

### DIP 是 OCP 的实现机制
OCP 要求 "对扩展开放、对修改关闭",但如何实现?答案就是 DIP:**通过抽象层(接口)解耦,新功能通过新增实现类加入,而非修改现有代码**。OCP 是目标,DIP 是手段。**没有 DIP 的 OCP 是空谈**——如果你直接 `new MySQLOrderRepository()`,再谈 "加 PostgreSQL 实现不动业务代码" 就是纸上谈兵。

### DIP 是 LSP 的应用场景
LSP 要求 "子类型可替换父类型",DIP 中的抽象层(接口)正是 LSP 的载体。`MySQLUserRepository`、`PostgreSQLUserRepository` 都是 `UserRepository` 的合法子类型,业务代码可以透明替换——这就是 DIP + LSP 的组合威力。

### DIP 与 ISP 互相成就
ISP 要求 "接口小而精",DIP 要求 "依赖抽象"。**两者结合就是 "依赖小而精的抽象"**。如果抽象是个胖接口,DIP 让业务代码被迫依赖所有方法,违背 ISP 初衷。接口拆分小,依赖也更精准。

### DIP 让 SRP 更容易落地
DIP 让业务层不依赖基础设施层(SRP 的体现)。**业务层只关心业务逻辑,不关心数据库怎么连、消息怎么发**——这是 SRP 在架构层面的体现。

### 五大原则的融合
SOLID 五原则不是孤立的,而是一个 **有机的整体**:
- **SRP** 是基础(拆分类)
- **OCP** 是目标(可扩展)
- **LSP** 是保证(替换安全)
- **ISP** 是策略(接口设计)
- **DIP** 是机制(依赖抽象)

**DIP 是连接四者的纽带**——它提供了具体的代码组织方式(抽象归属调用方),让前面四个原则真正落地。

## 实战思考

### Q1:抽象到底归属于调用方还是实现方?

**答:** DIP 明确要求 **抽象归属于调用方**。这是 OO 设计的一个关键洞察:**接口应该是调用方的 "需求",而不是实现方的 "能力"**。

```java
// ❌ 反例:实现方定义接口
// 在 mysql-repository 模块里
public interface UserRepository {
    void executeMysqlQuery(String sql); // MySQL 视角
    Connection getMysqlConnection();
}

// ✅ 正例:调用方定义接口
// 在 business 模块里
public interface UserRepository {
    Optional<User> findById(Long id); // 业务视角
    List<User> findActiveUsers();
}
```

这不只是代码风格问题,而是 **设计哲学问题**。调用方定义的接口反映 "我想怎么用业务",实现方定义的接口反映 "我能提供什么技术能力"。前者更稳定(业务相对稳定),后者易变(技术栈常换)。所以抽象归属于调用方,业务才不会被技术绑架。

### Q2:依赖注入容器是不是 DIP 的必备条件?

**答:** 不是,DIP 是 **设计原则**,IoC 容器(DI 框架)是 **实现工具**。你可以完全不用 Spring/Guice,纯手写 DI:

```java
// 纯手写 DI(适合小型项目)
public class Application {
    public static void main(String[] args) {
        // 手动组装依赖
        UserRepository repo = new MySQLUserRepository(new JdbcTemplate(...));
        NotificationService notifier = new EmailNotificationService(new SmtpClient(...));
        UserService service = new UserService(repo, notifier);
        UserController controller = new UserController(service);
        // 启动 web 框架,把 controller 注册进去
    }
}
```

这就是 "Pure DI" 或 "Manual DI"——没有 IoC 容器,代码完全可读、可控。Spring 的好处是 **"自动管理大量对象的生命周期"**——当你的项目有几百个 Bean 时,手写 DI 就会变得繁琐。

**小型项目:** Pure DI 即可,启动快、无学习成本
**中型项目:** Spring/Guice 自动注入 + 少量配置
**大型项目:** Spring Boot 全家桶 + 模块化配置

### Q3:DIP 和 "贫血模型" 冲突吗?

**答:** 表面上有些张力,但本质上不冲突。**贫血模型** 把数据和行为分离(`User` 只有数据,行为在 `UserService`),**充血模型** 把数据和行为聚合(`User` 自己有 `changePassword()` 等方法)。

在 DIP 视角下,关键是:**抽象(`UserRepository`)归属调用方(`UserService`)**。不管你用贫血还是充血,DIP 都成立:
- **贫血 + DIP**:`User` 是纯数据,`UserService` 通过 `UserRepository` 接口操作数据,`UserRepository` 定义在 service 包
- **充血 + DDD 聚合**:`User` 是聚合根,自带 `changePassword()` 等领域方法,`UserRepository` 接口定义在 domain 包

两种风格都可以遵循 DIP,关键是抽象的归属和依赖方向。

## 小结

DIP 是 SOLID 中 **最具架构价值** 的一条原则。它不是 "不用 new 这么简单",而是 **"让抽象归属于调用方,让业务不被技术绑架"** 的设计哲学。

DIP 的核心要点:
1. **高层 ↔ 低层,都依赖抽象** —— 不要让业务层直接依赖基础设施层
2. **抽象归属于调用方** —— 接口由调用方定义,不是由实现方定义
3. **细节依赖抽象** —— 实现类 `implements` 接口,而不是反过来
4. **IoC 容器是 DIP 的工业化** —— Spring 让 DIP 落地更轻松,但 DIP 本身不依赖任何框架

DIP 的真正价值在 **系统演化** 中体现:
- 换数据库?加一个新 `XxxUserRepository` 实现,业务代码不动
- 加缓存?在 Repository 上套一层装饰器,业务代码不动
- 单测?mock 接口,5 行搞定
- 团队协作?接口作为契约,多团队并行开发

**DIP 是 SOLID 的集大成者**——SRP 拆分类、OCP 要扩展、LSP 保正确、ISP 拆接口,所有这些最终都通过 DIP 的 "依赖抽象" 落地。掌握 DIP,就掌握了面向对象设计的精髓。

至此,SOLID 五原则全部学完。下一章我们将学习 **其他重要的设计原则**(KISS、YAGNI、DRY、组合优于继承、迪米特法则)——它们和 SOLID 互补,共同构成了软件设计的完整思想体系。
