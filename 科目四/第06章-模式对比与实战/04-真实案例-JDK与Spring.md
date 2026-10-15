---
title: 04-真实案例 - JDK 与 Spring 中的设计模式
tags:
  - JDK
  - Spring
  - 源码分析
  - 实战案例
description: 真实项目中的设计模式. 分析 JDK I/O / Collections / AWT / Spring IoC / AOP / MVC 中的模式应用
last_updated: 2026-09-11
status: ✅ v1.0
chapter_id: 06-04
---

# 04. 真实案例: JDK 与 Spring 中的设计模式

> 真实框架的设计模式应用是学习的最好教材. 本章剖析 JDK 和 Spring 框架中的核心模式.

---

## 1. JDK 中的设计模式

### 1.1 I/O 流体系: 装饰器模式典范

JDK 的 `InputStream` / `OutputStream` 是教科书级的装饰器模式应用:

```java
// 装饰器模式: 每个流装饰前一个流, 添加功能
public abstract class InputStream {
    public abstract int read() throws IOException;
    // ...
}

public class FileInputStream extends InputStream {
    private File file;
    public int read() throws IOException {
        // 从文件读
    }
}

public class FilterInputStream extends InputStream {
    protected volatile InputStream in;  // 被装饰者
    public int read() throws IOException {
        return in.read();  // 委托
    }
}

// 具体装饰器
public class BufferedInputStream extends FilterInputStream {
    private byte[] buf = new byte[8192];
    public int read() throws IOException {
        if (pos >= count) {
            // 批量填充 buf
            count = in.read(buf, 0, 8192);
            pos = 0;
        }
        return buf[pos++];
    }
}

public class DataInputStream extends FilterInputStream {
    public int readInt() throws IOException {
        // 读 4 字节, 高位在前
    }
}
```

**典型使用**:

```java
// 嵌套装饰: 文件流 → 缓冲流 → 数据流
DataInputStream dis = new DataInputStream(
    new BufferedInputStream(
        new FileInputStream("data.bin")
    )
);
dis.readInt();  // 高效读取
```

**模式角色**:
- `InputStream`: 抽象组件 (Component)
- `FileInputStream`: 具体组件 (Concrete Component)
- `FilterInputStream`: 装饰器基类 (Decorator)
- `BufferedInputStream`: 具体装饰器 (Concrete Decorator)

**启示**: 装饰器模式让 I/O 流的组合像搭积木, 加新功能只需新装饰器, 不动现有代码.

### 1.2 Collections 框架: 迭代器 + 模板方法

```java
// 抽象集合定义模板方法
public abstract class AbstractList<E> extends AbstractCollection<E> implements List<E> {
    
    // 模板方法
    public Iterator<E> iterator() {
        return new Itr();  // 调用内部类
    }
    
    // 抽象方法, 由子类实现
    public abstract E get(int index);
    public abstract int size();
    
    // 通用实现, 用模板方法
    public boolean add(E e) {
        add(size(), e);  // 调用抽象方法
        return true;
    }
    
    public boolean equals(Object o) {
        if (o == this) return true;
        if (!(o instanceof List)) return false;
        // ... 用 iterator() 遍历
    }
    
    // Itr 是迭代器的具体实现
    private class Itr implements Iterator<E> {
        int cursor = 0;
        int lastRet = -1;
        public boolean hasNext() { return cursor != size(); }
        public E next() {
            checkForComodification();
            // 用 get(int) 取元素
        }
    }
}
```

**模式角色**:
- `AbstractList`: 抽象模板类 (定义算法骨架)
- `ArrayList` / `LinkedList`: 具体子类 (实现抽象方法)
- `Itr`: 内部类迭代器 (具体迭代器)

### 1.3 Runtime: 单例模式

```java
public class Runtime {
    // 饿汉式单例 - 类加载时创建
    private static Runtime currentRuntime = new Runtime();
    
    // 私有构造器, 禁止外部 new
    private Runtime() {}
    
    // 全局访问点
    public static Runtime getRuntime() {
        return currentRuntime;
    }
}
```

**使用**:
```java
Runtime.getRuntime().exec("ls");     // 执行命令
Runtime.getRuntime().totalMemory();   // 总内存
Runtime.getRuntime().availableProcessors();
```

**为什么是单例**: 一个 JVM 只有一个运行时环境, 多实例无意义.

### 1.4 AWT/Swing: 适配器 + 桥接 + 观察者

```java
// 适配器: WindowAdapter 把 30+ 方法适配成空实现
public abstract class WindowAdapter
    implements WindowListener, WindowStateListener, WindowFocusListener {
    public void windowOpened(WindowEvent e) {}
    public void windowClosing(WindowEvent e) {}
    public void windowClosed(WindowEvent e) {}
    public void windowIconified(WindowEvent e) {}
    // ... 30 个空方法
}

// 用户继承 WindowAdapter, 只需实现关心的事件
class MyFrame extends WindowAdapter {
    public void windowClosing(WindowEvent e) {
        System.exit(0);
    }
}
```

**桥接**: AWT 的 Component 是抽象, Peer 是实现
```java
// Component (抽象)
public abstract class Component {
    private Peer peer;  // 桥接!
    public void paint(Graphics g) {
        peer.paint(g);  // 委托给 Peer
    }
}

// Button (抽象 + 平台实现)
public class Button extends Component {
    public void paint(Graphics g) {
        peer.paint(g);  // Windows/Mac 的 Peer 不同
    }
}
```

**观察者**: AWT 事件模型
```java
button.addActionListener(e -> System.out.println("点击"));
// button 是 Subject, ActionListener 是 Observer
```

### 1.5 ThreadPoolExecutor: 策略模式

```java
public class ThreadPoolExecutor {
    // 策略字段
    private final RejectedExecutionHandler handler;
    
    public ThreadPoolExecutor(int corePoolSize, int maxPoolSize,
                                long keepAliveTime, TimeUnit unit,
                                BlockingQueue<Runnable> queue,
                                ThreadFactory threadFactory,
                                RejectedExecutionHandler handler) {
        this.handler = handler;
    }
    
    // 触发策略
    final void reject(Runnable r) {
        handler.rejectedExecution(r, this);
    }
}

// 4 种内置策略
public static class AbortPolicy implements RejectedExecutionHandler {
    public void rejectedExecution(Runnable r, ThreadPoolExecutor e) {
        throw new RejectedExecutionException();
    }
}

public static class DiscardPolicy implements RejectedExecutionHandler {
    public void rejectedExecution(Runnable r, ThreadPoolExecutor e) {
        // 静默丢弃
    }
}

// 使用时选择策略
new ThreadPoolExecutor(5, 10, 60, TimeUnit.SECONDS,
    new LinkedBlockingQueue<>(),
    new ThreadPoolExecutor.CallerRunsPolicy());
```

**策略的灵活性**: 运行时切换, 不修改 ThreadPoolExecutor 代码.

---

## 2. Spring 中的设计模式

### 2.1 IoC 容器: 工厂方法 + 单例

```java
// BeanFactory: 抽象工厂
public interface BeanFactory {
    Object getBean(String name);
}

// ApplicationContext: 具体工厂
public class ClassPathXmlApplicationContext implements BeanFactory {
    private Map<String, Object> singletonObjects = new ConcurrentHashMap<>();
    
    public Object getBean(String name) {
        // 单例模式: 缓存 Bean
        Object bean = singletonObjects.get(name);
        if (bean == null) {
            bean = createBean(name);  // 工厂方法: 创建对象
            singletonObjects.put(name, bean);
        }
        return bean;
    }
}
```

**核心思想**: Bean 的创建由容器管理, 应用代码不直接 `new`.

### 2.2 AOP: 代理模式

```java
// JDK 动态代理
UserService target = new UserServiceImpl();
UserService proxy = (UserService) Proxy.newProxyInstance(
    classLoader,
    new Class[]{UserService.class},
    (proxyObj, method, args) -> {
        // 前置通知 (责任链)
        System.out.println("Before: " + method.getName());
        
        // 目标方法调用
        Object result = method.invoke(target, args);
        
        // 后置通知
        System.out.println("After: " + method.getName());
        return result;
    }
);
```

**AOP 通知链 = 责任链模式**:
```java
public class MethodInterceptorChain {
    private List<MethodInterceptor> interceptors = new ArrayList<>();
    
    public Object proceed(MethodInvocation invocation) throws Throwable {
        if (interceptors.isEmpty()) {
            return invocation.proceed();
        }
        // 递归调用, 形成链
        MethodInterceptor first = interceptors.remove(0);
        return first.invoke(invocation);
    }
}
```

### 2.3 MVC: 观察者 + 适配器 + 模板方法

```java
// DispatcherServlet: 前端控制器
public class DispatcherServlet extends HttpServlet {
    protected void doService(HttpServletRequest req, HttpServletResponse resp) {
        // 模板方法: 定义请求处理流程
        HandlerExecutionChain chain = getHandler(req);  // 1. 找 handler
        ModelAndView mv = handlerAdapter.handle(req, resp, chain);  // 2. 适配
        processDispatchResult(req, resp, mv);  // 3. 渲染
    }
}

// HandlerAdapter: 适配器模式
public interface HandlerAdapter {
    boolean supports(Object handler);
    ModelAndView handle(HttpServletRequest req, HttpServletResponse resp, Object handler);
}

// 多种适配器
public class SimpleControllerHandlerAdapter implements HandlerAdapter { ... }
public class HttpRequestHandlerAdapter implements HandlerAdapter { ... }
public class RequestMappingHandlerAdapter implements HandlerAdapter { ... }
```

**观察者**: ApplicationEvent 机制
```java
// 自定义事件
public class OrderCreatedEvent extends ApplicationEvent {
    public OrderCreatedEvent(Object source) { super(source); }
}

// 发布
@Service
public class OrderService {
    @Autowired
    private ApplicationEventPublisher publisher;
    
    public void createOrder() {
        // ... 创建订单逻辑
        publisher.publishEvent(new OrderCreatedEvent(this));
    }
}

// 监听
@Component
public class EmailListener implements ApplicationListener<OrderCreatedEvent> {
    public void onApplicationEvent(OrderCreatedEvent event) {
        sendEmail();  // 收到事件后发邮件
    }
}
```

### 2.4 JdbcTemplate: 模板方法

```java
public abstract class JdbcTemplate {
    // 模板方法: 定义查询流程
    public <T> T query(String sql, RowMapper<T> rowMapper, Object... args) {
        // 1. 获取连接
        Connection conn = DataSourceUtils.getConnection(dataSource);
        PreparedStatement ps = null;
        try {
            // 2. 准备语句 (变化点)
            ps = conn.prepareStatement(sql);
            for (int i = 0; i < args.length; i++) {
                ps.setObject(i + 1, args[i]);  // 抽象
            }
            // 3. 执行
            ResultSet rs = ps.executeQuery();
            // 4. 映射结果 (变化点)
            return mapResult(rs, rowMapper);
        } catch (SQLException e) {
            throw new DataAccessException(...);
        } finally {
            close(ps);
        }
    }
    
    // 4. 抽象方法, 由 RowMapper 实现
    protected abstract <T> T mapResult(ResultSet rs, RowMapper<T> rowMapper);
}

// 使用 - 只需提供映射策略
List<User> users = jdbcTemplate.query(
    "SELECT * FROM users WHERE age > ?",
    (rs, rowNum) -> new User(rs.getString("name"), rs.getInt("age")),
    18
);
```

**模板方法的好处**: 固定流程 (获取连接/准备/执行/关闭) 不变, 变化部分 (如何映射结果) 由用户定制.

### 2.5 Spring Data Access: 策略

```java
// PlatformTransactionManager 是策略接口
public interface PlatformTransactionManager {
    TransactionStatus getTransaction(TransactionDefinition definition);
    void commit(TransactionStatus status);
    void rollback(TransactionStatus status);
}

// 不同实现
public class DataSourceTransactionManager implements PlatformTransactionManager { ... }
public class JpaTransactionManager implements PlatformTransactionManager { ... }
public class JmsTransactionManager implements PlatformTransactionManager { ... }

// 运行时选择
@Bean
public PlatformTransactionManager transactionManager(DataSource ds) {
    return new DataSourceTransactionManager(ds);  // JDBC 用这个
}
```

**策略的好处**: 切换事务管理器 (JDBC → JPA → JMS) 不改业务代码.

### 2.6 Spring Security: 责任链 + 装饰器

```java
// SecurityFilterChain 是责任链
public class SecurityFilterChain {
    private List<Filter> filters = new ArrayList<>();
    
    public void doFilter(ServletRequest req, ServletResponse resp) {
        // 责任链: 依次调用每个 filter
        for (Filter filter : filters) {
            filter.doFilter(req, resp, chain);
        }
    }
}

// 具体 filter
public class AuthenticationFilter implements Filter {
    public void doFilter(req, resp, chain) {
        // 认证逻辑
        if (authenticated(req)) chain.doFilter(req, resp);
        else resp.sendError(401);
    }
}

public class AuthorizationFilter implements Filter {
    public void doFilter(req, resp, chain) {
        // 授权逻辑
        if (hasPermission(req)) chain.doFilter(req, resp);
        else resp.sendError(403);
    }
}
```

---

## 3. JDK vs Spring 模式应用对比

| 框架 | 主要模式 | 给开发者的启示 |
| :--- | :--- | :--- |
| **JDK I/O** | 装饰器 | 灵活组合优于继承 |
| **JDK Collections** | 迭代器 + 模板方法 | 抽象基类提取公共逻辑 |
| **JDK Runtime** | 单例 | 一个进程唯一对象 |
| **JDK AWT** | 适配器 + 桥接 + 观察者 | UI 跨平台 |
| **JDK ThreadPool** | 策略 | 算法可替换 |
| **Spring IoC** | 工厂方法 + 单例 | 解耦对象创建 |
| **Spring AOP** | 代理 + 责任链 | 不修改源码增强功能 |
| **Spring MVC** | 适配器 + 观察者 + 模板方法 | 多种 Controller 统一处理 |
| **Spring Data** | 策略 + 模板方法 | 切换数据库不改业务代码 |
| **Spring Security** | 责任链 + 装饰器 | 认证授权分层 |

---

## 4. 小结

✅ **JDK**: 装饰器 (I/O) + 迭代器 (Collections) + 单例 (Runtime) + 策略 (ThreadPool)

✅ **Spring**: 工厂 + 单例 (IoC) + 代理 + 责任链 (AOP) + 适配器 + 观察者 + 模板方法 (MVC)

✅ **启示**: 学习设计模式要看真实框架的应用, 不是死记硬背
