# Spring Boot Demo Project

基于 Spring Boot 2.7.x + Maven 构建的 Java 开发项目模板。

## 技术栈

| 技术 | 版本 | 说明 |
|------|------|------|
| Spring Boot | 2.7.18 | 核心框架（2.x 最后一个稳定版本） |
| Spring Data JPA | 2.7.18 | ORM 框架 |
| H2 Database | - | 内存数据库（开发环境） |
| MySQL | 8.0.33 | 关系型数据库（测试/生产环境） |
| Lombok | - | 简化代码 |
| Hutool | 5.8.18 | Java 工具包 |
| Fastjson | 1.2.83 | JSON 处理 |
| Maven | 3.x | 项目构建工具 |
| Java | 1.8+ | 编程语言 |

## 项目结构

```
springboot-demo/
├── pom.xml                              # Maven 配置文件
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/example/demo/
│   │   │       ├── DemoApplication.java        # 主应用程序入口
│   │   │       ├── common/
│   │   │       │   └── Result.java             # 统一响应结果类
│   │   │       ├── controller/
│   │   │       │   ├── HelloController.java    # 示例控制器
│   │   │       │   └── UserController.java     # 用户控制器
│   │   │       ├── entity/
│   │   │       │   └── User.java               # 用户实体类
│   │   │       ├── repository/
│   │   │       │   └── UserRepository.java     # 数据访问层
│   │   │       └── service/
│   │   │           ├── UserService.java        # 用户服务接口
│   │   │           └── impl/
│   │   │               └── UserServiceImpl.java # 用户服务实现
│   │   └── resources/
│   │       ├── application.yml          # 主配置文件
│   │       ├── application-dev.yml      # 开发环境配置
│   │       ├── application-test.yml     # 测试环境配置
│   │       └── application-prod.yml     # 生产环境配置
│   └── test/
│       └── java/
│           └── com/example/demo/
│               ├── DemoApplicationTests.java        # 应用测试
│               └── controller/
│                   └── HelloControllerTest.java     # 控制器测试
└── README.md
```

## 快速开始

### 环境要求

- JDK 1.8+
- Maven 3.6+
- IDE（推荐 IntelliJ IDEA）

### 克隆并运行

```bash
# 进入项目目录
cd springboot-demo

# 使用 Maven 构建
mvn clean install

# 运行项目
mvn spring-boot:run
```

或者直接运行主类：
```bash
mvn compile
mvn exec:java -Dexec.mainClass="com.example.demo.DemoApplication"
```

### IDE 运行

1. 使用 IntelliJ IDEA 打开项目
2. 等待 Maven 依赖下载完成
3. 找到 `DemoApplication.java`，右键运行

## API 接口

应用启动后访问地址：`http://localhost:8080`

### 基础接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/hello` | GET | Hello 示例 |
| `/api/health` | GET | 健康检查 |
| `/api/info` | GET | 应用信息 |
| `/actuator/health` | GET | Actuator 健康监控 |

### 用户管理接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/users` | GET | 获取所有用户 |
| `/api/users/page` | GET | 分页获取用户 |
| `/api/users/{id}` | GET | 根据 ID 获取用户 |
| `/api/users/username/{username}` | GET | 根据用户名获取用户 |
| `/api/users` | POST | 创建用户 |
| `/api/users/{id}` | PUT | 更新用户 |
| `/api/users/{id}` | DELETE | 删除用户 |

### H2 数据库控制台

开发环境可访问 H2 控制台：`http://localhost:8080/h2-console`

- JDBC URL: `jdbc:h2:mem:devdb`
- Username: `sa`
- Password: (空)

### 接口示例

#### 1. 访问 Hello 接口

**请求：**
```
GET http://localhost:8080/api/hello
```

**响应：**
```json
{
    "code": 200,
    "message": "操作成功",
    "data": {
        "message": "Hello, Spring Boot!",
        "timestamp": 1713436800000,
        "status": "ok"
    },
    "timestamp": 1713436800000
}
```

#### 2. 创建用户

**请求：**
```
POST http://localhost:8080/api/users
Content-Type: application/json

{
    "username": "admin",
    "password": "123456",
    "email": "admin@example.com",
    "nickname": "管理员",
    "phone": "13800138000",
    "status": 1
}
```

#### 3. 分页查询用户

**请求：**
```
GET http://localhost:8080/api/users/page?page=0&size=10&sort=id&direction=desc
```

## 多环境配置

项目支持多环境配置，通过 Maven Profile 或 Spring Boot Profile 切换。

### 配置文件说明

| 配置文件 | 环境 | 数据库 | 说明 |
|---------|------|--------|------|
| application-dev.yml | 开发 | H2 内存数据库 | 默认激活 |
| application-test.yml | 测试 | MySQL | 需配置本地 MySQL |
| application-prod.yml | 生产 | MySQL | 推荐使用环境变量 |

### 切换环境方式

#### 方式1：Maven Profile（推荐）

```bash
# 开发环境（默认）
mvn spring-boot:run -P dev

# 测试环境
mvn spring-boot:run -P test

# 生产环境
mvn spring-boot:run -P prod
```

#### 方式2：Spring Boot Profile

```bash
# 开发环境
mvn spring-boot:run -Dspring-boot.run.profiles=dev

# 测试环境
mvn spring-boot:run -Dspring-boot.run.profiles=test

# 生产环境
mvn spring-boot:run -Dspring-boot.run.profiles=prod
```

#### 方式3：打包后指定环境

```bash
# 打包
mvn clean package -DskipTests

# 运行时指定环境
java -jar target/springboot-demo-1.0.0.jar --spring.profiles.active=prod
```

## Maven 常用命令

```bash
# 清理项目
mvn clean

# 编译项目
mvn compile

# 执行测试
mvn test

# 打包（跳过测试）
mvn clean package -DskipTests

# 安装到本地仓库
mvn clean install -DskipTests

# 运行项目
mvn spring-boot:run

# 查看依赖树
mvn dependency:tree
```

## 主要依赖说明

### Spring Boot Starters

```xml
<!-- Web 开发 -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>

<!-- 参数校验 -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-validation</artifactId>
</dependency>

<!-- JPA 数据访问 -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-jpa</artifactId>
</dependency>

<!-- AOP 支持 -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-aop</artifactId>
</dependency>

<!-- 应用监控 -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>
```

### 数据库驱动

```xml
<!-- H2 内存数据库（开发环境） -->
<dependency>
    <groupId>com.h2database</groupId>
    <artifactId>h2</artifactId>
    <scope>runtime</scope>
</dependency>

<!-- MySQL 驱动 -->
<dependency>
    <groupId>mysql</groupId>
    <artifactId>mysql-connector-java</artifactId>
    <version>8.0.33</version>
    <scope>runtime</scope>
</dependency>
```

### 工具类库

```xml
<!-- Lombok 简化代码 -->
<dependency>
    <groupId>org.projectlombok</groupId>
    <artifactId>lombok</artifactId>
    <optional>true</optional>
</dependency>

<!-- Hutool Java 工具包 -->
<dependency>
    <groupId>cn.hutool</groupId>
    <artifactId>hutool-all</artifactId>
    <version>5.8.18</version>
</dependency>

<!-- Fastjson JSON 处理 -->
<dependency>
    <groupId>com.alibaba</groupId>
    <artifactId>fastjson</artifactId>
    <version>1.2.83</version>
</dependency>
```

## 配置说明

### 主配置文件 (application.yml)

```yaml
server:
  port: 8080                    # 服务端口

spring:
  application:
    name: springboot-demo       # 应用名称
  
  profiles:
    active: dev                  # 默认激活的环境

  jackson:
    date-format: yyyy-MM-dd HH:mm:ss  # 日期格式
    time-zone: GMT+8                   # 时区

  jpa:
    hibernate:
      ddl-auto: update           # 自动建表策略
    show-sql: true               # 显示 SQL
```

### 生产环境建议配置

```yaml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/demo_prod?useUnicode=true&characterEncoding=utf8&useSSL=true&serverTimezone=Asia/Shanghai
    username: ${DB_USERNAME:root}      # 使用环境变量
    password: ${DB_PASSWORD:root}       # 使用环境变量

  jpa:
    hibernate:
      ddl-auto: none             # 生产环境禁止自动建表
    show-sql: false              # 生产环境不显示 SQL

logging:
  file:
    name: logs/springboot-demo.log  # 日志文件路径
```

## 部署说明

### Docker 部署（推荐）

创建 `Dockerfile`：

```dockerfile
FROM openjdk:8-jdk-alpine
VOLUME /tmp
COPY target/springboot-demo-1.0.0.jar app.jar
EXPOSE 8080
ENV JAVA_OPTS=""
ENTRYPOINT [ "sh", "-c", "java $JAVA_OPTS -Djava.security.egd=file:/dev/./urandom -jar /app.jar" ]
```

构建并运行：

```bash
# 构建镜像
docker build -t springboot-demo:1.0.0 .

# 运行容器
docker run -d -p 8080:8080 \
    --name springboot-demo \
    -e SPRING_PROFILES_ACTIVE=prod \
    -e DB_USERNAME=root \
    -e DB_PASSWORD=your_password \
    springboot-demo:1.0.0
```

### 传统部署

```bash
# 打包
mvn clean package -DskipTests

# 运行
java -jar target/springboot-demo-1.0.0.jar \
    --spring.profiles.active=prod \
    --spring.datasource.username=root \
    --spring.datasource.password=your_password
```

## 开发建议

### 1. 代码规范

- 使用 Lombok 简化实体类代码
- Controller 层只做参数校验和响应封装
- Service 层处理业务逻辑
- Repository 层负责数据访问
- 统一使用 `Result<T>` 作为响应格式

### 2. 开发环境

开发环境默认使用 H2 内存数据库，无需额外配置，开箱即用。如果需要持久化数据，可以修改为：

```yaml
spring:
  datasource:
    url: jdbc:h2:file:./data/devdb;DB_CLOSE_ON_EXIT=FALSE
```

### 3. 添加新模块

建议按照以下结构添加新功能：

```
com.example.demo/
├── controller/    # 控制器层
├── service/       # 服务层接口
│   └── impl/      # 服务层实现
├── repository/    # 数据访问层
├── entity/        # 实体类
├── dto/           # 数据传输对象
├── vo/            # 视图对象
├── config/        # 配置类
├── aspect/        # 切面
├── exception/     # 异常处理
└── common/        # 通用类
```

## 常见问题

### Q1: Maven 依赖下载慢

**解决方案**：配置阿里云镜像

在 `~/.m2/settings.xml` 中添加：

```xml
<mirrors>
    <mirror>
        <id>aliyun</id>
        <mirrorOf>central</mirrorOf>
        <name>Aliyun Maven Mirror</name>
        <url>https://maven.aliyun.com/repository/central</url>
    </mirror>
</mirrors>
```

### Q2: Lombok 不生效

**解决方案**：

1. 确保 IDE 安装了 Lombok 插件
2. IntelliJ IDEA: 设置 → Build, Execution, Deployment → Compiler → Annotation Processors → Enable annotation processing

### Q3: 端口被占用

**解决方案**：

```bash
# Windows 查看端口占用
netstat -ano | findstr :8080

# 或者修改端口
mvn spring-boot:run -Dserver.port=8081
```

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交 Issue 或 PR。
