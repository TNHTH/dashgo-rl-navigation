# 编码规范规则

> **创建时间**: 2026-01-22 15:05:00
> **最后修改**: 2026-01-22 15:15:00
> **适用**: Python + C/C++ + 通用编码原则

---

## 📝 文档时间戳规则

**所有创建的文档必须包含精确到秒的时间戳。**

**时间戳格式**：
```markdown
> **创建时间**: YYYY-MM-DD HH:MM:SS
```

**示例**：
- ✅ 正确：`> **创建时间**: 2026-01-28 14:30:55`
- ❌ 错误：`> **日期**: 2026-01-28`
- ❌ 错误：`> **创建时间**: 2026-01-28`

**原因**：只有日期无法分辨文档修改的先后顺序，精确到秒可以明确版本顺序。

---

## 核心原则

### 1. 不可变性优先 (Immutability First)

**原则**: 除非必要，否则使用不可变数据结构

```python
# ✅ 好: 使用tuple（不可变）
coordinates = (1, 2, 3)
colors = ("red", "green", "blue")

# ❌ 差: 使用list（可变）
coordinates = [1, 2, 3]
colors = ["red", "green", "blue"]

# ✅ 好: 返回新对象而非修改原对象
new_users = users + [new_user]
new_dict = {**old_dict, "key": "value"}

# ❌ 差: 修改原对象
users.append(new_user)
old_dict["key"] = "value"
```

### 2. 纯函数优先 (Pure Functions First)

**原则**: 函数应该是纯函数（相同输入→相同输出，无副作用）

```python
# ✅ 好: 纯函数
def add_user(users: list[User], user: User) -> list[User]:
    return users + [user]

# ❌ 差: 有副作用（修改输入参数）
def add_user(users: list[User], user: User) -> None:
    users.append(user)
```

### 3. 显式错误处理 (Explicit Error Handling)

```python
# ✅ 好: 显式处理，具体异常
def fetch_data(url: str) -> dict:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        logger.error(f"Timeout fetching {url}")
        raise
    except requests.HTTPError as e:
        logger.error(f"HTTP {e.response.status_code} for {url}")
        raise
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise

# ❌ 差: 裸except，吞噬错误
def fetch_data(url: str) -> dict:
    try:
        return requests.get(url).json()
    except:
        return None  # 错误被吞噬
```

### 4. 类型安全 (Type Safety)

```python
# ✅ 好: 使用类型提示
def process_users(users: list[User]) -> dict[str, int]:
    return {u.name: u.age for u in users}

# ❌ 差: 无类型提示
def process_users(users):
    return {u.name: u.age for u in users}
```

### 5. 命名清晰 (Clear Naming)

```python
# ✅ 好: 清晰
user_ids = [1, 2, 3]
fetch_user_data()
class UserRepository:

# ❌ 差: 模糊
data = [1, 2, 3]
process()
class Manager:
```

---

## Python特定规范

### 遵循PEP 8

1. **函数命名**: snake_case
2. **类命名**: PascalCase
3. **常量**: UPPER_SNAKE_CASE
4. **缩进**: 4空格（非Tab）

```python
# ✅ 好
def get_user_data(user_id: int) -> dict:
    MAX_RETRIES = 3

class UserService:

# ❌ 差
def getUserData(userID: int) -> dict:
    max_retries = 3

class user_service:
```

### 使用列表推导（简单情况）

```python
# ✅ 好: 简单列表推导
squares = [x**2 for x in range(10)]
names = [user.name for user in users if user.active]

# ❌ 差: 复杂列表推导（难以阅读）
result = [complex_func(x) for x in items if cond1(x) and cond2(x)]
# 改为普通循环
result = []
for x in items:
    if cond1(x) and cond2(x):
        result.append(complex_func(x))
```

### 使用Context Manager

```python
# ✅ 好: 自动关闭资源
with open("file.txt", "r") as f:
    content = f.read()

# ❌ 差: 需要手动关闭
f = open("file.txt", "r")
content = f.read()
f.close()
```

### 使用dataclass（Python 3.7+）

```python
# ✅ 好: 使用dataclass
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str
    email: str

# ❌ 差: 手写__init__
class User:
    def __init__(self, id: int, name: str, email: str):
        self.id = id
        self.name = name
        self.email = email
```

---

## C/C++特定规范

### 遵循命名约定

1. **函数命名**: snake_case
2. **类/结构体命名**: PascalCase
3. **常量**: UPPER_SNAKE_CASE
4. **宏**: UPPER_SNAKE_CASE
5. **成员变量**: m_snake_case 或 trailing_underscore_

```cpp
// ✅ 好
void get_user_data(int user_id);
const int MAX_RETRIES = 3;
class UserService { };
struct UserProfile { };

// ❌ 差
void GetUserData(int userID);  // 不是Java风格
int max_retries = 3;           // 常量应该大写
class userService { };         // 类名应该PascalCase
```

### 使用const和constexpr

```cpp
// ✅ 好: 使用const确保不可变
const int MAX_SIZE = 100;
const std::string& getName() const;  // 成员函数不修改对象
constexpr int PI = 314;              // 编译时常量

// ❌ 差
int MAX_SIZE = 100;  // 应该是const
std::string& getName();  // 应该是const
```

### 使用智能指针（C++11+）

```cpp
// ✅ 好: 使用智能指针自动管理内存
#include <memory>
std::unique_ptr<User> user = std::make_unique<User>();
std::shared_ptr<Data> data = std::make_shared<Data>();

// ❌ 差: 手动管理内存
User* user = new User();
// ... 忘记delete
delete user;  // 容易遗漏
```

### 使用RAII（资源获取即初始化）

```cpp
// ✅ 好: 使用RAII自动管理资源
{
    std::ifstream file("data.txt");
    // 文件在作用域结束时自动关闭
    std::lock_guard<std::mutex> lock(mutex);
    // 互斥锁在作用域结束时自动释放
}

// ❌ 差: 手动管理资源
FILE* file = fopen("data.txt", "r");
// ... 必须记得fclose(file)
fclose(file);
```

### 避免内存泄漏

```cpp
// ❌ 危险: 原始指针容易泄漏
void process() {
    User* user = new User();
    // 如果中间抛异常，user不会被delete
    delete user;
}

// ✅ 安全: 智能指针自动管理
void process() {
    auto user = std::make_unique<User>();
    // 即使抛异常，user也会被自动删除
}
```

### 使用enum class而非enum（C++11）

```cpp
// ✅ 好: enum class避免命名冲突
enum class Color { RED, GREEN, BLUE };
enum class Status { ACTIVE, INACTIVE };
Color c = Color::RED;  // 明确的作用域

// ❌ 差: enum容易命名冲突
enum Color { RED, GREEN, BLUE };
enum Status { ACTIVE, INACTIVE };  // ACTIVE可能与其他枚举冲突
Color c = RED;
```

### 使用nullptr而非NULL（C++11）

```cpp
// ✅ 好: 使用nullptr（类型安全）
User* user = nullptr;

// ❌ 差: 使用NULL（实际上是整数0）
User* user = NULL;
```

### 使用auto（适度）

```cpp
// ✅ 好: auto简化复杂类型
std::map<std::string, std::vector<int>>::iterator it = data.begin();  // 太长
auto it = data.begin();  // 清晰

// ❌ 差: auto降低可读性
auto x = 5;  // 类型不明确
auto result = process();  // 无法知道返回类型
```

### 使用范围for循环（C++11）

```cpp
// ✅ 好: 范围for循环简洁
std::vector<int> numbers = {1, 2, 3};
for (int n : numbers) {
    std::cout << n << std::endl;
}

// 使用引用避免拷贝
for (const auto& user : users) {
    std::cout << user.name << std::endl;
}

// ❌ 差: 传统循环繁琐
for (std::vector<int>::iterator it = numbers.begin(); it != numbers.end(); ++it) {
    std::cout << *it << std::endl;
}
```

### C语言特定规范

#### 使用typedef定义类型别名

```c
// ✅ 好
typedef struct {
    int id;
    char name[50];
} User;

// ✅ 更好: C++可用using
using User = struct {
    int id;
    char name[50];
};
```

#### 动态内存管理

```c
// ✅ 好: 配对的malloc/free
int* arr = (int*)malloc(10 * sizeof(int));
if (arr) {
    // 使用arr
    free(arr);
}

// ❌ 差: 内存泄漏
int* arr = (int*)malloc(10 * sizeof(int));
// 忘记free(arr)
```

#### 避免全局变量

```c
// ❌ 差: 全局变量
int global_counter = 0;

// ✅ 好: 使用static限制作用域
static int file_counter = 0;  // 只在本文件可见

// ✅ 更好: 使用函数封装
int get_counter() {
    static int counter = 0;  // 函数内静态变量
    return counter++;
}
```

---

## 通用规范（跨语言）

### 1. 魔法数字 → 命名常量

```python
# ❌ 差
if user.status == 3:
    timeout = 86400

# ✅ 好
USER_STATUS_ACTIVE = 3
SECONDS_PER_DAY = 86400

if user.status == USER_STATUS_ACTIVE:
    timeout = SECONDS_PER_DAY
```

### 2. 注释掉的代码 → 删除

```python
# ❌ 差
# def old_function():
#     pass

def new_function():
    pass

# ✅ 好: 用Git管理历史，不需要注释旧代码
def new_function():
    pass
```

### 3. 嵌套过深 → 提前返回

```python
# ❌ 差: 嵌套4层
def process(user):
    if user:
        if user.active:
            if user.has_permission:
                # do something
                pass

# ✅ 好: 提前返回
def process(user):
    if not user:
        return
    if not user.active:
        return
    if not user.has_permission:
        return
    # do something
```

---

## 禁止模式

1. **禁止**吞噬异常（裸except）
2. **禁止**使用全局变量（除非必要）
3. **禁止**循环中修改正在迭代的列表
4. **禁止**使用`type(var) == Type`（用`isinstance()`）
5. **禁止**可变默认参数（`def func(items=[])`）

```python
# ❌ 全部禁止
try:
    risky_operation()
except:  # 禁止裸except
    pass

GLOBAL_DATA = []  # 禁止全局变量

for item in items:
    if item.bad:
        items.remove(item)  # 禁止修改迭代中的列表

if type(user) == User:  # 禁止type比较
    pass

def func(items=[]):  # 禁止可变默认参数
    pass
```

---

## 检查清单

### Python代码
- [ ] 函数有类型提示
- [ ] 无全局变量（除非明确必要）
- [ ] 异常处理具体（非裸except）
- [ ] 遵循PEP 8命名规范
- [ ] 使用with语句管理资源
- [ ] 使用dataclass定义数据结构

### C++代码
- [ ] 使用const/constexpr确保不可变
- [ ] 使用智能指针（unique_ptr/shared_ptr）
- [ ] 使用nullptr而非NULL
- [ ] 使用enum class而非enum
- [ ] 成员函数标记const（如果不修改对象）
- [ ] 避免使用new/delete（优先智能指针）
- [ ] 使用RAII管理资源

### C代码
- [ ] malloc/free配对
- [ ] 避免全局变量（使用static限制作用域）
- [ ] 使用typedef定义类型别名
- [ ] 检查指针是否为NULL
- [ ] 使用const修饰不修改的参数

### 通用（所有语言）
- [ ] 无魔法数字（使用常量）
- [ ] 函数/变量命名清晰
- [ ] 无注释掉的代码块
- [ ] 嵌套不超过3层（提前返回）
