import streamlit as st

# Insert containers separated into tabs:
tab1, tab2, tab3, tab4 = st.tabs(["Variables", "Control Structures", "Functions + OOP", "Files + Libraries"])

with tab1:
    st.header('Variables and Data Types', divider='rainbow')
    st.write(f'''

Python supports multiple data types:

- **Numeric**: `int`, `float`, `complex`
- **Text**: `str`
- **Sequence**: `list`, `tuple`, `range`
- **Mapping**: `dict`
- **Set Types**: `set`, `frozenset`
- **Boolean**: `bool`
- **Binary**: `bytes`, `bytearray`, `memoryview`''')
    st.code(f'''
name = "Alice"     # String
age = 25           # Integer
is_student = True  # Boolean
''')
with tab2:
    st.header('Control Structures', divider='rainbow')
    st.markdown(f'''
            #### **Conditionals**

```python
x = 10
if x > 5:
    print("x is greater than 5")
else:
    print("x is 5 or less")
```

#### **Loops**

- **For Loop**:

```python
for i in range(5):
    print(i)
```

- **While Loop**:

```python
count = 0
while count < 5:
    print(count)
    count += 1
```
''')

with tab3:
    st.header('Functions and Object Oriented Programming', divider='rainbow')

    st.markdown(f'''


## 1. **Key OOP Concepts**

| Concept        | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| **Class**      | Blueprint for creating objects.                                            |
| **Object**     | Instance of a class.                                                      |
| **Attribute**  | Data stored in an object, defined in a class.                              |
| **Method**     | Function defined in a class to manipulate or use attributes.              |
| **Inheritance**| Mechanism to create a new class using features of an existing class.      |
| **Encapsulation**| Restricting access to some components of an object.                     |
| **Polymorphism**| Ability to use a common interface for different data types.              |

---

## 2. **Defining Classes and Objects**

### Create a Class
```python
class MyClass:
    # Constructor
    def __init__(self, attribute):
        self.attribute = attribute  # Instance attribute

    # Method
    def my_method(self):
        return f"Attribute value: self.attribute"
```

### Create an Object
```python
# Create an instance of MyClass
obj = MyClass("Hello")
print(obj.my_method())  # Output: Attribute value: Hello
```

---

## 3. **Class vs Instance Attributes**

- **Class Attributes**: Shared across all instances.
- **Instance Attributes**: Unique to each instance.

```python
class Example:
    class_attribute = "I am shared"

    def __init__(self, instance_value):
        self.instance_attribute = instance_value

obj1 = Example("Instance 1")
obj2 = Example("Instance 2")

print(obj1.class_attribute)       # Output: I am shared
print(obj2.instance_attribute)    # Output: Instance 2
```

---

## 4. **Methods**

### Instance Method
```python
class Example:
    def greet(self, name):
        return f"Hello, name!"

obj = Example()
print(obj.greet("Alice"))  # Output: Hello, Alice!
```

### Class Method
```python
class Example:
    @classmethod
    def class_method(cls):
        return "This is a class method!"

print(Example.class_method())  # Output: This is a class method!
```

### Static Method
```python
class Example:
    @staticmethod
    def static_method():
        return "This is a static method!"

print(Example.static_method())  # Output: This is a static method!
```

---

## 5. **Inheritance**

```python
# Base Class
class Animal:
    def speak(self):
        return "I make a sound"

# Derived Class
class Dog(Animal):
    def speak(self):
        return "I bark"

dog = Dog()
print(dog.speak())  # Output: I bark
```

---

## 6. **Encapsulation**

### Public, Protected, and Private Attributes
```python
class Example:
    def __init__(self):
        self.public = "Public"       # Accessible everywhere
        self._protected = "Protected" # Accessible in subclasses
        self.__private = "Private"   # Not directly accessible

    def get_private(self):
        return self.__private  # Access private attribute via a method

obj = Example()
print(obj.public)             # Output: Public
print(obj._protected)         # Output: Protected
print(obj.get_private())      # Output: Private
```

---

## 7. **Polymorphism**

### Method Overriding
```python
class Parent:
    def show(self):
        return "I am a Parent"

class Child(Parent):
    def show(self):
        return "I am a Child"

obj = Child()
print(obj.show())  # Output: I am a Child
```

---

## 8. **Special (Magic/Dunder) Methods**

| Method         | Description                            |
|----------------|----------------------------------------|
| `__init__`     | Constructor to initialize attributes. |
| `__str__`      | String representation of an object.   |
| `__repr__`     | Official string representation.       |
| `__add__`      | Overloads the `+` operator.           |
| `__len__`      | Returns length using `len()`.         |

### Example: Magic Methods
```python
class Example:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Value: self.value"

obj = Example(10)
print(obj)  # Output: Value: 10
```

---

## 9. **Abstract Classes and Interfaces**

### Using `abc` Module
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2

circle = Circle(5)
print(circle.area())  # Output: 78.5
```

---

## 10. **Key Functions**

| Function       | Description                                       |
|----------------|---------------------------------------------------|
| `isinstance()` | Checks if an object is an instance of a class.    |
| `issubclass()` | Checks if a class is a subclass of another class. |

### Examples:
```python
print(isinstance(circle, Circle))  # Output: True
print(issubclass(Circle, Shape))   # Output: True
```

---

## 11. **OOP Best Practices**

- Use meaningful class and method names.
- Encapsulate data to protect attributes.
- Avoid large classes; use inheritance and composition.
- Follow the **Single Responsibility Principle** (SRP).

---

## 12. **Quick Example: OOP in Action**

```python
class Vehicle:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def start(self):
        return f"self.brand self.model is starting."

class Car(Vehicle):
    def start(self):
        return f"self.brand self.model is roaring to life!"

car = Car("Tesla", "Model S")
print(car.start())  # Output: Tesla Model S is roaring to life!
```

---

                ''')

with tab4:
    st.header('File Handling and Popular Libraries', divider='rainbow')
    st.markdown(f'''

## File Handling in Python

File handling allows you to work with files for tasks such as:
- **Reading and Writing**: Open, modify, and save files in formats like `.txt`, `.csv`, and `.json`.
- **Modes**: Work in various modes like read (`'r'`), write (`'w'`), append (`'a'`), and binary (`'b'`).
- **Context Management**: Ensure resources are properly managed with the `with` statement.

### Example:
```python
# Writing to a file
with open("example.txt", "w") as file:
    file.write("Hello, Python!")

# Reading from a file
with open("example.txt", "r") as file:
    content = file.read()
    print(content)
```

---

## Popular Libraries in Python

Python’s ecosystem is rich with libraries that simplify complex tasks, including:

### 1. **Django**
A high-level web framework for rapid development of secure and scalable web applications. Django emphasizes:
- **Built-in Features**: Authentication, ORM, and Admin Interface.
- **Scalability**: Handles traffic-heavy websites efficiently.
- **Security**: Protects against common vulnerabilities like SQL injection.

### 2. **Streamlit**
A lightweight library for creating interactive web applications for data visualization and machine learning.
- **Ease of Use**: Build apps with just Python.
- **Focus on Data**: Perfect for data scientists and analysts.
- **Real-Time Updates**: Automatically reflects changes during development.

### 3. **Other Popular Libraries**
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib/Seaborn**: For data visualization.
- **Flask**: A micro-framework for web development, an alternative to Django.

---

## What’s Next?

In the following sections, we’ll dive deeper into **Django** and **Streamlit**, exploring their features, use cases, and practical implementations.
                ''')
