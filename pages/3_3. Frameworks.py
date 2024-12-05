import streamlit as st

# Insert containers separated into tabs:
tab1, tab2 = st.tabs(["Django", "Streamlit"])

with tab1:
    st.header('Django', divider='rainbow')
    st.image("assets/DjangoLogo.png", use_container_width=True)
    st.markdown(f'''

## 1. **Setting Up Django**

### Installation
```bash
# Install Django
pip install django

# Check Django version
django-admin --version

### Starting a Project
```bash
# Create a new Django project
django-admin startproject project_name

# Navigate to the project directory
cd project_name

# Start the development server
python manage.py runserver
```

### Project Structure
```
project_name/
│
├── manage.py          # Command-line utility for administrative tasks
├── project_name/      # Project settings and configuration
│   ├── __init__.py
│   ├── settings.py    # Main settings file
│   ├── urls.py        # URL configurations
│   ├── asgi.py        # ASGI entry point
│   └── wsgi.py        # WSGI entry point
└── app_name/          # Your Django app (created manually)
```

---

## 2. **Creating and Managing Apps**

### Create an App
```bash
python manage.py startapp app_name
```

### Register App in `settings.py`
```python
# Add your app to the INSTALLED_APPS list
INSTALLED_APPS = [
    ...
    'app_name',
]
```

---

## 3. **Models**

### Define a Model
```python
from django.db import models

class Post(models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
```

### Apply Migrations
```bash
# Create migration files
python manage.py makemigrations

# Apply migrations to the database
python manage.py migrate
```

---

## 4. **Admin Interface**

### Register Models in Admin
```python
from django.contrib import admin
from .models import Post

admin.site.register(Post)
```

### Access Admin Panel
- Run the server: `python manage.py runserver`
- Open in a browser: [http://127.0.0.1:8000/admin](http://127.0.0.1:8000/admin)

---

## 5. **Views**

### Function-Based Views
```python
from django.http import HttpResponse

def home(request):
    return HttpResponse("Welcome to Django!")
```

### Class-Based Views
```python
from django.views import View
from django.http import HttpResponse

class HomeView(View):
    def get(self, request):
        return HttpResponse("Welcome to Django!")
```

---

## 6. **URLs**

### Configure URLs in `urls.py`
```python
from django.contrib import admin
from django.urls import path
from app_name import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),  # Home view
]
```

---

## 7. **Templates**

### Create a Template
- Place templates in `app_name/templates/app_name/`.

```html
<!-- app_name/templates/app_name/home.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Home</title>
</head>
<body>
    <h1>Welcome to Django</h1>
</body>
</html>
```

### Render a Template
```python
from django.shortcuts import render

def home(request):
    return render(request, 'app_name/home.html')
```

---

## 8. **Static Files**

### Configure Static Files in `settings.py`
```python
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / "static"]
```

### Use Static Files
- Create a `static` folder in your app.
- Load static files in templates:
```html
% load static %
<img src="% static 'images/logo.png' %" alt="Logo">
```

---

## 9. **Forms**

### Create a Form
```python
from django import forms

class PostForm(forms.Form):
    title = forms.CharField(max_length=100)
    content = forms.CharField(widget=forms.Textarea)
```

### Handle Form in Views
```python
from django.shortcuts import render
from .forms import PostForm

def create_post(request):
    if request.method == 'POST':
        form = PostForm(request.POST)
        if form.is_valid():
            # Process the form
            pass
    else:
        form = PostForm()
    return render(request, 'app_name/create_post.html', 'form': form)
```

---

## 10. **User Authentication**

### Add Login and Logout Views
```python
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
]
```

### Create Login Template
```html
<!-- registration/login.html -->
<form method="post">
    % csrf_token %
    <input type="text" name="username" placeholder="Username">
    <input type="password" name="password" placeholder="Password">
    <button type="submit">Login</button>
</form>
```

---

## 11. **Testing**

### Write Tests
```python
from django.test import TestCase
from .models import Post

class PostModelTest(TestCase):
    def test_string_representation(self):
        post = Post(title="My Post")
        self.assertEqual(str(post), "My Post")
```

### Run Tests
```bash
python manage.py test
```

---

## 12. **Common Commands**

| Command                            | Description                               |
|------------------------------------|-------------------------------------------|
| `python manage.py startproject`    | Create a new project.                    |
| `python manage.py startapp`        | Create a new app.                        |
| `python manage.py runserver`       | Start the development server.            |
| `python manage.py makemigrations`  | Create migrations for changes to models. |
| `python manage.py migrate`         | Apply migrations to the database.        |
| `python manage.py createsuperuser`| Create an admin user.                    |
| `python manage.py test`            | Run tests.                               |

                ''')
    
with tab2:
    st.header('Streamlit', divider='rainbow')
    st.image("assets/StreamlitLogo.png", use_container_width=True)
    st.markdown(f'''

## 1. **Setting Up Streamlit**

### Installation
```bash
# Install Streamlit
pip install streamlit

# Check Streamlit version
streamlit --version

### Running a Streamlit App
```bash
# Run the Streamlit app
streamlit run app.py
```

---

## 2. **Basic Streamlit App Structure**

### Example: `app.py`
```python
import streamlit as st

st.title("Streamlit App")
st.header("Welcome to Streamlit!")
st.text("This is a simple app.")
```

Run it using:
```bash
streamlit run app.py
```

---

## 3. **Core Components**

### Text Elements
```python
st.title("This is a title")
st.header("This is a header")
st.subheader("This is a subheader")
st.text("This is some text")
st.markdown("**Markdown** is *also* supported!")
st.code("print('Hello, Streamlit!')", language="python")
```

### Media Elements
```python
st.image("image.png", caption="This is an image")
st.audio("audio.mp3")
st.video("video.mp4")
```

### Data Display
```python
import pandas as pd

data = pd.DataFrame(
    'Column A': [1, 2, 3],
    'Column B': [4, 5, 6]
)

st.dataframe(data)  # Interactive table
st.table(data)      # Static table
st.json("key": "value")
```

---

## 4. **Widgets**

### Input Widgets
```python
name = st.text_input("Enter your name")
age = st.number_input("Enter your age", min_value=0, max_value=100, step=1)
hobby = st.selectbox("Select your hobby", ["Reading", "Gaming", "Hiking"])
agree = st.checkbox("I agree")
date = st.date_input("Pick a date")
```

### Button Widgets
```python
if st.button("Click me!"):
    st.write("Button clicked!")
```

### Slider
```python
value = st.slider("Pick a number", min_value=0, max_value=100, value=50)
```

### File Uploader
```python
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    st.write("File uploaded!")
```

---

## 5. **Layouts and Containers**

### Columns
```python
col1, col2 = st.columns(2)

with col1:
    st.write("Column 1 content")

with col2:
    st.write("Column 2 content")
```

### Expander
```python
with st.expander("Click to expand"):
    st.write("This is hidden content")
```

### Sidebar
```python
st.sidebar.title("Sidebar Title")
st.sidebar.button("Click me!")
```

---

## 6. **State Management**

### Session State
```python
if "counter" not in st.session_state:
    st.session_state.counter = 0

if st.button("Increase"):
    st.session_state.counter += 1

st.write("Counter value:", st.session_state.counter)
```

---

## 7. **Charts and Visualizations**

### Built-in Charting
```python
import pandas as pd
import numpy as np

data = pd.DataFrame(
    np.random.randn(10, 2),
    columns=["A", "B"]
)

st.line_chart(data)
st.bar_chart(data)
st.area_chart(data)
```

### Custom Plotting with Matplotlib
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
st.pyplot(fig)
```

### Plotly
```python
import plotly.express as px

data = px.data.iris()
fig = px.scatter(data, x="sepal_width", y="sepal_length", color="species")
st.plotly_chart(fig)
```

---

## 8. **Interactivity**

### Widgets and Callbacks
```python
name = st.text_input("Enter your name")
if st.button("Submit"):
    st.write(f"Hello, name!")
```

### Real-Time Updates
```python
import time

for i in range(100):
    st.write(f"Processing i%")
    time.sleep(0.1)
```

---

## 9. **Advanced Features**

### Caching
```python
@st.cache
def expensive_computation(x):
    time.sleep(2)  # Simulates a long computation
    return x * 2

result = expensive_computation(10)
st.write("Result:", result)
```

### Theming
```bash
# In the terminal
streamlit config show
```
Edit `~/.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#ff4b4b"
backgroundColor = "#f0f0f0"
textColor = "#333333"
```

---

## 10. **Deployment**

### Deploy with Streamlit Cloud
1. Push your code to a GitHub repository.
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud).
3. Create a new app and link your repository.

---

## 11. **Common Commands**

| Command                     | Description                              |
|-----------------------------|------------------------------------------|
| `streamlit run app.py`      | Run the app.                            |
| `streamlit hello`           | Launch the Streamlit demo app.          |
| `streamlit config show`     | Show the current configuration.         |
| `streamlit cache clear`     | Clear the cache.                        |
                ''')
