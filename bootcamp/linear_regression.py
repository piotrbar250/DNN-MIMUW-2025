import random
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def mse_loss(ys, ps):
    assert len(ys) == len(ps)
    return sum((y-p) ** 2 for y, p in zip(ys, ps, strict=True)) / len(ys)

def predict(a, b, xs):
    return [a * x + b for x in xs]

def train(xs, ys):
    a = 0.
    b = 0.
    lr = 0.5
    n_epochs = 4000

    for i in range(n_epochs):
        ps = predict(a, b, xs)

        if i % 50 == 0:
            print(f"Loss: {mse_loss(ys, ps)}")

        grad_a = sum((y - p) * x for y, p, x in zip(ys, ps, xs, strict=True)) * (-2) / len(ys)
        grad_b = sum(y - p for y, p in zip(ys, ps, strict=True)) * (-2) / len(ys)
        
        a = a - lr * grad_a
        b = b - lr * grad_b
    
    return a, b

_a = -0.3
_b = 0.5

f = lambda x: _a * x + _b
g = lambda x: f(x) + random.gauss(0, 0.01)

n = 50
xs = [random.random() for _ in range(n)]

ys = list(map(g, xs))
ts = list(map(f, xs))

a, b = train(xs, ys)

fig = go.Figure()

fig = px.scatter(x=xs, y=ys)
dense_x = np.linspace(np.min(xs), np.max(xs), 100)
fig.add_trace(go.Scatter(x=dense_x, y=predict(a, b, dense_x), name='linear fit', mode='lines'))
fig.add_trace(go.Scatter(x=xs, y=ts, name='y without noise', mode='markers'))

fig.show()