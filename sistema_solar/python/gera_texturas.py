from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "images"
WIDTH = 512
HEIGHT = 256


def gerar_sol():
  rng = random.Random(10)
  image = Image.new("RGB", (WIDTH, HEIGHT), (245, 165, 35))
  draw = ImageDraw.Draw(image)
  for _ in range(180):
    x = rng.randrange(WIDTH)
    y = rng.randrange(HEIGHT)
    radius = rng.randint(3, 16)
    color = rng.choice([(230, 125, 20), (250, 190, 50), (255, 210, 80)])
    for offset in [-WIDTH, 0, WIDTH]:
      draw.ellipse((x + offset - radius, y - radius,
                    x + offset + radius, y + radius), fill=color)
  image.filter(ImageFilter.GaussianBlur(2)).save(OUT / "sun.png")


def gerar_mapa_normal(height_image, filename):
  height = np.asarray(height_image, dtype=np.float32) / 255.0
  dy, dx = np.gradient(height)
  # A bitangente da esfera aponta no sentido contrário ao eixo y da imagem.
  normal = np.stack((-dx * 6, dy * 6, np.ones_like(height)), axis=-1)
  normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
  pixels = np.round((normal * 0.5 + 0.5) * 255).astype(np.uint8)
  Image.fromarray(pixels).save(OUT / filename)


def gerar_rochoso(name, color, seed):
  rng = random.Random(seed)
  image = Image.new("RGB", (WIDTH, HEIGHT), color)
  heights = Image.new("L", (WIDTH, HEIGHT), 128)
  draw = ImageDraw.Draw(image)
  height_draw = ImageDraw.Draw(heights)

  for _ in range(110):
    radius = rng.randint(3, 15)
    x = rng.randrange(WIDTH)
    y = rng.randint(radius, HEIGHT - radius - 1)
    shade = rng.randint(-30, 10)
    crater_color = tuple(max(0, min(255, c + shade)) for c in color)
    rim_color = tuple(min(255, c + 30) for c in color)
    for offset in [-WIDTH, 0, WIDTH]:
      cx = x + offset
      outer = (cx - radius, y - radius, cx + radius, y + radius)
      inner = (cx - radius + 2, y - radius + 2,
               cx + radius - 2, y + radius - 2)
      draw.ellipse(outer, fill=crater_color, outline=rim_color, width=2)
      height_draw.ellipse(outer, fill=155)
      height_draw.ellipse(inner, fill=100)

  image.filter(ImageFilter.GaussianBlur(0.5)).save(OUT / f"{name}.png")
  heights = heights.filter(ImageFilter.GaussianBlur(1))
  gerar_mapa_normal(heights, f"{name}-normal.png")


def gerar_fundo():
  rng = random.Random(30)
  face = 512
  image = Image.new("RGB", (4 * face, 3 * face), (2, 3, 9))
  draw = ImageDraw.Draw(image)
  # As seis imagens ficam na cruz utilizada pela classe TexCube.
  cells = [(2, 1), (0, 1), (1, 0), (1, 2), (1, 1), (3, 1)]
  for column, row in cells:
    for _ in range(500):
      x = column * face + rng.randrange(2, face - 2)
      y = row * face + rng.randrange(2, face - 2)
      brightness = rng.randint(90, 220)
      color = (brightness, brightness, min(255, brightness + 15))
      if rng.random() < 0.04:
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
      else:
        draw.point((x, y), fill=color)
  image.save(OUT / "space.png")


if __name__ == "__main__":
  OUT.mkdir(parents=True, exist_ok=True)
  gerar_sol()
  gerar_rochoso("moon", (155, 155, 150), 20)
  gerar_rochoso("mercury", (145, 130, 115), 21)
  gerar_fundo()
  print("Texturas salvas em", OUT)
