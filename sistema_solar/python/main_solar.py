from __future__ import annotations


import time
from pathlib import Path
from typing import Any

import graphics_math as gm
import wgpu
from rendercanvas.glfw import RenderCanvas, loop

from camera3d import Camera3D
from pointlight import PointLight
from shader import Shader
from pipeline import Pipeline
from phongmaterial import PhongMaterial
from texture import Texture
from texcube import TexCube
from sampler import Sampler
from textureset import TextureSet
from transform import Transform
from node import Node
from scene import Scene
from renderer import Renderer
from engine import Engine
from sphere import Sphere
from skybox import SkyBox, SkyBoxTransform
from clipplane import ClipPlane

canvas: RenderCanvas
device: wgpu.GPUDevice
context: Any
renderer: Renderer
scene: Scene
global_camera: Camera3D
moon_camera: Camera3D
camera: Camera3D
solar: SolarSystemEngine
clip_planes: list[ClipPlane]
shd_planet: Shader
last_t: float = 0.0

ASSETS_DIR = Path(__file__).resolve().parent.parent

clip_enabled = False
bump_enabled = True

# Dimensões fora de escala para facilitar a visualização.
SUN_RADIUS = 2.5
MERCURY_RADIUS = 0.35
MERCURY_DISTANCE = 5.0
EARTH_RADIUS = 0.8
EARTH_DISTANCE = 10.0
EARTH_TILT = 23.4
MOON_RADIUS = 0.25
MOON_DISTANCE = 2.0
MOON_INCLINATION = 5.1

CLIP_PLANE = (0.0, -1.0, 0.0, 0.0)
NO_CLIP_PLANE = (0.0, 0.0, 0.0, 1.0)


class SolarSystemEngine (Engine):

  SUN_SPIN = 25.0
  MERCURY_YEAR = 8.0
  MERCURY_SPIN = 20.0
  EARTH_YEAR = 40.0
  EARTH_DAY = 4.0
  MOON_MONTH = 10.0

  def __init__ (self, trf_sun: Transform,
                trf_mercury_orbit: Transform, trf_mercury: Transform,
                trf_earth_orbit: Transform, trf_earth: Transform,
                trf_moon_orbit: Transform, trf_moon: Transform) -> None:
    self.trf_sun = trf_sun
    self.trf_mercury_orbit = trf_mercury_orbit
    self.trf_mercury = trf_mercury
    self.trf_earth_orbit = trf_earth_orbit
    self.trf_earth = trf_earth
    self.trf_moon_orbit = trf_moon_orbit
    self.trf_moon = trf_moon
    self.time = 0.0
    self.speed = 1.0
    self.paused = False
    self.update(0.0)

  def angle (self, period: float, phase: float = 0.0) -> float:
    return (360.0 * self.time / period + phase) % 360.0

  def update (self, dt: float) -> None:
    if not self.paused:
      self.time += dt * self.speed

    self.trf_sun.load_identity()
    self.trf_sun.rotate(self.angle(self.SUN_SPIN), 0, 1, 0)
    self.trf_sun.scale(SUN_RADIUS, SUN_RADIUS, SUN_RADIUS)

    self.trf_mercury_orbit.load_identity()
    self.trf_mercury_orbit.rotate(7.0, 1, 0, 0)
    self.trf_mercury_orbit.rotate(self.angle(self.MERCURY_YEAR, 120.0), 0, 1, 0)
    self.trf_mercury_orbit.translate(MERCURY_DISTANCE, 0, 0)
    self.trf_mercury.load_identity()
    self.trf_mercury.rotate(self.angle(self.MERCURY_SPIN), 0, 1, 0)
    self.trf_mercury.scale(MERCURY_RADIUS, MERCURY_RADIUS, MERCURY_RADIUS)

    year = self.angle(self.EARTH_YEAR)
    self.trf_earth_orbit.load_identity()
    self.trf_earth_orbit.rotate(year, 0, 1, 0)
    self.trf_earth_orbit.translate(EARTH_DISTANCE, 0, 0)
    self.trf_earth.load_identity()
    # Mantém a direção do eixo da Terra durante a translação.
    self.trf_earth.rotate(-year, 0, 1, 0)
    self.trf_earth.rotate(EARTH_TILT, 0, 0, 1)
    self.trf_earth.rotate(self.angle(self.EARTH_DAY), 0, 1, 0)
    self.trf_earth.scale(EARTH_RADIUS, EARTH_RADIUS, EARTH_RADIUS)

    # A Lua acompanha a órbita da Terra, sem herdar a rotação diária.
    self.trf_moon_orbit.load_identity()
    self.trf_moon_orbit.rotate(-year, 0, 1, 0)
    self.trf_moon_orbit.rotate(MOON_INCLINATION, 1, 0, 0)
    self.trf_moon_orbit.rotate(self.angle(self.MOON_MONTH, 90.0), 0, 1, 0)
    self.trf_moon_orbit.translate(MOON_DISTANCE, 0, 0)
    self.trf_moon.load_identity()
    self.trf_moon.scale(MOON_RADIUS, MOON_RADIUS, MOON_RADIUS)


class EarthMoonCameraEngine (Engine):

  def __init__ (self, camera: Camera3D, earth: Node, moon: Node) -> None:
    self.camera = camera
    self.earth = earth
    self.moon = moon

  def update (self, dt: float) -> None:
    origin = gm.vec4(0, 0, 0, 1)
    e = (self.earth.get_model_matrix() @ origin)[:3]
    m = (self.moon.get_model_matrix() @ origin)[:3]
    d = gm.normalize(m - e)
    # O olho fica um pouco acima da superfície, voltado para a Lua.
    eye = e + d * (EARTH_RADIUS * 1.02)
    self.camera.set_eye(eye[0], eye[1], eye[2])
    self.camera.set_center(m[0], m[1], m[2])


def initialize (device: wgpu.GPUDevice, target_format: str) -> None:
  global scene, global_camera, moon_camera, camera, solar, clip_planes, shd_planet

  global_camera = Camera3D(0.0, 10.0, 20.0)
  global_camera.set_z_planes(0.1, 200.0)
  arcball = global_camera.create_arcball()
  arcball.attach(canvas)
  moon_camera = Camera3D(0.0, 0.0, 1.0)
  moon_camera.set_angle(30.0)
  moon_camera.set_z_planes(0.01, 200.0)
  camera = global_camera

  light = PointLight(0.0, 0.0, 0.0, space="world")
  light.set_ambient(0.04, 0.04, 0.05)
  light.set_diffuse(1.0, 1.0, 1.0)
  light.set_specular(1.0, 1.0, 1.0)

  sphere_buffers = [
    {"array_stride": 3 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x3", "offset": 0, "var_name": "coord"}]},
    {"array_stride": 3 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x3", "offset": 0, "var_name": "normal"}]},
    {"array_stride": 3 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x3", "offset": 0, "var_name": "tangent"}]},
    {"array_stride": 2 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x2", "offset": 0, "var_name": "texcoord"}]},
  ]
  shd_planet = Shader(device, str(ASSETS_DIR / "shaders/solar/planet.wgsl"), light=light, space="world")
  shd_planet.set_vertex_buffers(sphere_buffers)
  shd_planet.set_value("bump_enabled", 1.0 if bump_enabled else 0.0)
  # Desenha os dois lados para mostrar o interior no corte.
  pip_planet = Pipeline(shd_planet, target_format, primitive={"cull_mode": "none"})

  shd_sun = Shader(device, str(ASSETS_DIR / "shaders/solar/sun.wgsl"), space="world")
  shd_sun.set_vertex_buffers(sphere_buffers)
  pip_sun = Pipeline(shd_sun, target_format, primitive={"cull_mode": "none"})

  shd_sky = Shader(device, str(ASSETS_DIR / "shaders/solar/sky.wgsl"), space="world")
  shd_sky.set_vertex_buffers([
    {"array_stride": 3 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x3", "offset": 0, "var_name": "coord"}]},
  ])
  pip_sky = Pipeline(shd_sky, target_format, primitive={"cull_mode": "none"},
                     depth_stencil={"depth_write_enabled": False})

  clip_planes = [ClipPlane(shd_planet, *NO_CLIP_PLANE), ClipPlane(shd_sun, *NO_CLIP_PLANE)]
  clip_planes[0].set_color(0.9, 0.35, 0.1)
  clip_planes[1].set_color(1.0, 0.85, 0.3)

  sampler = Sampler(device, "tex_sampler", address_mode_v="clamp-to-edge")
  flat = Texture(device, "normal_texture", None, texel=gm.vec3(0.5, 0.5, 1.0), srgb=False)

  def planet_textures (decal: str, normal: str | None) -> TextureSet:
    normal_tex = Texture(device, "normal_texture", normal, srgb=False) if normal else flat
    ts = TextureSet([Texture(device, "decal_texture", decal), normal_tex, sampler])
    shd_planet.add_texture_set(ts)
    return ts

  tex_mercury = planet_textures(str(ASSETS_DIR / "images/mercury.png"), str(ASSETS_DIR / "images/mercury-normal.png"))
  tex_earth = planet_textures(str(ASSETS_DIR / "images/earth.jpg"), str(ASSETS_DIR / "images/earth-normal.png"))
  tex_moon = planet_textures(str(ASSETS_DIR / "images/moon.png"), str(ASSETS_DIR / "images/moon-normal.png"))

  tex_sun = TextureSet([Texture(device, "decal_texture", str(ASSETS_DIR / "images/sun.png")),
                        Sampler(device, "tex_sampler", address_mode_v="clamp-to-edge")])
  shd_sun.add_texture_set(tex_sun)
  tex_sky = TextureSet([TexCube(device, "sky_texture", str(ASSETS_DIR / "images/space.png")),
                        Sampler(device, "sky_sampler")])
  shd_sky.add_texture_set(tex_sky)

  rocky = PhongMaterial(1.0, 1.0, 1.0)
  rocky.set_specular(0.05, 0.05, 0.05)
  rocky.set_shininess(8.0)
  earth_mat = PhongMaterial(1.0, 1.0, 1.0)
  earth_mat.set_specular(0.35, 0.35, 0.35)
  earth_mat.set_shininess(40.0)
  shd_planet.add_material(rocky)
  shd_planet.add_material(earth_mat)

  sphere = Sphere(device)
  trf_sun = Transform()
  trf_mercury_orbit = Transform()
  trf_mercury = Transform()
  trf_earth_orbit = Transform()
  trf_earth = Transform()
  trf_moon_orbit = Transform()
  trf_moon = Transform()

  moon = Node(None, trf_moon, [rocky, tex_moon], [sphere])
  earth_orbit = Node(None, trf_earth_orbit, nodes=[
                       Node(None, trf_earth, [earth_mat, tex_earth], [sphere]),
                       Node(None, trf_moon_orbit, nodes=[moon]),
                     ])
  root = Node(nodes=[
           Node(pip_sky, SkyBoxTransform(), [tex_sky], [SkyBox(device)]),
           Node(pip_sun, trf_sun, [tex_sun], [sphere]),
           Node(pip_planet, nodes=[
             Node(None, trf_mercury_orbit, nodes=[
               Node(None, trf_mercury, [rocky, tex_mercury], [sphere]),
             ]),
             earth_orbit,
           ]),
         ])
  scene = Scene(root)

  solar = SolarSystemEngine(trf_sun, trf_mercury_orbit, trf_mercury,
                            trf_earth_orbit, trf_earth, trf_moon_orbit, trf_moon)
  scene.add_engine(solar)
  # A câmera lê as posições depois que a animação foi atualizada.
  scene.add_engine(EarthMoonCameraEngine(moon_camera, earth_orbit, moon))


def update_title () -> None:
  name = "global (arcball)" if camera is global_camera else "Terra -> Lua"
  flags = []
  if clip_enabled:
    flags.append("plano de corte")
  if bump_enabled:
    flags.append("rugosidade")
  if solar.paused:
    flags.append("pausado")
  canvas.set_title(f"Sistema solar - camera {name} - {solar.speed:g}x" +
                   (" - " + ", ".join(flags) if flags else ""))

def draw () -> None:
  global last_t
  t = time.perf_counter()
  scene.update(t - last_t)
  last_t = t

  target_texture = context.get_current_texture()
  renderer.render(target_texture, scene, camera)

def on_key (event: Any) -> None:
  global camera, clip_enabled, bump_enabled
  key = event["key"]
  if key == "q":
    canvas.close()
    return
  if key == "c":
    camera = moon_camera if camera is global_camera else global_camera
  elif key == "p":
    clip_enabled = not clip_enabled
    for cp in clip_planes:
      cp.set_plane(*(CLIP_PLANE if clip_enabled else NO_CLIP_PLANE))
  elif key == "b":
    bump_enabled = not bump_enabled
    shd_planet.set_value("bump_enabled", 1.0 if bump_enabled else 0.0)
  elif key == " ":
    solar.paused = not solar.paused
  elif key == "ArrowUp":
    solar.speed = min(solar.speed * 2.0, 64.0)
  elif key == "ArrowDown":
    solar.speed = max(solar.speed / 2.0, 1.0 / 16.0)
  update_title()

def on_wheel (event: Any) -> None:
  fovy = camera.get_angle() * (1.0 + event["dy"] * 0.001)
  camera.set_angle(min(max(fovy, 5.0), 90.0))

def main () -> None:
  global canvas, device, context, renderer, last_t

  canvas = RenderCanvas(size=(1024, 640), title="Sistema solar", update_mode="continuous", max_fps=60)
  adapter = wgpu.gpu.request_adapter_sync()
  device = adapter.request_device_sync()
  context = canvas.get_context("wgpu")
  target_format = context.get_preferred_format(device.adapter)
  context.configure(device=device, format=target_format)

  renderer = Renderer(device, depth_test=True, clear_value=(0.0, 0.0, 0.0, 1.0))

  initialize(device, target_format)
  update_title()
  print("Teclas: C camera | P plano de corte | B rugosidade | espaco pausa | "
        "setas cima/baixo velocidade | roda zoom | Q sai")

  canvas.add_event_handler(on_key, "key_down")
  canvas.add_event_handler(on_wheel, "wheel")
  last_t = time.perf_counter()
  canvas.request_draw(draw)
  loop.run()

if __name__ == "__main__":
  main()
