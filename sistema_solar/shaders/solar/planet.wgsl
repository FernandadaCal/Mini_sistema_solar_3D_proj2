struct Matrix {
  vertex: mat4x4<f32>,
  normal: mat4x4<f32>,
}
@group(1) @binding(0) var<storage, read> matrix: array<Matrix>;

struct Global {
  projection: mat4x4<f32>,
  light_ambient: vec3<f32>,
  light_diffuse: vec3<f32>,
  light_specular: vec3<f32>,
  light_position: vec4<f32>,
  camera_position: vec4<f32>,
  clip_plane: vec4<f32>,
  clip_plane_color: vec4<f32>,
  bump_enabled: f32,
}
@group(0) @binding(0) var<uniform> global: Global;

struct Material {
  base_color: vec3<f32>,
  opacity: f32,

  specular_color: vec3<f32>,
  shininess: f32,
}
@group(2) @binding(0) var<uniform> material: Material;

@group(3) @binding(0) var decal_texture: texture_2d<f32>;
@group(3) @binding(1) var normal_texture: texture_2d<f32>;
@group(3) @binding(2) var tex_sampler: sampler;

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) tangent: vec3<f32>,
  @location(3) texcoord: vec2<f32>,
  @location(4) clip_dist: f32,
}

@vertex
fn vs_main (@builtin(instance_index) instance_index: u32, @location(0) coord: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) tangent: vec3<f32>, @location(3) texcoord: vec2<f32>) -> VertexOutput {
  let m = matrix[instance_index];
  var out: VertexOutput;
  let p = m.vertex * vec4<f32>(coord, 1.0);
  out.position = p.xyz;
  out.clip_dist = dot(global.clip_plane, p);
  out.normal = (m.normal * vec4<f32>(normal, 0.0)).xyz;
  out.tangent = (m.vertex * vec4<f32>(tangent, 0.0)).xyz;
  out.texcoord = texcoord;
  out.clip_position = global.projection * p;
  return out;
}

@fragment
fn fs_main (in: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
  let decal = textureSample(decal_texture, tex_sampler, in.texcoord);
  let bump = textureSample(normal_texture, tex_sampler, in.texcoord).xyz * 2.0 - 1.0;

  if (in.clip_dist < 0.0) {
    discard;
  }
  if (!front_facing) {
    return global.clip_plane_color;
  }

  let ngeom = normalize(in.normal);
  var normal = ngeom;
  if (global.bump_enabled > 0.5) {
    // Base tangente para levar a normal da textura ao espaço do mundo.
    let t = normalize(in.tangent - dot(in.tangent, ngeom) * ngeom);
    let b = cross(ngeom, t);
    normal = normalize(bump.x * t + bump.y * b + bump.z * ngeom);
  }
  let light = normalize(global.light_position.xyz - in.position);
  let view = normalize(global.camera_position.xyz - in.position);

  let base = material.base_color * decal.rgb;
  var color = base * global.light_ambient;
  let ndotl = dot(normal, light);
  if (ndotl > 0.0 && dot(ngeom, light) > 0.0) {
    color += base * global.light_diffuse * ndotl;
    let refl = normalize(reflect(-light, normal));
    color += material.specular_color * global.light_specular * pow(max(0.0, dot(refl, view)), material.shininess);
  }
  return vec4<f32>(color, 1.0);
}
