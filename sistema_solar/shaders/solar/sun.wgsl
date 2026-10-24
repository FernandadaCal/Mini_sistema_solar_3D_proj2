struct Matrix {
  vertex: mat4x4<f32>,
  normal: mat4x4<f32>,
}
@group(1) @binding(0) var<storage, read> matrix: array<Matrix>;

struct Global {
  projection: mat4x4<f32>,
  clip_plane: vec4<f32>,
  clip_plane_color: vec4<f32>,
}
@group(0) @binding(0) var<uniform> global: Global;

@group(2) @binding(0) var decal_texture: texture_2d<f32>;
@group(2) @binding(1) var tex_sampler: sampler;

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) texcoord: vec2<f32>,
  @location(1) clip_dist: f32,
}

@vertex
fn vs_main(
  @builtin(instance_index) instance_index: u32,
  @location(0) coord: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) tangent: vec3<f32>,
  @location(3) texcoord: vec2<f32>
) -> VertexOutput {
  let p = matrix[instance_index].vertex * vec4<f32>(coord, 1.0);
  var out: VertexOutput;
  out.clip_position = global.projection * p;
  out.texcoord = texcoord;
  out.clip_dist = dot(global.clip_plane, p);
  return out;
}

@fragment
fn fs_main(
  in: VertexOutput,
  @builtin(front_facing) front_facing: bool
) -> @location(0) vec4<f32> {
  let decal = textureSample(decal_texture, tex_sampler, in.texcoord);
  if (in.clip_dist < 0.0) {
    discard;
  }
  if (!front_facing) {
    return global.clip_plane_color;
  }
  // O Sol exibe a textura sem receber a iluminação dos planetas.
  return vec4<f32>(decal.rgb, 1.0);
}
