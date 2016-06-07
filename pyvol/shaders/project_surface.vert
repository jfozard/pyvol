
attribute vec3 position;
attribute vec3 normal;
attribute vec3 color;

varying vec3 v_texcoord;
varying vec3 v_texnormal;
            
varying vec3 v_normal;
varying vec3 v_color;

uniform mat4 tex_matrix;
uniform mat4 mv_matrix;
uniform mat4 p_matrix;

uniform float depth_start;
uniform bool move_surface;

void main() {
  vec3 new_pos;
  if(move_surface) {
    new_pos = position + depth_start*normal;
  } else {
    new_pos = position;
  }
  vec4 eye =  mv_matrix * vec4(new_pos, 1.0);
  v_color = color;
  v_normal = (mv_matrix * vec4(normal, 0.0)).xyz;

  v_texcoord = (tex_matrix *vec4(position, 1.0)).xyz;
  v_texnormal = (tex_matrix *vec4(normalize(normal), 0.0)).xyz;

  gl_Position = p_matrix * eye;
}
