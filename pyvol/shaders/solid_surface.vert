attribute vec3 position;
attribute vec3 normal;
attribute vec3 color;
            
varying vec3 v_normal;
varying vec3 v_color;

uniform mat4 mv_matrix;
uniform mat4 p_matrix;


void main() {
  vec4 eye =  mv_matrix * vec4(position, 1.0);
  v_color = color;
  v_normal = (mv_matrix * vec4(normal, 0.0)).xyz;
  gl_Position = p_matrix * eye;
}
