attribute vec3 position;
attribute vec3 texcoord;

varying vec3 v_texcoord;
varying vec4 v_pos;

uniform mat4 mv_matrix;
uniform mat4 p_matrix;
void main() {
    vec4 eye =  mv_matrix * vec4(position,1.0);
    v_pos = p_matrix * eye;
    gl_Position = v_pos;
    v_texcoord = texcoord;
}
