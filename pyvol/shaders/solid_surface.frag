varying vec3 v_color;
varying vec3 v_normal;

const vec3 light_direction = vec3(0., 0., -1.);       
const vec4 light_diffuse = vec4(0.7, 0.7, 0.7, 0.0);
const vec4 light_ambient = vec4(0.3, 0.3, 0.3, 1.0);   

void main() {
  // Find surface color
  vec3 normal = normalize(v_normal);
  vec4 diffuse_factor = max(-dot(normal, light_direction), 0.0) * light_diffuse;
  vec4 diffuse_color = (diffuse_factor + light_ambient)*vec4(v_color, 1.0);
  // Combine surface and projected color
  gl_FragColor = diffuse_color;                
}
