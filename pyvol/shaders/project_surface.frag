
varying vec3 v_texcoord;
varying vec3 v_texnormal;
            
varying vec3 v_color;
varying vec3 v_normal;

uniform mat4 tex_matrix;

uniform sampler3D texture_3d;

uniform float depth_start;
uniform float depth_end;
            
const vec3 light_direction =  vec3(0., 0., -1.);       
const vec4 light_diffuse = vec4(0.7, 0.7, 0.7, 0.0);
const vec4 light_ambient = vec4(0.3, 0.3, 0.3, 1.0);   

uniform float sample_gain;
uniform float alpha_project;

void main() {
  vec3 tn = v_texnormal;
  vec3 startPos = v_texcoord + depth_start*tn;
  vec3 step = (depth_end - depth_start)*tn/19.0;
  vec4 colAcc = vec4(0,0,0,0);
  vec3 currentPos = startPos;
          
  float total_sample = 0.0;
  // Sample stack (3D texture) along ray
  for (int i=0; i<10; i++) {
    total_sample += texture3D(texture_3d, currentPos.xyz).x;
    currentPos += step;
  }
  // Average and scale samples
  float mean_sample = clamp(0.1*sample_gain*total_sample, 0.0, 1.0);

  vec4 projected_color = vec4(startPos.y, mean_sample, 0.0, 1.0);
  // Find surface color
  vec3 normal = normalize(v_normal);
  vec4 diffuse_factor = max(-dot(normal, light_direction), 0.0) * light_diffuse;
  vec4 diffuse_color = (diffuse_factor + light_ambient)*vec4(v_color, 1.0);
  // Combine surface and projected color
  gl_FragColor = mix(projected_color, diffuse_color, alpha_project);
}
