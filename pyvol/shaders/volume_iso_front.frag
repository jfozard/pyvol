
varying vec3 v_texcoord;
varying vec4 v_pos;

uniform float isolevel;   
uniform sampler2D backfaceTex;
uniform sampler3D texture_3d;

uniform mat4 mv_matrix;
uniform mat4 tex_inv_matrix;

const float eps = 0.001;

vec3 normal_calc(vec3 p, float u) {
    float dx = texture3D(texture_3d, p + vec3(eps,0,0)).x - u;
    float dy = texture3D(texture_3d, p + vec3(0,eps,0)).x - u;
    float dz = texture3D(texture_3d, p + vec3(0,0,eps)).x - u;
    return vec3(dx, dy, dz);              
}

void main() {
  vec2 texc = (v_pos.xy/v_pos.w +1.0)/2.0; //((/gl_FragCoord.w) + 1) / 2;
  vec3 startPos = v_texcoord;
  vec3 endPos = texture2D(backfaceTex, texc).rgb;
  vec3 ray = endPos - startPos;
  float rayLength = length(ray);
  vec3 step = normalize(ray)*(2.0/1200.0);
  vec4 col;
  float sample;
  vec3 samplePos = vec3(0,0,0); 
  vec4 sp;
  gl_FragDepth = 1.0; //gl_FragCoord.z;
  for(int i=0; i<1200;i++) {
    if ((length(samplePos) >= rayLength)) {
      discard;
      break;
    }
    sample = texture3D(texture_3d, startPos + samplePos).x;
    if(sample>isolevel) {
      vec3 n = normal_calc(startPos + samplePos, sample);
      n = normalize((mv_matrix * vec4(n, 0.0)).xyz);
      col = vec4(0.5*(1.0+n.x)*vec3((startPos+samplePos).x, (startPos+samplePos).y, 1.0), 1.0);
      gl_FragColor = col;
      sp = tex_inv_matrix*vec4(startPos + samplePos, 1.0);
      gl_FragDepth = 0.5*(1.0+sp.z/sp.w);
      break;
    }
    samplePos += step;
  }
}
