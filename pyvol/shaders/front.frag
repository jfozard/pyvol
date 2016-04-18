varying vec3 v_texcoord;
varying vec4 v_pos;

uniform sampler2D backfaceTex;            
uniform sampler3D texture_3d;
const float falloff = 0.995;

void main() {
    vec2 texc = (v_pos.xy/v_pos.w +1.0)/2.0; //((/gl_FragCoord.w) + 1) / 2;
    vec3 startPos = v_texcoord;
    vec3 endPos = texture2D(backfaceTex, texc).rgb;
    vec3 ray = endPos - startPos;
    float rayLength = length(ray);
    vec3 step = normalize(ray)*(2.0/600.0);
    vec4 colAcc = vec4(0,0,0,0);
    float sample;
    vec3 samplePos = vec3(0,0,0); 
    for (int i=0; i<1000; i++)
    {
        sample = texture3D(texture_3d, endPos - samplePos).x;
        colAcc.rgb = mix(colAcc.rgb, vec3(1.0, 0.0, 0.0), sample*0.1);
        colAcc.a = mix(colAcc.a, 1.0, sample*0.1);
        colAcc *= falloff;

        if ((length(samplePos) >= rayLength))
            break;
        //if(colAcc.a>0.99) {
        //    colAcc.a = 1.0;
        //    colAcc.rgb = vec3(0,1,0);
        //    break;
        //}
        samplePos += step;
    }
    gl_FragColor = colAcc;

}
