
float sdf(in vec3 p) {
    //vec2 q = vec2(length(p.xz) - 3.0, p.y);
    //return length(q) - 2.0;

    pMod3(p, vec3(5.0));
    
    float box = fBox(p,vec3(1));
    float sphere = length(p-vec3(1))-1;

    float r = 0.3;
    float n = 4;

    return fOpUnionStairs(box,sphere,r,n);
}
