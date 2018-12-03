const float twopi = 6.283;

float torus(in vec3 p) {
    vec2 q = vec2(length(p.xz) - 3.0, p.y);
    return length(q) - 2.0;
}

float columnridges(in vec3 p, in float radius, in float height, in float ridgeradius, in int numridges){
    float ridges = 1e10;
    for (int i = 0; i<numridges; i++){
        pR(p.xz, twopi/numridges);
        float ridge = fCylinder(p+vec3(radius,0,0), ridgeradius, height);
        ridges = min(ridges, ridge);
    }
    return ridges;
}

float column(in vec3 p, in float radius, in float height, float baseoffset, float baseheight, in float ridgeradius, in int numridges) {
    float cylinder = fCylinder(p, radius, height);
    if (ridgeradius>0){
        float ridges = columnridges(p, radius, height, ridgeradius, numridges);
        cylinder = fOpDifferenceRound(cylinder, ridges, 0.01);
    }
    float base1 = fCylinder(p-vec3(0,height,0), radius+baseoffset, baseheight);
    float basebox1 = fBox(p-vec3(0,height+2*baseheight,0), vec3(radius+baseoffset,baseheight, radius+baseoffset+0.1));
    base1 = fOpUnionColumns(base1, basebox1, 0.1, 3);
    float base2 = fCylinder(p+vec3(0,height,0), radius+baseoffset, baseheight);
    float basebox2 = fBox(p+vec3(0,height+2*baseheight,0), vec3(radius+baseoffset,baseheight, radius+baseoffset+0.1));
    base2 = fOpUnionColumns(base2, basebox2, 0.1, 3);
    float bases = min(base1, base2);
    return min(cylinder,bases);
}


float sdf(in vec3 p) {
    // What does this do?
    //pMod3(p, vec3(35.0));
    float radius = max(cos(time),0.5);//0.5;
    float height = 3.;
    float baseoffset = max(cos(5*time)*0.15*radius,0.);
    float baseheight = max(cos(3.1*time)*0.05*height,0.)+0.1;
    float ridgeradius = max(cos(3*time)*0.15*radius,0.);
    int numridges = 10;
    float shape1 = column(p, radius, height, baseoffset, baseheight, ridgeradius, numridges);
    //float shape1 = columnridges(p, radius, height, ridgeradius, numridges);
    return shape1;
}
