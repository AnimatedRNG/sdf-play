const float twopi = 6.283;

float torus(in vec3 p) {
    vec2 q = vec2(length(p.xz) - 3.0, p.y);
    return length(q) - 2.0;
}

float columnridges(in vec3 p, in float radius, in float height,
                   in float ridgeradius, in int numridges,
                   float baseheight){
    float ridges = 1e10;
    for (int i = 0; i<numridges; i++){
        pR(p.xz, twopi/numridges);
        float ridge = fCapsule(p+vec3(radius,0,0), ridgeradius, height-2.3*baseheight);
        ridges = min(ridges, ridge);
    }
    return ridges;
}

float column(in vec3 p, in float radius, in float height,
            float baseoffset, float baseheight,
            in float ridgeradius, in int numridges) {
    float cylinder = fCylinder(p, radius, height);

    if (ridgeradius>0){
        float ridges = columnridges(p, radius, height, ridgeradius, numridges, baseheight);
        cylinder = fOpDifferenceRound(cylinder, ridges, 0.01);
    }
    float base1 = fCylinder(p-vec3(0,height,0), radius+baseoffset, baseheight);
    float basebox1 = fBox(p-vec3(0,height+2*baseheight,0), vec3(radius+baseoffset+0.1,baseheight, radius+baseoffset+0.1));
    base1 = fOpUnionColumns(base1, basebox1, 0.1, 3);
    float base2 = fCylinder(p+vec3(0,height,0), radius+baseoffset, baseheight);
    float basebox2 = fBox(p+vec3(0,height+2*baseheight,0), vec3(radius+baseoffset+0.1,baseheight, radius+baseoffset+0.1));
    base2 = fOpUnionColumns(base2, basebox2, 0.1, 3);
    float bases = min(base1, base2);
    return fOpUnionColumns(cylinder,bases, 0.1, 3);
}

float random (vec3 st) {
    return fract(sin(dot(st.xy,
                         vec2(12.9898,78.233)))*
        43758.5453123);
}


float window(in vec3 p, in float wall, in float width, in float height, in float depth,
             in bool arc, in int widthdivisions, in int heightdivisions, in float dividerthickness){
    float indent = fBox(p, vec3(width, height, depth));
    if (arc&&height>width){
        pR(p.yz, 3.1414/2);
        float arc = fCylinder(p+vec3(0,0,height),width, depth);
        pR(p.yz, -3.1414/2);
        indent = min(indent, arc);
    }
    float dividers = 1e10;
    vec3 prepeat = p+vec3(0,0,depth-dividerthickness/2);
    pMod2(prepeat.xy, vec2(width/widthdivisions, height/heightdivisions));

    float widthbars=fBox(prepeat,vec3(width/widthdivisions,dividerthickness, dividerthickness));
    float heightbars=fBox(prepeat,vec3(dividerthickness,height/heightdivisions, dividerthickness));
    indent = max(indent,-widthbars);
    indent = max(indent, -heightbars);
    //return indent;
    //return wall;
    //return dividers;
    return indent;//fOpDifferenceStairs(wall,indent,max(depth*0.3*cos(0.1*time),0.1),3);
}

float subtractblock(in vec3 p, in float width, in float height, in float length, float windowwidth, float windowheight, float windowdepth){
    float block = fBox(p, vec3(width, height, length));
    bool arc = true;
    int heightdivisions = 6;
    int widthdivisions = 3;

    float dividerthickness = 0.05;
    float betweenwidth = width;
    float windows = 1e10;
    for (int i=0; i<4; i++){
        pR(p.xz, i*3.1414/2);
        vec3 prepeat = p+vec3(0,+float(arc)*windowwidth/2,+length-windowdepth);

        pMod2(prepeat.xy, vec2(windowwidth*3, (windowheight+windowwidth*float(arc))*2));
        windows = min(windows, window(prepeat, block, windowwidth, windowheight, windowdepth,
                               arc, widthdivisions, heightdivisions, dividerthickness));

    }
    //return max(windows, block);
    float subblock = fBox(p, vec3(width-windowdepth, height, length-windowdepth));
    return min(max(windows, block),subblock);
}

float subtractcolumnblock(in vec3 p, in float width, in float height, in float length){
    float columns = 1e10;
    for (int i=0; i<4; i++){
        pR(p.xz, 3.1414/2);
        vec3 prepeat = p+vec3(width/2,0,length/2);

        pMod2(prepeat.xy, vec2(5,height*2));

        columns = min(columns,column(prepeat,1,height,0.2,0.3,0.05,20));
    }
            pR(p.xz, 3.1414/2);
            //return columns;
    columns = max(columns, fBox(p,vec3(width,height,length)));
    //return columns;
    return max(fBox(p, vec3(width, height, length)),-columns);
}

float sdf(in vec3 p) {
     p = p + vec3(2,-3,15);
    // What does this do?
    //pMod3(p, vec3(35.0));
    float radius = max(cos(time),0.5);//0.5;
    float height = 3.;
    float baseoffset = max(cos(5*time)*0.15*radius,0.);
    float baseheight = max(cos(3.1*time)*0.05*height,0.)+0.1;
    float ridgeradius = max(cos(3*time)*0.1*radius,0.);
    int numridges = int(max(cos(0.5*time)*10+10,0));
    float columnobj = column(p, radius, height, baseoffset, baseheight, ridgeradius, numridges);
    //float shape1 = columnridges(p, radius, height, ridgeradius, numridges);
    //return columnobj;
    float windowheight = max(cos(time)*3,1);
    float windowwidth = max(sin(0.9*time)*5,2);
    float windowdepth = max(cos(0.8*time)*1,0.5);
    bool arc = cos(time*1.2)>0;
    float divisionwidth = max(cos(0.7*time)*windowwidth,0.5);
    float divisionheight =max(cos(0.6*time)*windowheight,0.2);
    int widthdivisions = int(windowwidth/divisionwidth)-1;
    int heightdivisions = int(windowheight/divisionheight)-1;
    float dividerthickness = max(cos(time*0.5)*windowdepth/2,0.05);
    float wall = fBox(p+vec3(0,0,3), vec3(6,8,3));
    float windowobj = window(p, wall, windowwidth, windowheight, windowdepth, arc, widthdivisions, heightdivisions, dividerthickness);
    //return windowobj;
    //return wall;
    //return subtractblock(p-vec3(0,0,5));
    float building = fBox(p,vec3(30,16,5));
    //building = max(building,-subtractblock(p-vec3(13,5.5,5),8,18,8,1,1.5,0.5));

    //building = max(building, -subtractblock(p-vec3(-18,5.5,5),8,12.5,8,2,3,0.5));
    //building = max(building, -subtractblock(p-vec3(0,-2,12),20,20,8,2,1,0.5));
    //return building;
    //return subtractcolumnblock(p, 20, 10, 20);
    building =  max(building,-subtractcolumnblock(p, 15, 8, 15));
    building = max(building, -subtractblock(p-vec3(0,-2,12),20,20,8,2,2,0.5));

    //return building;
    float simmons = fBox(p-vec3(0,0,3), vec3(20,7,4));
    simmons = max(simmons, -fBox(p-vec3(-8.5,5,0), vec3(4,3,8)));
    simmons = max(simmons, -fBox(p-vec3(8,4.5,0), vec3(2,2.6,8)));
    simmons = max(simmons, -fBox(p-vec3(-18.5,-0.5,8), vec3(1.6,1.6,2.5)));
    simmons = max(simmons, -fBox(p-vec3(6,-4.5,0), vec3(2,2.6,8)));
    simmons = max(simmons, -fBox(p-vec3(18.5,1,8), vec3(4,1.5,2.01)));
    simmons = max(simmons, -fBox(p-vec3(-10.5,-6,8.7), vec3(4,3,2.01)));
    simmons = max(simmons, -fBox(p-vec3(-1.5,-7,8.7), vec3(3,2,2.01)));
    simmons = max(simmons, -fBox(p-vec3(-1.5,3,8), vec3(1.5,2,2.01)));
    simmons = max(simmons, -subtractblock(p-vec3(0,0,8.7),20,7,2,0.13,0.13,0.4));
    return simmons;
}
