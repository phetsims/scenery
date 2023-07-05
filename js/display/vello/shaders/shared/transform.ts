/* eslint-disable */

export default `struct aL{K:af,bh:B}fn c1(Y:aL,bT:B)->B{return Y.K.xy*bT.x+Y.K.zw*bT.y+Y.bh;}fn fp(Y:aL)->aL{let ir=d/(Y.K.x*Y.K.w-Y.K.y*Y.K.z);let fo=ir*vec4(Y.K.w,-Y.K.y,-Y.K.z,Y.K.x);let iq=mat2x2(fo.xy,fo.zw)*-Y.bh;return aL(fo,iq);}fn dL(a:aL,b:aL)->aL{return aL(a.K.xyxy*b.K.xxzz+a.K.zwzw*b.K.yyww,a.K.xy*b.bh.x+a.K.zw*b.bh.y+a.bh);}`
