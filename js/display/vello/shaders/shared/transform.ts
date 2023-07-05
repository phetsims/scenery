/* eslint-disable */

export default `struct aK{K:af,be:B}fn c1(Y:aK,bR:B)->B{return Y.K.xy*bR.x+Y.K.zw*bR.y+Y.be;}fn fp(Y:aK)->aK{let ir=d/(Y.K.x*Y.K.w-Y.K.y*Y.K.z);let fo=ir*vec4(Y.K.w,-Y.K.y,-Y.K.z,Y.K.x);let iq=mat2x2(fo.xy,fo.zw)*-Y.be;return aK(fo,iq);}fn dL(a:aK,b:aK)->aK{return aK(a.K.xyxy*b.K.xxzz+a.K.zwzw*b.K.yyww,a.K.xy*b.be.x+a.K.zw*b.be.y+a.be);}`
