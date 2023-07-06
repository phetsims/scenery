/* eslint-disable */

export default `struct aT{Q:L,b5:B}fn c3(Z:aT,bR:B)->B{return Z.Q.xy*bR.x+Z.Q.zw*bR.y+Z.b5;}fn fs(Z:aT)->aT{let iw=d/(Z.Q.x*Z.Q.w-Z.Q.y*Z.Q.z);let fr=iw*vec4(Z.Q.w,-Z.Q.y,-Z.Q.z,Z.Q.x);let iv=mat2x2(fr.xy,fr.zw)*-Z.b5;return aT(fr,iv);}fn dL(a:aT,b:aT)->aT{return aT(a.Q.xyxy*b.Q.xxzz+a.Q.zwzw*b.Q.yyww,a.Q.xy*b.b5.x+a.Q.zw*b.b5.y+a.b5);}fn gQ(eC:j,p:j)->aT{let aS=eC+p*6u;let Q=bitcast<L>(vec4(u[aS],u[aS+f],u[aS+2u],u[aS+3u]));let b5=bitcast<B>(vec2(u[aS+4u],u[aS+5u]));return aT(Q,b5);}`
