/* eslint-disable */

export default `struct aG{G:Z,a8:t}fn cY(U:aG,bN:t)->t{return U.G.xy*bN.x+U.G.zw*bN.y+U.a8;}fn fk(U:aG)->aG{let in=1./(U.G.x*U.G.w-U.G.y*U.G.z);let fj=in*vec4(U.G.w,-U.G.y,-U.G.z,U.G.x);let im=mat2x2(fj.xy,fj.zw)*-U.a8;return aG(fj,im);}fn dH(a:aG,b:aG)->aG{return aG(a.G.xyxy*b.G.xxzz+a.G.zwzw*b.G.yyww,a.G.xy*b.a8.x+a.G.zw*b.a8.y+a.a8);}`
