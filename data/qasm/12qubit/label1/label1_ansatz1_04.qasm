OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.03130142388151924) q[0];
rz(-2.2382459619406827) q[0];
ry(-2.7829740581790268) q[1];
rz(3.082670439846175) q[1];
ry(2.768222779855077) q[2];
rz(0.9793076133442352) q[2];
ry(1.8419851493783632) q[3];
rz(1.508097644607358) q[3];
ry(1.8023769207515772) q[4];
rz(-0.21847545575534794) q[4];
ry(-0.5800824772503271) q[5];
rz(-3.1311348920612185) q[5];
ry(-0.6177151285614442) q[6];
rz(-3.1099059846107395) q[6];
ry(1.5445497536140909) q[7];
rz(0.12109367826390069) q[7];
ry(0.8311956746949001) q[8];
rz(1.7766890319936675) q[8];
ry(0.12252530002025205) q[9];
rz(-1.5210945510863356) q[9];
ry(0.35921266536321284) q[10];
rz(0.9644439764904938) q[10];
ry(-1.5250060009170934) q[11];
rz(1.9326418772153149) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-3.0974407780411957) q[0];
rz(2.7544509318362236) q[0];
ry(-0.6950353994923736) q[1];
rz(0.14822341477785478) q[1];
ry(1.6067047861734716) q[2];
rz(1.7856069692045882) q[2];
ry(-0.9995576251852466) q[3];
rz(-1.0485313066042314) q[3];
ry(2.023363504297781) q[4];
rz(3.090468319762398) q[4];
ry(1.7361749587272177) q[5];
rz(-0.0038358414917900336) q[5];
ry(1.5116889632092985) q[6];
rz(0.03938458352447457) q[6];
ry(-1.6388120594148523) q[7];
rz(3.0996198027485) q[7];
ry(0.8248089583819072) q[8];
rz(0.21033629947071653) q[8];
ry(-1.6304073480609982) q[9];
rz(-0.8850654347516671) q[9];
ry(0.20709271479455005) q[10];
rz(2.0319043460825954) q[10];
ry(-0.07944228108238961) q[11];
rz(1.2720631733419419) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.03996691352491499) q[0];
rz(0.20194274001110737) q[0];
ry(1.6527223869160643) q[1];
rz(-0.13728769864337728) q[1];
ry(-2.1523424021726534) q[2];
rz(2.937075647655714) q[2];
ry(-3.0948180669434104) q[3];
rz(2.3905121717653204) q[3];
ry(-1.1851571506302498) q[4];
rz(0.06034192354298796) q[4];
ry(-2.796235273328966) q[5];
rz(-1.620660648422571) q[5];
ry(-0.23862135904313497) q[6];
rz(1.626570630115221) q[6];
ry(2.003055766371403) q[7];
rz(-3.0524337078396457) q[7];
ry(-0.25604432412385375) q[8];
rz(-0.2668517773604515) q[8];
ry(0.619908361649853) q[9];
rz(-1.0748233557132778) q[9];
ry(-1.117484468897029) q[10];
rz(2.676243994916379) q[10];
ry(-1.7386550619996801) q[11];
rz(3.0583576568062143) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.2206469785951226) q[0];
rz(0.5067356711994755) q[0];
ry(-0.9093392935640613) q[1];
rz(-0.6757303882162118) q[1];
ry(-1.6970309502175152) q[2];
rz(2.604140308478331) q[2];
ry(-1.3913406719010153) q[3];
rz(-1.2206454221872052) q[3];
ry(-2.6499011503808867) q[4];
rz(1.5644652297796942) q[4];
ry(-1.5838923010733754) q[5];
rz(-1.9110424688655199) q[5];
ry(-1.6042395158499971) q[6];
rz(1.1769674414836406) q[6];
ry(2.9672448893310404) q[7];
rz(2.4296783767095054) q[7];
ry(-1.372620567878898) q[8];
rz(-3.113945094151172) q[8];
ry(-2.6593859643326003) q[9];
rz(-3.0889948212132037) q[9];
ry(1.2595536389907482) q[10];
rz(-0.1586775366525064) q[10];
ry(-0.14776926787070455) q[11];
rz(0.9425454237085757) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.5419307197594582) q[0];
rz(1.1495750747036608) q[0];
ry(2.675554951633123) q[1];
rz(-0.385129530110305) q[1];
ry(-1.7296446939840608) q[2];
rz(1.5424817471354553) q[2];
ry(-0.003693441254725006) q[3];
rz(-1.9043076468667888) q[3];
ry(0.15297430033410464) q[4];
rz(-1.532517287807574) q[4];
ry(2.9631483155828704) q[5];
rz(-0.02924074071625028) q[5];
ry(0.0723607861726406) q[6];
rz(-2.4564104208719124) q[6];
ry(0.008259875665713068) q[7];
rz(0.8139112794046036) q[7];
ry(-0.25779130639600467) q[8];
rz(-2.940479521146085) q[8];
ry(1.5303071685102678) q[9];
rz(-3.009266856100484) q[9];
ry(2.365457176288741) q[10];
rz(0.3638048519635966) q[10];
ry(2.703609103137846) q[11];
rz(-2.9457137709518086) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.2813663308144587) q[0];
rz(-1.813156449280772) q[0];
ry(1.4819128610507253) q[1];
rz(-1.7333616295778411) q[1];
ry(-1.497346468974436) q[2];
rz(2.292417747481579) q[2];
ry(2.1017157542686045) q[3];
rz(-1.604625954334713) q[3];
ry(-1.37652760021256) q[4];
rz(3.1410418994696054) q[4];
ry(1.618876409858753) q[5];
rz(1.5921259602297848) q[5];
ry(1.6362085322073145) q[6];
rz(-1.5204757587434177) q[6];
ry(-1.498620215649055) q[7];
rz(-2.9223177044450637) q[7];
ry(0.27199893107158485) q[8];
rz(2.9837428210160724) q[8];
ry(2.8781673536290437) q[9];
rz(-3.1073729417567506) q[9];
ry(1.4002572379866205) q[10];
rz(-0.43150062524453414) q[10];
ry(1.104339734569626) q[11];
rz(1.5466093397644824) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.6242054806300625) q[0];
rz(1.5517043367090153) q[0];
ry(-0.08769621054364188) q[1];
rz(0.15026426239071555) q[1];
ry(-3.140345679334426) q[2];
rz(-0.6862844469652363) q[2];
ry(-3.036951906710794) q[3];
rz(-0.07493847497871232) q[3];
ry(-1.106470748509616) q[4];
rz(-1.5258422036720856) q[4];
ry(-0.3480295086553436) q[5];
rz(1.5812592728937356) q[5];
ry(-0.3286251857982914) q[6];
rz(1.5599285831738465) q[6];
ry(0.09364047186883476) q[7];
rz(1.3177275052396888) q[7];
ry(-3.087875745999893) q[8];
rz(-1.5783280048722217) q[8];
ry(-1.1442033078745597) q[9];
rz(-1.3809689947716688) q[9];
ry(-2.870794460744076) q[10];
rz(-1.7434201944468157) q[10];
ry(1.4314273172093364) q[11];
rz(1.5019773986949678) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.5450624030051585) q[0];
rz(-1.3479939797215275) q[0];
ry(1.5660933262137993) q[1];
rz(1.3098326120894201) q[1];
ry(3.05783513961741) q[2];
rz(-0.789326217841656) q[2];
ry(-1.551123599823673) q[3];
rz(2.40361040591094) q[3];
ry(-1.5825893079577302) q[4];
rz(-0.9217970668187003) q[4];
ry(1.6009307985742607) q[5];
rz(-2.2829861669592795) q[5];
ry(-1.5827448597030473) q[6];
rz(-2.8218281912415217) q[6];
ry(-1.5946362502874685) q[7];
rz(-0.48228648307045496) q[7];
ry(1.6485357693867586) q[8];
rz(1.1717351613667168) q[8];
ry(-1.5867493575508664) q[9];
rz(2.836942620679163) q[9];
ry(-1.5648796087148131) q[10];
rz(-3.114774340139328) q[10];
ry(-1.4711584895104945) q[11];
rz(-1.5507491379790466) q[11];